"""Child-process entry point for single-image SqueakPose prediction."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
from typing import Any, Callable, Optional

from prediction_ops import serialize_prediction_result
from squeakpose_core import model_task_mismatch_message
from squeakpose.workers.protocol import read_config, write_event

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    write_event(payload)


def _load_prediction_dependencies(
    model_factory: Optional[Callable[[str], Any]],
    cv2_module: Any = None,
    numpy_module: Any = None,
) -> tuple[Optional[Callable[[str], Any]], Any, Any, str]:
    if model_factory is None:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from ultralytics import YOLO
        except Exception as exc:
            return None, cv2_module, numpy_module, f"Could not import ultralytics YOLO: {exc}"
        model_factory = YOLO
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except Exception:
            cv2_module = None
    if numpy_module is None:
        try:
            import numpy as numpy_module
        except Exception:
            numpy_module = None
    return model_factory, cv2_module, numpy_module, ""


def run_predict_server(
    request_lines,
    *,
    model_factory: Optional[Callable[[str], Any]] = None,
    cv2_module: Any = None,
    numpy_module: Any = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    """Serve newline-delimited prediction requests while keeping one model warm."""
    factory, cv2_module, numpy_module, dependency_error = _load_prediction_dependencies(
        model_factory,
        cv2_module,
        numpy_module,
    )
    if factory is None:
        event_writer({"event": "error", "request_id": None, "error_message": dependency_error})
        return 1

    cached_model = None
    cached_model_path = ""
    event_writer({"event": "ready"})

    for raw_line in request_lines:
        line = str(raw_line).strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            event_writer({"event": "error", "request_id": None, "error_message": f"Invalid request: {exc}"})
            continue

        request_id = request.get("request_id")
        command = str(request.get("command") or "predict").lower()
        if command == "shutdown":
            event_writer({"event": "stopped", "request_id": request_id})
            return 0

        model_path = str(request.get("model_path") or "")
        workflow = str(request.get("workflow") or "pose")
        layer_id = str(request.get("layer_id") or workflow)
        device = str(request.get("device") or "cpu")
        image_path = str(request.get("image_path") or "")
        if not model_path:
            event_writer({"event": "error", "request_id": request_id, "error_message": "model_path is required"})
            continue
        if command == "predict" and not image_path:
            event_writer({"event": "error", "request_id": request_id, "error_message": "image_path is required"})
            continue

        try:
            if cached_model is None or cached_model_path != model_path:
                event_writer({"event": "loading", "request_id": request_id, "model_path": model_path})
                with contextlib.redirect_stdout(sys.stderr):
                    cached_model = factory(model_path)
                cached_model_path = model_path

            task_error = model_task_mismatch_message(
                getattr(cached_model, "task", None),
                workflow,
                subject="Prediction model",
            )
            if task_error:
                event_writer({"event": "error", "request_id": request_id, "error_message": task_error})
                continue

            if command == "load":
                # Run no dummy prediction here; the first real image initializes
                # device-specific predictor state without guessing an input shape.
                event_writer({"event": "loaded", "request_id": request_id, "model_path": model_path})
                continue
            if command != "predict":
                event_writer(
                    {
                        "event": "error",
                        "request_id": request_id,
                        "error_message": f"Unsupported command: {command}",
                    }
                )
                continue

            event_writer({"event": "started", "request_id": request_id, "image_path": image_path})
            with contextlib.redirect_stdout(sys.stderr):
                results_list = cached_model.predict(
                    source=image_path,
                    imgsz=640,
                    conf=0.25,
                    iou=0.5,
                    device=device,
                    verbose=False,
                )
            results = list(results_list or [])
            if not results:
                raise RuntimeError("Prediction returned no results.")
            prediction = serialize_prediction_result(
                results[0],
                workflow=workflow,
                layer_id=layer_id,
                cv2_module=cv2_module,
                numpy_module=numpy_module,
            )
            event_writer(
                {
                    "event": "result",
                    "request_id": request_id,
                    "canceled": False,
                    "had_error": False,
                    "error_message": "",
                    "prediction": prediction,
                }
            )
        except Exception as exc:
            event_writer(
                {
                    "event": "result",
                    "request_id": request_id,
                    "canceled": False,
                    "had_error": True,
                    "error_message": str(exc),
                    "prediction": None,
                }
            )
    return 0


def run_predict_worker(
    config: dict[str, Any],
    *,
    model_factory: Optional[Callable[[str], Any]] = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = False

    model_path = str(config.get("model_path") or "")
    image_path = str(config.get("image_path") or "")
    workflow = str(config.get("workflow") or "pose")
    layer_id = str(config.get("layer_id") or workflow)
    device = str(config.get("device") or "cpu")

    if not model_path:
        event_writer({"event": "error", "error_message": "model_path is required"})
        return 1
    if not image_path:
        event_writer({"event": "error", "error_message": "image_path is required"})
        return 1

    model_factory, cv2_module, numpy_module, dependency_error = _load_prediction_dependencies(model_factory)
    if model_factory is None:
        event_writer({"event": "error", "error_message": dependency_error})
        return 1

    event_writer({"event": "started", "image_path": image_path})
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = model_factory(model_path)
            task_error = model_task_mismatch_message(
                getattr(model, "task", None),
                workflow,
                subject="Prediction model",
            )
            if task_error:
                event_writer({"event": "error", "error_message": task_error})
                return 1
            results_list = model.predict(
                source=image_path,
                imgsz=640,
                conf=0.25,
                iou=0.5,
                device=device,
                verbose=False,
            )
        if _CANCEL_REQUESTED:
            event_writer({"event": "result", "canceled": True, "had_error": False, "prediction": None})
            return 0
        results = list(results_list or [])
        if not results:
            event_writer(
                {
                    "event": "result",
                    "canceled": False,
                    "had_error": True,
                    "error_message": "Prediction returned no results.",
                    "prediction": None,
                }
            )
            return 1
        prediction = serialize_prediction_result(
            results[0],
            workflow=workflow,
            layer_id=layer_id,
            cv2_module=cv2_module,
            numpy_module=numpy_module,
        )
        event_writer(
            {
                "event": "result",
                "canceled": False,
                "had_error": False,
                "error_message": "",
                "prediction": prediction,
            }
        )
        return 0
    except Exception as exc:
        event_writer(
            {
                "event": "result",
                "canceled": bool(_CANCEL_REQUESTED),
                "had_error": not _CANCEL_REQUESTED,
                "error_message": "" if _CANCEL_REQUESTED else str(exc),
                "prediction": None,
            }
        )
        return 1 if not _CANCEL_REQUESTED else 0


def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _handle_cancel_signal)
    signal.signal(signal.SIGINT, _handle_cancel_signal)

    parser = argparse.ArgumentParser(description="Run SqueakPose single-image prediction in a child process.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="Path to JSON prediction config.")
    mode.add_argument("--server", action="store_true", help="Serve newline-delimited requests on stdin.")
    args = parser.parse_args(argv)

    if args.server:
        return run_predict_server(sys.stdin)

    try:
        config = read_config(args.config)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_predict_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
