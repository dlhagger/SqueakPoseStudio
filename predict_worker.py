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

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


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
    device = str(config.get("device") or "cpu")

    if not model_path:
        event_writer({"event": "error", "error_message": "model_path is required"})
        return 1
    if not image_path:
        event_writer({"event": "error", "error_message": "image_path is required"})
        return 1

    if model_factory is None:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from ultralytics import YOLO
        except Exception as exc:
            event_writer({"event": "error", "error_message": f"Could not import ultralytics YOLO: {exc}"})
            return 1
        model_factory = YOLO

    cv2_module = None
    numpy_module = None
    try:
        import cv2 as cv2_module
    except Exception:
        cv2_module = None
    try:
        import numpy as numpy_module
    except Exception:
        numpy_module = None

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
    parser.add_argument("--config", required=True, help="Path to JSON prediction config.")
    args = parser.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_predict_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
