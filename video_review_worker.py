"""Child-process entry point for Video Reviewer range prediction."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
from typing import Any, Callable, Optional

from prediction_ops import serialize_prediction_result, top_prediction_from_payload

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _load_model_factory(model_factory: Optional[Callable[[str], Any]]) -> Optional[Callable[[str], Any]]:
    if model_factory is not None:
        return model_factory
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from ultralytics import YOLO
    except Exception:
        return None
    return YOLO


def _load_cv2(cv2_module: Any = None) -> Any:
    if cv2_module is not None:
        return cv2_module
    try:
        import cv2
    except Exception:
        return None
    return cv2


def run_video_review_worker(
    config: dict[str, Any],
    *,
    model_factory: Optional[Callable[[str], Any]] = None,
    cv2_module: Any = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = False

    model_path = str(config.get("model_path") or "")
    video_path = str(config.get("video_path") or "")
    workflow = str(config.get("workflow") or "pose")
    device = str(config.get("device") or "cpu")
    start = max(0, int(config.get("start") or 0))
    end = max(start, int(config.get("end") if config.get("end") is not None else start))
    stride = max(1, int(config.get("stride") or 1))
    imgsz = int(config.get("imgsz") or 640)
    conf = float(config.get("conf") if config.get("conf") is not None else 0.25)
    iou = float(config.get("iou") if config.get("iou") is not None else 0.5)
    requested_batch = int(config.get("batch") if config.get("batch") is not None else 1)
    effective_batch = int(config.get("effective_batch") if config.get("effective_batch") is not None else requested_batch)
    effective_batch = max(1, effective_batch)
    batch_kwargs = {} if requested_batch <= 0 else {"batch": requested_batch}

    if not model_path:
        event_writer({"event": "error", "error_message": "model_path is required"})
        return 1
    if not video_path:
        event_writer({"event": "error", "error_message": "video_path is required"})
        return 1

    cv2 = _load_cv2(cv2_module)
    if cv2 is None:
        event_writer({"event": "error", "error_message": "Could not import OpenCV"})
        return 1

    factory = _load_model_factory(model_factory)
    if factory is None:
        event_writer({"event": "error", "error_message": "Could not import ultralytics YOLO"})
        return 1

    numpy_module = None
    try:
        import numpy as numpy_module
    except Exception:
        numpy_module = None

    cap = cv2.VideoCapture(video_path)
    if cap is None or not cap.isOpened():
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        event_writer({"event": "error", "error_message": f"Could not open video: {video_path}"})
        return 1

    total_steps = max(1, ((end - start) // stride) + 1)
    preds: dict[int, dict[str, Any]] = {}
    prediction_errors: list[str] = []
    processed = 0
    event_writer({"event": "started", "video_path": video_path, "total": total_steps})

    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = factory(model_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        frames: list[Any] = []
        frame_indices: list[int] = []

        while idx <= end:
            if _CANCEL_REQUESTED:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frames.append(frame)
            frame_indices.append(idx)

            if len(frames) >= effective_batch or (idx + stride) > end:
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        results_list = model.predict(
                            source=frames,
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            device=device,
                            verbose=False,
                            **batch_kwargs,
                        )
                    results = list(results_list or [])
                    for pos, fi in enumerate(frame_indices):
                        if pos >= len(results):
                            err = "Prediction returned fewer results than frames."
                            preds[fi] = {"ok": False, "error": err}
                            prediction_errors.append(f"frame {fi}: {err}")
                            continue
                        payload = serialize_prediction_result(
                            results[pos],
                            workflow=workflow,
                            cv2_module=cv2,
                            numpy_module=numpy_module,
                        )
                        preds[fi] = top_prediction_from_payload(payload, workflow=workflow)
                except Exception as exc:
                    error_text = str(exc)
                    for fi in frame_indices:
                        preds[fi] = {"ok": False, "error": error_text}
                    prediction_errors.append(f"frames {frame_indices[0]}-{frame_indices[-1]}: {error_text}")

                processed += len(frame_indices)
                event_writer(
                    {
                        "event": "progress",
                        "processed": processed,
                        "total": total_steps,
                        "message": f"Predicting frames {frame_indices[0]}-{frame_indices[-1]}",
                    }
                )
                frames = []
                frame_indices = []

            idx += stride
            if idx <= end:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        had_error = bool(prediction_errors) and not _CANCEL_REQUESTED
        error_message = "; ".join(prediction_errors[:3])
        if len(prediction_errors) > 3:
            error_message += f"; ...{len(prediction_errors) - 3} more"
        event_writer(
            {
                "event": "result",
                "canceled": bool(_CANCEL_REQUESTED),
                "had_error": had_error,
                "error_message": error_message,
                "preds": {str(k): v for k, v in preds.items()},
            }
        )
        return 1 if had_error else 0
    except Exception as exc:
        event_writer(
            {
                "event": "result",
                "canceled": bool(_CANCEL_REQUESTED),
                "had_error": not _CANCEL_REQUESTED,
                "error_message": "" if _CANCEL_REQUESTED else str(exc),
                "preds": {str(k): v for k, v in preds.items()},
            }
        )
        return 1 if not _CANCEL_REQUESTED else 0
    finally:
        try:
            cap.release()
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _handle_cancel_signal)
    signal.signal(signal.SIGINT, _handle_cancel_signal)

    parser = argparse.ArgumentParser(description="Run SqueakPose Video Reviewer prediction in a child process.")
    parser.add_argument("--config", required=True, help="Path to JSON video review prediction config.")
    args = parser.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_video_review_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
