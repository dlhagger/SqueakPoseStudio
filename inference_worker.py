"""Child-process entry point for SqueakPose video inference."""

from __future__ import annotations

import argparse
import os
import signal
from typing import Any, Callable, Optional

from squeakpose.core import model_task_mismatch_message
from squeakpose.project.layers import normalize_layer_id
from squeakpose.services.inference_runtime import (
    run_depth_video_inference,
    run_pose_video_inference,
    run_segmentation_video_inference,
)
from squeakpose.workers.protocol import read_config, write_event

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _cancel_requested() -> bool:
    return _CANCEL_REQUESTED


def _emit_event(event_writer: Callable[[dict[str, Any]], None], payload: dict[str, Any]) -> None:
    event_writer(payload)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    write_event(payload)


def run_inference_worker(
    config: dict[str, Any],
    *,
    model_factory: Optional[Callable[[str], Any]] = None,
    cv2_module: Any = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    """Run one inference job and emit JSON-compatible event payloads."""
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = False

    mode = str(config.get("mode", "pose")).lower()
    layer_id = normalize_layer_id(config.get("layer_id") or mode)
    model_path = str(config.get("model_path") or "")
    video_path = str(config.get("video_path") or "")
    csv_path = str(config.get("csv_path") or "")
    classes = list(config.get("classes") or [])
    kp_names = list(config.get("kp_names") or [])
    device = str(config.get("device") or "cpu")
    batch_size = int(config.get("batch_size") or 1)
    total_frames = int(config.get("total_frames") or 0)
    fps = float(config.get("fps") or 0.0)

    if not model_path:
        _emit_event(event_writer, {"event": "error", "error_message": "model_path is required"})
        return 1
    if not video_path:
        _emit_event(event_writer, {"event": "error", "error_message": "video_path is required"})
        return 1
    if not csv_path:
        _emit_event(event_writer, {"event": "error", "error_message": "csv_path is required"})
        return 1

    if model_factory is None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            _emit_event(
                event_writer,
                {"event": "error", "error_message": f"Could not import ultralytics YOLO: {exc}"},
            )
            return 1
        model_factory = YOLO

    if cv2_module is None and mode in {"pose", "depth"}:
        try:
            import cv2 as cv2_module
        except Exception as exc:
            _emit_event(
                event_writer, {"event": "error", "error_message": f"Could not import OpenCV: {exc}"}
            )
            return 1

    _emit_event(
        event_writer,
        {
            "event": "started",
            "csv_path": csv_path,
            "mode": mode,
            "layer_id": layer_id,
        },
    )
    try:
        model = model_factory(model_path)
    except Exception as exc:
        _emit_event(
            event_writer, {"event": "error", "error_message": f"Could not load model: {exc}"}
        )
        return 1

    task_error = model_task_mismatch_message(
        getattr(model, "task", None),
        mode,
        subject="Inference model",
    )
    if task_error:
        _emit_event(event_writer, {"event": "error", "error_message": task_error})
        return 1

    def progress(processed_frames: int, total: int, message: str) -> None:
        _emit_event(
            event_writer,
            {
                "event": "progress",
                "processed_frames": int(processed_frames),
                "total_frames": int(total),
                "message": str(message),
            },
        )

    if mode == "depth":
        try:
            import numpy as np
        except Exception as exc:
            _emit_event(
                event_writer, {"event": "error", "error_message": f"Could not import NumPy: {exc}"}
            )
            return 1
        preview_path = str(config.get("preview_path") or "")
        if not preview_path:
            root, _ext = os.path.splitext(csv_path)
            preview_path = root + "_preview.mp4"
        result = run_depth_video_inference(
            model=model,
            cv2_module=cv2_module,
            numpy_module=np,
            video_path=video_path,
            csv_path=csv_path,
            preview_path=preview_path,
            model_path=model_path,
            device=device,
            total_frames=total_frames,
            fps=fps,
            progress_callback=progress,
            cancel_requested=_cancel_requested,
        )
    elif mode in {"segment", "segmentation"}:
        result = run_segmentation_video_inference(
            model=model,
            video_path=video_path,
            csv_path=csv_path,
            classes=classes,
            device=device,
            total_frames=total_frames,
            progress_callback=progress,
            cancel_requested=_cancel_requested,
        )
    else:
        result = run_pose_video_inference(
            model=model,
            cv2_module=cv2_module,
            video_path=video_path,
            csv_path=csv_path,
            model_path=model_path,
            classes=classes,
            kp_names=kp_names,
            device=device,
            batch_size=batch_size,
            total_frames=total_frames,
            fps=fps,
            progress_callback=progress,
            cancel_requested=_cancel_requested,
        )

    _emit_event(
        event_writer,
        {
            "event": "result",
            "csv_path": result.csv_path,
            "preview_path": result.preview_path,
            "rows_written": int(result.rows_written),
            "processed_frames": int(result.processed_frames),
            "canceled": bool(result.canceled),
            "had_error": bool(result.had_error),
            "error_message": str(result.error_message or ""),
            "mode": mode,
            "layer_id": layer_id,
        },
    )
    return 1 if result.had_error else 0


def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _handle_cancel_signal)
    signal.signal(signal.SIGINT, _handle_cancel_signal)

    parser = argparse.ArgumentParser(
        description="Run SqueakPose video inference in a child process."
    )
    parser.add_argument("--config", required=True, help="Path to JSON inference config.")
    args = parser.parse_args(argv)

    try:
        config = read_config(args.config)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_inference_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
