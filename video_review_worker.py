"""Child-process entry point for Video Reviewer range prediction."""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from typing import Any, Callable, Optional

from prediction_ops import best_predictions_by_class_from_payload, serialize_prediction_result, top_prediction_from_payload
from squeakpose_core import model_task_mismatch_message
from squeakpose.workers.protocol import read_config, write_event

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    write_event(payload)


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


def _is_device_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "mps backend out of memory" in text
    )


def _clear_device_cache(device: str) -> None:
    try:
        import torch

        normalized = (device or "").lower()
        if normalized.startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        elif normalized == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


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
    layer_id = str(config.get("layer_id") or workflow)
    device = str(config.get("device") or "cpu")
    start = max(0, int(config.get("start") or 0))
    end = max(start, int(config.get("end") if config.get("end") is not None else start))
    stride = max(1, int(config.get("stride") or 1))
    imgsz = int(config.get("imgsz") or 640)
    conf = float(config.get("conf") if config.get("conf") is not None else 0.25)
    iou = float(config.get("iou") if config.get("iou") is not None else 0.5)
    requested_batch = int(config.get("batch") if config.get("batch") is not None else 0)
    effective_batch = int(config.get("effective_batch") if config.get("effective_batch") is not None else 1)
    effective_batch = max(1, effective_batch)
    auto_batch = requested_batch <= 0 and (device or "").lower().split(":", 1)[0] in {"cuda", "mps"}
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

        task_error = model_task_mismatch_message(
            getattr(model, "task", None),
            workflow,
            subject="Video review model",
        )
        if task_error:
            event_writer({"event": "error", "error_message": task_error})
            return 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        frames: list[Any] = []
        frame_indices: list[int] = []

        def flush_frames() -> None:
            nonlocal processed, effective_batch
            offset = 0
            while offset < len(frames) and not _CANCEL_REQUESTED:
                chunk_size = min(effective_batch, len(frames) - offset)
                chunk_frames = frames[offset : offset + chunk_size]
                chunk_indices = frame_indices[offset : offset + chunk_size]
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        results_list = model.predict(
                            source=chunk_frames,
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            end2end=False,
                            device=device,
                            batch=chunk_size,
                            verbose=False,
                        )
                    results = list(results_list or [])
                    if len(results) != len(chunk_frames):
                        raise RuntimeError(
                            f"Prediction returned {len(results)} results for {len(chunk_frames)} frames."
                        )
                    completed: dict[str, dict[str, Any]] = {}
                    for frame_idx, result in zip(chunk_indices, results):
                        payload = serialize_prediction_result(
                            result,
                            workflow=workflow,
                            layer_id=layer_id,
                            cv2_module=cv2,
                            numpy_module=numpy_module,
                        )
                        prediction = top_prediction_from_payload(payload, workflow=workflow)
                        prediction["detections"] = best_predictions_by_class_from_payload(
                            payload,
                            workflow=workflow,
                        )
                        preds[frame_idx] = prediction
                        completed[str(frame_idx)] = prediction
                except Exception as exc:
                    if auto_batch and chunk_size > 1 and _is_device_memory_error(exc):
                        previous_batch = effective_batch
                        effective_batch = max(1, chunk_size // 2)
                        _clear_device_cache(device)
                        event_writer(
                            {
                                "event": "batch_adjusted",
                                "previous_batch": previous_batch,
                                "effective_batch": effective_batch,
                                "message": (
                                    f"Device memory limit: reducing batch "
                                    f"{previous_batch} → {effective_batch}"
                                ),
                            }
                        )
                        continue
                    error_text = str(exc)
                    completed = {}
                    for frame_idx in chunk_indices:
                        prediction = {"ok": False, "error": error_text}
                        preds[frame_idx] = prediction
                        completed[str(frame_idx)] = prediction
                    prediction_errors.append(
                        f"frames {chunk_indices[0]}-{chunk_indices[-1]}: {error_text}"
                    )

                processed += chunk_size
                event_writer(
                    {
                        "event": "progress",
                        "processed": processed,
                        "total": total_steps,
                        "message": (
                            f"Predicted frames {chunk_indices[0]}-{chunk_indices[-1]} "
                            f"(batch {chunk_size})"
                        ),
                        "effective_batch": effective_batch,
                        "predictions": completed,
                    }
                )
                offset += chunk_size

        while idx <= end:
            if _CANCEL_REQUESTED:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            frame_indices.append(idx)

            if len(frames) >= effective_batch:
                flush_frames()
                frames = []
                frame_indices = []

            if _CANCEL_REQUESTED:
                break
            next_idx = idx + stride
            for _ in range(stride - 1):
                if next_idx > end or not cap.grab():
                    next_idx = end + 1
                    break
            idx = next_idx

        if frames and not _CANCEL_REQUESTED:
            flush_frames()

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
                # Predictions are streamed in bounded progress events as each
                # batch completes. Re-emitting a large range here can create a
                # final JSON line hundreds of megabytes long and prevent the Qt
                # process from ever reaching its finished handler.
                "preds": {},
                "preds_streamed": True,
                "prediction_count": len(preds),
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
                "preds": {},
                "preds_streamed": True,
                "prediction_count": len(preds),
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
        config = read_config(args.config)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_video_review_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
