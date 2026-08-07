"""Child-process entry point for SqueakPose YOLO training."""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from typing import Any, Callable, Optional

from layer_ops import normalize_layer_id
from squeakpose.workers.protocol import read_config, write_event
from squeakpose_core import model_task_mismatch_message

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    write_event(payload)


def _emit_event(event_writer: Callable[[dict[str, Any]], None], payload: dict[str, Any]) -> None:
    event_writer(payload)


def run_training_worker(
    config: dict[str, Any],
    *,
    model_factory: Optional[Callable[[str], Any]] = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    """Run one YOLO training job and emit JSON-compatible event payloads."""
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = False

    model_cfg = str(config.get("model_cfg") or "")
    layer_id = normalize_layer_id(
        config.get("layer_id") or dict(config.get("params") or {}).get("task")
    )
    params = dict(config.get("params") or {})
    if not model_cfg:
        _emit_event(event_writer, {"event": "error", "error_message": "model_cfg is required"})
        return 1

    if model_factory is None:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from ultralytics import YOLO
        except Exception as exc:
            _emit_event(
                event_writer,
                {"event": "error", "error_message": f"Could not import ultralytics YOLO: {exc}"},
            )
            return 1
        model_factory = YOLO

    _emit_event(
        event_writer,
        {"event": "started", "model_cfg": model_cfg, "layer_id": layer_id},
    )
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = model_factory(model_cfg)
    except Exception as exc:
        _emit_event(
            event_writer,
            {
                "event": "error",
                "error_message": f"Could not load model config '{model_cfg}': {exc}",
            },
        )
        return 1

    task_error = model_task_mismatch_message(
        getattr(model, "task", None),
        params.get("task"),
        subject="Training model",
    )
    if task_error:
        _emit_event(event_writer, {"event": "error", "error_message": task_error})
        return 1

    if _CANCEL_REQUESTED:
        _emit_event(
            event_writer,
            {
                "event": "result",
                "canceled": True,
                "had_error": False,
                "error_message": "",
                "save_dir": "",
            },
        )
        return 0

    _emit_event(event_writer, {"event": "training", "message": "Training started"})
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results = model.train(**params)
        save_dir = str(getattr(results, "save_dir", "") or "")
        _emit_event(
            event_writer,
            {
                "event": "result",
                "canceled": False,
                "had_error": False,
                "error_message": "",
                "save_dir": save_dir,
            },
        )
        return 0
    except Exception as exc:
        _emit_event(
            event_writer,
            {
                "event": "result",
                "canceled": bool(_CANCEL_REQUESTED),
                "had_error": not _CANCEL_REQUESTED,
                "error_message": "" if _CANCEL_REQUESTED else str(exc),
                "save_dir": "",
            },
        )
        return 1 if not _CANCEL_REQUESTED else 0


def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _handle_cancel_signal)
    signal.signal(signal.SIGINT, _handle_cancel_signal)

    parser = argparse.ArgumentParser(description="Run SqueakPose YOLO training in a child process.")
    parser.add_argument("--config", required=True, help="Path to JSON training config.")
    args = parser.parse_args(argv)

    try:
        config = read_config(args.config)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_training_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
