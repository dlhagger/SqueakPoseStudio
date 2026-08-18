"""Child-process entry point for SqueakPose YOLO training."""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional

from squeakpose.core import model_task_mismatch_message
from squeakpose.project.layers import normalize_layer_id
from squeakpose.workers.protocol import read_config, write_event

_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    # Ultralytics output is redirected to stderr while this stream remains the
    # worker's newline-delimited JSON protocol channel.
    write_event(payload, stream=sys.__stdout__)


def _emit_event(event_writer: Callable[[dict[str, Any]], None], payload: dict[str, Any]) -> None:
    event_writer(payload)


def _safe_number(value: Any) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _safe_metrics(values: Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in values.items():
        number = _safe_number(value)
        if number is not None:
            result[str(key)] = number
    return result


def _trainer_losses(trainer: Any) -> dict[str, float]:
    losses = _safe_metrics(getattr(trainer, "tloss", None))
    if losses:
        return losses
    labeler = getattr(trainer, "label_loss_items", None)
    if callable(labeler):
        try:
            return _safe_metrics(labeler(getattr(trainer, "tloss", None), prefix="train"))
        except Exception:
            return {}
    return {}


def _register_training_callbacks(
    model: Any,
    event_writer: Callable[[dict[str, Any]], None],
) -> bool:
    """Bridge Ultralytics lifecycle callbacks to stable GUI progress events."""
    add_callback = getattr(model, "add_callback", None)
    if not callable(add_callback):
        return False

    state: dict[str, Any] = {
        "batch": 0,
        "epoch_started": time.monotonic(),
        "last_batch_emit": 0.0,
    }

    def emit(payload: dict[str, Any]) -> None:
        try:
            _emit_event(event_writer, payload)
        except Exception:
            pass

    def dimensions(trainer: Any) -> tuple[int, int, int]:
        epoch = int(getattr(trainer, "epoch", 0) or 0) + 1
        epochs = max(epoch, int(getattr(trainer, "epochs", epoch) or epoch))
        try:
            batches = len(getattr(trainer, "train_loader", ()))
        except TypeError:
            batches = 0
        return epoch, epochs, max(0, int(batches))

    def on_train_start(trainer: Any) -> None:
        epoch, epochs, batches = dimensions(trainer)
        emit(
            {
                "event": "training_setup",
                "epoch": epoch,
                "epochs": epochs,
                "batches": batches,
                "device": str(getattr(trainer, "device", "")),
                "save_dir": str(getattr(trainer, "save_dir", "") or ""),
            }
        )

    def on_train_epoch_start(trainer: Any) -> None:
        state["batch"] = 0
        state["epoch_started"] = time.monotonic()
        state["last_batch_emit"] = 0.0
        epoch, epochs, batches = dimensions(trainer)
        emit({"event": "epoch_start", "epoch": epoch, "epochs": epochs, "batches": batches})

    def on_train_batch_end(trainer: Any) -> None:
        epoch, epochs, batches = dimensions(trainer)
        state["batch"] = int(state.get("batch", 0)) + 1
        batch = int(state["batch"])
        now = time.monotonic()
        should_emit = (
            batch == 1 or batch >= batches or now - float(state.get("last_batch_emit", 0.0)) >= 0.2
        )
        if not should_emit:
            return
        state["last_batch_emit"] = now
        elapsed = max(0.0, now - float(state.get("epoch_started", now)))
        rate = batch / elapsed if elapsed > 0 else 0.0
        remaining = (batches - batch) / rate if rate > 0 and batches > batch else 0.0
        memory = None
        memory_reader = getattr(trainer, "_get_memory", None)
        if callable(memory_reader):
            try:
                memory = _safe_number(memory_reader())
            except Exception:
                memory = None
        emit(
            {
                "event": "batch_progress",
                "epoch": epoch,
                "epochs": epochs,
                "batch": batch,
                "batches": batches,
                "losses": _trainer_losses(trainer),
                "memory_gb": memory,
                "elapsed_seconds": elapsed,
                "eta_seconds": remaining,
            }
        )

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch, epochs, batches = dimensions(trainer)
        elapsed = max(
            0.0,
            time.time() - float(getattr(trainer, "train_time_start", time.time())),
        )
        completed = max(1, epoch - int(getattr(trainer, "start_epoch", 0) or 0))
        remaining = max(0, epochs - epoch) * (elapsed / completed)
        emit(
            {
                "event": "epoch_end",
                "epoch": epoch,
                "epochs": epochs,
                "batches": batches,
                "epoch_seconds": _safe_number(getattr(trainer, "epoch_time", None)),
                "elapsed_seconds": elapsed,
                "eta_seconds": remaining,
                "losses": _trainer_losses(trainer),
                "metrics": _safe_metrics(getattr(trainer, "metrics", None)),
                "learning_rates": _safe_metrics(getattr(trainer, "lr", None)),
                "fitness": _safe_number(getattr(trainer, "fitness", None)),
                "best_fitness": _safe_number(getattr(trainer, "best_fitness", None)),
                "save_dir": str(getattr(trainer, "save_dir", "") or ""),
            }
        )

    add_callback("on_train_start", on_train_start)
    add_callback("on_train_epoch_start", on_train_epoch_start)
    add_callback("on_train_batch_end", on_train_batch_end)
    add_callback("on_fit_epoch_end", on_fit_epoch_end)
    return True


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
        _register_training_callbacks(model, event_writer)
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
