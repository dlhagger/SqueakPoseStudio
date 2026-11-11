"""Child-process entry point for SqueakPose YOLO training."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import sys
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional

from squeakpose.core import model_task_mismatch_message
from squeakpose.project.layers import normalize_layer_id
from squeakpose.services.training import FINALIZED_CHECKPOINT_RESUME_MESSAGE
from squeakpose.workers.protocol import (
    cancel_marker_path,
    read_config,
    start_cancel_file_watcher,
    write_event,
)

_CANCEL_REQUESTED = False
_CANCEL_SIGNAL_COUNT = 0


class TrainingCanceled(Exception):
    """Raised from a callback when a cancel arrives before the first training batch."""


def _request_cancel() -> None:
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True


def _handle_cancel_signal(_signum, _frame):
    # The first signal only sets the flag. The training callbacks translate it
    # into Ultralytics' own ``trainer.stop`` so the current batch finishes, the
    # model is validated, and last.pt/best.pt are saved before ``train()``
    # returns. A second signal (the owner's forced-close escalation) raises
    # SystemExit so ``finally`` blocks still run before the kill would arrive.
    global _CANCEL_SIGNAL_COUNT
    _CANCEL_SIGNAL_COUNT += 1
    _request_cancel()
    if _CANCEL_SIGNAL_COUNT > 1:
        raise SystemExit(130)


def _cancel_requested() -> bool:
    return _CANCEL_REQUESTED


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


RESUMABLE_CHECKPOINT_NAME = "last_resumable.pt"


def _snapshot_resumable_checkpoint(trainer: Any) -> str:
    """Copy ``last.pt`` to ``last_resumable.pt`` before final evaluation strips it.

    Ultralytics' ``final_eval`` runs ``strip_optimizer(last.pt)``, which drops
    the optimizer state and sets ``epoch`` to -1, so a canceled run could no
    longer be resumed exactly. The copy keeps the full training checkpoint
    (``train_args`` including ``save_dir``) next to it in the weights folder.
    """
    last = getattr(trainer, "last", None)
    if not last:
        save_dir = str(getattr(trainer, "save_dir", "") or "")
        if not save_dir:
            return ""
        last = os.path.join(save_dir, "weights", "last.pt")
    last = os.fspath(last)
    if not os.path.isfile(last):
        return ""
    target = os.path.join(os.path.dirname(last), RESUMABLE_CHECKPOINT_NAME)
    temp_path = f"{target}.partial"
    try:
        shutil.copy2(last, temp_path)
        os.replace(temp_path, target)
    except OSError as exc:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        print(f"Could not save resumable checkpoint {target}: {exc}", file=sys.stderr)
        return ""
    return target


def _remove_stale_resumable_checkpoint(save_dir: str) -> None:
    path = os.path.join(save_dir, "weights", RESUMABLE_CHECKPOINT_NAME)
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError as exc:
        print(f"Could not remove stale resumable checkpoint {path}: {exc}", file=sys.stderr)


def _register_training_callbacks(
    model: Any,
    event_writer: Callable[[dict[str, Any]], None],
    cancel_requested: Callable[[], bool] = _cancel_requested,
    *,
    run_state: dict[str, Any] | None = None,
) -> bool:
    """Bridge Ultralytics lifecycle callbacks to stable GUI progress events.

    The same callbacks honor cancel requests: before the first batch training
    is aborted (nothing worth keeping yet); afterwards ``trainer.stop`` is set
    so Ultralytics ends at the next batch boundary (or right after the current
    validation) and saves checkpoints. ``run_state`` receives
    ``cancel_shortened`` (the stop actually ended training early) and
    ``resume_checkpoint`` (the ``last_resumable.pt`` copy, when one was made).
    """
    add_callback = getattr(model, "add_callback", None)
    if not callable(add_callback):
        return False

    state: dict[str, Any] = run_state if run_state is not None else {}
    state.update(
        {
            "batch": 0,
            "epoch_started": time.monotonic(),
            "last_batch_emit": 0.0,
            "batches_run": 0,
            "stop_announced": False,
            "cancel_shortened": False,
            "resume_checkpoint": "",
            "epoch_validated": False,
        }
    )

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

    def is_final_epoch(trainer: Any) -> bool:
        epoch = int(getattr(trainer, "epoch", 0) or 0)
        epochs = int(getattr(trainer, "epochs", epoch + 1) or epoch + 1)
        return epoch + 1 >= epochs

    def honor_cancel(trainer: Any, *, epoch_complete: bool = False) -> None:
        if not cancel_requested():
            return
        if int(state.get("batches_run", 0)) == 0:
            raise TrainingCanceled("Training canceled before the first batch.")
        if state.get("stop_announced"):
            trainer.stop = True
            return
        # The stop shortens training only when Ultralytics was not already
        # stopping (final epoch, early stopping, time limit) and work remains.
        already_stopping = bool(getattr(trainer, "stop", False))
        state["cancel_shortened"] = not already_stopping and not (
            epoch_complete and is_final_epoch(trainer)
        )
        trainer.stop = True
        state["stop_announced"] = True
        if not state["cancel_shortened"]:
            return
        emit(
            {
                "event": "training_stopping",
                "message": (
                    "Stopping after the current batch; validating and saving "
                    "last.pt/best.pt before exiting."
                ),
                "save_dir": str(getattr(trainer, "save_dir", "") or ""),
            }
        )

    def snapshot_for_resume(trainer: Any) -> None:
        # Only a stop that skips remaining epochs leaves something to resume;
        # Ultralytics refuses to resume a checkpoint of the final epoch.
        if not state.get("cancel_shortened") or is_final_epoch(trainer):
            return
        path = _snapshot_resumable_checkpoint(trainer)
        if path:
            state["resume_checkpoint"] = path

    def on_pretrain_routine_end(trainer: Any) -> None:
        honor_cancel(trainer)

    def on_train_start(trainer: Any) -> None:
        honor_cancel(trainer)
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
        honor_cancel(trainer)
        state["batch"] = 0
        state["epoch_started"] = time.monotonic()
        state["last_batch_emit"] = 0.0
        state["epoch_validated"] = False
        epoch, epochs, batches = dimensions(trainer)
        emit({"event": "epoch_start", "epoch": epoch, "epochs": epochs, "batches": batches})

    def on_train_batch_end(trainer: Any) -> None:
        state["batches_run"] = int(state.get("batches_run", 0)) + 1
        epoch, epochs, batches = dimensions(trainer)
        state["batch"] = int(state.get("batch", 0)) + 1
        batch = int(state["batch"])
        honor_cancel(trainer, epoch_complete=batches > 0 and batch >= batches)
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

    def on_model_save(trainer: Any) -> None:
        # Runs right after save_model() wrote last.pt and before final_eval()
        # strips it, so this is the last moment the full checkpoint exists.
        honor_cancel(trainer, epoch_complete=True)
        snapshot_for_resume(trainer)

    def emit_final_validation(trainer: Any) -> None:
        # Ultralytics' final_eval() validates best.pt and calls
        # on_fit_epoch_end with ``epoch`` temporarily advanced by one.
        completed = max(0, int(getattr(trainer, "epoch", 0) or 0))
        epochs = max(completed, int(getattr(trainer, "epochs", completed) or completed))
        best = getattr(trainer, "best", None)
        emit(
            {
                "event": "final_validation",
                "epochs_completed": completed,
                "epochs": epochs,
                "metrics": _safe_metrics(getattr(trainer, "metrics", None)),
                "best_fitness": _safe_number(getattr(trainer, "best_fitness", None)),
                "model_path": os.fspath(best) if best else "",
                "save_dir": str(getattr(trainer, "save_dir", "") or ""),
            }
        )

    def on_fit_epoch_end(trainer: Any) -> None:
        # The training loop reports each started epoch exactly once; a second
        # call for the same started epoch can only be final_eval().
        if state.get("epoch_validated"):
            emit_final_validation(trainer)
            return
        state["epoch_validated"] = True
        # A cancel that arrived during validation stops here, at Ultralytics'
        # ``if self.stop: break``, instead of running another epoch.
        honor_cancel(trainer, epoch_complete=True)
        if not state.get("resume_checkpoint"):
            snapshot_for_resume(trainer)
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

    add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    add_callback("on_train_start", on_train_start)
    add_callback("on_train_epoch_start", on_train_epoch_start)
    add_callback("on_train_batch_end", on_train_batch_end)
    add_callback("on_model_save", on_model_save)
    add_callback("on_fit_epoch_end", on_fit_epoch_end)
    return True


def _finalized_resume_error(model: Any, params: Mapping[str, Any]) -> str:
    """Return an error when exact resume targets a checkpoint without training state.

    Ultralytics silently starts a brand-new run (in its global runs folder)
    when ``resume=True`` names a checkpoint whose optimizer was stripped.
    """
    if params.get("resume") is not True or not hasattr(model, "ckpt"):
        return ""
    ckpt = getattr(model, "ckpt", None)
    if not isinstance(ckpt, Mapping):
        return FINALIZED_CHECKPOINT_RESUME_MESSAGE
    try:
        epoch = int(ckpt.get("epoch", -1))
    except (TypeError, ValueError):
        epoch = -1
    if epoch < 0 or ckpt.get("optimizer") is None:
        return FINALIZED_CHECKPOINT_RESUME_MESSAGE
    return ""


def run_training_worker(
    config: dict[str, Any],
    *,
    model_factory: Optional[Callable[[str], Any]] = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
    reset_cancel: bool = True,
) -> int:
    """Run one YOLO training job and emit JSON-compatible event payloads."""
    global _CANCEL_REQUESTED
    if reset_cancel:
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

    resume_error = _finalized_resume_error(model, params)
    if resume_error:
        _emit_event(event_writer, {"event": "error", "error_message": resume_error})
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
    run_state: dict[str, Any] = {}
    try:
        _register_training_callbacks(model, event_writer, run_state=run_state)
        with contextlib.redirect_stdout(sys.stderr):
            results = model.train(**params)
        save_dir = str(getattr(results, "save_dir", "") or "")
        if not save_dir:
            trainer = getattr(model, "trainer", None)
            save_dir = str(getattr(trainer, "save_dir", "") or "")
        # A cancel honored via trainer.stop returns normally with checkpoints
        # saved; report it as canceled (only when it actually ended training
        # early) and keep save_dir so the UI can point at the partial run. A
        # cancel that arrives during the final validation of a completed run
        # is a normal completion.
        canceled = bool(run_state.get("cancel_shortened"))
        payload: dict[str, Any] = {
            "event": "result",
            "canceled": canceled,
            "had_error": False,
            "error_message": "",
            "save_dir": save_dir,
        }
        if canceled:
            payload["resume_checkpoint"] = str(run_state.get("resume_checkpoint") or "")
        elif save_dir:
            # A completed (or resumed-to-completion) run must not leave an older
            # resumable snapshot that would rewind the run if resumed again.
            _remove_stale_resumable_checkpoint(save_dir)
        _emit_event(event_writer, payload)
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
    # Portable cancel channel: the owner creates ``<config>.cancel``.
    start_cancel_file_watcher(cancel_marker_path(args.config), _request_cancel)

    try:
        config = read_config(args.config)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    # Keep a cancel that arrived while the config was loading.
    return run_training_worker(config, reset_cancel=False)


if __name__ == "__main__":
    raise SystemExit(main())
