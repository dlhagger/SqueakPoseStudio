"""Common child-process lifecycle helpers used by Qt owners."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def remove_file_quietly(path: str | None) -> None:
    """Best-effort cleanup for an owned temporary configuration file."""
    if not path:
        return
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
    except OSError:
        pass


def shutdown_qprocess(
    process: Any,
    *,
    terminate_timeout_ms: int = 2000,
    kill_timeout_ms: int = 1000,
) -> bool:
    """Synchronously stop a QProcess-like object owned by a closing window."""
    if process is None:
        return True

    from PyQt6.QtCore import QProcess

    if process.state() == QProcess.ProcessState.NotRunning:
        return True
    try:
        process.blockSignals(True)
    except (AttributeError, RuntimeError):
        pass
    process.terminate()
    if process.waitForFinished(terminate_timeout_ms):
        return True
    process.kill()
    process.waitForFinished(kill_timeout_ms)
    return process.state() == QProcess.ProcessState.NotRunning


def request_qprocess_stop(
    process: Any,
    *,
    schedule: Callable[[int, Callable[[], None]], None],
    force_kill: Callable[[], None],
    kill_after_ms: int = 5000,
) -> bool:
    """Request graceful termination and schedule an owned force-kill check."""
    if process is None:
        return False

    from PyQt6.QtCore import QProcess

    if process.state() == QProcess.ProcessState.NotRunning:
        return False
    process.terminate()
    schedule(max(0, int(kill_after_ms)), force_kill)
    return True
