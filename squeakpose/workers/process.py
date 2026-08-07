"""Common child-process lifecycle helpers used by Qt owners."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from typing import Any

from squeakpose.project.safety import require_path_within_project

logger = logging.getLogger(__name__)


def create_worker_config(
    project_root: str,
    directory: str,
    kind: str,
    payload: Mapping[str, Any],
) -> str:
    """Create a unique, owner-readable worker configuration inside a project."""

    root = os.path.abspath(project_root)
    config_dir = require_path_within_project(
        root,
        directory,
        purpose="worker configuration directory",
    )
    os.makedirs(config_dir, exist_ok=True)
    safe_kind = re.sub(r"[^a-z0-9_-]+", "_", str(kind).strip().lower()).strip("_")
    if not safe_kind:
        raise ValueError("worker configuration kind must not be empty")

    fd = -1
    path = ""
    try:
        fd, path = tempfile.mkstemp(
            prefix=f".{safe_kind}_config_",
            suffix=".json",
            dir=config_dir,
        )
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(dict(payload), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except Exception:
        if fd >= 0:
            os.close(fd)
        if path:
            remove_file_quietly(path)
        raise


def remove_file_quietly(path: str | None) -> None:
    """Best-effort cleanup for an owned temporary configuration file."""
    if not path:
        return
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
    except OSError:
        logger.warning(
            "Could not remove owned temporary file",
            exc_info=True,
            extra={
                "event": "owned_temporary_cleanup_failed",
                "operation": "remove_owned_temporary",
                "target_path": path,
            },
        )


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

    def force_kill_original_process() -> None:
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                force_kill()
        except RuntimeError:
            return

    schedule(max(0, int(kill_after_ms)), force_kill_original_process)
    return True
