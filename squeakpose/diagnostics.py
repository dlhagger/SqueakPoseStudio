"""Project-local structured logging for SqueakPose Studio."""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
import os
from typing import Any

PROJECT_LOG_FILENAME = "squeakpose.jsonl"
_HANDLER_MARKER = "_squeakpose_project_handler"
_previous_root_level: int | None = None
_STRUCTURED_FIELDS = (
    "event",
    "operation",
    "project_root",
    "source_path",
    "target_path",
    "recovery_path",
    "worker",
    "request_id",
)


class JsonLineFormatter(logging.Formatter):
    """Serialize one logging record as a stable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value not in (None, ""):
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def project_log_path(project_root: str) -> str:
    """Return the canonical structured log path for a project."""

    return os.path.join(os.path.abspath(project_root), "logs", PROJECT_LOG_FILENAME)


def reset_project_logging() -> None:
    """Detach and close the active project handler, if any."""

    global _previous_root_level
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()
    if _previous_root_level is not None:
        root_logger.setLevel(_previous_root_level)
        _previous_root_level = None
    logging.captureWarnings(False)


def configure_project_logging(
    project_root: str,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> str:
    """Route application logs to a rotating JSON-lines file in the project."""

    global _previous_root_level
    root = os.path.abspath(project_root)
    log_path = project_log_path(root)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    reset_project_logging()
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max(1, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
    )
    setattr(handler, _HANDLER_MARKER, True)
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonLineFormatter())

    root_logger = logging.getLogger()
    _previous_root_level = root_logger.level
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    logging.captureWarnings(True)

    try:
        os.chmod(log_path, 0o600)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not restrict project log permissions",
            exc_info=True,
            extra={"event": "log_permissions_failed", "target_path": log_path},
        )

    logging.getLogger(__name__).info(
        "Project logging configured",
        extra={
            "event": "logging_configured",
            "operation": "configure_logging",
            "project_root": root,
            "target_path": log_path,
        },
    )
    return log_path
