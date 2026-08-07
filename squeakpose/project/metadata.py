"""Atomic project metadata persistence and recovery."""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from layer_ops import layer_worker_mode, normalize_layer_id
from squeakpose_core import (
    CURRENT_PROJECT_SCHEMA_VERSION,
    atomic_write_text,
    migrate_project_metadata,
)

from .paths import PROJECT_META_FILE

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetadataReadResult:
    data: dict[str, Any]
    recovery_path: str = ""
    recovery_error: str = ""


class ProjectMetadataStore:
    """Own metadata loading, migration, recovery, and path serialization."""

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.path = os.path.join(self.project_root, PROJECT_META_FILE)

    def read(self) -> MetadataReadResult:
        if not os.path.isfile(self.path):
            return MetadataReadResult({})
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("project metadata must contain a JSON object")
        except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
            backup_path = self._corrupt_backup_path()
            try:
                os.replace(self.path, backup_path)
            except OSError:
                logger.error(
                    "Could not preserve invalid project metadata",
                    exc_info=True,
                    extra={
                        "event": "metadata_recovery_backup_failed",
                        "operation": "read_metadata",
                        "project_root": self.project_root,
                        "source_path": self.path,
                        "recovery_path": backup_path,
                    },
                )
                backup_path = ""
            logger.warning(
                "Invalid project metadata detected",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "event": "metadata_recovery_started",
                    "operation": "read_metadata",
                    "project_root": self.project_root,
                    "source_path": self.path,
                    "recovery_path": backup_path,
                },
            )
            return MetadataReadResult(
                {},
                recovery_path=backup_path,
                recovery_error=str(exc),
            )

        migrated, changed = migrate_project_metadata(
            data,
            created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        if changed:
            atomic_write_text(self.path, json.dumps(migrated, indent=2))
            logger.info(
                "Project metadata migrated",
                extra={
                    "event": "metadata_migrated",
                    "operation": "migrate_metadata",
                    "project_root": self.project_root,
                    "target_path": self.path,
                },
            )
        return MetadataReadResult(migrated)

    def update(self, updates: dict[str, Any]) -> MetadataReadResult:
        result = self.read()
        payload = dict(result.data)
        if not payload:
            payload = {
                "schema_version": CURRENT_PROJECT_SCHEMA_VERSION,
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        normalized_updates = dict(updates)
        if "active_workflow" in normalized_updates and "active_layer" not in normalized_updates:
            normalized_updates["active_layer"] = normalize_layer_id(
                normalized_updates["active_workflow"]
            )
        if "active_layer" in normalized_updates:
            normalized_updates["active_layer"] = normalize_layer_id(
                normalized_updates["active_layer"]
            )
            normalized_updates["active_workflow"] = layer_worker_mode(
                normalized_updates["active_layer"]
            )
        for key, value in normalized_updates.items():
            if value is None:
                payload.pop(str(key), None)
            else:
                payload[str(key)] = value
        atomic_write_text(self.path, json.dumps(payload, indent=2))
        return MetadataReadResult(
            payload,
            recovery_path=result.recovery_path,
            recovery_error=result.recovery_error,
        )

    def resolve_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        return os.path.abspath(os.path.join(self.project_root, raw))

    def store_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        abs_path = os.path.abspath(raw)
        try:
            relative = os.path.relpath(abs_path, self.project_root)
        except ValueError:
            return abs_path
        if relative == ".":
            return os.path.basename(abs_path)
        if relative != ".." and not relative.startswith(f"..{os.sep}"):
            return relative
        return abs_path

    def _corrupt_backup_path(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(
            self.project_root,
            f"squeakpose_project.corrupt-{timestamp}.json",
        )
        suffix = 1
        while os.path.exists(backup_path):
            backup_path = os.path.join(
                self.project_root,
                f"squeakpose_project.corrupt-{timestamp}-{suffix}.json",
            )
            suffix += 1
        return backup_path
