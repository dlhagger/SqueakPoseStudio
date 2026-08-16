"""Conservative discovery and recovery of interrupted project transactions."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from squeakpose.core import remove_path
from squeakpose.project.safety import ProjectPathError, require_path_within_project

logger = logging.getLogger(__name__)

_BACKUP_RE = re.compile(r"^(?P<target>.+)\.backup-(?P<token>[0-9a-f]{32})$")
_TEMPFILE_TOKEN = r"[a-z0-9_]{8}"
_STAGING_FILE_RE = re.compile(rf"^\..+\.{_TEMPFILE_TOKEN}\.tmp(?:\.[^./\\]+)?$")
_EXPORT_STAGING_DIR_RE = re.compile(rf"^\.(?:pose|segment|detect)-export-{_TEMPFILE_TOKEN}$")
_PRUNED_DIRECTORIES = {".git", ".venv", "__pycache__"}


@dataclass(frozen=True, slots=True)
class TransactionBackup:
    backup_path: str
    target_path: str


@dataclass(slots=True)
class TransactionRecoveryReport:
    project_root: str
    staging_paths: list[str] = field(default_factory=list)
    restorable_backups: list[TransactionBackup] = field(default_factory=list)
    preserved_backups: list[TransactionBackup] = field(default_factory=list)


@dataclass(slots=True)
class TransactionRecoveryResult:
    restored_paths: list[str] = field(default_factory=list)
    removed_staging_paths: list[str] = field(default_factory=list)
    preserved_backup_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _backup_from_path(project_root: str, path: str) -> TransactionBackup | None:
    match = _BACKUP_RE.match(os.path.basename(path))
    if match is None:
        return None
    target_path = os.path.join(os.path.dirname(path), match.group("target"))
    try:
        backup_path = require_path_within_project(
            project_root,
            path,
            purpose="transaction backup",
            allow_root=False,
        )
        target_path = require_path_within_project(
            project_root,
            target_path,
            purpose="transaction backup target",
            allow_root=False,
        )
    except ProjectPathError:
        logger.warning(
            "Ignored unsafe transaction backup candidate",
            extra={
                "event": "transaction_backup_unsafe",
                "operation": "scan_transaction",
                "project_root": project_root,
                "source_path": path,
            },
        )
        return None
    return TransactionBackup(backup_path=backup_path, target_path=target_path)


def scan_transaction_artifacts(project_root: str) -> TransactionRecoveryReport:
    """Classify only artifacts matching names emitted by transaction helpers."""

    root = os.path.abspath(project_root)
    report = TransactionRecoveryReport(project_root=root)
    if not os.path.isdir(root):
        return report

    backups_by_target: dict[str, list[TransactionBackup]] = {}
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        retained_directories: list[str] = []
        for name in directory_names:
            path = os.path.join(current, name)
            if name in _PRUNED_DIRECTORIES or os.path.islink(path):
                continue
            backup = _backup_from_path(root, path)
            if backup is not None:
                backups_by_target.setdefault(backup.target_path, []).append(backup)
                continue
            if _EXPORT_STAGING_DIR_RE.fullmatch(name):
                report.staging_paths.append(
                    require_path_within_project(
                        root,
                        path,
                        purpose="dataset export staging directory",
                        allow_root=False,
                    )
                )
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in file_names:
            path = os.path.join(current, name)
            backup = _backup_from_path(root, path)
            if backup is not None:
                backups_by_target.setdefault(backup.target_path, []).append(backup)
            elif _STAGING_FILE_RE.fullmatch(name) and not os.path.islink(path):
                report.staging_paths.append(
                    require_path_within_project(
                        root,
                        path,
                        purpose="transaction staging file",
                        allow_root=False,
                    )
                )

    for target_path, backups in backups_by_target.items():
        ordered = sorted(backups, key=lambda item: item.backup_path.casefold())
        if len(ordered) == 1 and not os.path.lexists(target_path):
            report.restorable_backups.extend(ordered)
        else:
            report.preserved_backups.extend(ordered)

    report.staging_paths = sorted(set(report.staging_paths), key=str.casefold)
    report.restorable_backups.sort(key=lambda item: item.backup_path.casefold())
    report.preserved_backups.sort(key=lambda item: item.backup_path.casefold())
    return report


def restore_missing_transaction_targets(project_root: str) -> TransactionRecoveryResult:
    """Restore only a sole recognized backup whose original target is missing."""

    report = scan_transaction_artifacts(project_root)
    result = TransactionRecoveryResult(
        preserved_backup_paths=[item.backup_path for item in report.preserved_backups]
    )
    for item in report.restorable_backups:
        try:
            if os.path.lexists(item.target_path):
                result.preserved_backup_paths.append(item.backup_path)
                continue
            os.replace(item.backup_path, item.target_path)
            result.restored_paths.append(item.target_path)
            logger.warning(
                "Restored missing transaction target from backup",
                extra={
                    "event": "transaction_target_restored",
                    "operation": "recover_transaction",
                    "project_root": report.project_root,
                    "source_path": item.backup_path,
                    "target_path": item.target_path,
                },
            )
        except OSError as exc:
            logger.exception(
                "Could not restore transaction backup",
                extra={
                    "event": "transaction_restore_failed",
                    "operation": "recover_transaction",
                    "project_root": report.project_root,
                    "source_path": item.backup_path,
                    "target_path": item.target_path,
                },
            )
            result.errors.append(f"{item.backup_path}: {exc}")
    result.preserved_backup_paths = sorted(set(result.preserved_backup_paths), key=str.casefold)
    return result


def cleanup_transaction_staging(project_root: str) -> TransactionRecoveryResult:
    """Remove only currently recognized staging files and export directories."""

    report = scan_transaction_artifacts(project_root)
    result = TransactionRecoveryResult(
        preserved_backup_paths=[
            item.backup_path for item in report.restorable_backups + report.preserved_backups
        ]
    )
    for path in report.staging_paths:
        try:
            remove_path(path)
            result.removed_staging_paths.append(path)
            logger.info(
                "Removed abandoned transaction staging artifact",
                extra={
                    "event": "transaction_staging_removed",
                    "operation": "cleanup_transaction",
                    "project_root": report.project_root,
                    "target_path": path,
                },
            )
        except OSError as exc:
            logger.exception(
                "Could not remove transaction staging artifact",
                extra={
                    "event": "transaction_staging_cleanup_failed",
                    "operation": "cleanup_transaction",
                    "project_root": report.project_root,
                    "target_path": path,
                },
            )
            result.errors.append(f"{path}: {exc}")
    return result
