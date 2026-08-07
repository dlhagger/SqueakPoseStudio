"""Filesystem containment and single-writer ownership for project data."""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import uuid
from dataclasses import asdict, dataclass

PROJECT_LOCK_FILENAME = ".squeakpose.lock"
logger = logging.getLogger(__name__)


class ProjectPathError(ValueError):
    """Raised when a managed path escapes its project root."""


@dataclass(frozen=True, slots=True)
class ProjectLockInfo:
    pid: int
    hostname: str
    created_at: str
    token: str
    version: str = ""


class ProjectLockedError(RuntimeError):
    """Raised when a project already has a lock that cannot be replaced."""

    def __init__(self, lock_path: str, info: ProjectLockInfo | None, *, stale: bool):
        self.lock_path = lock_path
        self.info = info
        self.stale = bool(stale)
        if info is None:
            detail = "the lock file is invalid or unreadable"
        else:
            detail = f"PID {info.pid} on {info.hostname} since {info.created_at}"
        state = "stale" if stale else "active"
        super().__init__(f"Project has a {state} writer lock: {detail}")


def canonical_path(path: str) -> str:
    """Return an absolute, symlink-resolved path suitable for comparisons."""

    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def is_path_within_project(project_root: str, path: str, *, allow_root: bool = True) -> bool:
    """Return whether a path remains inside the project after symlink resolution."""

    root = canonical_path(project_root)
    candidate = canonical_path(path)
    try:
        contained = os.path.commonpath((root, candidate)) == root
    except ValueError:
        return False
    return contained and (allow_root or candidate != root)


def require_path_within_project(
    project_root: str,
    path: str,
    *,
    purpose: str = "managed project path",
    allow_root: bool = True,
) -> str:
    """Return an absolute path or reject it when it escapes the project."""

    absolute = os.path.abspath(os.fspath(path))
    if not is_path_within_project(project_root, absolute, allow_root=allow_root):
        raise ProjectPathError(f"{purpose} escapes the project root: {absolute}")
    return absolute


def project_lock_path(project_root: str) -> str:
    return require_path_within_project(
        project_root,
        os.path.join(os.path.abspath(project_root), PROJECT_LOCK_FILENAME),
        purpose="project lock path",
        allow_root=False,
    )


def _read_lock_info(lock_path: str) -> ProjectLockInfo | None:
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return ProjectLockInfo(
            pid=int(payload["pid"]),
            hostname=str(payload["hostname"]),
            created_at=str(payload["created_at"]),
            token=str(payload["token"]),
            version=str(payload.get("version") or ""),
        )
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def lock_is_stale(info: ProjectLockInfo | None) -> bool:
    """Return true only when a lock is provably stale on this host."""

    if info is None or info.hostname != socket.gethostname():
        return False
    return not _pid_is_running(info.pid)


def inspect_project_lock(project_root: str) -> tuple[ProjectLockInfo | None, bool] | None:
    """Return lock information and stale status, or ``None`` when unlocked."""

    path = project_lock_path(project_root)
    if not os.path.lexists(path):
        return None
    info = _read_lock_info(path)
    return info, lock_is_stale(info)


def break_stale_project_lock(project_root: str) -> ProjectLockInfo:
    """Remove a provably stale lock after its identity is revalidated."""

    path = project_lock_path(project_root)
    info = _read_lock_info(path)
    if info is None or not lock_is_stale(info):
        raise ProjectLockedError(path, info, stale=False)
    current = _read_lock_info(path)
    if current != info:
        raise ProjectLockedError(path, current, stale=lock_is_stale(current))
    os.unlink(path)
    logger.warning(
        "Removed stale project lock",
        extra={
            "event": "stale_project_lock_removed",
            "operation": "break_project_lock",
            "project_root": os.path.abspath(project_root),
            "target_path": path,
        },
    )
    return info


class ProjectLock:
    """Ownership-token lock that prevents concurrent project writers."""

    def __init__(self, project_root: str, *, version: str = ""):
        self.project_root = os.path.abspath(project_root)
        self.path = project_lock_path(self.project_root)
        self.info = ProjectLockInfo(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            created_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            token=uuid.uuid4().hex,
            version=str(version or ""),
        )
        self.acquired = False

    def acquire(self) -> "ProjectLock":
        if self.acquired:
            return self
        os.makedirs(self.project_root, exist_ok=True)
        payload = json.dumps(asdict(self.info), indent=2) + "\n"
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            existing = _read_lock_info(self.path)
            raise ProjectLockedError(
                self.path,
                existing,
                stale=lock_is_stale(existing),
            ) from exc
        handle = None
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001 - cleanup must preserve arbitrary encoder/I/O failures
            if handle is None:
                try:
                    os.close(descriptor)
                except OSError:
                    logger.warning(
                        "Could not close incomplete project lock descriptor",
                        exc_info=True,
                        extra={
                            "event": "project_lock_descriptor_cleanup_failed",
                            "operation": "acquire_project_lock",
                            "project_root": self.project_root,
                            "target_path": self.path,
                        },
                    )
            try:
                os.unlink(self.path)
            except OSError:
                logger.warning(
                    "Could not remove incomplete project lock",
                    exc_info=True,
                    extra={
                        "event": "project_lock_cleanup_failed",
                        "operation": "acquire_project_lock",
                        "project_root": self.project_root,
                        "target_path": self.path,
                    },
                )
            raise
        self.acquired = True
        logger.info(
            "Project writer lock acquired",
            extra={
                "event": "project_lock_acquired",
                "operation": "acquire_project_lock",
                "project_root": self.project_root,
                "target_path": self.path,
            },
        )
        return self

    def release(self) -> None:
        if not self.acquired:
            return
        current = _read_lock_info(self.path)
        if current is not None and current.token == self.info.token:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "Could not release project writer lock",
                    exc_info=True,
                    extra={
                        "event": "project_lock_release_failed",
                        "operation": "release_project_lock",
                        "project_root": self.project_root,
                        "target_path": self.path,
                    },
                )
                return
        elif os.path.lexists(self.path):
            logger.error(
                "Project lock ownership changed before release",
                extra={
                    "event": "project_lock_ownership_changed",
                    "operation": "release_project_lock",
                    "project_root": self.project_root,
                    "target_path": self.path,
                },
            )
            return
        self.acquired = False

    def __enter__(self) -> "ProjectLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: BLE001 - destructors must never raise
            pass
