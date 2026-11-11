"""Filesystem containment and single-writer ownership for project data."""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import uuid
from dataclasses import asdict, dataclass

from squeakpose.json_io import read_json_file

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

    def __init__(
        self,
        lock_path: str,
        info: ProjectLockInfo | None,
        *,
        stale: bool,
        claimed_path: str | None = None,
    ):
        self.lock_path = lock_path
        self.info = info
        self.stale = bool(stale)
        self.claimed_path = claimed_path
        if info is None:
            detail = "the lock file is invalid or unreadable"
        else:
            detail = f"PID {info.pid} on {info.hostname} since {info.created_at}"
        state = "stale" if stale else "active"
        message = f"Project has a {state} writer lock: {detail}"
        if claimed_path:
            message += (
                f". The lock changed hands while being broken and could not be restored; "
                f"it was left at {claimed_path}. Rename it back to {lock_path} only after "
                "confirming no other SqueakPose session is using this project."
            )
        super().__init__(message)


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
        payload = read_json_file(lock_path, max_bytes=64 * 1024, require_object=True)
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


def _restore_claimed_lock(claimed: str, path: str, project_root: str) -> bool:
    """Put back a lock that changed hands while it was being broken, never clobbering.

    Returns ``False`` when the claimed lock could not be restored and was therefore left
    in place at ``claimed``; a live lock is never deleted in that case.
    """

    log_extra = {
        "operation": "break_project_lock",
        "project_root": os.path.abspath(project_root),
        "target_path": path,
    }
    try:
        os.link(claimed, path)
    except FileExistsError:
        pass
    except OSError:
        # Filesystems such as FAT/exFAT and many SMB shares have no hard links. Fall
        # back to a rename only while the lock path is still free so that a lock created
        # in the meantime is never clobbered.
        if not os.path.lexists(path):
            try:
                os.rename(claimed, path)
            except OSError:
                logger.warning(
                    "Could not restore a project lock that changed while being broken",
                    exc_info=True,
                    extra={**log_extra, "event": "project_lock_restore_failed"},
                )
                return False
            return True
    try:
        os.unlink(claimed)
    except OSError:
        logger.warning(
            "Could not remove a claimed project lock",
            exc_info=True,
            extra={
                **log_extra,
                "event": "project_lock_claim_cleanup_failed",
                "target_path": claimed,
            },
        )
    return True


def break_stale_project_lock(project_root: str) -> ProjectLockInfo:
    """Remove a provably stale lock after its identity is revalidated.

    The lock is first renamed to a unique sibling so that only one breaker can claim
    it; the claimed file is re-read and only unlinked when it still carries the stale
    identity. Losing the race (the lock vanished or was replaced) reports the project
    as still locked instead of removing another writer's lock.
    """

    path = project_lock_path(project_root)
    info = _read_lock_info(path)
    if info is None or not lock_is_stale(info):
        raise ProjectLockedError(path, info, stale=False)
    claimed = f"{path}.breaking-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        os.rename(path, claimed)
    except FileNotFoundError as exc:
        current = _read_lock_info(path)
        raise ProjectLockedError(path, current, stale=lock_is_stale(current)) from exc
    current = _read_lock_info(claimed)
    if current != info:
        if not _restore_claimed_lock(claimed, path, project_root):
            raise ProjectLockedError(
                path, current, stale=lock_is_stale(current), claimed_path=claimed
            )
        raise ProjectLockedError(path, current, stale=lock_is_stale(current))
    os.unlink(claimed)
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
