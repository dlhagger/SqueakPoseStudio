"""Bounded, symlink-safe JSON file loading for managed application data."""

from __future__ import annotations

import json
import os
import stat
from typing import Any

DEFAULT_MAX_JSON_BYTES = 4 * 1024 * 1024


class JsonFileError(ValueError):
    """Raised when a managed JSON file cannot be loaded safely."""


def read_json_file(
    path: str,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
    require_object: bool = False,
) -> Any:
    """Read one regular JSON file without following symlinks or growing unbounded."""

    limit = max(1, int(max_bytes))
    try:
        initial_stat = os.lstat(path)
    except OSError as exc:
        raise JsonFileError(f"could not inspect JSON file: {exc}") from exc
    if stat.S_ISLNK(initial_stat.st_mode):
        raise JsonFileError("JSON path must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise JsonFileError(f"could not open JSON file: {exc}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise JsonFileError("JSON path must be a regular file")
        if file_stat.st_size > limit:
            raise JsonFileError(f"JSON file exceeds {limit} bytes")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            raw = handle.read(limit + 1)
    except (OSError, UnicodeError) as exc:
        raise JsonFileError(f"could not read JSON file: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)

    if len(raw.encode("utf-8")) > limit:
        raise JsonFileError(f"JSON file exceeds {limit} bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonFileError(f"invalid JSON: {exc}") from exc
    if require_object and not isinstance(payload, dict):
        raise JsonFileError("JSON file must contain an object")
    return payload
