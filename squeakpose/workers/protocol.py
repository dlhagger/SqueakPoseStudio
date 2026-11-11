"""Shared newline-delimited JSON protocol for child workers."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, TextIO

from squeakpose.json_io import JsonFileError, read_json_file


class WorkerProtocolError(ValueError):
    """Raised when a worker request or event violates the common envelope."""


MAX_CONFIG_BYTES = 4 * 1024 * 1024
CANCEL_FILE_SUFFIX = ".cancel"
CANCEL_POLL_INTERVAL_SECONDS = 0.25


def cancel_marker_path(config_path: str) -> str:
    """Return the cancel marker a Qt owner creates next to a one-shot worker config.

    ``QProcess.terminate()`` does not reach console children on Windows, and a
    worker thread blocked reading stdin makes inherited-stdin grandchildren
    (Ultralytics runs ``nvidia-smi``/``git``) hang during CRT start-up there.
    A marker file beside the project-local config is the portable graceful
    channel; SIGTERM remains the POSIX fallback.
    """
    return f"{config_path}{CANCEL_FILE_SUFFIX}"


def start_cancel_file_watcher(
    path: str,
    on_cancel: Callable[[], None],
    *,
    interval: float = CANCEL_POLL_INTERVAL_SECONDS,
    exists: Callable[[str], bool] = os.path.exists,
) -> threading.Thread:
    """Call ``on_cancel`` once when ``path`` appears, polling every ``interval`` seconds.

    Runs on a daemon thread so a worker that is never canceled exits normally.
    """

    def watch() -> None:
        while True:
            try:
                present = exists(path)
            except OSError, ValueError:
                present = False
            if present:
                on_cancel()
                return
            time.sleep(max(0.01, float(interval)))

    thread = threading.Thread(target=watch, name="squeakpose-cancel-watcher", daemon=True)
    thread.start()
    return thread


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    event: str
    payload: dict[str, Any]
    request_id: Any = None

    @property
    def is_terminal(self) -> bool:
        return self.event in {"result", "error", "stopped"}

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def validate_event(payload: Mapping[str, Any]) -> WorkerEvent:
    if not isinstance(payload, Mapping):
        raise WorkerProtocolError("worker event must be a JSON object")
    event_name = str(payload.get("event") or "").strip()
    if not event_name:
        raise WorkerProtocolError("worker event is missing a non-empty 'event' field")
    data = dict(payload)
    return WorkerEvent(
        event=event_name,
        payload=data,
        request_id=data.get("request_id"),
    )


def parse_event_line(line: str) -> WorkerEvent:
    raw = str(line or "").strip()
    if not raw:
        raise WorkerProtocolError("worker event line is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerProtocolError(f"invalid worker JSON: {exc}") from exc
    return validate_event(payload)


def write_event(payload: Mapping[str, Any], *, stream: TextIO | None = None) -> None:
    """Write one complete event line, suitable for a QProcess stdout channel."""
    if stream is None:
        import sys

        stream = sys.stdout
    event = validate_event(payload)
    print(json.dumps(event.payload, sort_keys=True), file=stream, flush=True)


def read_config(path: str, *, max_bytes: int = MAX_CONFIG_BYTES) -> dict[str, Any]:
    """Load a worker configuration object from JSON."""
    try:
        return read_json_file(path, max_bytes=max_bytes, require_object=True)
    except JsonFileError as exc:
        raise WorkerProtocolError(f"invalid worker config: {exc}") from exc


class JsonLineBuffer:
    """Incrementally split text chunks into complete protocol lines."""

    def __init__(self) -> None:
        self._buffer = ""

    @property
    def pending(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> list[str]:
        # Split only on "\n", "\r" and "\r\n". ``str.splitlines`` would also
        # break on "\x0c", "\x1c"-"\x1e", "\x85", "\u2028" and "\u2029", which
        # third-party stdout (progress bars, form feeds) can emit mid-line;
        # treating those as terminators would truncate the pending text and
        # corrupt the next protocol line.
        self._buffer += str(chunk or "")
        text = self._buffer
        holdback = ""
        if text.endswith("\r"):
            # A trailing "\r" may be the first half of "\r\n"; keep it pending
            # so the following "\n" does not produce a spurious empty line.
            text = text[:-1]
            holdback = "\r"
        parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._buffer = parts.pop() + holdback
        complete: list[str] = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                complete.append(stripped)
        return complete

    def finish(self) -> str:
        pending = self._buffer.strip()
        self._buffer = ""
        return pending
