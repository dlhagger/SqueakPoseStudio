"""Shared newline-delimited JSON protocol for child workers."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, TextIO

from squeakpose.json_io import JsonFileError, read_json_file


class WorkerProtocolError(ValueError):
    """Raised when a worker request or event violates the common envelope."""


MAX_CONFIG_BYTES = 4 * 1024 * 1024
CANCEL_COMMAND = "cancel"


def cancel_command_bytes() -> bytes:
    """Return the stdin request a Qt owner sends to ask a one-shot worker to stop.

    ``QProcess.terminate()`` is a no-op for console children on Windows, so a
    stdin line is the portable graceful-cancel channel; SIGTERM remains the
    POSIX fallback.
    """
    return (json.dumps({"command": CANCEL_COMMAND}) + "\n").encode("utf-8")


def start_cancel_listener(
    on_cancel: Callable[[], None],
    *,
    stream: TextIO | None = None,
) -> threading.Thread:
    """Call ``on_cancel`` once when a ``{"command": "cancel"}`` line arrives on stdin.

    Runs on a daemon thread so a worker that never receives the request still
    exits normally. Other lines are ignored; EOF ends the listener.
    """

    source = stream if stream is not None else sys.stdin

    def listen() -> None:
        if source is None:
            return
        try:
            for raw_line in source:
                try:
                    request = json.loads(raw_line)
                except (TypeError, ValueError):
                    continue
                if isinstance(request, dict) and request.get("command") == CANCEL_COMMAND:
                    on_cancel()
                    return
        except (OSError, ValueError):
            return

    thread = threading.Thread(target=listen, name="squeakpose-cancel-listener", daemon=True)
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
