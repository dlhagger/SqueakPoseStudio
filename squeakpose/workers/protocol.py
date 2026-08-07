"""Shared newline-delimited JSON protocol for child workers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, TextIO

from squeakpose.json_io import JsonFileError, read_json_file


class WorkerProtocolError(ValueError):
    """Raised when a worker request or event violates the common envelope."""


MAX_CONFIG_BYTES = 4 * 1024 * 1024


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
        self._buffer += str(chunk or "")
        lines = self._buffer.splitlines(keepends=True)
        self._buffer = ""
        complete: list[str] = []
        for line in lines:
            if line.endswith(("\n", "\r")):
                stripped = line.strip()
                if stripped:
                    complete.append(stripped)
            else:
                self._buffer = line
        return complete

    def finish(self) -> str:
        pending = self._buffer.strip()
        self._buffer = ""
        return pending
