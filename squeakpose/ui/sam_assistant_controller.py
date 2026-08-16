"""Persistent child-process coordination for interactive SAM prompts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from squeakpose.annotation.segmentation_assistant import SamPromptRequest
from squeakpose.services.sam_assistant import (
    SamAssistantDecision,
    build_sam_load_request,
    build_sam_prediction_request,
    correlate_sam_event,
)
from squeakpose.workers.process import PersistentWorkerResult, PersistentWorkerSession


class SamAssistantController(QObject):
    """Own one warm SAM worker generation and correlate prompt responses."""

    status_changed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    event_received = pyqtSignal(dict)
    output_received = pyqtSignal(str)
    decision_ready = pyqtSignal(object)
    terminal = pyqtSignal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        displayed_image_path: Callable[[], str],
        session_factory: Callable[[QObject], Any] = PersistentWorkerSession,
        program: str = sys.executable,
        arguments: tuple[str, ...] = ("-m", "sam_worker", "--server"),
        working_directory: str | None = None,
        schedule: Callable[[Callable[[], None]], None] = lambda callback: QTimer.singleShot(
            0, callback
        ),
    ) -> None:
        super().__init__(parent)
        self._displayed_image_path = displayed_image_path
        self._session_factory = session_factory
        self._program = program
        self._arguments = list(arguments)
        self._working_directory = working_directory
        self._schedule = schedule
        self._session: Any = None
        self._request_counter = 0
        self._pending_request: dict[str, Any] | None = None
        self._current_request_id: Any = None
        self._requested_image_path = ""
        self._busy = False
        self._expected_stop = False
        self._cancel_requested = False
        self._restart_request: dict[str, str] | None = None

    @property
    def session(self) -> Any:
        return self._session

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def current_request_id(self) -> Any:
        return self._current_request_id

    def submit_prompt(
        self,
        *,
        model_path: str,
        prompt: SamPromptRequest,
        device: str = "",
    ) -> Any:
        if self._busy or self._pending_request is not None:
            raise RuntimeError("SAM request already running")
        request_id = self._next_request_id()
        request = build_sam_prediction_request(
            request_id=request_id,
            model_path=model_path,
            prompt=prompt,
            device=device,
        ).as_worker_payload()
        self._current_request_id = request_id
        self._requested_image_path = prompt.source
        self._set_busy(True)
        self.status_changed.emit("Running SAM segmentation...")
        self._send_or_queue(request)
        return request_id

    def warm_model(self, *, model_path: str, device: str = "") -> Any:
        if self._busy or self._pending_request is not None:
            raise RuntimeError("SAM request already pending")
        request_id = self._next_request_id()
        request = build_sam_load_request(
            request_id=request_id,
            model_path=model_path,
            device=device,
        ).as_worker_payload()
        self._send_or_queue(request)
        return request_id

    def cancel(self) -> bool:
        session = self._session
        if session is None or not session.is_running:
            return False
        self._restart_request = None
        self._cancel_requested = True
        self._expected_stop = True
        return bool(session.shutdown(terminate_after_ms=250, kill_after_ms=3000))

    def restart_model(
        self,
        *,
        model_path: str,
        device: str = "",
        warm: bool = True,
    ) -> bool:
        self._restart_request = (
            {"model_path": model_path, "device": device} if warm and model_path else None
        )
        self._pending_request = None
        session = self._session
        if session is not None and session.is_running:
            self._expected_stop = True
            return bool(session.shutdown(terminate_after_ms=250, kill_after_ms=3000))
        self._resume_restart()
        return False

    def shutdown(self) -> bool:
        session = self._session
        if session is None or not session.is_running:
            return False
        self._restart_request = None
        self._expected_stop = True
        return bool(session.shutdown(terminate_after_ms=250, kill_after_ms=3000))

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _ensure_session(self) -> Any:
        if self._session is None:
            session = self._session_factory(self)
            session.ready.connect(self._on_ready)
            session.event_received.connect(self._on_event)
            session.output_received.connect(self.output_received.emit)
            session.stderr_received.connect(self.output_received.emit)
            session.terminal.connect(self._on_terminal)
            self._session = session
        session = self._session
        if not session.is_running:
            self._expected_stop = False
            self._cancel_requested = False
            session.start(
                self._program,
                list(self._arguments),
                working_directory=self._working_directory,
                start_timeout_ms=1000,
            )
        return session

    def _send_or_queue(self, request: dict[str, Any]) -> None:
        self._pending_request = dict(request)
        session = self._ensure_session()
        if not session.is_running or not session.is_ready:
            return
        if session.send_request(request):
            self._pending_request = None
        else:
            self.output_received.emit("Could not write SAM request to worker.")

    def _on_ready(self) -> None:
        pending = self._pending_request
        self._pending_request = None
        if pending is not None and not self._session.send_request(pending):
            self._pending_request = pending
            self.output_received.emit("Could not write SAM request to worker.")

    def _on_event(self, event: dict[str, Any]) -> None:
        self.event_received.emit(dict(event))
        event_type = str(event.get("event") or "")
        if event_type == "loading":
            self.status_changed.emit("Loading SAM model...")
            return
        if event_type == "loaded":
            self.status_changed.emit("SAM model ready.")
            return
        if event_type == "started":
            self.status_changed.emit("SAM worker started...")
            return
        if event_type not in {"error", "result"}:
            return
        decision = correlate_sam_event(
            event,
            current_request_id=self._current_request_id,
            requested_image_path=self._requested_image_path,
            displayed_image_path=self._displayed_image_path(),
        )
        if decision.action == "ignore":
            return
        if decision.action == "background_error":
            self.status_changed.emit(f"SAM model error: {decision.error_message}")
            self.decision_ready.emit(decision)
            return
        self._clear_active_request()
        self.decision_ready.emit(decision)

    def _on_terminal(self, result: PersistentWorkerResult) -> None:
        cancel_requested = self._cancel_requested
        expected_stop = self._expected_stop
        self._pending_request = None
        self._cancel_requested = False
        self._expected_stop = False
        if self._busy:
            if cancel_requested:
                decision = SamAssistantDecision(
                    "cancel",
                    request_id=self._current_request_id,
                )
            elif not expected_stop:
                decision = SamAssistantDecision(
                    "error",
                    request_id=self._current_request_id,
                    error_message=(result.error_message or result.stderr or "SAM worker stopped."),
                )
            else:
                decision = None
            self._clear_active_request()
            if decision is not None:
                self.decision_ready.emit(decision)
        self.terminal.emit(result)
        if expected_stop and self._restart_request is not None:
            self._schedule(self._resume_restart)

    def _resume_restart(self) -> None:
        request = self._restart_request
        self._restart_request = None
        if request is not None:
            self.warm_model(**request)

    def _clear_active_request(self) -> None:
        self._current_request_id = None
        self._requested_image_path = ""
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        value = bool(busy)
        if value == self._busy:
            return
        self._busy = value
        self.busy_changed.emit(value)


__all__ = ["SamAssistantController"]
