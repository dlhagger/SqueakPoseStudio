"""Qt coordination for ChatGPT sign-in and remote pose proposals."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from PyQt6.QtCore import QObject, pyqtSignal

from squeakpose.services.openai_auth import (
    ChatGPTCredentialStore,
    ChatGPTProfile,
    complete_authorization,
    create_authorization_attempt,
    revoke_profile,
    usable_access_token,
)
from squeakpose.services.openai_labeling import (
    OpenAIModelChoice,
    list_models,
    request_pose_proposal,
)


class OpenAIController(QObject):
    """Keep blocking OAuth and streaming HTTP away from the Qt event loop."""

    authorization_url = pyqtSignal(str)
    profile_changed = pyqtSignal(object)
    models_ready = pyqtSignal(object)
    proposal_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        store: ChatGPTCredentialStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store or ChatGPTCredentialStore()
        self._busy = False
        self._closed = threading.Event()
        self._generation = 0
        self.models: tuple[OpenAIModelChoice, ...] = ()

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def profile(self) -> ChatGPTProfile | None:
        return self.store.load_profile()

    def sign_in(self) -> None:
        self._start("Waiting for ChatGPT sign-in…", self._sign_in_task)

    def refresh_models(self) -> None:
        self._start("Loading OpenAI models…", self._models_task)

    def request_pose(
        self,
        *,
        model: str,
        image_path: str,
        image_width: float,
        image_height: float,
        class_name: str,
        keypoint_names: tuple[str, ...],
    ) -> None:
        def task() -> None:
            _profile, token = usable_access_token(self.store)
            proposal = request_pose_proposal(
                access_token=token,
                model=model,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                class_name=class_name,
                keypoint_names=keypoint_names,
            )
            if not self._closed.is_set():
                self.proposal_ready.emit(proposal)

        self._start(f"Asking {model} to label the current image…", task)

    def sign_out(self) -> None:
        def task() -> None:
            warning = ""
            try:
                revoke_profile(self.store)
            except Exception as exc:  # local tokens are cleared by the service
                warning = str(exc)
            self.models = ()
            if not self._closed.is_set():
                self.profile_changed.emit(self.store.load_profile())
                self.models_ready.emit(self.models)
                if warning:
                    self.error.emit(warning)

        self._start("Signing out of ChatGPT…", task)

    def shutdown(self) -> None:
        self._closed.set()
        self._generation += 1

    def _start(self, status: str, task: Callable[[], None]) -> None:
        if self._busy or self._closed.is_set():
            return
        self._busy = True
        self._generation += 1
        generation = self._generation
        self.busy_changed.emit(True)
        self.status_changed.emit(status)

        def runner() -> None:
            try:
                task()
            except Exception as exc:
                if not self._closed.is_set() and generation == self._generation:
                    self.error.emit(str(exc) or "OpenAI operation failed.")
            finally:
                if not self._closed.is_set() and generation == self._generation:
                    self._busy = False
                    self.busy_changed.emit(False)

        threading.Thread(target=runner, name="squeakpose-openai", daemon=True).start()

    def _sign_in_task(self) -> None:
        callback: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler_self) -> None:  # noqa: N802 - stdlib protocol name
                parsed = urlparse(handler_self.path)
                if parsed.path != "/auth/callback":
                    handler_self.send_error(404)
                    return
                callback.update(
                    {key: values[0] for key, values in parse_qs(parsed.query).items() if values}
                )
                body = (
                    b"<!doctype html><title>SqueakPose Studio</title>"
                    b"<h1>Return to SqueakPose Studio</h1>"
                    b"<p>ChatGPT sign-in has returned to the app. You can close this tab.</p>"
                )
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "text/html; charset=utf-8")
                handler_self.send_header("Content-Length", str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.timeout = 0.5
        try:
            port = int(server.server_address[1])
            attempt = create_authorization_attempt(
                redirect_uri=f"http://127.0.0.1:{port}/auth/callback",
                store=self.store,
            )
            self.authorization_url.emit(attempt.authorization_url)
            deadline = time.monotonic() + 180.0
            while not callback and not self._closed.is_set() and time.monotonic() < deadline:
                server.handle_request()
            if self._closed.is_set():
                return
            if not callback:
                raise RuntimeError("ChatGPT sign-in timed out. Try connecting again.")
            profile = complete_authorization(attempt, callback, store=self.store)
            if not profile.plan_usage_enabled:
                raise RuntimeError(
                    "ChatGPT connected, but plan usage was not enabled. Reconnect and approve "
                    "AI requests for SqueakPose Studio."
                )
            self.profile_changed.emit(profile)
            self.models = list_models(profile.access_token)
            self.models_ready.emit(self.models)
            self.status_changed.emit(f"Connected to ChatGPT as {profile.email or 'account'}.")
        finally:
            server.server_close()

    def _models_task(self) -> None:
        _profile, token = usable_access_token(self.store)
        self.models = list_models(token)
        if not self._closed.is_set():
            self.models_ready.emit(self.models)
            self.status_changed.emit(f"Loaded {len(self.models)} OpenAI model choices.")


__all__ = ["OpenAIController"]
