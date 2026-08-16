"""Common child-process lifecycle helpers used by Qt owners."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from squeakpose.project.safety import require_path_within_project
from squeakpose.workers.protocol import JsonLineBuffer, WorkerProtocolError, parse_event_line

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerJobResult:
    """Final state emitted exactly once by a :class:`WorkerJobController`."""

    state: str
    exit_code: int | None = None
    exit_status: Any = None
    error_message: str = ""
    stderr: str = ""

    @property
    def succeeded(self) -> bool:
        return self.state == "finished" and self.exit_code == 0


class WorkerJobController(QObject):
    """Own one newline-delimited JSON worker process from start through cleanup."""

    event_received = pyqtSignal(dict)
    output_received = pyqtSignal(str)
    stderr_received = pyqtSignal(str)
    terminal = pyqtSignal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        process_factory: Callable[[QObject], Any] | None = None,
        schedule: Callable[[int, Callable[[], None]], None] | None = None,
    ):
        super().__init__(parent)
        self._process_factory = process_factory or QProcess
        self._schedule = schedule or QTimer.singleShot
        self._process: Any = None
        self._config_path: str | None = None
        self._stdout = JsonLineBuffer()
        self._stderr_pending = ""
        self._stderr_parts: list[str] = []
        self._started = False
        self._cancel_requested = False
        self._process_error = ""
        self._terminal_result: WorkerJobResult | None = None

    @property
    def process(self) -> Any:
        return self._process

    @property
    def config_path(self) -> str | None:
        return self._config_path

    @property
    def terminal_result(self) -> WorkerJobResult | None:
        return self._terminal_result

    @property
    def is_running(self) -> bool:
        if self._process is None or self._terminal_result is not None:
            return False
        try:
            return self._process.state() != QProcess.ProcessState.NotRunning
        except RuntimeError:
            return False

    def start(
        self,
        program: str,
        arguments: list[str],
        *,
        config_path: str | None = None,
        working_directory: str | None = None,
        start_timeout_ms: int = 3000,
    ) -> bool:
        """Start this one-shot controller and synchronously confirm process startup."""
        if self._started:
            raise RuntimeError("worker job controllers are one-shot")
        self._started = True
        self._config_path = config_path
        try:
            process = self._process_factory(self)
            self._process = process
            process.setProgram(program)
            process.setArguments(list(arguments))
            if working_directory:
                process.setWorkingDirectory(working_directory)
            process.readyReadStandardOutput.connect(self._read_stdout)
            process.readyReadStandardError.connect(self._read_stderr)
            process.errorOccurred.connect(self._process_failed)
            process.finished.connect(self._process_finished)
            process.start()
            started = process.waitForStarted(max(0, int(start_timeout_ms)))
        except Exception as exc:
            self._process_error = str(exc) or "Could not start worker process."
            self._complete("start_failed", error_message=self._process_error)
            return False
        if started:
            return True

        if not self._process_error:
            self._process_error = self._error_string(process) or "Could not start worker process."
        self._complete("start_failed", error_message=self._process_error)
        return False

    def cancel(self, *, kill_after_ms: int = 5000) -> bool:
        """Request termination and force-kill this job if it remains active."""
        process = self._process
        if process is None or self._terminal_result is not None:
            return False
        self._cancel_requested = True
        return request_qprocess_stop(
            process,
            schedule=self._schedule,
            force_kill=self._force_kill,
            kill_after_ms=kill_after_ms,
        )

    def shutdown(
        self,
        *,
        terminate_timeout_ms: int = 2000,
        kill_timeout_ms: int = 1000,
    ) -> bool:
        """Synchronously stop an owned job, clean up, and emit its terminal result."""
        process = self._process
        self._cancel_requested = True
        stopped = shutdown_qprocess(
            process,
            terminate_timeout_ms=terminate_timeout_ms,
            kill_timeout_ms=kill_timeout_ms,
        )
        if self._terminal_result is None:
            self._drain_output()
            self._complete(
                "cancelled" if stopped else "failed",
                error_message="" if stopped else "Worker process did not stop.",
            )
        return stopped

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or self._terminal_result is not None:
            return
        try:
            data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        for line in self._stdout.feed(data):
            self._deliver_stdout_line(line.strip())

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or self._terminal_result is not None:
            return
        try:
            data = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        if not data:
            return
        self._stderr_parts.append(data)
        self._stderr_pending += data
        while "\n" in self._stderr_pending:
            line, self._stderr_pending = self._stderr_pending.split("\n", 1)
            if line.strip():
                self.stderr_received.emit(line.strip())

    def _deliver_stdout_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self.output_received.emit(line)
            return
        self.event_received.emit(event)

    def _process_failed(self, error: Any) -> None:
        process = self._process
        self._process_error = self._error_string(process) or "Worker process error."
        if error == QProcess.ProcessError.FailedToStart and not self._cancel_requested:
            self._drain_output()
            self._complete("start_failed", error_message=self._process_error)

    def _process_finished(self, exit_code: int, exit_status: Any) -> None:
        if self._terminal_result is not None:
            return
        self._drain_output()
        if self._cancel_requested:
            state = "cancelled"
        elif int(exit_code) == 0:
            state = "finished"
        else:
            state = "failed"
        self._complete(
            state,
            exit_code=int(exit_code),
            exit_status=exit_status,
            error_message=self._process_error,
        )

    def _drain_output(self) -> None:
        self._read_stdout()
        self._read_stderr()
        pending_stdout = self._stdout.finish()
        if pending_stdout:
            self._deliver_stdout_line(pending_stdout.strip())
        if self._stderr_pending.strip():
            self.stderr_received.emit(self._stderr_pending.strip())
        self._stderr_pending = ""

    def _force_kill(self) -> None:
        process = self._process
        if process is None or self._terminal_result is not None:
            return
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
        except RuntimeError:
            return

    def _complete(
        self,
        state: str,
        *,
        exit_code: int | None = None,
        exit_status: Any = None,
        error_message: str = "",
    ) -> None:
        if self._terminal_result is not None:
            return
        config_path = self._config_path
        self._config_path = None
        remove_file_quietly(config_path)
        result = WorkerJobResult(
            state=state,
            exit_code=exit_code,
            exit_status=exit_status,
            error_message=error_message,
            stderr="".join(self._stderr_parts),
        )
        self._terminal_result = result
        self.terminal.emit(result)
        process = self._process
        self._process = None
        if process is not None:
            try:
                process.deleteLater()
            except (AttributeError, RuntimeError):
                pass

    @staticmethod
    def _error_string(process: Any) -> str:
        if process is None:
            return ""
        try:
            return str(process.errorString() or "")
        except (AttributeError, RuntimeError):
            return ""


@dataclass(frozen=True, slots=True)
class PersistentWorkerResult:
    """Final state for one generation of a persistent worker session."""

    state: str
    exit_code: int | None = None
    exit_status: Any = None
    error_message: str = ""
    stderr: str = ""

    @property
    def succeeded(self) -> bool:
        return self.state in {"finished", "stopped"} and self.exit_code in {None, 0}


class PersistentWorkerSession(QObject):
    """Own a restartable newline-delimited JSON request/response worker."""

    started = pyqtSignal()
    ready = pyqtSignal()
    event_received = pyqtSignal(dict)
    output_received = pyqtSignal(str)
    stderr_received = pyqtSignal(str)
    stopped = pyqtSignal(object)
    terminal = pyqtSignal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        process_factory: Callable[[QObject], Any] | None = None,
        schedule: Callable[[int, Callable[[], None]], None] | None = None,
    ):
        super().__init__(parent)
        self._process_factory = process_factory or QProcess
        self._schedule = schedule or QTimer.singleShot
        self._process: Any = None
        self._stdout = JsonLineBuffer()
        self._stderr_pending = ""
        self._stderr_parts: list[str] = []
        self._ready = False
        self._shutdown_requested = False
        self._process_error = ""
        self._terminal_result: PersistentWorkerResult | None = None
        self._generation = 0
        self._emitting_terminal = False

    @property
    def process(self) -> Any:
        return self._process

    @property
    def is_ready(self) -> bool:
        return self._ready and self.is_running

    @property
    def is_running(self) -> bool:
        process = self._process
        if process is None or self._terminal_result is not None:
            return False
        try:
            return process.state() != QProcess.ProcessState.NotRunning
        except RuntimeError:
            return False

    @property
    def stderr(self) -> str:
        return "".join(self._stderr_parts)

    @property
    def terminal_result(self) -> PersistentWorkerResult | None:
        return self._terminal_result

    def start(
        self,
        program: str,
        arguments: list[str],
        *,
        working_directory: str | None = None,
        start_timeout_ms: int = 3000,
    ) -> bool:
        """Start a new worker generation after any prior generation has stopped."""

        if self._emitting_terminal:
            raise RuntimeError("cannot restart a persistent worker from its terminal signal")
        if self._process is not None:
            raise RuntimeError("persistent worker session is already active")

        self._generation += 1
        generation = self._generation
        self._reset_generation_state()
        try:
            process = self._process_factory(self)
            self._process = process
            process.setProgram(program)
            process.setArguments(list(arguments))
            if working_directory:
                process.setWorkingDirectory(working_directory)
            process.readyReadStandardOutput.connect(
                lambda process=process, generation=generation: self._read_stdout(
                    process, generation
                )
            )
            process.readyReadStandardError.connect(
                lambda process=process, generation=generation: self._read_stderr(
                    process, generation
                )
            )
            process.errorOccurred.connect(
                lambda error, process=process, generation=generation: self._process_failed(
                    process, generation, error
                )
            )
            process.finished.connect(
                lambda exit_code, exit_status, process=process, generation=generation: (
                    self._process_finished(
                        process,
                        generation,
                        exit_code,
                        exit_status,
                    )
                )
            )
            process.start()
            started = process.waitForStarted(max(0, int(start_timeout_ms)))
        except Exception as exc:
            self._process_error = str(exc) or "Could not start worker process."
            self._complete_generation(
                process=self._process,
                generation=generation,
                state="start_failed",
                error_message=self._process_error,
            )
            return False
        if self._terminal_result is not None:
            return False
        if not started:
            self._process_error = self._error_string(process) or "Could not start worker process."
            self._complete_generation(
                process=process,
                generation=generation,
                state="start_failed",
                error_message=self._process_error,
            )
            return False
        self.started.emit()
        return True

    def send_request(
        self,
        request: Mapping[str, Any],
        *,
        require_ready: bool = True,
    ) -> bool:
        """Write one compact JSON request line to the active worker."""

        process = self._process
        if (
            process is None
            or not self.is_running
            or self._shutdown_requested
            or (require_ready and not self._ready)
        ):
            return False
        payload = (json.dumps(dict(request), separators=(",", ":")) + "\n").encode("utf-8")
        try:
            return int(process.write(payload)) >= 0
        except (AttributeError, RuntimeError):
            return False

    def shutdown(
        self,
        *,
        request: Mapping[str, Any] | None = None,
        terminate_after_ms: int = 1000,
        kill_after_ms: int = 4000,
    ) -> bool:
        """Request protocol shutdown, then schedule terminate and kill escalation."""

        process = self._process
        if process is None or not self.is_running or self._shutdown_requested:
            return False
        generation = self._generation
        shutdown_request = request if request is not None else {"command": "shutdown"}
        self.send_request(shutdown_request, require_ready=False)
        self._shutdown_requested = True

        terminate_delay = max(0, int(terminate_after_ms))
        kill_delay = max(terminate_delay, int(kill_after_ms))

        def terminate_active() -> None:
            self._terminate_if_active(process, generation)

        def kill_active() -> None:
            self._kill_if_active(process, generation)

        self._schedule(
            terminate_delay,
            terminate_active,
        )
        self._schedule(
            kill_delay,
            kill_active,
        )
        return True

    def _reset_generation_state(self) -> None:
        self._stdout = JsonLineBuffer()
        self._stderr_pending = ""
        self._stderr_parts = []
        self._ready = False
        self._shutdown_requested = False
        self._process_error = ""
        self._terminal_result = None

    def _is_active_generation(self, process: Any, generation: int) -> bool:
        return process is self._process and generation == self._generation

    def _read_stdout(self, process: Any, generation: int) -> None:
        if not self._is_active_generation(process, generation):
            return
        try:
            data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        for line in self._stdout.feed(data):
            self._deliver_stdout_line(line.strip())

    def _read_stderr(self, process: Any, generation: int) -> None:
        if not self._is_active_generation(process, generation):
            return
        try:
            data = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        if not data:
            return
        self._stderr_parts.append(data)
        self._stderr_pending += data
        while "\n" in self._stderr_pending:
            line, self._stderr_pending = self._stderr_pending.split("\n", 1)
            if line.strip():
                self.stderr_received.emit(line.strip())

    def _deliver_stdout_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self.output_received.emit(line)
            return
        if event.get("event") == "ready" and not self._ready:
            self._ready = True
            self.ready.emit()
        self.event_received.emit(event)

    def _process_failed(self, process: Any, generation: int, error: Any) -> None:
        if not self._is_active_generation(process, generation):
            return
        self._process_error = self._error_string(process) or "Worker process error."
        if error == QProcess.ProcessError.FailedToStart and not self._shutdown_requested:
            self._drain_output(process, generation)
            self._complete_generation(
                process=process,
                generation=generation,
                state="start_failed",
                error_message=self._process_error,
            )

    def _process_finished(
        self,
        process: Any,
        generation: int,
        exit_code: int,
        exit_status: Any,
    ) -> None:
        if not self._is_active_generation(process, generation):
            return
        self._drain_output(process, generation)
        if self._shutdown_requested:
            state = "stopped"
        elif int(exit_code) == 0:
            state = "finished"
        else:
            state = "failed"
        self._complete_generation(
            process=process,
            generation=generation,
            state=state,
            exit_code=int(exit_code),
            exit_status=exit_status,
            error_message=self._process_error,
        )

    def _drain_output(self, process: Any, generation: int) -> None:
        self._read_stdout(process, generation)
        self._read_stderr(process, generation)
        pending_stdout = self._stdout.finish()
        if pending_stdout:
            self._deliver_stdout_line(pending_stdout.strip())
        if self._stderr_pending.strip():
            self.stderr_received.emit(self._stderr_pending.strip())
        self._stderr_pending = ""

    def _terminate_if_active(self, process: Any, generation: int) -> None:
        if not self._is_active_generation(process, generation):
            return
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.terminate()
        except RuntimeError:
            return

    def _kill_if_active(self, process: Any, generation: int) -> None:
        if not self._is_active_generation(process, generation):
            return
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
        except RuntimeError:
            return

    def _complete_generation(
        self,
        *,
        process: Any,
        generation: int,
        state: str,
        exit_code: int | None = None,
        exit_status: Any = None,
        error_message: str = "",
    ) -> None:
        if not self._is_active_generation(process, generation) or self._terminal_result is not None:
            return
        result = PersistentWorkerResult(
            state=state,
            exit_code=exit_code,
            exit_status=exit_status,
            error_message=error_message,
            stderr=self.stderr,
        )
        self._terminal_result = result
        self._ready = False
        self._process = None
        self._emitting_terminal = True
        try:
            self.terminal.emit(result)
            self.stopped.emit(result)
        finally:
            self._emitting_terminal = False
        if process is not None:
            try:
                process.deleteLater()
            except (AttributeError, RuntimeError):
                pass

    @staticmethod
    def _error_string(process: Any) -> str:
        if process is None:
            return ""
        try:
            return str(process.errorString() or "")
        except (AttributeError, RuntimeError):
            return ""


def create_worker_config(
    project_root: str,
    directory: str,
    kind: str,
    payload: Mapping[str, Any],
) -> str:
    """Create a unique, owner-readable worker configuration inside a project."""

    root = os.path.abspath(project_root)
    config_dir = require_path_within_project(
        root,
        directory,
        purpose="worker configuration directory",
    )
    os.makedirs(config_dir, exist_ok=True)
    safe_kind = re.sub(r"[^a-z0-9_-]+", "_", str(kind).strip().lower()).strip("_")
    if not safe_kind:
        raise ValueError("worker configuration kind must not be empty")

    fd = -1
    path = ""
    try:
        fd, path = tempfile.mkstemp(
            prefix=f".{safe_kind}_config_",
            suffix=".json",
            dir=config_dir,
        )
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(dict(payload), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except Exception:
        if fd >= 0:
            os.close(fd)
        if path:
            remove_file_quietly(path)
        raise


def remove_file_quietly(path: str | None) -> None:
    """Best-effort cleanup for an owned temporary configuration file."""
    if not path:
        return
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
    except OSError:
        logger.warning(
            "Could not remove owned temporary file",
            exc_info=True,
            extra={
                "event": "owned_temporary_cleanup_failed",
                "operation": "remove_owned_temporary",
                "target_path": path,
            },
        )


def shutdown_qprocess(
    process: Any,
    *,
    terminate_timeout_ms: int = 2000,
    kill_timeout_ms: int = 1000,
) -> bool:
    """Synchronously stop a QProcess-like object owned by a closing window."""
    if process is None:
        return True

    from PyQt6.QtCore import QProcess

    if process.state() == QProcess.ProcessState.NotRunning:
        return True
    try:
        process.blockSignals(True)
    except (AttributeError, RuntimeError):
        pass
    process.terminate()
    if process.waitForFinished(terminate_timeout_ms):
        return True
    process.kill()
    process.waitForFinished(kill_timeout_ms)
    return process.state() == QProcess.ProcessState.NotRunning


def request_qprocess_stop(
    process: Any,
    *,
    schedule: Callable[[int, Callable[[], None]], None],
    force_kill: Callable[[], None],
    kill_after_ms: int = 5000,
) -> bool:
    """Request graceful termination and schedule an owned force-kill check."""
    if process is None:
        return False

    from PyQt6.QtCore import QProcess

    if process.state() == QProcess.ProcessState.NotRunning:
        return False
    process.terminate()

    def force_kill_original_process() -> None:
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                force_kill()
        except RuntimeError:
            return

    schedule(max(0, int(kill_after_ms)), force_kill_original_process)
    return True
