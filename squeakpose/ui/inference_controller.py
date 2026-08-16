"""Standalone orchestration of planned one-shot video inference jobs."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from squeakpose.core import atomic_write_text
from squeakpose.services.inference import (
    InferenceJobPlan,
    InferencePassResult,
    InferenceRunAccumulator,
    InferenceRunPlan,
    InferenceRunSummary,
    prepare_inference_run,
)
from squeakpose.workers.process import WorkerJobController, WorkerJobResult, create_worker_config


class InferenceController(QObject):
    """Run an :class:`InferenceRunPlan` sequentially with exact job correlation."""

    busy_changed = pyqtSignal(bool)
    run_started = pyqtSignal(object)
    job_started = pyqtSignal(object)
    event_received = pyqtSignal(object, dict)
    progress = pyqtSignal(object, dict)
    output_received = pyqtSignal(str)
    pass_finished = pyqtSignal(object)
    completed = pyqtSignal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        controller_factory: Callable[[QObject], Any] = WorkerJobController,
        config_writer: Callable[[str, str, str, dict[str, Any]], str] = create_worker_config,
        manifest_writer: Callable[[str, str], None] = atomic_write_text,
        discard_outputs: Callable[[InferencePassResult], None] = lambda _result: None,
        schedule: Callable[[Callable[[], None]], None] = lambda callback: QTimer.singleShot(
            0, callback
        ),
        program: str = sys.executable,
        worker_module: str = "inference_worker",
        working_directory: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller_factory = controller_factory
        self._config_writer = config_writer
        self._manifest_writer = manifest_writer
        self._discard_outputs = discard_outputs
        self._schedule = schedule
        self._program = program
        self._worker_module = worker_module
        self._working_directory = working_directory
        self._plan: InferenceRunPlan | None = None
        self._accumulator: InferenceRunAccumulator | None = None
        self._pending: list[InferenceJobPlan] = []
        self._active_job: InferenceJobPlan | None = None
        self._active_controller: Any = None
        self._result_event: dict[str, Any] | None = None
        self._output_parts: list[str] = []
        self._cancel_requested = False
        self._run_canceled = False
        self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def active_job(self) -> InferenceJobPlan | None:
        return self._active_job

    @property
    def pending_jobs(self) -> tuple[InferenceJobPlan, ...]:
        return tuple(self._pending)

    @property
    def process_controller(self) -> Any:
        return self._active_controller

    def start(self, plan: InferenceRunPlan) -> None:
        if self._busy:
            raise RuntimeError("inference run already active")
        prepare_inference_run(plan)
        self._plan = plan
        self._accumulator = InferenceRunAccumulator(plan)
        self._pending = list(plan.jobs)
        self._run_canceled = False
        self._set_busy(True)
        self.run_started.emit(plan)
        self._start_next_job()

    def cancel(self) -> bool:
        controller = self._active_controller
        if controller is None or not controller.is_running:
            if self._busy and self._active_job is None:
                self._run_canceled = True
                self._pending = []
                self._finish_run()
                return True
            return False
        self._cancel_requested = True
        return bool(controller.cancel(kill_after_ms=5000))

    def shutdown(self) -> bool:
        controller = self._active_controller
        if controller is None:
            if self._busy:
                self._run_canceled = True
                self._pending = []
                self._finish_run()
                return True
            return False
        self._cancel_requested = True
        return bool(controller.shutdown())

    def _start_next_job(self) -> None:
        if not self._busy:
            return
        if not self._pending:
            self._finish_run()
            return
        job = self._pending.pop(0)
        self._active_job = job
        self._result_event = None
        self._output_parts = []
        self._cancel_requested = False
        try:
            config_path = self._config_writer(
                self._plan.project_root,
                os.path.dirname(job.csv_path),
                "inference",
                job.worker_config(),
            )
        except Exception as exc:
            self._record_job_failure(job, f"Could not write worker config: {exc}")
            self._schedule(self._start_next_job)
            return

        controller = self._controller_factory(self)
        self._active_controller = controller
        controller.event_received.connect(
            lambda event, owner=controller, expected=job: self._on_event(owner, expected, event)
        )
        controller.output_received.connect(
            lambda text, owner=controller, expected=job: self._capture_output(owner, expected, text)
        )
        controller.stderr_received.connect(
            lambda text, owner=controller, expected=job: self._capture_output(owner, expected, text)
        )
        controller.terminal.connect(
            lambda result, owner=controller, expected=job: self._on_terminal(
                owner, expected, result
            )
        )
        self.job_started.emit(job)
        controller.start(
            self._program,
            ["-m", self._worker_module, "--config", config_path],
            config_path=config_path,
            working_directory=self._working_directory,
            start_timeout_ms=1000,
        )

    def _on_event(self, owner: Any, job: InferenceJobPlan, event: dict[str, Any]) -> None:
        if owner is not self._active_controller or job != self._active_job:
            return
        payload = dict(event)
        self.event_received.emit(job, payload)
        event_type = str(payload.get("event") or "")
        if event_type == "progress":
            self.progress.emit(job, payload)
        elif event_type == "result":
            self._result_event = payload
        elif event_type == "error":
            self._result_event = {
                "event": "result",
                "csv_path": job.csv_path,
                "rows_written": 0,
                "processed_frames": 0,
                "canceled": False,
                "had_error": True,
                "error_message": str(payload.get("error_message") or "Inference worker error"),
                "mode": job.workflow,
            }

    def _capture_output(self, owner: Any, job: InferenceJobPlan, text: str) -> None:
        if owner is not self._active_controller or job != self._active_job:
            return
        if text:
            value = str(text).rstrip()
            self._output_parts.append(value)
            self.output_received.emit(value)

    def _on_terminal(
        self,
        owner: Any,
        job: InferenceJobPlan,
        process_result: WorkerJobResult,
    ) -> None:
        if owner is not self._active_controller or job != self._active_job:
            return
        canceled = self._cancel_requested or process_result.state == "cancelled"
        exit_code = process_result.exit_code if process_result.exit_code is not None else 1
        stderr = (process_result.stderr or "\n".join(self._output_parts)).strip()
        result = self._accumulator.record(
            job,
            self._result_event,
            exit_code=exit_code,
            crashed=process_result.state == "failed",
            cancel_requested=canceled,
            stderr=stderr,
        )
        self._discard_outputs(result)
        self.pass_finished.emit(result)
        self._active_job = None
        self._active_controller = None
        self._result_event = None
        self._output_parts = []
        self._cancel_requested = False
        if result.canceled:
            self._run_canceled = True
            self._pending = []
        self._schedule(self._start_next_job)

    def _record_job_failure(self, job: InferenceJobPlan, message: str) -> None:
        result = self._accumulator.record(
            job,
            {"event": "error", "error_message": message},
            exit_code=1,
        )
        self._discard_outputs(result)
        self.pass_finished.emit(result)
        self._active_job = None

    def _finish_run(self) -> None:
        accumulator = self._accumulator
        if accumulator is None:
            return
        summary = accumulator.finalize(
            canceled=self._run_canceled,
            writer=self._manifest_writer,
        )
        self._plan = None
        self._accumulator = None
        self._pending = []
        self._active_job = None
        self._active_controller = None
        self._set_busy(False)
        self.completed.emit(summary)

    def _set_busy(self, busy: bool) -> None:
        value = bool(busy)
        if value == self._busy:
            return
        self._busy = value
        self.busy_changed.emit(value)


__all__ = ["InferenceController", "InferenceRunSummary"]
