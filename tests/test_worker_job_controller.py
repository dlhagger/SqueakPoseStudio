import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PyQt6.QtCore import QCoreApplication, QProcess

from squeakpose.workers.process import WorkerJobController
from squeakpose.workers.protocol import cancel_command_bytes


class _FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self._callbacks):
            callback(*args)


class _FakeProcess:
    def __init__(self, _parent, *, starts=True):
        self.readyReadStandardOutput = _FakeSignal()
        self.readyReadStandardError = _FakeSignal()
        self.errorOccurred = _FakeSignal()
        self.finished = _FakeSignal()
        self.starts = starts
        self.program = ""
        self.arguments = []
        self.working_directory = ""
        self.process_state = QProcess.ProcessState.NotRunning
        self.stdout = b""
        self.stderr = b""
        self.terminate_calls = 0
        self.kill_calls = 0
        self.signals_blocked = False
        self.stdin = b""
        self.call_order = []

    def write(self, data):
        self.stdin += bytes(data)
        self.call_order.append("write")
        return len(data)

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = list(arguments)

    def setWorkingDirectory(self, working_directory):
        self.working_directory = working_directory

    def start(self):
        if self.starts:
            self.process_state = QProcess.ProcessState.Running

    def waitForStarted(self, _timeout):
        return self.starts

    def state(self):
        return self.process_state

    def readAllStandardOutput(self):
        data, self.stdout = self.stdout, b""
        return data

    def readAllStandardError(self):
        data, self.stderr = self.stderr, b""
        return data

    def errorString(self):
        return "injected start failure"

    def terminate(self):
        self.terminate_calls += 1
        self.call_order.append("terminate")

    def blockSignals(self, blocked):
        self.signals_blocked = bool(blocked)

    def waitForFinished(self, _timeout):
        self.process_state = QProcess.ProcessState.NotRunning
        return True

    def kill(self):
        self.kill_calls += 1
        self.process_state = QProcess.ProcessState.NotRunning


class WorkerJobControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_buffers_protocol_and_stderr_and_completes_once(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".analysis_config_test.json"
            config_path.write_text("{}", encoding="utf-8")
            processes = []

            def factory(parent):
                process = _FakeProcess(parent)
                processes.append(process)
                return process

            controller = WorkerJobController(process_factory=factory)
            events = []
            output = []
            stderr = []
            terminal = []
            controller.event_received.connect(events.append)
            controller.output_received.connect(output.append)
            controller.stderr_received.connect(stderr.append)
            controller.terminal.connect(terminal.append)

            self.assertTrue(
                controller.start(
                    "/python",
                    ["worker.py", "--config", str(config_path)],
                    config_path=str(config_path),
                    working_directory=tmp,
                )
            )
            process = processes[0]
            self.assertEqual(process.program, "/python")
            self.assertEqual(process.arguments[-1], str(config_path))
            self.assertEqual(process.working_directory, tmp)

            process.stdout = b'{"event":"started"}\nplain'
            process.readyReadStandardOutput.emit()
            process.stdout = b' output\n{"event":"result","value":1}'
            process.readyReadStandardOutput.emit()
            process.stderr = b"warning\npartial"
            process.readyReadStandardError.emit()
            process.process_state = QProcess.ProcessState.NotRunning
            process.finished.emit(0, QProcess.ExitStatus.NormalExit)
            process.finished.emit(0, QProcess.ExitStatus.NormalExit)

            self.assertEqual([event["event"] for event in events], ["started", "result"])
            self.assertEqual(output, ["plain output"])
            self.assertEqual(stderr, ["warning", "partial"])
            self.assertEqual(len(terminal), 1)
            self.assertTrue(terminal[0].succeeded)
            self.assertEqual(terminal[0].stderr, "warning\npartial")
            self.assertFalse(config_path.exists())

    def test_start_failure_cleans_config_and_notifies_once(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, ".analysis_config_test.json")
            Path(config_path).write_text("{}", encoding="utf-8")
            process = _FakeProcess(None, starts=False)
            controller = WorkerJobController(process_factory=lambda _parent: process)
            terminal = []
            controller.terminal.connect(terminal.append)

            self.assertFalse(
                controller.start("/missing", [], config_path=config_path, start_timeout_ms=1)
            )
            process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
            process.finished.emit(-1, QProcess.ExitStatus.CrashExit)

            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0].state, "start_failed")
            self.assertIn("injected", terminal[0].error_message)
            self.assertFalse(os.path.exists(config_path))

    def test_construction_failure_still_cleans_owned_config(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, ".analysis_config_test.json")
            Path(config_path).write_text("{}", encoding="utf-8")

            def fail_factory(_parent):
                raise RuntimeError("injected construction failure")

            controller = WorkerJobController(process_factory=fail_factory)
            terminal = []
            controller.terminal.connect(terminal.append)

            self.assertFalse(controller.start("/python", [], config_path=config_path))
            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0].state, "start_failed")
            self.assertIn("construction", terminal[0].error_message)
            self.assertFalse(os.path.exists(config_path))

    def test_cancel_escalates_to_kill_and_reports_cancelled(self):
        scheduled = []
        process = _FakeProcess(None)
        controller = WorkerJobController(
            process_factory=lambda _parent: process,
            schedule=lambda delay, callback: scheduled.append((delay, callback)),
        )
        terminal = []
        controller.terminal.connect(terminal.append)
        self.assertTrue(controller.start("/python", ["worker.py"]))

        self.assertTrue(controller.cancel(kill_after_ms=25))
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(scheduled[0][0], 25)
        scheduled[0][1]()
        self.assertEqual(process.kill_calls, 1)

        process.finished.emit(-1, QProcess.ExitStatus.CrashExit)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].state, "cancelled")

    def test_cancel_sends_stdin_cancel_request_before_terminate(self):
        process = _FakeProcess(None)
        controller = WorkerJobController(
            process_factory=lambda _parent: process,
            schedule=lambda _delay, _callback: None,
        )
        self.assertTrue(controller.start("/python", ["worker.py"]))

        self.assertTrue(controller.cancel(kill_after_ms=60_000))

        # The stdin line is the portable channel (Windows terminate() is a no-op
        # for console children); SIGTERM follows for POSIX.
        self.assertEqual(process.stdin, cancel_command_bytes())
        self.assertEqual(process.call_order, ["write", "terminate"])
        self.assertEqual(process.kill_calls, 0)

    def test_force_stop_kills_immediately_and_reports_cancelled(self):
        process = _FakeProcess(None)
        controller = WorkerJobController(
            process_factory=lambda _parent: process,
            schedule=lambda _delay, _callback: None,
        )
        terminal = []
        controller.terminal.connect(terminal.append)
        self.assertTrue(controller.start("/python", ["worker.py"]))
        controller.cancel(kill_after_ms=60_000)

        self.assertTrue(controller.force_stop())
        self.assertEqual(process.kill_calls, 1)
        process.finished.emit(-1, QProcess.ExitStatus.CrashExit)

        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].state, "cancelled")
        self.assertFalse(controller.force_stop())

    def test_shutdown_sends_cancel_request_first(self):
        process = _FakeProcess(None)
        controller = WorkerJobController(process_factory=lambda _parent: process)
        self.assertTrue(controller.start("/python", ["worker.py"]))

        controller.shutdown()

        self.assertEqual(process.stdin, cancel_command_bytes())
        self.assertEqual(process.call_order[:2], ["write", "terminate"])

    def test_cancel_tolerates_process_without_write(self):
        process = _FakeProcess(None)
        process.write = None
        controller = WorkerJobController(
            process_factory=lambda _parent: process,
            schedule=lambda _delay, _callback: None,
        )
        self.assertTrue(controller.start("/python", ["worker.py"]))

        self.assertTrue(controller.cancel())
        self.assertEqual(process.terminate_calls, 1)

    def test_reentrant_finished_during_event_delivery_keeps_result_before_terminal(self):
        process = _FakeProcess(None)
        controller = WorkerJobController(process_factory=lambda _parent: process)
        timeline = []

        def on_event(event):
            timeline.append(("event", event["event"]))
            if event["event"] == "progress":
                # Simulate a consumer slot that re-enters the event loop
                # (``QApplication.processEvents()``) after the child already
                # exited: Qt dispatches ``finished`` while the outer
                # ``_read_stdout`` is still delivering the chunk.
                process.process_state = QProcess.ProcessState.NotRunning
                process.finished.emit(0, QProcess.ExitStatus.NormalExit)

        controller.event_received.connect(on_event)
        controller.terminal.connect(lambda result: timeline.append(("terminal", result)))
        self.assertTrue(controller.start("/python", ["worker.py"]))

        process.stdout = b'{"event":"started"}\n{"event":"progress"}\n{"event":"result","ok":1}\n'
        process.readyReadStandardOutput.emit()

        kinds = [kind for kind, _ in timeline]
        self.assertEqual(kinds.count("terminal"), 1)
        self.assertEqual(
            [name for kind, name in timeline if kind == "event"],
            ["started", "progress", "result"],
        )
        self.assertGreater(kinds.index("terminal"), timeline.index(("event", "result")))
        self.assertEqual(timeline[-1][0], "terminal")
        self.assertTrue(timeline[-1][1].succeeded)
        self.assertIs(controller.terminal_result, timeline[-1][1])

    def test_reentrant_finished_also_flushes_unterminated_trailing_result(self):
        process = _FakeProcess(None)
        controller = WorkerJobController(process_factory=lambda _parent: process)
        events = []
        terminal = []

        def on_event(event):
            events.append(event["event"])
            if event["event"] == "progress":
                process.process_state = QProcess.ProcessState.NotRunning
                process.finished.emit(0, QProcess.ExitStatus.NormalExit)

        controller.event_received.connect(on_event)
        controller.terminal.connect(terminal.append)
        self.assertTrue(controller.start("/python", ["worker.py"]))

        process.stdout = b'{"event":"progress"}\n{"event":"result"}'
        process.readyReadStandardOutput.emit()

        self.assertEqual(events, ["progress", "result"])
        self.assertEqual(len(terminal), 1)
        self.assertTrue(terminal[0].succeeded)

    def test_shutdown_stops_process_cleans_config_and_notifies_once(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, ".analysis_config_test.json")
            Path(config_path).write_text("{}", encoding="utf-8")
            process = _FakeProcess(None)
            controller = WorkerJobController(process_factory=lambda _parent: process)
            terminal = []
            controller.terminal.connect(terminal.append)
            self.assertTrue(controller.start("/python", [], config_path=config_path))

            self.assertTrue(controller.shutdown())
            self.assertTrue(process.signals_blocked)
            self.assertEqual(process.terminate_calls, 1)
            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0].state, "cancelled")
            self.assertFalse(os.path.exists(config_path))


if __name__ == "__main__":
    unittest.main()
