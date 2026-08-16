import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PyQt6.QtCore import QCoreApplication, QProcess

from squeakpose.workers.process import WorkerJobController


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
