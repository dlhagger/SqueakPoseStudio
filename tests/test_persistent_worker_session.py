import unittest

from PyQt6.QtCore import QCoreApplication, QProcess

from squeakpose.workers.process import PersistentWorkerSession


class _FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self._callbacks):
            callback(*args)


class _FakePersistentProcess:
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
        self.writes = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.deleted = False

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = list(arguments)

    def setWorkingDirectory(self, directory):
        self.working_directory = directory

    def start(self):
        if self.starts:
            self.process_state = QProcess.ProcessState.Running

    def waitForStarted(self, _timeout):
        return self.starts

    def state(self):
        return self.process_state

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)

    def readAllStandardOutput(self):
        data, self.stdout = self.stdout, b""
        return data

    def readAllStandardError(self):
        data, self.stderr = self.stderr, b""
        return data

    def errorString(self):
        return "injected persistent worker failure"

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.process_state = QProcess.ProcessState.NotRunning

    def deleteLater(self):
        self.deleted = True


class PersistentWorkerSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_start_readiness_json_io_and_buffered_protocol_delivery(self):
        process = _FakePersistentProcess(None)
        session = PersistentWorkerSession(process_factory=lambda _parent: process)
        started = []
        ready = []
        events = []
        output = []
        stderr_lines = []
        terminal = []
        stopped = []
        session.started.connect(lambda: started.append(True))
        session.ready.connect(lambda: ready.append(True))
        session.event_received.connect(events.append)
        session.output_received.connect(output.append)
        session.stderr_received.connect(stderr_lines.append)
        session.terminal.connect(terminal.append)
        session.stopped.connect(stopped.append)

        self.assertTrue(session.start("/python", ["-m", "worker"], working_directory="/repo"))
        self.assertEqual(started, [True])
        self.assertEqual(process.program, "/python")
        self.assertEqual(process.arguments, ["-m", "worker"])
        self.assertEqual(process.working_directory, "/repo")
        self.assertFalse(session.is_ready)
        self.assertFalse(session.send_request({"command": "predict", "request_id": 1}))

        process.stdout = b'{"event":"ready"}\nplain'
        process.readyReadStandardOutput.emit()
        process.stdout = b' text\n{"event":"result","request_id":1}'
        process.readyReadStandardOutput.emit()
        self.assertTrue(session.is_ready)
        self.assertEqual(ready, [True])
        self.assertEqual(output, ["plain text"])
        self.assertEqual([event["event"] for event in events], ["ready"])

        self.assertTrue(session.send_request({"command": "predict", "request_id": 1}))
        self.assertEqual(
            process.writes,
            [b'{"command":"predict","request_id":1}\n'],
        )

        process.stderr = b"warning one\npartial"
        process.readyReadStandardError.emit()
        process.stderr = b" warning\n"
        process.readyReadStandardError.emit()
        process.process_state = QProcess.ProcessState.NotRunning
        process.finished.emit(0, QProcess.ExitStatus.NormalExit)
        process.finished.emit(0, QProcess.ExitStatus.NormalExit)

        self.assertEqual([event["event"] for event in events], ["ready", "result"])
        self.assertEqual(stderr_lines, ["warning one", "partial warning"])
        self.assertEqual(len(terminal), 1)
        self.assertEqual(len(stopped), 1)
        self.assertIs(terminal[0], stopped[0])
        self.assertEqual(terminal[0].state, "finished")
        self.assertEqual(terminal[0].stderr, "warning one\npartial warning\n")
        self.assertTrue(process.deleted)

    def test_shutdown_writes_command_then_terminates_and_kills(self):
        scheduled = []
        process = _FakePersistentProcess(None)
        session = PersistentWorkerSession(
            process_factory=lambda _parent: process,
            schedule=lambda delay, callback: scheduled.append((delay, callback)),
        )
        terminal = []
        stopped = []
        session.terminal.connect(terminal.append)
        session.stopped.connect(stopped.append)
        self.assertTrue(session.start("/python", ["worker.py"]))

        self.assertTrue(
            session.shutdown(
                request={"command": "shutdown", "request_id": 9},
                terminate_after_ms=25,
                kill_after_ms=75,
            )
        )
        self.assertFalse(session.shutdown())
        self.assertEqual(process.writes, [b'{"command":"shutdown","request_id":9}\n'])
        self.assertEqual([delay for delay, _callback in scheduled], [25, 75])

        scheduled[0][1]()
        self.assertEqual(process.terminate_calls, 1)
        scheduled[1][1]()
        self.assertEqual(process.kill_calls, 1)
        process.finished.emit(-1, QProcess.ExitStatus.CrashExit)
        process.finished.emit(-1, QProcess.ExitStatus.CrashExit)

        self.assertEqual(len(terminal), 1)
        self.assertEqual(len(stopped), 1)
        self.assertEqual(terminal[0].state, "stopped")

    def test_graceful_protocol_stop_prevents_escalation(self):
        scheduled = []
        process = _FakePersistentProcess(None)
        session = PersistentWorkerSession(
            process_factory=lambda _parent: process,
            schedule=lambda delay, callback: scheduled.append((delay, callback)),
        )
        events = []
        terminal = []
        session.event_received.connect(events.append)
        session.terminal.connect(terminal.append)
        self.assertTrue(session.start("/python", ["worker.py"]))
        self.assertTrue(session.shutdown())

        process.stdout = b'{"event":"stopped"}\n'
        process.process_state = QProcess.ProcessState.NotRunning
        process.finished.emit(0, QProcess.ExitStatus.NormalExit)
        for _delay, callback in scheduled:
            callback()

        self.assertEqual([event["event"] for event in events], ["stopped"])
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].state, "stopped")
        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(process.kill_calls, 0)

    def test_restart_ignores_old_process_signals_and_escalation_callbacks(self):
        scheduled = []
        processes = []

        def factory(parent):
            process = _FakePersistentProcess(parent)
            processes.append(process)
            return process

        session = PersistentWorkerSession(
            process_factory=factory,
            schedule=lambda delay, callback: scheduled.append((delay, callback)),
        )
        terminal = []
        session.terminal.connect(terminal.append)
        self.assertTrue(session.start("/python", ["worker.py"]))
        first = processes[0]
        self.assertTrue(session.shutdown(terminate_after_ms=10, kill_after_ms=20))
        first.process_state = QProcess.ProcessState.NotRunning
        first.finished.emit(0, QProcess.ExitStatus.NormalExit)

        self.assertTrue(session.start("/python", ["worker.py"]))
        second = processes[1]
        first.stdout = b'{"event":"result","request_id":"old"}\n'
        first.readyReadStandardOutput.emit()
        first.finished.emit(1, QProcess.ExitStatus.CrashExit)
        for _delay, callback in scheduled:
            callback()

        self.assertIs(session.process, second)
        self.assertTrue(session.is_running)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(second.terminate_calls, 0)
        self.assertEqual(second.kill_calls, 0)

        second.process_state = QProcess.ProcessState.NotRunning
        second.finished.emit(0, QProcess.ExitStatus.NormalExit)
        self.assertEqual(len(terminal), 2)

    def test_start_failure_and_overlapping_start_are_safe(self):
        failed = _FakePersistentProcess(None, starts=False)
        session = PersistentWorkerSession(process_factory=lambda _parent: failed)
        terminal = []
        session.terminal.connect(terminal.append)

        self.assertFalse(session.start("/missing", []))
        failed.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
        failed.finished.emit(-1, QProcess.ExitStatus.CrashExit)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].state, "start_failed")
        self.assertIn("injected", terminal[0].error_message)

        running = _FakePersistentProcess(None)
        session = PersistentWorkerSession(process_factory=lambda _parent: running)
        self.assertTrue(session.start("/python", []))
        with self.assertRaisesRegex(RuntimeError, "already active"):
            session.start("/python", [])


if __name__ == "__main__":
    unittest.main()
