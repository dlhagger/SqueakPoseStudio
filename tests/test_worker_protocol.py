import io
import json
import os
import unittest
from tempfile import TemporaryDirectory

from PyQt6.QtCore import QProcess

from squeakpose.workers.process import create_worker_config, request_qprocess_stop
from squeakpose.workers.protocol import (
    JsonLineBuffer,
    WorkerProtocolError,
    parse_event_line,
    read_config,
    validate_event,
    write_event,
)


class WorkerProtocolTests(unittest.TestCase):
    def test_event_round_trip_preserves_request_id(self):
        stream = io.StringIO()
        write_event(
            {"event": "result", "request_id": "request-7", "value": 42},
            stream=stream,
        )

        event = parse_event_line(stream.getvalue())

        self.assertEqual(event.event, "result")
        self.assertEqual(event.request_id, "request-7")
        self.assertTrue(event.is_terminal)
        self.assertEqual(event.as_dict()["value"], 42)

    def test_validate_event_rejects_missing_event_name(self):
        with self.assertRaises(WorkerProtocolError):
            validate_event({"message": "missing envelope"})
        with self.assertRaises(WorkerProtocolError):
            parse_event_line("not json")

    def test_json_line_buffer_handles_partial_and_multiple_lines(self):
        buffer = JsonLineBuffer()

        first = buffer.feed('{"event":"started"}\n{"event":"pro')
        second = buffer.feed('gress"}\n\n{"event":"result"}')

        self.assertEqual(first, ['{"event":"started"}'])
        self.assertEqual(second, ['{"event":"progress"}'])
        self.assertEqual(buffer.finish(), '{"event":"result"}')
        self.assertEqual(buffer.pending, "")

    def test_read_config_requires_json_object(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(["not", "an", "object"], fh)

            with self.assertRaises(WorkerProtocolError):
                read_config(config_path)

    def test_worker_config_is_unique_project_contained_and_owner_only(self):
        with TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "logs")

            first = create_worker_config(tmp, config_dir, "video review", {"value": 1})
            second = create_worker_config(tmp, config_dir, "video review", {"value": 2})

            self.assertNotEqual(first, second)
            self.assertEqual(read_config(first), {"value": 1})
            self.assertEqual(read_config(second), {"value": 2})
            self.assertEqual(os.stat(first).st_mode & 0o777, 0o600)
            self.assertEqual(os.path.commonpath((tmp, first)), os.path.abspath(tmp))

    def test_worker_config_creation_rejects_directory_outside_project(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            with self.assertRaises(ValueError):
                create_worker_config(tmp, outside, "analysis", {"value": 1})

    def test_worker_config_serialization_failure_removes_partial_file(self):
        with TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "logs")

            with self.assertRaises(TypeError):
                create_worker_config(tmp, config_dir, "analysis", {"bad": object()})

            self.assertEqual(os.listdir(config_dir), [])

    def test_read_config_rejects_oversized_and_symlinked_files(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump({"payload": "too large"}, fh)
            with self.assertRaises(WorkerProtocolError):
                read_config(config_path, max_bytes=8)

            symlink_path = os.path.join(tmp, "linked.json")
            try:
                os.symlink(config_path, symlink_path)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaises(WorkerProtocolError):
                read_config(symlink_path)

            with self.assertRaises(WorkerProtocolError):
                read_config(tmp)

    def test_stop_request_terminates_and_schedules_force_kill(self):
        calls = []

        class FakeProcess:
            process_state = QProcess.ProcessState.Running

            def state(self):
                return self.process_state

            def terminate(self):
                calls.append("terminate")

        process = FakeProcess()
        force_kill = lambda: calls.append("kill")

        requested = request_qprocess_stop(
            process,
            schedule=lambda delay, callback: calls.append((delay, callback)),
            force_kill=force_kill,
            kill_after_ms=2500,
        )

        self.assertTrue(requested)
        self.assertEqual(calls[0], "terminate")
        delay, callback = calls[1]
        self.assertEqual(delay, 2500)

        process.process_state = QProcess.ProcessState.NotRunning
        callback()
        self.assertNotIn("kill", calls)

    def test_stop_request_force_kills_the_original_process_if_still_running(self):
        scheduled = []

        class FakeProcess:
            def state(self):
                return QProcess.ProcessState.Running

            def terminate(self):
                pass

        request_qprocess_stop(
            FakeProcess(),
            schedule=lambda _delay, callback: scheduled.append(callback),
            force_kill=lambda: scheduled.append("killed"),
        )

        scheduled[0]()

        self.assertEqual(scheduled[1], "killed")


if __name__ == "__main__":
    unittest.main()
