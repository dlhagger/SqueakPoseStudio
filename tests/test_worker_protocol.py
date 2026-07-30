import io
import json
import os
import unittest
from tempfile import TemporaryDirectory

from squeakpose.workers.protocol import (
    JsonLineBuffer,
    WorkerProtocolError,
    parse_event_line,
    read_config,
    validate_event,
    write_event,
)
from squeakpose.workers.process import request_qprocess_stop
from PyQt6.QtCore import QProcess


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

    def test_stop_request_terminates_and_schedules_force_kill(self):
        calls = []

        class FakeProcess:
            def state(self):
                return QProcess.ProcessState.Running

            def terminate(self):
                calls.append("terminate")

        force_kill = lambda: calls.append("kill")

        requested = request_qprocess_stop(
            FakeProcess(),
            schedule=lambda delay, callback: calls.append((delay, callback)),
            force_kill=force_kill,
            kill_after_ms=2500,
        )

        self.assertTrue(requested)
        self.assertEqual(calls[0], "terminate")
        self.assertEqual(calls[1], (2500, force_kill))


if __name__ == "__main__":
    unittest.main()
