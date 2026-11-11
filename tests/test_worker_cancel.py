import io
import json
import threading
import unittest

from squeakpose.services.cancellation import (
    OperationCanceled,
    cancel_check_scope,
    cancel_requested,
    raise_if_canceled,
)
from squeakpose.workers.protocol import (
    CANCEL_COMMAND,
    cancel_command_bytes,
    start_cancel_listener,
)


class CancelCommandTests(unittest.TestCase):
    def test_cancel_command_is_one_json_line(self):
        raw = cancel_command_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertEqual(json.loads(raw), {"command": CANCEL_COMMAND})

    def test_listener_fires_once_on_cancel_line_and_ignores_noise(self):
        stream = io.StringIO(
            "not json\n"
            + json.dumps({"command": "other"})
            + "\n"
            + cancel_command_bytes().decode("utf-8")
            + cancel_command_bytes().decode("utf-8")
        )
        fired = []
        done = threading.Event()

        def on_cancel():
            fired.append(True)
            done.set()

        thread = start_cancel_listener(on_cancel, stream=stream)
        thread.join(timeout=2)

        self.assertTrue(done.is_set())
        self.assertEqual(fired, [True])
        self.assertTrue(thread.daemon)

    def test_listener_exits_quietly_at_eof_without_cancel(self):
        fired = []
        thread = start_cancel_listener(lambda: fired.append(True), stream=io.StringIO("{}\n"))
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(fired, [])


class CancellationScopeTests(unittest.TestCase):
    def test_no_scope_never_cancels(self):
        self.assertFalse(cancel_requested())
        raise_if_canceled()  # does not raise

    def test_scope_raises_base_exception_that_except_exception_cannot_swallow(self):
        state = {"cancel": False}
        with cancel_check_scope(lambda: state["cancel"]):
            raise_if_canceled()
            state["cancel"] = True
            with self.assertRaises(OperationCanceled):
                try:
                    raise_if_canceled()
                except Exception:  # noqa: BLE001 - proves the fallback cannot catch it
                    self.fail("OperationCanceled must not be an Exception subclass")
        self.assertFalse(issubclass(OperationCanceled, Exception))

    def test_scope_restores_previous_check(self):
        with cancel_check_scope(lambda: True):
            with cancel_check_scope(lambda: False):
                self.assertFalse(cancel_requested())
            self.assertTrue(cancel_requested())
        self.assertFalse(cancel_requested())


if __name__ == "__main__":
    unittest.main()
