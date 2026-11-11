import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.services.cancellation import (
    OperationCanceled,
    cancel_check_scope,
    cancel_requested,
    raise_if_canceled,
)
from squeakpose.workers.protocol import cancel_marker_path, start_cancel_file_watcher


class CancelFileWatcherTests(unittest.TestCase):
    def test_marker_path_sits_next_to_config(self):
        config = os.path.join("project", "logs", ".analysis_config_abc.json")
        self.assertEqual(cancel_marker_path(config), f"{config}.cancel")

    def test_watcher_fires_once_when_marker_appears(self):
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / ".train_config_x.json.cancel"
            fired = []
            done = threading.Event()

            def on_cancel():
                fired.append(True)
                done.set()

            thread = start_cancel_file_watcher(str(marker), on_cancel, interval=0.01)
            self.assertTrue(thread.daemon)
            self.assertFalse(done.wait(0.05))
            marker.write_text("", encoding="utf-8")

            self.assertTrue(done.wait(2))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(fired, [True])

    def test_watcher_never_reads_stdin_and_keeps_polling_through_errors(self):
        calls = []
        done = threading.Event()

        def flaky_exists(path):
            calls.append(path)
            if len(calls) == 1:
                raise OSError("transient")
            return len(calls) >= 3

        thread = start_cancel_file_watcher(
            "/project/logs/.cfg.json.cancel",
            done.set,
            interval=0.01,
            exists=flaky_exists,
        )

        self.assertTrue(done.wait(2))
        thread.join(timeout=2)
        self.assertEqual(len(calls), 3)

    def test_watcher_without_marker_does_not_fire(self):
        with TemporaryDirectory() as tmp:
            fired = []
            start_cancel_file_watcher(
                os.path.join(tmp, "missing.cancel"), lambda: fired.append(True), interval=0.01
            )
            threading.Event().wait(0.05)
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
