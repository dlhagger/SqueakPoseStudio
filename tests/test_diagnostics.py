import json
import logging
import os
import stat
import unittest
from tempfile import TemporaryDirectory

from squeakpose.diagnostics import (
    configure_project_logging,
    reset_project_logging,
)


class ProjectLoggingTests(unittest.TestCase):
    def tearDown(self):
        reset_project_logging()

    @staticmethod
    def _flush_project_handlers() -> None:
        for handler in logging.getLogger().handlers:
            if getattr(handler, "_squeakpose_project_handler", False):
                handler.flush()

    def test_project_log_contains_structured_event_and_exception(self):
        with TemporaryDirectory() as tmp:
            log_path = configure_project_logging(tmp)
            logger = logging.getLogger("squeakpose.test")
            logger.info(
                "Saved an artifact",
                extra={
                    "event": "artifact_saved",
                    "operation": "test_logging",
                    "project_root": tmp,
                    "target_path": os.path.join(tmp, "artifact.txt"),
                },
            )
            try:
                raise OSError("injected failure")
            except OSError:
                logger.exception(
                    "Artifact failed",
                    extra={"event": "artifact_failed", "operation": "test_logging"},
                )
            self._flush_project_handlers()

            with open(log_path, "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]

            saved = next(event for event in events if event.get("event") == "artifact_saved")
            failed = next(event for event in events if event.get("event") == "artifact_failed")
            self.assertEqual(saved["level"], "INFO")
            self.assertEqual(saved["operation"], "test_logging")
            self.assertEqual(saved["project_root"], tmp)
            self.assertIn("OSError: injected failure", failed["exception"])
            self.assertEqual(stat.S_IMODE(os.stat(log_path).st_mode), 0o600)

    def test_switching_projects_routes_new_events_only_to_the_new_log(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            first_log = configure_project_logging(first)
            logging.getLogger("squeakpose.test").info(
                "First project event", extra={"event": "first_project"}
            )
            self._flush_project_handlers()

            second_log = configure_project_logging(second)
            logging.getLogger("squeakpose.test").info(
                "Second project event", extra={"event": "second_project"}
            )
            self._flush_project_handlers()

            with open(first_log, "r", encoding="utf-8") as handle:
                first_text = handle.read()
            with open(second_log, "r", encoding="utf-8") as handle:
                second_text = handle.read()

            self.assertIn("first_project", first_text)
            self.assertNotIn("second_project", first_text)
            self.assertIn("second_project", second_text)

    def test_small_log_limit_rotates_without_losing_active_log(self):
        with TemporaryDirectory() as tmp:
            log_path = configure_project_logging(tmp, max_bytes=256, backup_count=2)
            logger = logging.getLogger("squeakpose.test")
            for index in range(20):
                logger.info(
                    "Rotation payload %s %s",
                    index,
                    "x" * 80,
                    extra={"event": "rotation_test"},
                )
            self._flush_project_handlers()

            self.assertTrue(os.path.isfile(log_path))
            self.assertTrue(os.path.isfile(f"{log_path}.1"))


if __name__ == "__main__":
    unittest.main()
