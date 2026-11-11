import io
import unittest
from unittest.mock import patch

from squeakpose.app import _save_last_project_best_effort, make_console_output_lossless


class AppStartupTests(unittest.TestCase):
    def test_last_project_persistence_failure_does_not_escape(self):
        project_root = "/valid/project"

        with (
            patch(
                "squeakpose.app.save_last_project",
                side_effect=PermissionError("read-only state directory"),
            ),
            self.assertLogs("squeakpose.app", level="WARNING") as logs,
        ):
            saved = _save_last_project_best_effort(project_root)

        self.assertFalse(saved)
        self.assertIn("continuing to open /valid/project", "\n".join(logs.output))

    def test_last_project_persistence_success_is_reported(self):
        with patch("squeakpose.app.save_last_project") as save:
            saved = _save_last_project_best_effort("/valid/project")

        self.assertTrue(saved)
        save.assert_called_once_with("/valid/project")

    def test_console_output_escapes_characters_the_encoding_cannot_represent(self):
        # Simulates a redirected Windows console using the cp1252 ANSI code page.
        raw = io.BytesIO()
        stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        with patch("sys.stdout", stdout), patch("sys.stderr", stdout):
            make_console_output_lossless()
            print("\U0001f9e0 Inference device: cuda")
            stdout.flush()
        self.assertIn(b"\\U0001f9e0 Inference device: cuda", raw.getvalue())


if __name__ == "__main__":
    unittest.main()
