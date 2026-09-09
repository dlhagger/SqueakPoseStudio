import unittest
from unittest.mock import patch

from squeakpose.app import _save_last_project_best_effort


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


if __name__ == "__main__":
    unittest.main()
