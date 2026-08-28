import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from squeakpose.ui.project_video_picker import ProjectVideoPickerDialog


class ProjectVideoPickerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["project-video-picker-test"])

    def test_lists_project_links_and_selects_only_readable_entries(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as sources:
            videos_dir = Path(tmp) / "videos"
            videos_dir.mkdir()
            first_source = Path(sources) / "first.mp4"
            second_source = Path(sources) / "second.mov"
            first_source.write_bytes(b"first")
            second_source.write_bytes(b"second")
            (videos_dir / first_source.name).symlink_to(first_source)
            (videos_dir / second_source.name).symlink_to(second_source)
            (videos_dir / "missing.mp4").symlink_to(Path(sources) / "missing.mp4")

            dialog = ProjectVideoPickerDialog(str(videos_dir), selected_names={second_source.name})

            self.assertEqual(dialog.video_table.rowCount(), 3)
            self.assertEqual(
                [entry.name for entry in dialog.selected_entries],
                [second_source.name],
            )
            self.assertTrue(dialog.accept_button.isEnabled())

            dialog.select_all()
            self.assertEqual(
                [entry.name for entry in dialog.selected_entries],
                [first_source.name, second_source.name],
            )
            self.assertIn("1 missing", dialog.summary_label.text())

            dialog.clear_selection()
            self.assertEqual(dialog.selected_entries, ())
            self.assertFalse(dialog.accept_button.isEnabled())
            dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
