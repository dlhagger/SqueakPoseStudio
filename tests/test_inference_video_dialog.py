import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from squeakpose.ui.inference_video_dialog import InferenceVideoDialog


class InferenceVideoDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["inference-picker-test"])

    def test_lists_project_videos_marks_history_and_supports_all_or_clear(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            videos.mkdir()
            first = videos / "first.mp4"
            second = videos / "second.mov"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            runs = root / "inference outputs" / "runs"
            runs.mkdir(parents=True)
            (runs / "run.json").write_text(
                json.dumps(
                    {
                        "video_path": str(first),
                        "created_at": "2026-08-20T12:00:00",
                        "passes": [
                            {"layer_id": "keypoints", "had_error": False, "canceled": False}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dialog = InferenceVideoDialog(
                tmp, configured_layers=("keypoints",), default_batch_size=8
            )
            self.assertEqual(dialog.video_table.rowCount(), 2)
            self.assertIn("Complete", dialog.video_table.item(0, 2).text())
            self.assertEqual(dialog.batch_size, 8)
            self.assertFalse(dialog.run_button.isEnabled())

            dialog.select_all()
            self.assertEqual(dialog.selected_video_paths, (str(first), str(second)))
            self.assertTrue(dialog.run_button.isEnabled())

            dialog.clear_selection()
            self.assertEqual(dialog.selected_video_paths, ())
            self.assertFalse(dialog.run_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
