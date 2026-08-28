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
                        "schema_version": 2,
                        "tracking": {
                            "expected_animal_count": 2,
                            "requested_tracker": "botsort",
                            "resolved_tracker": "botsort",
                        },
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
            self.assertEqual(dialog.video_table.item(0, 2).text(), "✓ Keypoints")
            self.assertEqual(dialog.batch_size, 8)
            self.assertFalse(dialog.batch_spin.isEnabled())
            self.assertTrue(dialog.batch_spin.isHidden())
            self.assertFalse(dialog.tracking_defaults_panel.isHidden())
            self.assertGreaterEqual(dialog.video_table.rowHeight(0), 44)
            self.assertGreaterEqual(dialog.video_table.columnWidth(4), 190)
            self.assertFalse(dialog.run_button.isEnabled())
            first_animals = dialog.video_table.cellWidget(0, 3)
            first_tracker = dialog.video_table.cellWidget(0, 4)
            self.assertEqual(first_animals.value(), 2)
            self.assertEqual(first_tracker.currentData(), "botsort")

            dialog.select_all()
            self.assertEqual(dialog.selected_video_paths, (str(first), str(second)))
            self.assertTrue(dialog.run_button.isEnabled())
            settings = dialog.selected_video_settings
            self.assertEqual(settings[0].expected_animal_count, 2)
            self.assertEqual(settings[0].requested_tracker, "botsort")
            self.assertEqual(settings[1].expected_animal_count, 1)
            self.assertEqual(settings[1].requested_tracker, "auto")

            dialog.clear_selection()
            self.assertEqual(dialog.selected_video_paths, ())
            self.assertFalse(dialog.run_button.isEnabled())

    def test_bulk_tracking_defaults_apply_only_to_selected_videos(self):
        with TemporaryDirectory() as tmp:
            videos = Path(tmp) / "videos"
            videos.mkdir()
            first = videos / "first.mp4"
            second = videos / "second.mp4"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            dialog = InferenceVideoDialog(tmp, configured_layers=("keypoints",))

            dialog._set_checked_for_key(str(first.resolve()), True)
            dialog.default_animals_spin.setValue(3)
            dialog.default_tracker_combo.setCurrentIndex(
                dialog.default_tracker_combo.findData("botsort")
            )
            dialog.apply_tracking_defaults()

            first_animals = dialog.video_table.cellWidget(0, 3)
            first_tracker = dialog.video_table.cellWidget(0, 4)
            second_animals = dialog.video_table.cellWidget(1, 3)
            second_tracker = dialog.video_table.cellWidget(1, 4)
            self.assertEqual(first_animals.value(), 3)
            self.assertEqual(first_tracker.currentData(), "botsort")
            self.assertEqual(second_animals.value(), 1)
            self.assertEqual(second_tracker.currentData(), "auto")

    def test_auto_label_updates_with_expected_animal_count(self):
        with TemporaryDirectory() as tmp:
            videos = Path(tmp) / "videos"
            videos.mkdir()
            (videos / "mouse.mp4").write_bytes(b"video")
            dialog = InferenceVideoDialog(tmp, configured_layers=("keypoints",))
            animals = dialog.video_table.cellWidget(0, 3)
            tracker = dialog.video_table.cellWidget(0, 4)

            self.assertIn("ByteTrack", tracker.itemText(0))
            animals.setValue(2)
            self.assertIn("BoT-SORT", tracker.itemText(0))

    def test_depth_only_inference_preserves_batch_controls_and_disables_tracking(self):
        with TemporaryDirectory() as tmp:
            videos = Path(tmp) / "videos"
            videos.mkdir()
            (videos / "mouse.mp4").write_bytes(b"video")
            dialog = InferenceVideoDialog(tmp, configured_layers=("depth",), default_batch_size=7)

            self.assertTrue(dialog.batch_spin.isEnabled())
            self.assertEqual(dialog.batch_size, 7)
            self.assertFalse(dialog.batch_spin.isHidden())
            self.assertTrue(dialog.tracking_defaults_panel.isHidden())
            self.assertTrue(dialog.video_table.isColumnHidden(3))
            self.assertTrue(dialog.video_table.isColumnHidden(4))
            self.assertFalse(dialog.video_table.cellWidget(0, 3).isEnabled())
            self.assertFalse(dialog.video_table.cellWidget(0, 4).isEnabled())


if __name__ == "__main__":
    unittest.main()
