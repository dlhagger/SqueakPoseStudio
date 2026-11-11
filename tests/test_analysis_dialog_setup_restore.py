"""Analysis setup restore/autosave regressions and canceled multi-layer results."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from analysis_dialog import DEFAULT_REAL_DISTANCE_MM, AnalysisDialog
from squeakpose.services.video_analysis_setup import video_analysis_setup_path

ROI = {"type": "polygon", "points": [[10, 10], [100, 10], [100, 100], [10, 100]]}


def _make_project(root: Path) -> None:
    """One project video whose segmentation CSV reports a 1280x720 frame."""

    video = root / "videos" / "session.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    csv_path = root / "inference outputs" / "segmentation" / "run_segmentation.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "frame,det,mask_polygon,image_width,image_height\n0,-1,,1280,720\n",
        encoding="utf-8",
    )
    runs = root / "inference outputs" / "runs"
    runs.mkdir()
    (runs / "run.json").write_text(
        json.dumps(
            {
                "video_path": str(video),
                "created_at": "2026-08-21T12:00:00",
                "passes": [
                    {
                        "layer_id": "segmentation",
                        "csv_path": str(csv_path),
                        "had_error": False,
                        "canceled": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class AnalysisSetupRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["analysis-setup-restore-test"])

    def _dialog(self, root: Path) -> AnalysisDialog:
        dialog = AnalysisDialog(
            None, project_root=str(root), app_base_dir=str(root), layer_id="segmentation"
        )
        self.addCleanup(dialog.close)
        return dialog

    def _save_initial_setup(self, root: Path) -> Path:
        first = self._dialog(root)
        self.assertEqual(first.project_video_combo.currentIndex(), 1)
        first._set_scale_points([(20, 20), (80, 20)])
        first.real_distance_spin.setValue(50)
        first.annotation_state.add_roi(ROI, name="Open")
        first._refresh_roi_list()
        first._save_analysis_setup()
        video_name = first._selected_bundle.video_name
        first.close()
        path = Path(video_analysis_setup_path(str(root), video_name))
        self.assertTrue(path.is_file())
        return path

    def test_refused_restore_never_autosaves_over_the_saved_setup(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            path = self._save_initial_setup(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["frame"]["width"] = 640  # the video was re-encoded since
            path.write_text(json.dumps(payload), encoding="utf-8")
            saved = path.read_bytes()

            dialog = self._dialog(root)
            self.assertIn("frame size changed", dialog.setup_persistence_label.text())
            self.assertEqual(dialog.scale_points, [])
            self.assertEqual(dialog.real_distance_spin.value(), DEFAULT_REAL_DISTANCE_MM)

            # Switching away and closing are automatic saves: they must not write.
            dialog.project_video_combo.setCurrentIndex(0)
            self.assertEqual(path.read_bytes(), saved)
            dialog.project_video_combo.setCurrentIndex(1)
            self.assertIn("frame size changed", dialog.setup_persistence_label.text())
            dialog._cleanup_before_close()
            self.assertEqual(path.read_bytes(), saved)

            # An explicit edit deliberately replaces the unrestorable setup.
            dialog._set_scale_points([(0, 0), (30, 40)])
            rewritten = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(rewritten["frame"]["width"], 1280)
            self.assertEqual(rewritten["scale"]["points"], [[0.0, 0.0], [30.0, 40.0]])

    def test_unreadable_setup_is_kept_when_the_dialog_closes(self):
        for content in ("{not json", json.dumps({"schema_version": 99})):
            with self.subTest(content=content), TemporaryDirectory() as tmp:
                root = Path(tmp)
                _make_project(root)
                path = self._save_initial_setup(root)
                path.write_text(content, encoding="utf-8")

                dialog = self._dialog(root)
                self.assertIn(
                    "Could not restore video setup", dialog.setup_persistence_label.text()
                )
                self.assertEqual(dialog.real_distance_spin.value(), DEFAULT_REAL_DISTANCE_MM)
                dialog._cleanup_before_close()
                dialog.close()

                self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_placeholder_clears_the_preview_and_reference_length(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            dialog = self._dialog(root)
            dialog.real_distance_spin.setValue(50)
            self.assertFalse(dialog.frame_view._pixmap.isNull())

            with patch.object(dialog, "_load_preview_frame") as reload_preview:
                dialog.project_video_combo.setCurrentIndex(0)

            reload_preview.assert_not_called()
            self.assertEqual(dialog.video_edit.text(), "")
            self.assertEqual(dialog.csv_edit.text(), "")
            self.assertEqual(dialog.analysis_mode_combo.count(), 0)
            self.assertTrue(dialog.frame_view._pixmap.isNull())
            self.assertEqual(dialog.annotation_state.frame.width, 0)
            self.assertEqual(dialog.real_distance_spin.value(), DEFAULT_REAL_DISTANCE_MM)

    def test_video_without_saved_setup_starts_at_the_default_reference_length(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            dialog = self._dialog(root)
            video_name = dialog._selected_bundle.video_name
            dialog.project_video_combo.setCurrentIndex(0)
            setup = Path(video_analysis_setup_path(str(root), video_name))
            if setup.exists():
                setup.unlink()
            # Not attached to any video, so this value is never saved.
            dialog.real_distance_spin.setValue(75)

            dialog.project_video_combo.setCurrentIndex(1)

            self.assertIn("No saved setup", dialog.setup_persistence_label.text())
            self.assertEqual(dialog.real_distance_spin.value(), DEFAULT_REAL_DISTANCE_MM)

    def test_pixel_distance_label_follows_the_scale_points(self):
        with TemporaryDirectory() as tmp:
            dialog = self._dialog(Path(tmp))
            dialog._set_scale_points([(0, 0), (10, 0)])
            self.assertEqual(dialog.pixel_distance_label.text(), "10.0 px")

            dialog._set_scale_points([(5, 5)])  # third click restarts the line
            self.assertEqual(dialog.pixel_distance_label.text(), "Draw scale")

            dialog._set_scale_points([(0, 0), (10, 0)])
            dialog._set_scale_points([(5, 5), (5, 5)])  # zero-length line
            self.assertEqual(dialog.pixel_distance_label.text(), "Draw scale")


class CanceledAnalysisResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["analysis-setup-restore-test"])

    def test_canceled_run_with_published_layers_records_the_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            dialog = AnalysisDialog(
                None, project_root=tmp, app_base_dir=tmp, layer_id="segmentation"
            )
            self.addCleanup(dialog.close)
            self.assertIsNotNone(dialog._selected_bundle)
            output_dir = root / "analysis outputs" / "session" / "combined"
            output_dir.mkdir(parents=True)

            with patch("analysis_dialog.record_project_video_analysis_output") as record:
                dialog._handle_worker_event(
                    {
                        "event": "result",
                        "canceled": True,
                        "had_error": False,
                        "output_dir": str(output_dir),
                        "completed_layers": ["keypoints"],
                        "results_by_layer": {"keypoints": {"feature_csv": "features.csv"}},
                    }
                )

            record.assert_called_once_with(
                dialog.project_root, dialog._selected_bundle.video_name, str(output_dir)
            )
            self.assertEqual(dialog.last_output_dir, str(output_dir))
            self.assertTrue(dialog.open_output_btn.isEnabled())
            self.assertIn("Updated layers: keypoints", dialog.log_view.toPlainText())


if __name__ == "__main__":
    unittest.main()
