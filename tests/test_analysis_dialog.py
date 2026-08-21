import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from analysis_dialog import AnalysisDialog, FrameAnnotationView


class FrameAnnotationViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["analysis-dialog-test"])

    def setUp(self):
        self.view = FrameAnnotationView()
        self.view.resize(600, 400)
        frame = QPixmap(1000, 500)
        frame.fill(QColor("#000000"))
        self.view.set_frame(frame, 1000, 500)

    def tearDown(self):
        self.view.close()

    def test_zoom_keeps_cursor_anchored_in_image_coordinates(self):
        anchor = QPointF(450, 200)
        before = self.view._widget_to_image(anchor)

        self.view.set_zoom(2.0, anchor)
        after = self.view._widget_to_image(anchor)

        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(after[0], before[0])
        self.assertAlmostEqual(after[1], before[1])

    def test_segmentation_polygon_is_painted_over_frame(self):
        self.view.set_segmentation_polygons(
            [[(400.0, 150.0), (600.0, 150.0), (600.0, 350.0), (400.0, 350.0)]]
        )

        image = self.view.grab().toImage()
        center = image.pixelColor(300, 200)

        self.assertGreater(center.green(), 20)
        self.assertGreater(center.blue(), 20)

    def test_polygon_roi_clicks_finish_on_first_vertex(self):
        self.view.set_mode("roi")
        emitted = []
        self.view.roiDrawn.connect(emitted.append)

        for point in ((100, 100), (500, 100), (300, 300), (100, 100)):
            QTest.mouseClick(
                self.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPoint(*point),
            )

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["type"], "polygon")
        self.assertEqual(len(emitted[0]["points"]), 3)
        self.assertEqual(self.view.polygon_vertex_count, 0)

    def test_polygon_roi_backspace_undoes_and_escape_cancels(self):
        self.view.set_mode("roi")
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(500, 100))
        self.assertEqual(self.view.polygon_vertex_count, 2)

        QTest.keyClick(self.view, Qt.Key.Key_Backspace)
        self.assertEqual(self.view.polygon_vertex_count, 1)
        QTest.keyClick(self.view, Qt.Key.Key_Escape)
        self.assertEqual(self.view.polygon_vertex_count, 0)


class AnalysisDialogInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["analysis-input-test"])

    def test_project_video_selector_sets_both_video_and_inference_csv(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "videos" / "session.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            csv_path = root / "inference outputs" / "segmentation" / "run_segmentation.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("frame,det,mask_polygon\n", encoding="utf-8")
            runs = root / "inference outputs" / "runs"
            runs.mkdir()
            (runs / "run.json").write_text(
                json.dumps(
                    {
                        "video_path": str(video),
                        "created_at": "2026-08-20T12:00:00",
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

            dialog = AnalysisDialog(
                None, project_root=tmp, app_base_dir=tmp, layer_id="segmentation"
            )

            self.assertEqual(dialog.project_video_combo.count(), 2)
            self.assertEqual(dialog.video_edit.text(), str(video))
            self.assertEqual(dialog.csv_edit.text(), str(csv_path))
            self.assertIn("run_segmentation.csv", dialog.input_detail_label.text())
            dialog.close()

    def test_project_video_scale_rois_and_priority_restore_after_reopen(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            shape = {
                "type": "polygon",
                "points": [[10, 10], [100, 10], [100, 100], [10, 100]],
            }

            first = AnalysisDialog(
                None, project_root=tmp, app_base_dir=tmp, layer_id="segmentation"
            )
            first._set_scale_points([(20, 20), (80, 20)])
            first.real_distance_spin.setValue(50)
            first.annotation_state.add_roi(shape, name="Open")
            first.annotation_state.add_roi(shape, name="Center")
            first._refresh_roi_list()
            first.roi_list.setCurrentRow(1)
            first._move_selected_roi(-1)
            first.close()

            reopened = AnalysisDialog(
                None, project_root=tmp, app_base_dir=tmp, layer_id="segmentation"
            )

            self.assertEqual(reopened.scale_points, [(20.0, 20.0), (80.0, 20.0)])
            self.assertEqual(reopened.real_distance_spin.value(), 50.0)
            self.assertEqual(
                [roi["name"] for roi in reopened.rois],
                ["Center", "Open"],
            )
            self.assertIn("Restored saved scale", reopened.setup_persistence_label.text())
            reopened.close()


if __name__ == "__main__":
    unittest.main()
