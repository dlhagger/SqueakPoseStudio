import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from squeakpose.ui.video_reviewer import VideoReviewDialog


class PoseVideoReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["pose-video-review-tests"])

    def dialog(self, tmp):
        dialog = VideoReviewDialog(
            None,
            "cpu",
            ["nose"],
            ["mouse"],
            layer_id="keypoints",
            model_paths={"keypoints": "pose.pt"},
        )
        dialog.project_root = tmp
        dialog.keypoints_overlay_check.setChecked(True)
        dialog.layer_schemas["keypoints"] = {
            "classes": ["mouse"],
            "kp_names": ["nose"],
            "class_keypoints": {"mouse": ["nose"]},
        }
        return dialog

    def prediction(self):
        return {
            "ok": True,
            "detections": [
                {"ok": True, "cls": 0, "conf": 0.9, "xyxy": [0, 0, 10, 10], "kps": [[3, 4, 0.9]]},
                {
                    "ok": True,
                    "cls": 0,
                    "conf": 0.8,
                    "xyxy": [50, 50, 60, 60],
                    "kps": [[53, 54, 0.8]],
                },
            ],
        }

    def test_both_boxes_keypoints_and_names_draw_and_redraw(self):
        with TemporaryDirectory() as tmp:
            dialog = self.dialog(tmp)
            dialog.preds_by_layer["keypoints"] = {0: self.prediction()}
            for _ in range(2):
                dialog._draw_overlay_for(0)
                items = dialog.scene.items()
                self.assertEqual(sum(isinstance(item, QGraphicsRectItem) for item in items), 2)
                dots = [item for item in items if isinstance(item, QGraphicsEllipseItem)]
                self.assertEqual({(dot.x(), dot.y()) for dot in dots}, {(3, 4), (53, 54)})
                self.assertEqual(
                    sum(
                        isinstance(item, QGraphicsSimpleTextItem) and item.text() == "nose"
                        for item in items
                    ),
                    2,
                )
            dialog.close()

    def test_invalid_keypoints_do_not_draw_at_origin_or_nan(self):
        with TemporaryDirectory() as tmp:
            dialog = self.dialog(tmp)
            prediction = self.prediction()
            prediction["detections"][0]["kps"] = [[0, 0, 0]]
            prediction["detections"][1]["kps"] = [[float("nan"), 4, 0.9]]
            dialog.preds_by_layer["keypoints"] = {0: prediction}
            dialog._draw_overlay_for(0)
            self.assertEqual(
                sum(isinstance(item, QGraphicsEllipseItem) for item in dialog.scene.items()), 0
            )
            self.assertEqual(
                sum(isinstance(item, QGraphicsRectItem) for item in dialog.scene.items()), 2
            )
            dialog.close()

    def test_export_saves_raw_frame_and_keeps_both_predictions(self):
        with TemporaryDirectory() as tmp:
            dialog = self.dialog(tmp)
            dialog.preds_by_layer["keypoints"] = {0: self.prediction()}
            dialog._last_frame_bgr = np.zeros((80, 80, 3), dtype=np.uint8)
            dialog.cur = 0
            dialog.base = "video"
            dialog.video_source_id = "test"
            dialog._draw_overlay_for(0)
            with (
                patch.object(dialog, "_labeler_image_dir", return_value=tmp),
                patch("squeakpose.ui.video_reviewer.QMessageBox.information"),
                patch("squeakpose.ui.video_reviewer.QMessageBox.warning") as warning,
            ):
                dialog._export_current_frame_to_images()
                warning.assert_not_called()
            exported = list(Path(tmp).glob("*.png"))
            self.assertEqual(len(exported), 1)
            self.assertTrue(np.array_equal(cv2.imread(str(exported[0])), dialog._last_frame_bgr))
            self.assertEqual(len(dialog.preds_by_layer["keypoints"][0]["detections"]), 2)
            dialog.close()
