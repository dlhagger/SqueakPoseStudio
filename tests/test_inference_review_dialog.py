import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from squeakpose.services.inference_review import (
    InferenceReviewCandidate,
    InferenceReviewScanResult,
)
from squeakpose.ui.inference_review_dialog import InferenceReviewDialog


def _candidate(
    frame: int, status: str, *, video_path: str = "/missing/session.mp4"
) -> InferenceReviewCandidate:
    return InferenceReviewCandidate(
        layer_id="keypoints",
        model_path="/models/pose/best.pt",
        video_name="session.mp4",
        video_path=video_path,
        inference_csv="/project/inference.csv",
        frame_index=frame,
        time_seconds=frame / 10,
        status=status,
        reasons=("low_detection_confidence",) if status != "good" else (),
        detection_confidence=0.2 if status != "good" else 0.9,
        keypoint_confidence=0.4,
        detections_in_frame=1,
        tracks_in_frame=1,
        expected_animal_count=1,
        priority_score=80 if status != "good" else 5,
    )


class InferenceReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["inference-review-test"])

    def test_populates_project_candidates_and_problem_filter(self):
        with patch("squeakpose.ui.inference_review_dialog.cv2", None):
            dialog = InferenceReviewDialog("/project", auto_scan=False)
            dialog._scan_finished(
                InferenceReviewScanResult(
                    layer_id="keypoints",
                    candidates=(_candidate(1, "warning"), _candidate(20, "good")),
                    videos_total=1,
                    videos_with_inference=1,
                    frames_scanned=100,
                    warning_frames=1,
                    bad_frames=0,
                    model_paths=("/models/pose/best.pt",),
                )
            )
            self.assertEqual(dialog.table.rowCount(), 2)
            self.assertIn("100 frames scanned", dialog.summary_label.text())
            dialog.status_combo.setCurrentIndex(dialog.status_combo.findData("problem"))
            self.assertEqual(dialog.table.rowCount(), 1)
            self.assertEqual(dialog.table.item(0, 3).text(), "Warning")
            self.assertTrue(dialog.export_shown_button.isEnabled())
            dialog.close()

    def test_exports_original_frame_once_to_project_label_queue(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            video_path = project / "session.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (48, 32)
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.full((32, 48, 3), 90, dtype=np.uint8))
            writer.release()
            candidate = _candidate(0, "warning", video_path=str(video_path))
            dialog = InferenceReviewDialog(str(project), auto_scan=False)
            with patch("squeakpose.ui.inference_review_dialog.QMessageBox.information"):
                dialog._export_candidates((candidate,))
                dialog._export_candidates((candidate,))
            exported = list(Path(project, "images_to_label").glob("*.png"))
            self.assertEqual(len(exported), 1)
            image = cv2.imread(str(exported[0]))
            self.assertIsNotNone(image)
            self.assertAlmostEqual(float(image.mean()), 90.0, delta=3.0)
            dialog.close()


if __name__ == "__main__":
    unittest.main()
