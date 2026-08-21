import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.services.video_analysis_setup import (
    load_video_analysis_setup,
    save_video_analysis_setup,
    video_analysis_setup_path,
)


class VideoAnalysisSetupTests(unittest.TestCase):
    def test_scale_rois_and_priority_order_round_trip(self):
        with TemporaryDirectory() as tmp:
            rois = [
                {
                    "name": "Center",
                    "type": "polygon",
                    "points": [[40, 20], [60, 20], [60, 40], [40, 40]],
                },
                {
                    "name": "Open",
                    "type": "polygon",
                    "points": [[0, 0], [100, 0], [100, 80], [0, 80]],
                },
            ]

            path = save_video_analysis_setup(
                tmp,
                "session.mp4",
                frame_width=100,
                frame_height=80,
                scale_points=[(10, 10), (60, 10)],
                real_world_distance_mm=50,
                rois=rois,
            )
            setup = load_video_analysis_setup(tmp, "session.mp4")

            self.assertTrue(path.startswith(os.path.join(tmp, "analysis settings", "videos")))
            self.assertIsNotNone(setup)
            self.assertEqual(setup.scale_points, ((10.0, 10.0), (60.0, 10.0)))
            self.assertEqual(setup.real_world_distance_mm, 50.0)
            self.assertEqual([roi["name"] for roi in setup.rois], ["Center", "Open"])

    def test_missing_setup_and_unsafe_video_names(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(load_video_analysis_setup(tmp, "missing.mp4"))
            with self.assertRaises(ValueError):
                video_analysis_setup_path(tmp, "../escape.mp4")

    def test_corrupt_or_mismatched_metadata_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(video_analysis_setup_path(tmp, "session.mp4"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"schema_version": 1, "video_name": "other.mp4"}))

            with self.assertRaisesRegex(ValueError, "different project video"):
                load_video_analysis_setup(tmp, "session.mp4")


if __name__ == "__main__":
    unittest.main()
