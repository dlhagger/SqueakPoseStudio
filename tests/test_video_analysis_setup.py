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

    def test_empty_legacy_setup_recovers_rois_from_moved_analysis_summary_once(self):
        with TemporaryDirectory() as tmp:
            setup_path = Path(video_analysis_setup_path(tmp, "session.mp4"))
            setup_path.parent.mkdir(parents=True)
            setup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "video_name": "session.mp4",
                        "frame": {"width": 100, "height": 80},
                        "scale": {"points": [], "real_world_distance_mm": 1},
                        "rois": [],
                    }
                ),
                encoding="utf-8",
            )
            summary = Path(tmp, "analysis outputs", "segmentation", "old-run")
            summary.mkdir(parents=True)
            (summary / "analysis_summary.json").write_text(
                json.dumps(
                    {
                        "video_path": "/Users/old/project/videos/session.mp4",
                        "rois": [
                            {
                                "name": "Center",
                                "type": "polygon",
                                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            recovered = load_video_analysis_setup(tmp, "session.mp4")
            self.assertEqual([roi["name"] for roi in recovered.rois], ["Center"])

            save_video_analysis_setup(
                tmp,
                "session.mp4",
                frame_width=100,
                frame_height=80,
                scale_points=[],
                real_world_distance_mm=1,
                rois=[],
                rois_cleared=True,
            )
            self.assertEqual(load_video_analysis_setup(tmp, "session.mp4").rois, ())

    @staticmethod
    def _write_legacy_summary(tmp, run_name, roi_name):
        summary = Path(tmp, "analysis outputs", "segmentation", run_name)
        summary.mkdir(parents=True)
        (summary / "analysis_summary.json").write_text(
            json.dumps(
                {
                    "video_path": "/Users/old/project/videos/session.mp4",
                    "rois": [
                        {
                            "name": roi_name,
                            "type": "polygon",
                            "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return summary

    def test_legacy_roi_recovery_preserves_saved_scale_calibration(self):
        with TemporaryDirectory() as tmp:
            setup_path = Path(video_analysis_setup_path(tmp, "session.mp4"))
            setup_path.parent.mkdir(parents=True)
            # A setup file written before SUMMARY_RECOVERY_FIELD existed, with a
            # scale calibration but no ROIs yet.
            setup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "video_name": "session.mp4",
                        "frame": {"width": 640, "height": 480},
                        "scale": {"points": [[10, 10], [110, 10]], "real_world_distance_mm": 25},
                        "rois": [],
                    }
                ),
                encoding="utf-8",
            )
            self._write_legacy_summary(tmp, "old-run", "Center")

            recovered = load_video_analysis_setup(tmp, "session.mp4")

            self.assertEqual([roi["name"] for roi in recovered.rois], ["Center"])
            self.assertEqual(recovered.scale_points, ((10.0, 10.0), (110.0, 10.0)))
            self.assertEqual(recovered.real_world_distance_mm, 25.0)
            self.assertEqual((recovered.frame_width, recovered.frame_height), (640, 480))

            persisted = json.loads(setup_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["scale"]["real_world_distance_mm"], 25.0)
            self.assertEqual(persisted["scale"]["points"], [[10.0, 10.0], [110.0, 10.0]])
            self.assertEqual(persisted["frame"], {"width": 640, "height": 480})
            self.assertTrue(persisted["analysis_summary_recovery_checked"])

    def test_legacy_roi_recovery_is_not_repeated_after_being_checked(self):
        with TemporaryDirectory() as tmp:
            setup_path = Path(video_analysis_setup_path(tmp, "session.mp4"))
            setup_path.parent.mkdir(parents=True)
            setup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "video_name": "session.mp4",
                        "frame": {"width": 100, "height": 80},
                        "scale": {"points": [], "real_world_distance_mm": 1},
                        "rois": [],
                    }
                ),
                encoding="utf-8",
            )
            summary = self._write_legacy_summary(tmp, "old-run", "Center")

            first = load_video_analysis_setup(tmp, "session.mp4")
            self.assertEqual([roi["name"] for roi in first.rois], ["Center"])

            # The user removes the recovered ROI; the dialog autosaves with an empty
            # ROI list but without the explicit "cleared" flag (as most save sites do).
            save_video_analysis_setup(
                tmp,
                "session.mp4",
                frame_width=100,
                frame_height=80,
                scale_points=[(5, 5), (55, 5)],
                real_world_distance_mm=40,
                rois=[],
            )
            # Even a newer legacy summary must not re-arm recovery.
            (summary / "analysis_summary.json").unlink()
            summary.rmdir()
            self._write_legacy_summary(tmp, "newer-run", "Arena")

            reloaded = load_video_analysis_setup(tmp, "session.mp4")

            self.assertEqual(reloaded.rois, ())
            self.assertEqual(reloaded.scale_points, ((5.0, 5.0), (55.0, 5.0)))
            self.assertEqual(reloaded.real_world_distance_mm, 40.0)


if __name__ == "__main__":
    unittest.main()
