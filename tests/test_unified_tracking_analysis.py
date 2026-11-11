import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analysis_ops import AnalysisConfig
from unified_analysis_ops import build_unified_frame_table, run_unified_analysis_workflow


def _pose_row(frame, detection, track_id, x1, *, confidence=0.9, expected=2):
    return {
        "frame_index": frame,
        "detection_index": detection,
        "detections_in_frame": 2,
        "track_id": track_id,
        "tracks_in_frame": 2,
        "expected_animal_count": expected,
        "tracker_type": "bytetrack",
        "tracker_profile": "fixed_camera_v1",
        "class_id": 0,
        "class_name": "mouse",
        "confidence": confidence,
        "bbox_x1": x1,
        "bbox_y1": 10,
        "bbox_x2": x1 + 10,
        "bbox_y2": 20,
        "bbox_center_x": x1 + 5,
        "bbox_center_y": 15,
        "kp_nose_x": x1 + 3,
        "kp_nose_y": 12,
        "kp_nose_conf": 0.95,
    }


def _segmentation_row(frame, detection, track_id, x1, *, confidence=0.9, expected=2):
    return {
        "frame": frame,
        "frame_index": frame,
        "det": detection,
        "track_id": track_id,
        "tracks_in_frame": 2,
        "expected_animal_count": expected,
        "tracker_type": "bytetrack",
        "tracker_profile": "fixed_camera_v1",
        "class_id": 0,
        "class_name": "mouse",
        "conf": confidence,
        "x1": x1,
        "y1": 10,
        "x2": x1 + 10,
        "y2": 20,
        "mask_polygon": json.dumps([[x1, 10], [x1 + 10, 10], [x1 + 10, 20], [x1, 20]]),
    }


class UnifiedTrackingAnalysisTests(unittest.TestCase):
    def _config(self):
        return AnalysisConfig(
            detections_csv="",
            output_dir="unused",
            fps=10.0,
            smooth=False,
            make_plots=False,
        )

    def test_reconciles_different_layer_ids_through_crossing_and_ignores_duplicates(self):
        pose_rows = []
        segmentation_rows = []
        for frame, (left, right) in enumerate(((0, 30), (10, 20), (20, 10))):
            pose_rows.extend(
                [
                    _pose_row(frame, 0, 101, left + 1),
                    _pose_row(frame, 1, 202, right - 1),
                ]
            )
            segmentation_rows.extend(
                [
                    _segmentation_row(frame, 0, 10, left),
                    _segmentation_row(frame, 1, 20, right),
                ]
            )
        # High-confidence one-frame duplicates must not displace complete tracks.
        pose_rows.append(_pose_row(1, 2, 303, 70, confidence=0.99))
        segmentation_rows.append(_segmentation_row(1, 2, 99, 70, confidence=0.99))

        table = build_unified_frame_table(
            pd.DataFrame(pose_rows),
            pd.DataFrame(segmentation_rows),
            self._config(),
            video_path="",
            fps=10.0,
            scale=1.0,
        )

        self.assertEqual(len(table), 6)
        self.assertEqual(table.groupby("frame_index")["animal_id"].nunique().tolist(), [2, 2, 2])
        mapping = table.attrs["tracking_diagnostics"]["animal_track_mapping"]
        self.assertEqual(
            {(row["segmentation_track_id"], row["pose_track_id"]) for row in mapping},
            {("10", "101"), ("20", "202")},
        )
        self.assertNotIn("99", set(table["segmentation_track_id"].dropna()))
        self.assertNotIn("303", set(table["pose_track_id"].dropna()))
        for _, animal in table.groupby("animal_id"):
            self.assertTrue(pd.isna(animal.sort_values("frame_index").iloc[0]["distance_mm"]))
            self.assertEqual(animal["bbox_source"].tolist(), ["segmentation_bbox"] * 3)

    def test_missing_pose_frame_does_not_borrow_the_other_animals_pose(self):
        pose_rows = [
            _pose_row(0, 0, 101, 0),
            _pose_row(0, 1, 202, 30),
            _pose_row(1, 0, 202, 28),
        ]
        segmentation_rows = [
            _segmentation_row(0, 0, 10, 0),
            _segmentation_row(0, 1, 20, 30),
            _segmentation_row(1, 0, 10, 2),
            _segmentation_row(1, 1, 20, 28),
        ]

        table = build_unified_frame_table(
            pd.DataFrame(pose_rows),
            pd.DataFrame(segmentation_rows),
            self._config(),
            video_path="",
            fps=10.0,
            scale=1.0,
        )
        animal_for_seg_10 = table.loc[table["mapped_segmentation_track_id"].eq("10")]
        missing = animal_for_seg_10.loc[animal_for_seg_10["frame_index"].eq(1)].iloc[0]
        self.assertFalse(bool(missing["pose_valid"]))
        self.assertTrue(bool(missing["segmentation_valid"]))
        self.assertTrue(pd.isna(missing["pose_track_id"]))
        self.assertEqual(missing["bbox_source"], "segmentation_bbox")

    def test_legacy_csvs_retain_one_row_per_frame_confidence_fallback(self):
        pose = pd.DataFrame(
            [
                _pose_row(0, 0, "", 0, confidence=0.5, expected=1),
                _pose_row(0, 1, "", 20, confidence=0.9, expected=1),
                _pose_row(1, 0, "", 22, confidence=0.8, expected=1),
            ]
        ).drop(
            columns=[
                "track_id",
                "tracks_in_frame",
                "expected_animal_count",
                "tracker_type",
                "tracker_profile",
            ]
        )
        segmentation = pd.DataFrame(
            [
                _segmentation_row(0, 0, "", 1, confidence=0.4, expected=1),
                _segmentation_row(0, 1, "", 21, confidence=0.95, expected=1),
                _segmentation_row(1, 0, "", 23, confidence=0.8, expected=1),
            ]
        ).drop(
            columns=[
                "track_id",
                "tracks_in_frame",
                "expected_animal_count",
                "tracker_type",
                "tracker_profile",
            ]
        )

        table = build_unified_frame_table(
            pose,
            segmentation,
            self._config(),
            video_path="",
            fps=10.0,
            scale=1.0,
        )

        self.assertEqual(len(table), 2)
        self.assertEqual(table["animal_id"].tolist(), ["animal_1", "animal_1"])
        self.assertEqual(table["tracking_status"].tolist(), ["legacy_untracked"] * 2)
        self.assertEqual(table["segmentation_detection_index"].tolist(), [1, 0])
        self.assertEqual(table["pose_detection_index"].tolist(), [1, 0])
        self.assertEqual(table["bbox_source"].tolist(), ["segmentation_bbox"] * 2)

    def test_workflow_persists_long_schema_and_track_mapping_diagnostics(self):
        pose = pd.DataFrame(
            [
                _pose_row(0, 0, 501, 0),
                _pose_row(0, 1, 502, 30),
                _pose_row(1, 0, 501, 2),
                _pose_row(1, 1, 502, 28),
            ]
        )
        segmentation = pd.DataFrame(
            [
                _segmentation_row(0, 0, 31, 0),
                _segmentation_row(0, 1, 32, 30),
                _segmentation_row(1, 0, 31, 2),
                _segmentation_row(1, 1, 32, 28),
            ]
        )
        with TemporaryDirectory() as tmp:
            pose_path = Path(tmp, "pose.csv")
            segmentation_path = Path(tmp, "segmentation.csv")
            output_dir = Path(tmp, "analysis")
            pose.to_csv(pose_path, index=False)
            segmentation.to_csv(segmentation_path, index=False)

            result = run_unified_analysis_workflow(
                AnalysisConfig(
                    detections_csv="",
                    output_dir=str(output_dir),
                    fps=10.0,
                    smooth=False,
                    make_plots=True,
                    make_annotated_video=False,
                ),
                pose_csv=str(pose_path),
                segmentation_csv=str(segmentation_path),
            )

            analysis = pd.read_csv(result["feature_csv"])
            self.assertEqual(len(analysis), 4)
            self.assertEqual(
                analysis.groupby("frame_index")["animal_id"].nunique().tolist(), [2, 2]
            )
            self.assertEqual(result["summary"]["frames"], 2)
            self.assertEqual(result["summary"]["analysis_rows"], 4)
            self.assertEqual(result["summary"]["expected_animal_count"], 2)
            plot_names = {Path(path).name for path in result["plot_paths"]}
            self.assertIn("prediction_qc.png", plot_names)
            self.assertIn("acceleration_magnitude.png", plot_names)
            self.assertNotIn("speed_mm_per_sec.png", plot_names)
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 5)
            self.assertEqual(
                manifest["segmentation_bbox_definition"],
                "mask_polygon_bounds_with_inference_bbox_fallback",
            )
            self.assertEqual(
                {
                    (row["segmentation_track_id"], row["pose_track_id"])
                    for row in manifest["tracking"]["animal_track_mapping"]
                },
                {("31", "501"), ("32", "502")},
            )

    def test_plots_survive_animal_without_pose_rows_when_image_size_columns_exist(self):
        # Real pose CSVs carry image_width/image_height. In the unified table those
        # columns only come from pose rows, so an animal that has a segmentation track
        # but no matched pose track ends up with an all-NaN image size group; plotting
        # used to crash on ``int(NaN)`` for that animal.
        pose = pd.DataFrame([_pose_row(0, 0, 501, 0), _pose_row(1, 0, 501, 2)])
        pose["image_width"] = 640
        pose["image_height"] = 480
        segmentation = pd.DataFrame(
            [
                _segmentation_row(0, 0, 31, 0),
                _segmentation_row(0, 1, 32, 30),
                _segmentation_row(1, 0, 31, 2),
                _segmentation_row(1, 1, 32, 28),
            ]
        )
        with TemporaryDirectory() as tmp:
            pose_path = Path(tmp, "pose.csv")
            segmentation_path = Path(tmp, "segmentation.csv")
            pose.to_csv(pose_path, index=False)
            segmentation.to_csv(segmentation_path, index=False)

            result = run_unified_analysis_workflow(
                AnalysisConfig(
                    detections_csv="",
                    output_dir=str(Path(tmp, "analysis")),
                    fps=10.0,
                    smooth=False,
                    make_plots=True,
                    make_annotated_video=False,
                ),
                pose_csv=str(pose_path),
                segmentation_csv=str(segmentation_path),
            )

            analysis = pd.read_csv(result["feature_csv"])
            self.assertEqual(analysis["animal_id"].nunique(), 2)
            pose_free = analysis.groupby("animal_id")["image_width"].apply(
                lambda values: values.isna().all()
            )
            self.assertTrue(pose_free.any(), "expected one animal with no pose rows")
            trajectory_plots = {
                Path(path).parent.name
                for path in result["plot_paths"]
                if Path(path).name == "trajectory.png"
            }
            self.assertEqual(trajectory_plots, set(analysis["animal_id"].unique()))
            for path in result["plot_paths"]:
                self.assertTrue(Path(path).is_file())

    def test_single_mouse_fragment_recovery_applies_when_a_class_is_selected(self):
        # One mouse recorded alongside static objects; the tracker re-labels the
        # mouse from id 3 to id 9 halfway through. Selecting the mouse class must not
        # disable the one-animal fragment recovery that an unscoped run receives.
        rows = []
        for frame in range(4):
            mouse_track = 3 if frame < 2 else 9
            for detection, track, x1, name, confidence in [
                (0, 1, 100, "object", 0.99),
                (1, 2, 200, "object", 0.98),
                (2, mouse_track, frame * 2, "mouse", 0.8),
            ]:
                row = _segmentation_row(
                    frame, detection, track, x1, confidence=confidence, expected=1
                )
                row.update(
                    class_name=name,
                    class_id=int(name == "object"),
                    tracks_in_frame=1,
                    total_tracks_in_frame=3,
                )
                rows.append(row)
        segmentation = pd.DataFrame(rows)
        pose = pd.DataFrame(columns=list(_pose_row(0, 0, 0, 0)))
        config = AnalysisConfig(
            detections_csv="",
            output_dir="unused",
            fps=10.0,
            smooth=False,
            make_plots=False,
            subject_class="mouse",
        )

        table = build_unified_frame_table(
            pose, segmentation, config, video_path="", fps=10, scale=1
        )

        self.assertEqual(table["animal_id"].unique().tolist(), ["animal_1"])
        self.assertEqual(table["segmentation_valid"].tolist(), [True, True, True, True])
        self.assertEqual(table["segmentation_track_id"].astype(int).tolist(), [3, 3, 9, 9])
        self.assertEqual(set(table["segmentation_class_name"]), {"mouse"})
        self.assertEqual(table["extra_segmentation_tracks"].max(), 0)

    def test_prediction_qc_flags_extra_detections_without_discarding_primary_track(self):
        pose = pd.DataFrame(
            [
                _pose_row(0, 0, 101, 0, expected=1),
                _pose_row(0, 1, 909, 70, confidence=0.4, expected=1),
                _pose_row(1, 0, 101, 2, expected=1),
            ]
        )
        segmentation = pd.DataFrame(
            [
                _segmentation_row(0, 0, 10, 0, expected=1),
                _segmentation_row(0, 1, 99, 70, confidence=0.4, expected=1),
                _segmentation_row(1, 0, 10, 2, expected=1),
            ]
        )
        pose.loc[pose["frame_index"].eq(1), ["detections_in_frame", "tracks_in_frame"]] = 1
        segmentation.loc[segmentation["frame_index"].eq(1), "tracks_in_frame"] = 1

        table = build_unified_frame_table(
            pose,
            segmentation,
            self._config(),
            video_path="",
            fps=10.0,
            scale=1.0,
        )

        self.assertEqual(len(table), 2)
        first = table.loc[table["frame_index"].eq(0)].iloc[0]
        second = table.loc[table["frame_index"].eq(1)].iloc[0]
        self.assertEqual(first["prediction_qc_status"], "warning")
        self.assertEqual(first["extra_pose_detections"], 1)
        self.assertEqual(first["extra_segmentation_detections"], 1)
        self.assertEqual(first["extra_pose_tracks"], 1)
        self.assertEqual(first["extra_segmentation_tracks"], 1)
        self.assertIn("extra_pose_detection", first["prediction_qc_reasons"])
        self.assertIn("extra_segmentation_detection", first["prediction_qc_reasons"])
        self.assertIn("extra_pose_track", first["prediction_qc_reasons"])
        self.assertIn("extra_segmentation_track", first["prediction_qc_reasons"])
        self.assertEqual(first["pose_track_id"], "101")
        self.assertEqual(first["segmentation_track_id"], "10")
        self.assertEqual(second["prediction_qc_status"], "good")


if __name__ == "__main__":
    unittest.main()
