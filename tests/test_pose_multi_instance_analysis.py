import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analysis_ops import AnalysisConfig, compute_features, create_roi_outputs, run_analysis_workflow
from tests.test_unified_tracking_analysis import _pose_row


class PoseMultiInstanceAnalysisTests(unittest.TestCase):
    def config(self, **kwargs):
        return AnalysisConfig(
            detections_csv="", output_dir="unused", fps=10, make_plots=False, **kwargs
        )

    def crossing(self):
        return pd.DataFrame(
            [
                _pose_row(frame, det, track, x, expected=2)
                for frame in range(3)
                for det, track, x in [(0, 1, frame), (1, 2, 100 - frame)]
            ]
        )

    def test_both_mice_preserved_with_independent_motion_and_keypoints(self):
        raw = self.crossing()
        result = compute_features(raw, 10, 1, self.config(smooth=False))
        self.assertEqual(len(result), 6)
        self.assertEqual(result.groupby("frame_index").size().tolist(), [2, 2, 2])
        left = result.loc[result.track_id.eq(1)]
        right = result.loc[result.track_id.eq(2)]
        self.assertEqual(left.vx.dropna().tolist(), [10, 10])
        self.assertEqual(right.vx.dropna().tolist(), [-10, -10])
        self.assertEqual(left.cumulative_distance_mm.tolist(), [0, 1, 2])
        self.assertEqual(right.cumulative_distance_mm.tolist(), [0, 1, 2])
        actual = result.sort_values(["frame_index", "track_id"]).reset_index(drop=True)
        pd.testing.assert_series_equal(actual.kp_nose_x, raw.kp_nose_x)

    def test_smoothing_matches_each_track_run_independently(self):
        raw = self.crossing()
        config = self.config(smooth=True)
        together = compute_features(raw, 10, 1, config)
        for track in [1, 2]:
            isolated = compute_features(raw.loc[raw.track_id.eq(track)], 10, 1, config)
            actual = together.loc[together.track_id.eq(track)]
            np.testing.assert_allclose(actual.bbox_center_x_euro, isolated.bbox_center_x_euro)
            np.testing.assert_allclose(actual.distance_mm, isolated.distance_mm, equal_nan=True)

    def test_motion_plot_keeps_separate_track_lines(self):
        from unittest.mock import patch

        from analysis_ops import _plot_if_column, _setup_plotting

        features = compute_features(self.crossing(), 10, 1, self.config(smooth=False))
        plt, _sns = _setup_plotting()
        fig, ax = plt.subplots()
        with TemporaryDirectory() as tmp, patch.object(plt, "subplots", return_value=(fig, ax)):
            _plot_if_column(
                features,
                "frame_index",
                "speed_mm_per_sec",
                Path(tmp, "speed.png"),
                "Speed",
                "Tracks",
            )
        self.assertEqual(len(ax.lines), 2)
        for line in ax.lines:
            np.testing.assert_allclose(line.get_xdata(), [0, 1, 2])
            np.testing.assert_allclose(line.get_ydata(), [np.nan, 10, 10], equal_nan=True)

    def test_gaps_and_new_ids_never_create_movement_or_smoothing_jumps(self):
        raw = pd.DataFrame(
            [_pose_row(0, 0, 1, 0), _pose_row(2, 0, 1, 1000), _pose_row(3, 0, 9, 2000)]
        )
        result = compute_features(raw, 10, 1, self.config(smooth=True))
        self.assertTrue(result.distance_mm.isna().all())
        self.assertEqual(result.cumulative_distance_mm.tolist(), [0, 0, 0])
        self.assertEqual(result.missing_frame_gap_before.tolist(), [0, 1, 0])
        np.testing.assert_allclose(result.bbox_center_x_euro, result.bbox_center_x)

    def test_untracked_instances_preserved_without_assuming_detection_order_is_identity(self):
        raw = self.crossing().drop(columns="track_id")
        result = compute_features(raw, 10, 1, self.config(smooth=True))
        self.assertEqual(len(result), 6)
        self.assertTrue(result.distance_mm.isna().all())
        self.assertEqual(result.trajectory_id.nunique(), 6)

    def test_legacy_single_instance_keeps_motion_behavior(self):
        raw = self.crossing().loc[lambda df: df.track_id.eq(1)].drop(columns="track_id")
        result = compute_features(raw, 10, 1, self.config(smooth=False))
        self.assertEqual(result.distance_mm.dropna().tolist(), [1, 1])
        self.assertEqual(result.trajectory_id.nunique(), 1)

    def test_workflow_outputs_both_instances_and_distance_columns(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "pose.csv")
            self.crossing().to_csv(path, index=False)
            config = self.config(smooth=False)
            config.detections_csv = str(path)
            config.output_dir = str(Path(tmp, "analysis"))
            config.subject_class = "*"
            result = run_analysis_workflow(config)
            features = pd.read_csv(result["feature_csv"])
            self.assertEqual(len(features), 6)
            self.assertEqual(features.trajectory_id.nunique(), 2)
            self.assertEqual(
                features.center_distances.tolist(), features.center_distances_smoothed.tolist()
            )
            first = json.loads(features.center_distances.iloc[0])
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["distance_px"], 100)
            summary = result["summary"]["summaries_by_class"]["mouse"]
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["analysis_rows"], 6)
            self.assertEqual(summary["trajectory_count"], 2)
            self.assertEqual(summary["total_distance_mm"], 4)

    def test_behavior_windows_do_not_average_two_mice_or_bridge_gaps(self):
        from analysis_ops import _rolling_behavior_features

        frame = pd.DataFrame(
            [
                {"frame_index": f, "trajectory_id": identity, "speed": speed}
                for f in range(3)
                for identity, speed in [("mouse:1", 1), ("mouse:2", 100)]
            ]
        )
        rolled = _rolling_behavior_features(frame, ["speed"], 2)
        for identity, expected in [("mouse:1", 1), ("mouse:2", 100)]:
            values = rolled.loc[frame.trajectory_id.eq(identity), "speed_mean_1s"].dropna()
            self.assertTrue(len(values))
            self.assertTrue(values.eq(expected).all())
        gap = pd.DataFrame(
            {
                "frame_index": [0, 1, 4, 5],
                "trajectory_id": ["mouse:1"] * 4,
                "speed": [1, 1, 100, 100],
            }
        )
        self.assertTrue(_rolling_behavior_features(gap, ["speed"], 3).isna().all().all())

    def test_roi_transitions_never_connect_two_mice(self):
        features = compute_features(self.crossing(), 10, 1, self.config(smooth=False))
        features["roi_label"] = np.where(features.track_id.eq(1), "Left", "Right")
        with TemporaryDirectory() as tmp:
            result = create_roi_outputs(features, Path(tmp), 10)
            transitions = pd.read_csv(result["roi_transition_csv"], index_col=0)
            self.assertEqual(transitions.loc["Left", "Left"], 2)
            self.assertEqual(transitions.loc["Right", "Right"], 2)
            self.assertEqual(transitions.loc["Left", "Right"], 0)
            self.assertEqual(transitions.loc["Right", "Left"], 0)
