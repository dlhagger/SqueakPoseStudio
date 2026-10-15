"""Regression coverage for one subject and multiple same-class non-subjects."""

import csv
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis_ops import AnalysisConfig, AnalysisError
from segmentation_analysis_ops import (
    compute_segmentation_detection_features,
    compute_segmentation_track_features,
    run_segmentation_analysis_workflow,
)
from squeakpose.services.class_tracking import ClassIsolatedTracker
from squeakpose.services.inference_runtime import run_segmentation_video_inference
from squeakpose.services.tracking import resolve_subject_class
from tests.test_inference_ops import _Boxes, _Masks, _PoseModel, _Result
from tests.test_unified_tracking_analysis import _pose_row, _segmentation_row
from unified_analysis_ops import build_unified_frame_table


def mixed_rows():
    rows = []
    for frame in range(4):
        for det, track, x, name, conf in [
            (0, 1, 100, "object", 0.99),
            (1, 2, 200, "object", 0.98),
            (2, 3, frame * 2, "mouse", 0.8),
        ]:
            if name == "mouse" and frame == 2:
                continue
            row = _segmentation_row(frame, det, track, x, confidence=conf, expected=1)
            row.update(
                class_name=name,
                class_id=int(name == "object"),
                subject_class="mouse",
                tracks_in_frame=int(frame != 2),
                total_tracks_in_frame=3,
            )
            rows.append(row)
    return pd.DataFrame(rows)


class MixedClassInferenceTests(unittest.TestCase):
    def config(self, **kwargs):
        return AnalysisConfig(
            detections_csv="", output_dir="unused", smooth=False, make_plots=False, **kwargs
        )

    def test_subject_selection_is_optional_for_inference(self):
        self.assertEqual(resolve_subject_class(["mouse"]), "mouse")
        self.assertEqual(resolve_subject_class(["mouse", "object"], "mouse"), "mouse")
        self.assertEqual(resolve_subject_class(["mouse", "object"]), "")
        with self.assertRaises(ValueError):
            resolve_subject_class(["mouse", "object"], "rat")

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("ultralytics", "torch", "lap")),
        "Optional Ultralytics tracking dependencies are not installed.",
    )
    def test_real_bytetrack_isolates_classes_and_remaps_detection_indices(self):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
        wrapper = ClassIsolatedTracker(BYTETracker(args), [0, 1])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        first = np.array(
            [[10, 10, 30, 30, 0.9, 0], [10, 10, 30, 30, 0.95, 1], [60, 60, 80, 80, 0.9, 1]],
            dtype=np.float32,
        )
        initial = wrapper.update(Boxes(first, (100, 100)).numpy(), image)
        self.assertEqual(initial[:, -1].tolist(), [0, 1, 2])
        self.assertEqual(len(set(initial[:, 4])), 3)
        reordered = first[[2, 0, 1]]
        tracked = wrapper.update(Boxes(reordered, (100, 100)).numpy(), image)
        self.assertEqual(tracked[:, 4].tolist(), initial[[2, 0, 1], 4].tolist())
        self.assertEqual(tracked[:, 6].tolist(), [1, 0, 1])
        # Missing mouse does not prevent object tracking or advance its position.
        objects = wrapper.update(Boxes(first[[1, 2]], (100, 100)).numpy(), image)
        self.assertEqual(objects[:, 4].tolist(), initial[[1, 2], 4].tolist())
        wrapper.reset()
        restarted = wrapper.update(Boxes(first, (100, 100)).numpy(), image)
        self.assertEqual(restarted[:, 4].tolist(), [1, 2, 3])

    def test_quality_review_counts_subjects_but_retains_object_overlays(self):
        from squeakpose.services.inference_review import ModelInferenceSource, _FrameAccumulator

        source = ModelInferenceSource(
            "segmentation", "model.pt", "video", "video.mp4", "inference.csv"
        )
        accumulator = _FrameAccumulator(source, 0)
        for row in mixed_rows().loc[lambda df: df["frame"].eq(0)].to_dict("records"):
            accumulator.add({key: str(value) for key, value in row.items()})
        candidate = accumulator.finish()
        self.assertEqual(candidate.detections_in_frame, 1)
        self.assertEqual(candidate.tracks_in_frame, 1)
        self.assertEqual(candidate.track_ids, ("3",))
        self.assertEqual(len(candidate.overlays), 3)
        self.assertNotIn("extra_detection", candidate.reasons)
        self.assertNotIn("track_count_mismatch", candidate.reasons)

    def test_runtime_exports_all_masks_but_counts_only_mouse(self):
        polygons = [[[x, 0], [x + 5, 0], [x + 5, 5]] for x in (0, 20, 40)]
        results = []
        for ids, classes in [([3, 1, 2], [0, 1, 1]), ([1, 2], [1, 1])]:
            n = len(ids)
            results.append(
                _Result(
                    boxes=_Boxes(
                        xyxy=[[i * 20, 0, i * 20 + 5, 5] for i in range(n)],
                        conf=[0.9] * n,
                        cls=classes,
                        ids=ids,
                    ),
                    masks=_Masks(xy=polygons[:n]),
                    names={0: "mouse", 1: "object"},
                )
            )
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp, "inference.csv"))
            result = run_segmentation_video_inference(
                model=_PoseModel(results),
                video_path="unused.mp4",
                csv_path=path,
                classes=["mouse", "object"],
                subject_class="mouse",
                tracking_enabled=True,
                expected_animal_count=1,
                device="cpu",
                total_frames=2,
                fps=30,
            )
            self.assertFalse(result.had_error, result.error_message)
            with Path(path).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["tracks_in_frame"], "1")
            self.assertEqual(rows[0]["total_tracks_in_frame"], "3")
            self.assertEqual(rows[3]["tracks_in_frame"], "0")
            self.assertEqual(result.frames_with_track_count_mismatch, 1)
            self.assertEqual(result.frames_without_track_ids, 1)
            self.assertEqual([json.loads(row["mask_polygon"]) for row in rows[:3]], polygons)

    def test_analysis_selects_mouse_and_never_credits_movement_across_gap(self):
        raw = mixed_rows()
        features = compute_segmentation_detection_features(raw, 1)
        self.assertEqual(len(features), len(raw))
        self.assertIn("track_id", features)
        track = compute_segmentation_track_features(features, raw, 30, 1, self.config())
        self.assertEqual(track.class_name.tolist(), ["mouse"] * 3)
        self.assertEqual(track.frame_index.tolist(), [0, 1, 3])
        self.assertEqual(track.distance_mm.iloc[1], 2)
        self.assertTrue(pd.isna(track.distance_mm.iloc[2]))
        self.assertEqual(track.cumulative_distance_mm.iloc[-1], 2)

    def test_analysis_can_choose_mouse_or_objects_after_unscoped_inference(self):
        raw = mixed_rows().drop(columns="subject_class")
        features = compute_segmentation_detection_features(raw, 1)
        mouse = compute_segmentation_track_features(
            features, raw, 30, 1, self.config(subject_class="mouse")
        )
        objects = compute_segmentation_track_features(
            features, raw, 30, 1, self.config(subject_class="object")
        )
        self.assertEqual(mouse.class_name.unique().tolist(), ["mouse"])
        self.assertEqual(objects.class_name.unique().tolist(), ["object"])
        self.assertEqual(set(objects.trajectory_id), {"1", "2"})
        self.assertEqual(objects.distance_mm.sum(), 0)
        self.assertEqual(len(features), 11)

    def test_unified_analysis_selects_class_without_counting_other_classes(self):
        pose = pd.DataFrame(columns=list(_pose_row(0, 0, 0, 0)))
        raw = mixed_rows().drop(columns="subject_class")
        raw["tracks_in_frame"] = 3
        raw["expected_animal_count"] = 0
        table = build_unified_frame_table(
            pose, raw, self.config(subject_class="object"), video_path="", fps=30, scale=1
        )
        self.assertEqual(set(table.segmentation_class_name.dropna()), {"object"})
        self.assertEqual(len(table), 8)
        self.assertEqual(table.extra_segmentation_tracks.max(), 0)

    def test_mixed_class_runtime_runs_without_subject_or_count_checks(self):
        result = _Result(
            boxes=_Boxes(
                xyxy=[[0, 0, 5, 5], [10, 0, 15, 5], [20, 0, 25, 5]],
                conf=[0.9, 0.8, 0.7],
                cls=[0, 1, 1],
                ids=[1, 2, 3],
            ),
            masks=_Masks(xy=[[[0, 0], [5, 0], [5, 5]]] * 3),
            names={0: "mouse", 1: "object"},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "all.csv")
            output = run_segmentation_video_inference(
                model=_PoseModel([result]),
                video_path="video.mp4",
                csv_path=str(path),
                classes=["mouse", "object"],
                device="cpu",
                total_frames=1,
                tracking_enabled=True,
                expected_animal_count=0,
            )
            self.assertFalse(output.had_error, output.error_message)
            self.assertEqual(output.rows_written, 3)
            self.assertEqual(output.frames_with_track_count_mismatch, 0)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["subject_class"] for row in rows}, {""})
            self.assertEqual({row["tracks_in_frame"] for row in rows}, {"3"})

    def test_analysis_does_not_join_subject_track_fragments(self):
        raw = mixed_rows()
        raw.loc[raw["frame"].eq(3) & raw.class_name.eq("mouse"), "track_id"] = 9
        features = compute_segmentation_detection_features(raw, 1)
        track = compute_segmentation_track_features(features, raw, 30, 1, self.config())
        self.assertEqual(track.trajectory_id.tolist(), ["3", "3", "9"])
        self.assertTrue(pd.isna(track.distance_mm.iloc[-1]))

    def test_legacy_mixed_class_analysis_fails_instead_of_following_object(self):
        raw = mixed_rows().drop(columns="subject_class")
        features = compute_segmentation_detection_features(raw, 1)
        with self.assertRaisesRegex(AnalysisError, "Choose a class"):
            compute_segmentation_track_features(features, raw, 30, 1, self.config())

    def test_unified_analysis_keeps_missing_mouse_gap(self):
        pose = pd.DataFrame(columns=list(_pose_row(0, 0, 0, 0)))
        table = build_unified_frame_table(
            pose, mixed_rows(), self.config(), video_path="", fps=30, scale=1
        )
        self.assertEqual(table.segmentation_class_name.dropna().unique().tolist(), ["mouse"])
        self.assertFalse(table.loc[table.frame_index.eq(2), "segmentation_valid"].iloc[0])
        self.assertEqual(table.extra_segmentation_tracks.max(), 0)

    def test_all_classes_analyzes_both_classes_without_duplicate_detections(self):
        from analysis_ops import run_analysis_workflow

        with TemporaryDirectory() as tmp:
            path = Path(tmp, "input.csv")
            raw = mixed_rows().drop(columns="subject_class")
            raw.to_csv(path, index=False)
            config = self.config(subject_class="*")
            config.detections_csv = str(path)
            config.output_dir = str(Path(tmp, "analysis"))
            result = run_analysis_workflow(config)
            features = pd.read_csv(result["feature_csv"])
            self.assertEqual(
                features.analysis_class.value_counts().to_dict(), {"object": 8, "mouse": 3}
            )
            self.assertEqual(len(pd.read_csv(result["segmentation_detections_csv"])), 11)
            self.assertEqual(set(result["summary"]["summaries_by_class"]), {"mouse", "object"})
            self.assertFalse(Path(config.output_dir, "classes").exists())
            self.assertNotIn("results_by_class", result)
            self.assertEqual(set(features.track_id), {1, 2, 3})
            first_frame = features.loc[features.frame_index.eq(0)]
            self.assertTrue(
                all(len(json.loads(value)) == 2 for value in first_frame.center_distances)
            )
            mouse = first_frame.loc[first_frame.class_name.eq("mouse")].iloc[0]
            self.assertEqual(
                {entry["track_id"] for entry in json.loads(mouse.center_distances)}, {1, 2}
            )
            self.assertNotIn("squeakpose-class-work-", json.dumps(result))
            self.assertEqual(result["summary"]["analysis_kind"], "all_classes")
            self.assertNotIn("average_speed_mm_per_sec", result["summary"])

    def test_all_classes_preserves_plots_without_publishing_class_tables(self):
        from analysis_ops import run_analysis_workflow

        with TemporaryDirectory() as tmp:
            path = Path(tmp, "input.csv")
            mixed_rows().to_csv(path, index=False)
            config = self.config(subject_class="*")
            config.detections_csv = str(path)
            config.output_dir = str(Path(tmp, "analysis"))
            config.make_plots = True
            result = run_analysis_workflow(config)
            self.assertTrue(result["plot_paths"])
            for plot in result["plot_paths"]:
                self.assertTrue(Path(plot).is_file())
                self.assertEqual(Path(plot).parent.name, "plots")
            analysis_tables = list(Path(config.output_dir).rglob("analysis_features.csv"))
            self.assertEqual(analysis_tables, [Path(result["feature_csv"])])
            self.assertFalse(Path(config.output_dir, "classes").exists())

    def test_all_classes_renders_one_video_with_every_instance(self):
        from unittest.mock import patch

        from analysis_ops import run_analysis_workflow

        with TemporaryDirectory() as tmp:
            path = Path(tmp, "input.csv")
            mixed_rows().to_csv(path, index=False)
            config = self.config(subject_class="*")
            config.detections_csv = str(path)
            config.output_dir = str(Path(tmp, "analysis"))
            config.make_annotated_video = True

            def render(table, video_path, output_path, fps, **kwargs):
                self.assertEqual(set(table.analysis_class), {"mouse", "object"})
                self.assertEqual(len(table), 11)
                output_path.write_bytes(b"video-placeholder")
                return str(output_path)

            with patch(
                "segmentation_analysis_ops.render_segmentation_annotated_video", side_effect=render
            ) as renderer:
                result = run_analysis_workflow(config)
                renderer.assert_called_once()
            self.assertTrue(Path(result["annotated_video"]).is_file())

    def test_all_classes_rolls_back_if_one_class_fails(self):
        from analysis_ops import run_all_class_analysis, run_analysis_workflow

        with TemporaryDirectory() as tmp:
            path = Path(tmp, "input.csv")
            mixed_rows().to_csv(path, index=False)
            output = Path(tmp, "analysis")
            output.mkdir()
            previous = output / "analysis_features.csv"
            previous.write_text("previous output")
            config = self.config(subject_class="*")
            config.detections_csv = str(path)
            config.output_dir = str(output)

            def runner(selected, progress):
                if selected.subject_class == "object":
                    raise AnalysisError("deliberate failure")
                return run_analysis_workflow(selected, progress)

            with self.assertRaisesRegex(AnalysisError, "deliberate failure"):
                run_all_class_analysis(config, input_paths=[str(path)], runner=runner)
            self.assertEqual(previous.read_text(), "previous output")
            self.assertFalse((output / "classes").exists())

    def test_combined_all_classes_does_not_match_objects_to_mouse_pose(self):
        from unified_analysis_ops import run_unified_analysis_workflow

        with TemporaryDirectory() as tmp:
            pose_path = Path(tmp, "pose.csv")
            seg_path = Path(tmp, "seg.csv")
            pd.DataFrame(
                [_pose_row(frame, 0, 3, frame * 2, expected=1) for frame in range(4)]
            ).to_csv(pose_path, index=False)
            mixed_rows().drop(columns="subject_class").to_csv(seg_path, index=False)
            config = self.config(subject_class="*")
            config.output_dir = str(Path(tmp, "analysis"))
            result = run_unified_analysis_workflow(
                config, pose_csv=str(pose_path), segmentation_csv=str(seg_path)
            )
            features = pd.read_csv(result["feature_csv"])
            self.assertEqual(set(features.analysis_class), {"mouse", "object"})
            objects = features.loc[features.analysis_class.eq("object")]
            self.assertFalse(objects.pose_valid.any())
            self.assertEqual(len(objects), 8)
            self.assertTrue(Path(config.output_dir, "analysis.csv").is_file())
            manifest = json.loads(Path(config.output_dir, "analysis_manifest.json").read_text())
            self.assertEqual(manifest["analysis_scope"], "all_classes")
            self.assertEqual(set(manifest["summaries_by_class"]), {"mouse", "object"})
            self.assertNotIn("squeakpose-class-work-", json.dumps(manifest))

    def test_workflow_preserves_objects_in_detection_export(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "input.csv")
            mixed_rows().to_csv(path, index=False)
            config = self.config()
            config.detections_csv = str(path)
            config.output_dir = str(Path(tmp, "analysis"))
            result = run_segmentation_analysis_workflow(config)
            detections = pd.read_csv(Path(config.output_dir, "segmentation_detections.csv"))
            primary = pd.read_csv(Path(config.output_dir, "analysis_features.csv"))
            self.assertEqual(
                detections.class_name.value_counts().to_dict(), {"object": 8, "mouse": 3}
            )
            self.assertEqual(primary.class_name.unique().tolist(), ["mouse"])
            self.assertEqual(result["summary"]["subject_class"], "mouse")
