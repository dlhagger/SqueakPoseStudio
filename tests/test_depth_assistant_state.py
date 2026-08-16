import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.annotation.depth import (
    DEFAULT_PROBE_TEXT,
    DEFAULT_RANGE_TEXT,
    INVALID_RANGE_TEXT,
    DepthAssistantState,
    DepthPredictionTargetPlan,
    DepthProbe,
    DepthRangeSummary,
    depth_unit,
    format_depth_value,
    normalize_depth_view_mode,
    plan_depth_prediction_targets,
)

METRIC_METADATA = {
    "p02_depth": 0.42,
    "p98_depth": 4.81,
    "median_depth": 1.73,
    "min_depth": 0.2,
    "max_depth": 7.0,
    "valid_pixels": 120,
    "units": "estimated_meters",
    "scale_status": "model_default",
}


class DepthAssistantStateTests(unittest.TestCase):
    def test_view_mode_normalizes_known_values_and_falls_back_to_depth(self):
        state = DepthAssistantState()

        self.assertEqual(state.set_view_mode(" OVERLAY "), "overlay")
        self.assertEqual(state.view_mode, "overlay")
        self.assertEqual(state.set_view_mode("unknown"), "depth")
        self.assertEqual(normalize_depth_view_mode("original"), "original")

    def test_finite_metadata_builds_range_and_is_detached(self):
        metadata = dict(METRIC_METADATA)
        state = DepthAssistantState()

        state.set_metadata(metadata)
        metadata["median_depth"] = 99

        self.assertEqual(
            state.depth_range,
            DepthRangeSummary(
                p02_depth=0.42,
                p98_depth=4.81,
                median_depth=1.73,
                min_depth=0.2,
                max_depth=7.0,
                valid_pixels=120,
            ),
        )
        self.assertTrue(state.has_valid_metadata)

    def test_nonfinite_or_incomplete_range_metadata_is_invalid(self):
        state = DepthAssistantState(
            metadata={"p02_depth": math.nan, "p98_depth": 2, "median_depth": 1}
        )
        self.assertIsNone(state.depth_range)
        self.assertEqual(state.range_text(), INVALID_RANGE_TEXT)

        state.set_metadata({"p02_depth": 1, "p98_depth": 2})
        self.assertIsNone(state.depth_range)
        self.assertEqual(state.range_text(), INVALID_RANGE_TEXT)

    def test_range_text_preserves_metric_display_and_supports_relative_units(self):
        state = DepthAssistantState(metadata=dict(METRIC_METADATA))

        self.assertEqual(
            state.range_text(),
            "Range (2–98%): 0.420–4.810 m · median 1.730 m · Near = bright",
        )

        state.metadata["units"] = "relative"
        self.assertEqual(
            state.range_text(),
            "Range (2–98%): 0.420–4.810 relative · median 1.730 relative · Near = bright",
        )

    def test_missing_metadata_uses_current_default_range_text(self):
        self.assertEqual(DepthAssistantState().range_text(), DEFAULT_RANGE_TEXT)

    def test_probe_normalization_rejects_nonfinite_nonpositive_values(self):
        valid = DepthProbe.from_mapping({"x": 2, "y": 3, "depth": 1.25, "valid": True})
        invalid = DepthProbe.from_mapping({"x": 4, "y": 5, "depth": math.inf, "valid": True})
        nonpositive = DepthProbe.from_mapping({"x": 1, "y": 1, "depth": 0, "valid": True})

        self.assertEqual(valid.as_mapping(), {"x": 2, "y": 3, "depth": 1.25, "valid": True})
        self.assertIsNone(invalid.depth)
        self.assertFalse(invalid.valid)
        self.assertIsNone(nonpositive.depth)

    def test_probe_state_keeps_six_recent_samples_and_formats_delta(self):
        state = DepthAssistantState(metadata=dict(METRIC_METADATA))
        for index in range(7):
            state.add_probe({"x": index, "y": index + 1, "depth": index + 0.5, "valid": True})

        self.assertEqual([probe.x for probe in state.probes], [1, 2, 3, 4, 5, 6])
        self.assertIn("1. (1, 2): 1.500 m", state.probe_text())
        self.assertIn("Δ last two: 1.000 m", state.probe_text())

    def test_probe_text_uses_error_or_default_when_empty(self):
        state = DepthAssistantState()
        self.assertEqual(state.probe_text(), DEFAULT_PROBE_TEXT)

        state.probe_error = "No raw depth map."
        self.assertEqual(state.probe_text(), "No raw depth map.")

    def test_relative_probe_format_and_invalid_value(self):
        state = DepthAssistantState(metadata={"units": "relative"})
        state.add_probe({"x": 1, "y": 2, "depth": 0.75, "valid": True})
        state.add_probe({"x": 3, "y": 4, "depth": None, "valid": False})

        text = state.probe_text()
        self.assertIn("0.750 relative", text)
        self.assertIn("invalid", text)
        self.assertNotIn("Δ last two", text)

    def test_loading_new_image_clears_probes_and_undo_but_same_image_keeps_them(self):
        state = DepthAssistantState(image_name="frame1.png")
        state.add_probe({"x": 1, "y": 2, "depth": 1, "valid": True})
        state.push_undo_snapshot()

        state.load_image("frame1.png", metadata=METRIC_METADATA)
        self.assertEqual(len(state.probes), 1)
        self.assertTrue(state.can_undo)

        state.load_image("frame2.png", metadata=None, probe_error="missing")
        self.assertEqual(state.probes, [])
        self.assertFalse(state.can_undo)
        self.assertEqual(state.probe_error, "missing")

    def test_snapshot_is_detached_and_undo_restores_transitions(self):
        state = DepthAssistantState(view_mode="overlay", image_name="frame.png")
        state.set_metadata(METRIC_METADATA)
        state.add_probe({"x": 2, "y": 3, "depth": 1.2, "valid": True})
        snapshot = state.push_undo_snapshot()

        state.clear_probes()
        state.set_view_mode("original")
        snapshot.metadata["median_depth"] = 100

        self.assertTrue(state.undo())
        self.assertEqual(state.view_mode, "overlay")
        self.assertEqual(len(state.probes), 1)
        self.assertEqual(state.metadata["median_depth"], 1.73)
        self.assertFalse(state.undo())

    def test_clear_resets_image_data_but_preserves_view_preference(self):
        state = DepthAssistantState(view_mode="overlay", image_name="frame.png")
        state.set_metadata(METRIC_METADATA)
        state.add_probe({"x": 1, "y": 1, "depth": 1, "valid": True})

        state.clear()

        self.assertEqual(state.view_mode, "overlay")
        self.assertEqual(state.image_name, "")
        self.assertIsNone(state.metadata)
        self.assertEqual(state.probes, [])

    def test_depth_unit_and_value_formatting(self):
        self.assertEqual(depth_unit({"units": "estimated_meters"}), "m")
        self.assertEqual(depth_unit({"units": "arbitrary"}), "relative")
        self.assertEqual(format_depth_value(1.23456), "1.235 m")
        self.assertEqual(format_depth_value(1.23456, {"units": "relative"}), "1.235 relative")
        self.assertEqual(format_depth_value(math.nan), "invalid")
        self.assertEqual(format_depth_value(None), "invalid")

    def test_target_plan_matches_current_worker_and_commit_shapes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "depth maps" / "images"
            preview_dir = root / "depth maps" / "previews"

            plan = plan_depth_prediction_targets(
                project_root=tmp,
                depth_image_dir=str(image_dir),
                depth_preview_dir=str(preview_dir),
                image_path="/source/frame001.png",
                staging_factory=lambda target: f"{target}.stage",
            )

            self.assertEqual(plan.final_map, str(image_dir / "frame001.npy"))
            self.assertEqual(plan.final_preview, str(preview_dir / "frame001_depth.png"))
            self.assertEqual(plan.final_metadata, str(image_dir / "frame001_depth.json"))
            self.assertEqual(
                plan.worker_paths(),
                {
                    "depth_map_path": f"{plan.final_map}.stage",
                    "depth_preview_path": f"{plan.final_preview}.stage",
                    "depth_metadata_path": f"{plan.final_metadata}.stage",
                },
            )
            self.assertEqual(plan.replacements()[0], (plan.staged_map, plan.final_map))
            self.assertEqual(plan.as_mapping()["staged_metadata"], plan.staged_metadata)
            self.assertEqual(
                DepthPredictionTargetPlan.from_mapping(plan.as_mapping()),
                plan,
            )

    def test_incomplete_restored_target_plan_cannot_be_committed(self):
        plan = DepthPredictionTargetPlan.from_mapping({"final_map": "/tmp/map.npy"})

        with self.assertRaisesRegex(ValueError, "transaction is incomplete"):
            plan.replacements()

    def test_target_plan_rejects_outputs_outside_project(self):
        with TemporaryDirectory() as project, TemporaryDirectory() as outside:
            with self.assertRaises(ValueError):
                plan_depth_prediction_targets(
                    project_root=project,
                    depth_image_dir=outside,
                    depth_preview_dir=outside,
                    image_path="frame.png",
                    staging_factory=lambda target: f"{target}.stage",
                )

    def test_target_plan_cleans_owned_staging_files_after_partial_failure(self):
        with TemporaryDirectory() as tmp:
            calls = []

            def staging_factory(target):
                calls.append(target)
                if len(calls) == 2:
                    raise OSError("injected failure")
                staged = f"{target}.stage"
                Path(staged).parent.mkdir(parents=True, exist_ok=True)
                Path(staged).write_bytes(b"")
                return staged

            with self.assertRaisesRegex(OSError, "injected"):
                plan_depth_prediction_targets(
                    depth_image_dir=os.path.join(tmp, "images"),
                    depth_preview_dir=os.path.join(tmp, "previews"),
                    image_path="frame.png",
                    staging_factory=staging_factory,
                )

            self.assertFalse(Path(f"{calls[0]}.stage").exists())

    def test_target_plan_rejects_non_sibling_staging_paths_without_deleting_them(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            external = Path(outside) / "unowned.stage"
            external.write_bytes(b"keep")

            with self.assertRaisesRegex(ValueError, "siblings"):
                plan_depth_prediction_targets(
                    depth_image_dir=os.path.join(tmp, "images"),
                    depth_preview_dir=os.path.join(tmp, "previews"),
                    image_path="frame.png",
                    staging_factory=lambda _target: str(external),
                )

            self.assertEqual(external.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
