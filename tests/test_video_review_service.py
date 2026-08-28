import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.services.video_review import (
    available_export_frame_indices,
    build_video_review_cache_payload,
    build_video_review_pass_config,
    build_video_signature,
    complete_video_review_pass,
    decide_video_review_cache,
    exported_frame_indices,
    plan_confidence_export,
    plan_export_frame_path,
    plan_video_review_run,
    select_random_export_frames,
    video_review_cache_path,
)


class VideoReviewServiceTests(unittest.TestCase):
    def test_video_signature_uses_file_identity_and_video_metadata(self):
        with TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.mp4"
            video.write_bytes(b"video")

            signature = build_video_signature(str(video), total=120, fps=29.97)

            self.assertEqual(signature["path"], str(video))
            self.assertEqual(signature["size"], 5)
            self.assertEqual(signature["mtime"], video.stat().st_mtime)
            self.assertEqual(signature["total"], 120)
            self.assertEqual(signature["fps"], 29.97)

    def test_cache_path_is_stable_and_inside_project(self):
        with TemporaryDirectory() as tmp:
            first = video_review_cache_path(tmp, "/videos/session.mp4")
            second = video_review_cache_path(tmp, "/videos/session.mp4")

            self.assertEqual(first, second)
            self.assertEqual(os.path.commonpath([tmp, first]), tmp)
            self.assertTrue(first.endswith(".json"))
            self.assertIsNone(video_review_cache_path(tmp, None))

    def test_run_plan_preserves_meta_settings_and_step_counts(self):
        plan = plan_video_review_run(
            video_signature={"path": "/video.mp4", "size": 10},
            model_paths={LAYER_KEYPOINTS: "pose.pt", LAYER_SEGMENTATION: "seg.pt"},
            review_layers=[LAYER_KEYPOINTS, LAYER_SEGMENTATION],
            layer_schemas={LAYER_KEYPOINTS: {"classes": ["mouse"]}},
            start=2,
            end=10,
            stride=3,
            imgsz=640,
            conf=0.25,
            iou=0.7,
            kpvis=0.4,
            requested_batch=0,
            effective_batch=8,
            total=100,
            fps=30.0,
        )

        self.assertEqual(plan.steps_per_pass, 3)
        self.assertEqual(plan.total_steps, 6)
        self.assertEqual(plan.settings["effective_batch"], 8)
        self.assertEqual(plan.meta["model_paths"][LAYER_SEGMENTATION], "seg.pt")
        self.assertEqual(plan.meta["schemas"][LAYER_SEGMENTATION], {})
        self.assertEqual(plan.meta["initial_effective_batch"], 8)

    def test_single_layer_run_keeps_all_configured_model_cache_identities(self):
        plan = plan_video_review_run(
            video_signature={"path": "/video.mp4", "size": 10},
            model_paths={LAYER_KEYPOINTS: "pose.pt", LAYER_SEGMENTATION: "seg.pt"},
            review_layers=[LAYER_KEYPOINTS],
            layer_schemas={
                LAYER_KEYPOINTS: {"classes": ["mouse"]},
                LAYER_SEGMENTATION: {"classes": ["mouse"]},
            },
            start=2,
            end=10,
            stride=3,
            imgsz=640,
            conf=0.25,
            iou=0.7,
            kpvis=0.4,
            requested_batch=0,
            effective_batch=8,
            total=100,
            fps=30.0,
        )

        self.assertEqual(plan.meta["layers"], [LAYER_KEYPOINTS])
        self.assertEqual(
            plan.meta["model_paths"],
            {LAYER_KEYPOINTS: "pose.pt", LAYER_SEGMENTATION: "seg.pt"},
        )
        self.assertIn(LAYER_SEGMENTATION, plan.meta["schemas"])
        self.assertEqual(plan.total_steps, 3)

    def test_pass_config_keeps_worker_protocol(self):
        config = build_video_review_pass_config(
            layer_id=LAYER_SEGMENTATION,
            model_path="seg.pt",
            video_path="/video.mp4",
            device="mps",
            settings={"start": 1, "end": 9, "stride": 2},
        )

        self.assertEqual(
            config,
            {
                "model_path": "seg.pt",
                "video_path": "/video.mp4",
                "workflow": "segmentation",
                "layer_id": LAYER_SEGMENTATION,
                "device": "mps",
                "start": 1,
                "end": 9,
                "stride": 2,
            },
        )

    def test_layered_cache_accepts_matching_identity_and_models(self):
        video = {"path": "/video.mp4", "size": 10, "mtime": 100.0}
        decision = decide_video_review_cache(
            {
                "meta": {
                    "video": {**video, "mtime": 101.5},
                    "model_paths": {LAYER_KEYPOINTS: "pose.pt", LAYER_SEGMENTATION: "seg.pt"},
                },
                "preds_by_layer": {
                    LAYER_KEYPOINTS: {"2": {"ok": True}},
                    LAYER_SEGMENTATION: {"5": {"ok": True}},
                },
            },
            current_video=video,
            review_layers=[LAYER_KEYPOINTS, LAYER_SEGMENTATION],
            model_paths={LAYER_KEYPOINTS: "pose.pt", LAYER_SEGMENTATION: "seg.pt"},
            layer_id=LAYER_KEYPOINTS,
            model_path="pose.pt",
            workflow="pose",
        )

        self.assertTrue(decision.has_predictions)
        self.assertEqual(list(decision.predictions_by_layer[LAYER_KEYPOINTS]), [2])
        self.assertEqual(list(decision.predictions_by_layer[LAYER_SEGMENTATION]), [5])

    def test_cache_rejects_identity_model_and_index_mismatches(self):
        video = {"path": "/video.mp4", "size": 10, "mtime": 100.0}
        base = {
            "meta": {"video": video, "model_paths": {LAYER_KEYPOINTS: "pose.pt"}},
            "preds_by_layer": {LAYER_KEYPOINTS: {"2": {"ok": True}}},
        }
        arguments = {
            "current_video": video,
            "review_layers": [LAYER_KEYPOINTS],
            "model_paths": {LAYER_KEYPOINTS: "pose.pt"},
            "layer_id": LAYER_KEYPOINTS,
            "model_path": "pose.pt",
            "workflow": "pose",
        }

        changed_path = {**base, "meta": {**base["meta"], "video": {**video, "path": "/x"}}}
        self.assertIsNone(decide_video_review_cache(changed_path, **arguments))

        changed_model = {**base, "meta": {**base["meta"], "model_paths": {LAYER_KEYPOINTS: "x"}}}
        self.assertIsNone(decide_video_review_cache(changed_model, **arguments))

        bad_index = {**base, "preds_by_layer": {LAYER_KEYPOINTS: {"bad": {"ok": True}}}}
        self.assertIsNone(decide_video_review_cache(bad_index, **arguments))

    def test_legacy_cache_preserves_single_layer_contract(self):
        video = {"path": "/video.mp4", "size": 10, "mtime": 100.0}
        decision = decide_video_review_cache(
            {
                "meta": {"video": video, "model_path": "seg.pt", "workflow": "segmentation"},
                "preds": {"4": {"ok": True}},
            },
            current_video=video,
            review_layers=[],
            model_paths={},
            layer_id="segmentation",
            model_path="seg.pt",
            workflow="segmentation",
        )

        self.assertTrue(decision.has_predictions)
        self.assertEqual(decision.predictions_by_layer[LAYER_SEGMENTATION], {4: {"ok": True}})

    def test_cache_payload_stringifies_frame_indices_and_omits_empty_layers(self):
        payload = build_video_review_cache_payload(
            {"video": {"path": "/video.mp4"}},
            {LAYER_KEYPOINTS: {3: {"ok": True}}, LAYER_SEGMENTATION: {}},
        )

        self.assertEqual(payload["preds_by_layer"], {LAYER_KEYPOINTS: {"3": {"ok": True}}})

    def test_pass_completion_merges_streamed_and_final_predictions(self):
        completion = complete_video_review_pass(
            partial_predictions={1: {"ok": True, "conf": 0.2}, 2: {"ok": True}},
            result_event={
                "preds": {"1": {"ok": True, "conf": 0.9}, "4": None},
                "canceled": False,
                "had_error": False,
            },
            cancel_requested=False,
            worker_state="finished",
            exit_code=0,
            crashed=False,
            worker_error="",
            stderr="",
        )

        self.assertEqual(completion.predictions[1]["conf"], 0.9)
        self.assertEqual(completion.predictions[2], {"ok": True})
        self.assertEqual(completion.predictions[4], {"ok": False})
        self.assertFalse(completion.canceled)
        self.assertFalse(completion.had_error)

    def test_pass_completion_distinguishes_cancel_from_failure(self):
        canceled = complete_video_review_pass(
            partial_predictions={2: {"ok": True}},
            result_event=None,
            cancel_requested=True,
            worker_state="cancelled",
            exit_code=-1,
            crashed=True,
            worker_error="",
            stderr="",
        )
        failed = complete_video_review_pass(
            partial_predictions={},
            result_event=None,
            cancel_requested=False,
            worker_state="start_failed",
            exit_code=None,
            crashed=False,
            worker_error="could not start",
            stderr="",
        )

        self.assertTrue(canceled.canceled)
        self.assertFalse(canceled.had_error)
        self.assertEqual(canceled.predictions, {2: {"ok": True}})
        self.assertFalse(failed.canceled)
        self.assertTrue(failed.had_error)
        self.assertEqual(failed.error_message, "could not start")

    def test_export_index_scan_accepts_suffixes_and_supported_extensions(self):
        indices = exported_frame_indices(
            [
                "session_src_f000002.png",
                "session_src_f000005_1.JPG",
                "other_src_f000006.png",
                "session_src_f12.png",
            ],
            video_base="session",
            source_id="src",
        )

        self.assertEqual(indices, {2, 5})

    def test_export_path_stays_inside_destination_and_adds_collision_suffix(self):
        with TemporaryDirectory() as tmp:
            existing = Path(tmp) / "my video_src_f000007.png"
            existing.write_bytes(b"existing")

            path = plan_export_frame_path(
                tmp,
                video_base="../my video",
                source_id="src",
                frame_index=7,
            )

            self.assertEqual(path, str(Path(tmp) / "my video_src_f000007_1.png"))
            self.assertEqual(os.path.commonpath([tmp, path]), tmp)

    def test_random_export_selection_excludes_existing_and_sorts_sample(self):
        available = available_export_frame_indices(6, already_exported=[1, 4])
        selected = select_random_export_frames(
            6,
            already_exported=[1, 4],
            count=2,
            sampler=lambda values, count: [values[-1], values[0]][:count],
        )

        self.assertEqual(available, (0, 2, 3, 5))
        self.assertEqual(selected, (0, 5))

    def test_confidence_export_plan_ranks_filters_and_limits(self):
        predictions = {
            1: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.8}]},
            2: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.2}]},
            3: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.5}]},
        }
        plan = plan_confidence_export(
            predictions,
            class_ids=[0],
            order="low",
            balanced=False,
            already_exported=[2],
            count=1,
        )

        self.assertEqual(plan.candidates, ((2, 0.2, 0), (3, 0.5, 0), (1, 0.8, 0)))
        self.assertEqual(plan.pending, ((3, 0.5, 0), (1, 0.8, 0)))
        self.assertEqual(plan.selected, ((3, 0.5, 0),))


if __name__ == "__main__":
    unittest.main()
