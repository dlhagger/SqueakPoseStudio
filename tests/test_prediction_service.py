import os
import unittest

from squeakpose.services.prediction import (
    DepthPredictionTargets,
    PredictionValidationError,
    build_prediction_load_request,
    build_prediction_request,
    correlate_prediction_event,
    plan_prediction_application,
    validate_model_task_for_layer,
    validate_prediction_identity,
)


def _depth_targets() -> DepthPredictionTargets:
    return DepthPredictionTargets(
        final_map="final/frame.npy",
        final_preview="final/frame_depth.png",
        final_metadata="final/frame_depth.json",
        staged_map="staged/frame.npy",
        staged_preview="staged/frame_depth.png",
        staged_metadata="staged/frame_depth.json",
    )


class PredictionRequestTests(unittest.TestCase):
    def test_pose_request_matches_existing_worker_json(self):
        request = build_prediction_request(
            request_id=7,
            layer_id="keypoints",
            model_path="pose.pt",
            image_path="frame.png",
            device="cuda",
        )

        self.assertEqual(
            request.as_worker_payload(),
            {
                "command": "predict",
                "request_id": 7,
                "layer_id": "keypoints",
                "model_path": "pose.pt",
                "image_path": "frame.png",
                "workflow": "pose",
                "device": "cuda",
            },
        )

    def test_load_request_normalizes_legacy_layer_alias(self):
        request = build_prediction_load_request(
            request_id="warm-2",
            layer_id="segment",
            model_path="segment.pt",
            device="mps",
        )

        self.assertEqual(request.layer_id, "segmentation")
        self.assertEqual(request.workflow, "segmentation")
        self.assertNotIn("image_path", request.as_worker_payload())

    def test_depth_request_adds_staged_output_paths(self):
        targets = _depth_targets()
        request = build_prediction_request(
            request_id=8,
            layer_id="depth",
            model_path="depth.pt",
            image_path="frame.png",
            depth_targets=targets,
        )

        payload = request.as_worker_payload()
        self.assertEqual(payload["workflow"], "depth")
        self.assertEqual(payload["depth_map_path"], targets.staged_map)
        self.assertEqual(payload["depth_preview_path"], targets.staged_preview)
        self.assertEqual(payload["depth_metadata_path"], targets.staged_metadata)

    def test_request_validation_rejects_missing_or_incompatible_fields(self):
        with self.assertRaisesRegex(PredictionValidationError, "model_path"):
            build_prediction_request(
                request_id=1,
                layer_id="keypoints",
                model_path="",
                image_path="frame.png",
            )
        with self.assertRaisesRegex(PredictionValidationError, "image_path"):
            build_prediction_request(
                request_id=1,
                layer_id="keypoints",
                model_path="pose.pt",
                image_path="",
            )
        with self.assertRaisesRegex(PredictionValidationError, "output paths"):
            build_prediction_request(
                request_id=1,
                layer_id="depth",
                model_path="depth.pt",
                image_path="frame.png",
            )
        with self.assertRaisesRegex(PredictionValidationError, "only valid"):
            build_prediction_request(
                request_id=1,
                layer_id="keypoints",
                model_path="pose.pt",
                image_path="frame.png",
                depth_targets=_depth_targets(),
            )
        with self.assertRaisesRegex(PredictionValidationError, "Unsupported layer_id"):
            build_prediction_load_request(
                request_id=1,
                layer_id="detect",
                model_path="detect.pt",
            )

    def test_model_task_validation_matches_worker_rules(self):
        self.assertEqual(validate_model_task_for_layer("keypoints", "keypoints"), "pose")
        self.assertEqual(validate_model_task_for_layer(None, "segmentation"), "segment")
        with self.assertRaisesRegex(PredictionValidationError, "task mismatch"):
            validate_model_task_for_layer("detect", "segmentation")


class PredictionCorrelationTests(unittest.TestCase):
    def test_matching_result_is_applied_only_to_the_requested_image(self):
        requested = os.path.join("project", "frame.png")
        decision = correlate_prediction_event(
            {
                "event": "result",
                "request_id": 4,
                "canceled": False,
                "had_error": False,
                "prediction": {"workflow": "pose", "detections": []},
            },
            current_request_id=4,
            requested_image_path=requested,
            displayed_image_path=os.path.abspath(requested),
        )

        self.assertEqual(decision.action, "apply")
        self.assertTrue(decision.matched)
        self.assertEqual(decision.prediction, {"workflow": "pose", "detections": []})

    def test_stale_or_wrong_image_results_are_not_applied(self):
        stale = correlate_prediction_event(
            {"event": "result", "request_id": 3, "prediction": {}},
            current_request_id=4,
            requested_image_path="first.png",
            displayed_image_path="first.png",
        )
        wrong_image = correlate_prediction_event(
            {
                "event": "result",
                "request_id": 4,
                "canceled": False,
                "had_error": False,
                "prediction": {},
            },
            current_request_id=4,
            requested_image_path="first.png",
            displayed_image_path="second.png",
        )

        self.assertEqual(stale.action, "ignore")
        self.assertFalse(stale.matched)
        self.assertEqual(wrong_image.action, "discard")
        self.assertTrue(wrong_image.matched)

    def test_terminal_error_and_cancel_decisions_preserve_messages(self):
        canceled = correlate_prediction_event(
            {"event": "result", "request_id": 5, "canceled": True},
            current_request_id=5,
        )
        failed = correlate_prediction_event(
            {
                "event": "result",
                "request_id": 5,
                "had_error": True,
                "error_message": "model failed",
            },
            current_request_id=5,
        )
        malformed = correlate_prediction_event(
            {"event": "result", "request_id": 5, "prediction": None},
            current_request_id=5,
        )
        worker_error = correlate_prediction_event(
            {"event": "error", "request_id": None, "error_message": "worker failed"},
            current_request_id=5,
        )
        background_error = correlate_prediction_event(
            {"event": "error", "request_id": 2, "error_message": "warm-up failed"},
            current_request_id=5,
        )

        self.assertEqual(canceled.action, "cancel")
        self.assertEqual(failed.action, "error")
        self.assertEqual(failed.error_message, "model failed")
        self.assertEqual(malformed.action, "error")
        self.assertIn("no prediction payload", malformed.error_message)
        self.assertEqual(worker_error.action, "error")
        self.assertTrue(worker_error.matched)
        self.assertEqual(background_error.action, "background_error")


class PredictionApplicationPlanTests(unittest.TestCase):
    def test_identity_validation_accepts_legacy_fields_and_rejects_mismatch(self):
        self.assertEqual(
            validate_prediction_identity(
                {"layer_id": "keypoint", "workflow": "pose"},
                expected_layer="keypoints",
            ),
            "keypoints",
        )
        self.assertEqual(
            validate_prediction_identity({}, expected_layer="segmentation"), "segmentation"
        )
        with self.assertRaisesRegex(PredictionValidationError, "does not match"):
            validate_prediction_identity(
                {"layer_id": "segmentation", "workflow": "segmentation"},
                expected_layer="keypoints",
            )
        with self.assertRaisesRegex(PredictionValidationError, "Unsupported workflow"):
            validate_prediction_identity(
                {"workflow": "detect"},
                expected_layer="keypoints",
            )

    def test_pose_plan_selects_last_highest_confidence_detection_per_class(self):
        prediction = {
            "layer_id": "keypoints",
            "workflow": "pose",
            "detections": [
                {
                    "class_id": 0,
                    "confidence": 0.7,
                    "xyxy": [0, 1, 10, 11],
                    "keypoints": [[1, 2, 0.4], [3, 4, 0.5]],
                },
                {
                    "class_id": 0,
                    "confidence": 0.7,
                    "xyxy": [5, 6, 25, 26],
                    "keypoints": [[7, 8, 0.8], [9, 10, 0.9]],
                },
                {
                    "class_id": 1,
                    "confidence": 0.8,
                    "xyxy": [10, 20, 30, 50],
                    "keypoints": [[11, 21, 0.6], [12, 22, 0.95], [13, 23, 0.3]],
                },
                {"class_id": 99, "confidence": 1.0, "xyxy": [0, 0, 100, 100]},
            ],
        }

        plan = plan_prediction_application(
            prediction,
            expected_layer="keypoints",
            class_names=["mouse", "rat"],
            canonical_keypoints=["nose", "tail", "ear"],
            class_keypoints={"mouse": ["nose", "tail"], "rat": ["tail"]},
            active_class_id=0,
        )

        self.assertEqual(plan.outcome, "ready")
        self.assertEqual(plan.selected_classes, (0, 1))
        self.assertEqual(len(plan.pose), 2)
        mouse, rat = plan.pose
        self.assertEqual(
            (mouse.class_id, mouse.confidence, mouse.x, mouse.y, mouse.width, mouse.height),
            (0, 0.7, 5.0, 6.0, 20.0, 20.0),
        )
        self.assertEqual([point.name for point in mouse.keypoints], ["nose", "tail"])
        self.assertEqual([point.name for point in rat.keypoints], ["tail"])
        self.assertEqual(rat.keypoints[0].confidence, 0.95)

    def test_pose_plan_falls_back_to_active_class_and_reports_bad_boxes(self):
        prediction = {
            "detections": [
                {"class_id": 90, "confidence": 0.5, "xyxy": [0, 0, 0, 5]},
                {"class_id": 91, "confidence": 0.9, "xyxy": [0, 0, 5, 0]},
            ]
        }

        plan = plan_prediction_application(
            prediction,
            expected_layer="keypoints",
            class_names=["mouse", "rat"],
            active_class_id=1,
        )

        self.assertEqual(plan.selected_classes, (1,))
        self.assertEqual(plan.outcome, "no_usable_boxes")
        self.assertEqual(plan.pose, ())

    def test_segmentation_plan_counts_selected_detections_without_masks(self):
        prediction = {
            "layer_id": "segmentation",
            "workflow": "segmentation",
            "detections": [
                {"class_id": 0, "confidence": 0.9, "segments": [[0, 0], [1, 1]]},
                {
                    "class_id": 1,
                    "confidence": 0.8,
                    "segments": [[2, 2], [8, 2], [8, 8], [2, 8]],
                },
            ],
        }

        plan = plan_prediction_application(
            prediction,
            expected_layer="segmentation",
            class_names=["mouse", "rat"],
        )

        self.assertEqual(plan.outcome, "ready")
        self.assertEqual(plan.missing_mask_count, 1)
        self.assertEqual(len(plan.segmentation), 1)
        self.assertEqual(plan.segmentation[0].class_id, 1)
        self.assertEqual(plan.segmentation[0].points[2], (8.0, 8.0))

    def test_empty_and_unusable_detection_outcomes_are_distinct(self):
        empty = plan_prediction_application(
            {"detections": []},
            expected_layer="keypoints",
            class_names=["mouse"],
        )
        unusable = plan_prediction_application(
            {"detections": [None, "bad"]},
            expected_layer="segmentation",
            class_names=["mouse"],
        )
        no_masks = plan_prediction_application(
            {"detections": [{"class_id": 0, "segments": []}]},
            expected_layer="segmentation",
            class_names=["mouse"],
        )

        self.assertEqual(empty.outcome, "no_detections")
        self.assertEqual(unusable.outcome, "no_usable_detections")
        self.assertEqual(no_masks.outcome, "no_usable_masks")

    def test_depth_plan_validates_identity_and_builds_commit_replacements(self):
        targets = _depth_targets()
        plan = plan_prediction_application(
            {
                "ok": True,
                "layer_id": "depth",
                "workflow": "depth",
                "depth_metadata": {"median_depth": "1.75", "width": 32, "height": 24},
            },
            expected_layer="depth",
            depth_targets=targets,
        )

        self.assertEqual(plan.outcome, "ready")
        self.assertEqual(plan.depth.replacements, targets.replacements())
        self.assertEqual(plan.depth.median_depth, 1.75)
        with self.assertRaisesRegex(PredictionValidationError, "transaction is incomplete"):
            plan_prediction_application(
                {"workflow": "depth"},
                expected_layer="depth",
            )


if __name__ == "__main__":
    unittest.main()
