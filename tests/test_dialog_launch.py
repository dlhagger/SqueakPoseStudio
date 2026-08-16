import unittest

from squeakpose.ui.dialog_launch import (
    DialogUnavailableError,
    plan_analysis_dialog,
    plan_training_dialog,
    plan_video_review_dialog,
    require_dialog_support,
)


class DialogLaunchPlanTests(unittest.TestCase):
    def test_training_selects_layer_task_and_existing_specific_dataset(self):
        plan = plan_training_dialog(
            project_root="/project",
            layer_id="segmentation",
            is_directory=lambda path: path == "/project/datasets/segment",
        )
        self.assertEqual(plan.default_dataset, "/project/datasets/segment")
        self.assertEqual(plan.default_task, "segment")
        self.assertEqual(plan.layer_id, "segmentation")

        fallback = plan_training_dialog(
            project_root="/project",
            layer_id="keypoints",
            is_directory=lambda _path: False,
        )
        self.assertEqual(fallback.default_dataset, "/project/datasets")
        self.assertEqual(fallback.default_task, "pose")

    def test_depth_feature_notices_preserve_existing_text(self):
        expected = {
            "training": "Depth training is not included in the inference-only MVP.",
            "distillation": "Depth distillation is not included in the inference-only MVP.",
            "analysis": "Depth analysis tools are not included in the MVP yet.",
        }
        for feature, message in expected.items():
            with self.subTest(feature=feature), self.assertRaises(DialogUnavailableError) as raised:
                require_dialog_support(feature, "depth")
            self.assertEqual(raised.exception.title, "Depth MVP")
            self.assertEqual(raised.exception.message, message)

    def test_analysis_plan_contains_only_constructor_configuration(self):
        plan = plan_analysis_dialog(
            project_root="/project",
            app_base_dir="/app",
            layer_id="pose",
        )
        self.assertEqual(plan.project_root, "/project")
        self.assertEqual(plan.app_base_dir, "/app")
        self.assertEqual(plan.layer_id, "keypoints")

    def test_video_review_depth_fallback_and_schemas_match_current_contract(self):
        plan = plan_video_review_dialog(
            active_layer="depth",
            layer_model_paths={"keypoints": "", "segmentation": "segment.pt"},
            pose_classes=["mouse"],
            pose_keypoints=["nose"],
            pose_class_keypoints={"mouse": ["nose"]},
            segmentation_classes=["body"],
        )
        self.assertEqual(plan.layer_id, "segmentation")
        self.assertEqual(plan.workflow, "segmentation")
        self.assertEqual(
            plan.model_paths,
            {"keypoints": "", "segmentation": "segment.pt"},
        )
        self.assertEqual(plan.active_schema["classes"], ["body"])
        self.assertEqual(plan.layer_schemas["keypoints"]["class_keypoints"], {"mouse": ["nose"]})

        keypoints_default = plan_video_review_dialog(
            active_layer="depth",
            layer_model_paths={},
            pose_classes=["mouse"],
            pose_keypoints=["nose"],
            pose_class_keypoints={},
            segmentation_classes=["body"],
        )
        self.assertEqual(keypoints_default.layer_id, "keypoints")


if __name__ == "__main__":
    unittest.main()
