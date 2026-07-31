import unittest

from layer_ops import (
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    layer_model_paths,
    layer_worker_mode,
    normalize_layer_id,
    normalize_layer_settings,
)


class LayerOpsTests(unittest.TestCase):
    def test_legacy_workflow_names_normalize_to_layers(self):
        self.assertEqual(normalize_layer_id("pose"), LAYER_KEYPOINTS)
        self.assertEqual(normalize_layer_id("keypoint"), LAYER_KEYPOINTS)
        self.assertEqual(normalize_layer_id("segment"), LAYER_SEGMENTATION)
        self.assertEqual(normalize_layer_id("masks"), LAYER_SEGMENTATION)

    def test_layer_definitions_drive_worker_and_dataset_tasks(self):
        keypoints = layer_definition(LAYER_KEYPOINTS)
        segmentation = layer_definition(LAYER_SEGMENTATION)

        self.assertEqual(keypoints.model_task, "pose")
        self.assertEqual(keypoints.dataset_task, "pose")
        self.assertEqual(segmentation.model_task, "segment")
        self.assertEqual(segmentation.dataset_task, "segment")
        self.assertEqual(layer_worker_mode(LAYER_KEYPOINTS), "pose")

    def test_layer_settings_preserve_independent_model_paths(self):
        settings = normalize_layer_settings(
            {
                "pose": {"model_path": "pose.pt"},
                "segmentation": {
                    "model_path": "segment.pt",
                    "custom": True,
                },
            }
        )
        paths = layer_model_paths(settings, resolve_path=lambda path: f"/p/{path}")

        self.assertEqual(paths[LAYER_KEYPOINTS], "/p/pose.pt")
        self.assertEqual(paths[LAYER_SEGMENTATION], "/p/segment.pt")
        self.assertTrue(settings[LAYER_SEGMENTATION]["custom"])


if __name__ == "__main__":
    unittest.main()
