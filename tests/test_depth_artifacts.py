import os
import tempfile
import unittest

from squeakpose.annotation.depth import (
    load_depth_artifacts,
    plan_depth_artifacts,
)
from squeakpose.ui.depth_controller import DepthAssistantController


class _Array:
    def __init__(self, shape, ndim=2):
        self.shape = shape
        self.ndim = ndim


class DepthArtifactTests(unittest.TestCase):
    def test_plan_resolves_sidecars_and_loader_validates_alignment(self):
        with tempfile.TemporaryDirectory() as root:
            image_dir = os.path.join(root, "depth", "images")
            preview_dir = os.path.join(root, "depth", "previews")
            os.makedirs(image_dir)
            os.makedirs(preview_dir)
            plan = plan_depth_artifacts(
                depth_image_dir=image_dir,
                depth_preview_dir=preview_dir,
                image_name="frame.001.png",
                image_width=20,
                image_height=10,
                project_root=root,
            )
            for path in (plan.map_path, plan.metadata_path, plan.preview_path):
                open(path, "wb").close()

            result = load_depth_artifacts(
                plan,
                array_reader=lambda _path: _Array((10, 20)),
                metadata_reader=lambda _path: {
                    "p02_depth": 1,
                    "p98_depth": 4,
                    "median_depth": 2,
                },
            )

            self.assertTrue(plan.map_path.endswith("frame.001.npy"))
            self.assertEqual(result.depth_map.shape, (10, 20))
            self.assertEqual(result.metadata["median_depth"], 2)
            self.assertTrue(result.preview_available)
            self.assertEqual(result.map_error, "")

    def test_missing_reader_and_invalid_shape_produce_ui_ready_errors(self):
        plan = plan_depth_artifacts(
            depth_image_dir="/tmp/depth",
            depth_preview_dir="/tmp/preview",
            image_name="frame.png",
            image_width=20,
            image_height=10,
        )
        unavailable = load_depth_artifacts(
            plan,
            array_reader=None,
            metadata_reader=None,
            is_file=lambda _path: True,
        )
        invalid = load_depth_artifacts(
            plan,
            array_reader=lambda _path: _Array((20, 10)),
            metadata_reader=None,
            is_file=lambda _path: True,
        )

        self.assertIn("NumPy is unavailable", unavailable.map_error)
        self.assertIn("does not match image", invalid.map_error)
        self.assertIsNone(invalid.depth_map)

    def test_controller_binds_loaded_map_metadata_and_probe_error(self):
        plan = plan_depth_artifacts(
            depth_image_dir="/tmp/depth",
            depth_preview_dir="/tmp/preview",
            image_name="frame.png",
            image_width=4,
            image_height=3,
        )
        controller = DepthAssistantController(sampler=lambda *_args, **_kwargs: {})
        result = controller.load_artifacts(
            plan,
            array_reader=lambda _path: _Array((3, 4)),
            metadata_reader=lambda _path: {"units": "relative"},
            is_file=lambda _path: True,
        )

        self.assertIs(controller.depth_map, result.depth_map)
        self.assertEqual(controller.state.image_name, "frame.png")
        self.assertEqual(controller.state.metadata, {"units": "relative"})
        self.assertEqual(controller.state.probe_error, "")


if __name__ == "__main__":
    unittest.main()
