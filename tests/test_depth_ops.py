import json
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from depth_ops import (
    DepthMapError,
    colorize_depth_map,
    depth_array_from_result,
    keypoint_depth_label,
    sample_depth_map,
    serialize_depth_prediction_result,
)


class _Tensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._data)


class _Depth:
    def __init__(self, data):
        self.data = _Tensor(data)


class _Result:
    def __init__(self, data, *, orig_shape=None):
        self.depth = _Depth(data)
        self.orig_shape = orig_shape


class _Cv2:
    def __init__(self):
        self.images = []

    def imwrite(self, path, image):
        self.images.append(image.copy())
        with open(path, "wb") as handle:
            handle.write(b"preview")
        return True


class DepthOpsTests(unittest.TestCase):
    def test_sample_depth_map_uses_image_xy_without_transposing(self):
        depth = np.asarray(
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float32
        )

        sample = sample_depth_map(
            depth, x=2.8, y=1.2, numpy_module=np
        )

        self.assertEqual((sample["x"], sample["y"]), (2, 1))
        self.assertAlmostEqual(sample["depth"], 1.3, places=6)
        self.assertTrue(sample["valid"])

    def test_sample_depth_map_rejects_out_of_bounds_and_marks_invalid(self):
        depth = np.asarray([[0.0, 2.0]], dtype=np.float32)

        invalid = sample_depth_map(
            depth, x=0, y=0, numpy_module=np
        )

        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["depth"])
        with self.assertRaisesRegex(DepthMapError, "outside"):
            sample_depth_map(depth, x=2, y=0, numpy_module=np)

    def test_keypoint_depth_label_appends_aligned_metric_value(self):
        depth = np.asarray([[0.25, 0.5], [1.25, 1.5]], dtype=np.float32)

        label = keypoint_depth_label(
            "nose", depth, x=1.9, y=0.8, numpy_module=np
        )

        self.assertEqual(label, "nose · 0.500 m")

    def test_keypoint_depth_label_reports_invalid_sample(self):
        depth = np.asarray([[0.0]], dtype=np.float32)

        label = keypoint_depth_label(
            "tail", depth, x=0, y=0, numpy_module=np
        )

        self.assertEqual(label, "tail · invalid")

    def test_depth_array_is_float32_and_invalid_pixels_become_zero(self):
        result = _Result([[1.0, float("nan")], [-2.0, 4.0]])

        depth = depth_array_from_result(result, numpy_module=np)

        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_array_equal(depth, [[1.0, 0.0], [0.0, 4.0]])

    def test_depth_array_rejects_missing_or_empty_depth(self):
        with self.assertRaises(DepthMapError):
            depth_array_from_result(object(), numpy_module=np)
        with self.assertRaises(DepthMapError):
            depth_array_from_result(_Result([[0.0, -1.0]]), numpy_module=np)

    def test_serialization_rejects_transposed_source_alignment(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(DepthMapError, "not aligned"):
                serialize_depth_prediction_result(
                    _Result([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], orig_shape=(3, 2)),
                    map_path=os.path.join(tmp, "depth.npy"),
                    preview_path=os.path.join(tmp, "depth.png"),
                    metadata_path=os.path.join(tmp, "depth.json"),
                    model_path="yolo26n-depth.pt",
                    image_path="frame.png",
                    numpy_module=np,
                    cv2_module=_Cv2(),
                )

    def test_colorized_depth_is_rgb_and_keeps_invalid_pixels_black(self):
        depth = np.asarray([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)

        preview = colorize_depth_map(depth, numpy_module=np)

        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertEqual(preview.dtype, np.uint8)
        np.testing.assert_array_equal(preview[0, 0], [0, 0, 0])
        self.assertGreater(int(preview[0, 1].sum()), 0)

    def test_colorized_depth_preserves_orientation_and_makes_near_bright(self):
        depth = np.asarray(
            [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype=np.float32
        )

        preview = colorize_depth_map(depth, numpy_module=np)

        self.assertEqual(preview.shape, (2, 3, 3))
        self.assertGreater(
            int(preview[0, 0].sum()), int(preview[1, 2].sum())
        )

    def test_serialization_writes_raw_preview_and_metadata(self):
        with TemporaryDirectory() as tmp:
            map_path = os.path.join(tmp, "depth.npy")
            preview_path = os.path.join(tmp, "depth.png")
            metadata_path = os.path.join(tmp, "depth.json")
            cv2 = _Cv2()

            payload = serialize_depth_prediction_result(
                _Result([[1.0, 2.0], [3.0, 4.0]], orig_shape=(2, 2)),
                map_path=map_path,
                preview_path=preview_path,
                metadata_path=metadata_path,
                model_path="yolo26n-depth.pt",
                image_path="frame.png",
                numpy_module=np,
                cv2_module=cv2,
            )

            np.testing.assert_array_equal(
                np.load(map_path, allow_pickle=False),
                [[1.0, 2.0], [3.0, 4.0]],
            )
            self.assertTrue(os.path.isfile(preview_path))
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["units"], "estimated_meters")
            self.assertEqual(metadata["valid_pixels"], 4)
            self.assertTrue(metadata["aligned_to_source"])
            self.assertEqual((metadata["height"], metadata["width"]), (2, 2))
            self.assertEqual(cv2.images[0].shape, (2, 2, 3))
            self.assertEqual(payload["depth_map_path"], map_path)
            self.assertNotIn("depth_map", payload)


if __name__ == "__main__":
    unittest.main()
