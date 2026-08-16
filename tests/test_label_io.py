import os
import unittest
from tempfile import TemporaryDirectory

import label_io
import squeakpose.annotation as annotation_package
import squeakpose.annotation.serialization as serialization
from squeakpose.annotation.serialization import (
    load_pose_annotations_from_file,
    load_segmentation_annotations_from_file,
    parse_pose_label_line,
    parse_segmentation_label_line,
    pose_annotation_to_line,
    segmentation_annotation_to_line,
)


class LabelIoTests(unittest.TestCase):
    def test_legacy_and_package_exports_preserve_function_identity(self):
        public_names = (
            "load_pose_annotations_from_file",
            "load_segmentation_annotations_from_file",
            "parse_pose_label_line",
            "parse_segmentation_label_line",
            "pose_annotation_to_line",
            "segmentation_annotation_to_line",
        )

        for name in public_names:
            package_function = getattr(annotation_package, name)
            self.assertIs(getattr(label_io, name), package_function)
            self.assertEqual(package_function.__module__, "squeakpose.annotation.serialization")

        self.assertFalse(hasattr(label_io, "parse_yolo_pose_label_line"))
        self.assertFalse(hasattr(serialization, "parse_yolo_pose_label_line"))

    def test_pose_annotation_round_trips_through_canonical_keypoint_order(self):
        canonical_names = ["nose", "head", "tail"]
        class_lookup = [["nose", "tail"]]
        entry = {
            "class_id": 0,
            "bbox": {"x": 20.0, "y": 10.0, "w": 60.0, "h": 20.0},
            "keypoints": [
                {"idx": 0, "canon_idx": 0, "name": "nose", "x": 40.0, "y": 50.0, "vis": 2},
                {"idx": 1, "canon_idx": 2, "name": "tail", "x": 160.0, "y": 80.0, "vis": 1},
            ],
        }

        line = pose_annotation_to_line(
            entry,
            kp_names=canonical_names,
            img_w=200,
            img_h=100,
        )

        self.assertEqual(
            line,
            "0 0.250000 0.200000 0.300000 0.200000 "
            "0.200000 0.500000 2 0.000000 0.000000 0 0.800000 0.800000 1",
        )
        parsed, had_extra = parse_pose_label_line(
            line,
            classes_count=1,
            canonical_names=canonical_names,
            class_keypoint_lookup=class_lookup,
            img_w=200,
            img_h=100,
        )

        self.assertFalse(had_extra)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["class_id"], 0)
        self.assertEqual(
            [(kp["idx"], kp["canon_idx"], kp["name"], kp["vis"]) for kp in parsed["keypoints"]],
            [(0, 0, "nose", 2), (1, 2, "tail", 1)],
        )
        self.assertAlmostEqual(parsed["bbox"]["x"], 20.0)
        self.assertAlmostEqual(parsed["bbox"]["h"], 20.0)

    def test_load_pose_annotations_keeps_last_row_per_class_and_counts_extra_keypoints(self):
        canonical_names = ["nose"]
        class_lookup = [["nose"]]
        with TemporaryDirectory() as tmp:
            label_file = os.path.join(tmp, "frame001.txt")
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.8 0.9 1\n")
                f.write("0 0.4 0.3 0.2 0.1 0.2 0.3 1\n")

            cache, extra_rows = load_pose_annotations_from_file(
                label_file,
                classes_count=1,
                canonical_names=canonical_names,
                class_keypoint_lookup=class_lookup,
                img_w=100,
                img_h=200,
            )

        self.assertEqual(extra_rows, 1)
        self.assertEqual(sorted(cache.keys()), [0])
        self.assertAlmostEqual(cache[0]["bbox"]["x"], 30.0)
        self.assertAlmostEqual(cache[0]["bbox"]["y"], 50.0)
        self.assertEqual(cache[0]["keypoints"][0]["vis"], 1)

    def test_segmentation_annotation_round_trips(self):
        entry = {
            "class_id": 1,
            "segments": [(10.0, 5.0), (50.0, 5.0), (50.0, 25.0), (10.0, 25.0)],
        }

        line = segmentation_annotation_to_line(entry, img_w=100, img_h=50)
        parsed = parse_segmentation_label_line(
            line,
            classes_count=2,
            img_w=100,
            img_h=50,
        )

        self.assertEqual(
            line, "1 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000 0.100000 0.500000"
        )
        self.assertEqual(parsed, entry)

    def test_load_segmentation_annotations_keeps_last_row_per_class(self):
        with TemporaryDirectory() as tmp:
            label_file = os.path.join(tmp, "frame001.txt")
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
                f.write("2 0.1 0.1 0.5 0.1 0.5 0.5\n")
                f.write("0 0.2 0.2 0.6 0.2 0.6 0.6\n")

            cache = load_segmentation_annotations_from_file(
                label_file,
                classes_count=1,
                img_w=100,
                img_h=50,
            )

        self.assertEqual(sorted(cache.keys()), [0])
        self.assertEqual(cache[0]["segments"], [(20.0, 10.0), (60.0, 10.0), (60.0, 30.0)])


if __name__ == "__main__":
    unittest.main()
