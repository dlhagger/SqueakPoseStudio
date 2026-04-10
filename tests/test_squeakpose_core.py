import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose_core import (
    InferenceCsvWriter,
    build_segmentation_inference_rows,
    effective_prediction_batch,
    find_duplicate_names,
    parse_yolo_pose_label_line,
    resolve_default_training_dataset_path,
)


class _FakeWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


class CoreHelpersTests(unittest.TestCase):
    def test_find_duplicate_names_preserves_first_seen_order(self):
        dupes = find_duplicate_names(["nose", "tail", "nose", "ear", "tail", "tail"])
        self.assertEqual(dupes, ["nose", "tail"])

    def test_effective_prediction_batch_auto_and_explicit(self):
        self.assertEqual(effective_prediction_batch(12, "cpu"), 12)
        self.assertEqual(effective_prediction_batch(-1, "cuda"), 8)
        self.assertEqual(effective_prediction_batch(0, "mps"), 8)
        self.assertEqual(effective_prediction_batch(-1, "cpu"), 1)

    def test_parse_label_line_ignores_extra_keypoints_without_schema_mutation(self):
        canonical = ["nose", "tail"]
        line = "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.8 0.7 1 0.9 0.1 2"
        entry, had_extra = parse_yolo_pose_label_line(
            line,
            classes_count=1,
            canonical_names=canonical,
            class_keypoint_lookup=[["nose", "tail"]],
            img_w=100,
            img_h=200,
        )

        self.assertTrue(had_extra)
        self.assertIsNotNone(entry)
        self.assertEqual(canonical, ["nose", "tail"])
        self.assertEqual(len(entry["keypoints"]), 2)
        self.assertEqual(entry["keypoints"][0]["name"], "nose")
        self.assertEqual(entry["keypoints"][1]["name"], "tail")

    def test_parse_label_line_filters_to_class_keypoint_order(self):
        line = "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.8 0.7 1"
        entry, had_extra = parse_yolo_pose_label_line(
            line,
            classes_count=1,
            canonical_names=["nose", "tail"],
            class_keypoint_lookup=[["tail"]],
            img_w=100,
            img_h=100,
        )

        self.assertFalse(had_extra)
        self.assertIsNotNone(entry)
        self.assertEqual(len(entry["keypoints"]), 1)
        self.assertEqual(entry["keypoints"][0]["name"], "tail")
        self.assertEqual(entry["keypoints"][0]["idx"], 0)

    def test_inference_csv_writer_streams_incrementally(self):
        fake = _FakeWriter()
        stream = InferenceCsvWriter(fake)

        stream.write_row({"frame_index": 0, "confidence": 0.8})
        self.assertEqual(stream.rows_written, 1)
        self.assertEqual(len(fake.rows), 1)

        stream.write_row({"frame_index": 1, "confidence": 0.9})
        self.assertEqual(stream.rows_written, 2)
        self.assertEqual(len(fake.rows), 2)

    def test_build_segmentation_inference_rows_formats_pickle_schema(self):
        rows = build_segmentation_inference_rows(
            frame_index=7,
            detections=[
                {
                    "class_id": 1,
                    "conf": 0.91,
                    "box": [1, 2, 30, 40],
                    "mask_polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "binary_mask": [[1, 0], [0, 1]],
                }
            ],
            class_names={1: "mouse"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], 7)
        self.assertEqual(rows[0]["det"], 0)
        self.assertEqual(rows[0]["class_id"], 1)
        self.assertEqual(rows[0]["class_name"], "mouse")
        self.assertAlmostEqual(rows[0]["conf"], 0.91, places=5)
        self.assertEqual(rows[0]["x1"], 1.0)
        self.assertEqual(rows[0]["y2"], 40.0)
        self.assertEqual(rows[0]["mask_polygon"][2], [1.0, 1.0])
        self.assertEqual(rows[0]["binary_mask"], [[1, 0], [0, 1]])

    def test_resolve_default_training_dataset_path_prefers_pose_then_segment_then_detect(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = root / "datasets"
            pose = datasets / "pose"
            segment = datasets / "segment"
            detect = datasets / "detect"
            pose.mkdir(parents=True, exist_ok=True)
            segment.mkdir(parents=True, exist_ok=True)
            detect.mkdir(parents=True, exist_ok=True)

            (detect / "dataset.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(detect),
            )

            (segment / "dataset.yaml").write_text("task: segment\nnames: [mouse]\n", encoding="utf-8")
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(segment),
            )

            (pose / "dataset.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(pose),
            )

    def test_resolve_default_training_dataset_path_falls_back_to_datasets_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "datasets"
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(expected),
            )


if __name__ == "__main__":
    unittest.main()
