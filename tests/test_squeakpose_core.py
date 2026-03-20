import unittest

from squeakpose_core import (
    InferenceCsvWriter,
    effective_prediction_batch,
    find_duplicate_names,
    parse_yolo_pose_label_line,
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


if __name__ == "__main__":
    unittest.main()

