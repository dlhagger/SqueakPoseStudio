import unittest

from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    SegmentationAnnotationDocument,
)


class AnnotationDocumentTests(unittest.TestCase):
    def test_pose_document_copies_entries_and_checks_required_keypoints(self):
        source = {
            0: {
                "bbox": {"x": 0, "y": 0, "w": 10, "h": 5},
                "keypoints": [{"name": "nose"}, {"name": "tail"}],
            }
        }
        document = PoseAnnotationDocument(source)
        source[0]["bbox"]["w"] = 0

        self.assertTrue(document.is_complete(0, required_keypoints=["nose", "tail"]))
        self.assertFalse(document.is_complete(0, required_keypoints=["nose", "ear"]))
        self.assertEqual(document[0]["class_id"], 0)

    def test_segmentation_document_requires_three_polygon_points(self):
        document = SegmentationAnnotationDocument({2: {"segments": [(0, 0), (1, 0)]}})
        self.assertFalse(document.is_complete(2))

        document[2] = {"segments": [(0, 0), (1, 0), (0, 1)]}

        self.assertTrue(document.is_complete(2))

    def test_snapshot_does_not_expose_mutable_internal_state(self):
        document = PoseAnnotationDocument({0: {"bbox": {"w": 2, "h": 2}, "keypoints": []}})

        snapshot = document.snapshot()
        snapshot[0]["bbox"]["w"] = 99

        self.assertEqual(document[0]["bbox"]["w"], 2)


if __name__ == "__main__":
    unittest.main()
