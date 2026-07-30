import unittest

from squeakpose.annotation.models import (
    Annotation,
    BoundingBox,
    Keypoint,
    KeypointEntry,
)


class AnnotationModelTests(unittest.TestCase):
    def test_bounding_box_converts_to_yolo_coordinates(self):
        box = BoundingBox(x=10, y=20, w=40, h=20, class_id=2)

        self.assertEqual(box.to_yolo(100, 100), (2, 0.3, 0.3, 0.4, 0.2))

    def test_keypoint_converts_to_yolo_coordinates(self):
        point = Keypoint(x=25, y=75, class_id=1, name="nose")

        self.assertEqual(point.to_yolo(100, 100), (1, 0.25, 0.75, "nose"))

    def test_conversion_rejects_invalid_image_dimensions(self):
        box = BoundingBox(x=0, y=0, w=1, h=1, class_id=0)
        point = Keypoint(x=0, y=0, class_id=0, name="nose")

        with self.assertRaises(ValueError):
            box.to_yolo(0, 100)
        with self.assertRaises(ValueError):
            point.to_yolo(100, 0)

    def test_annotation_groups_keypoints_in_declared_order(self):
        point = Keypoint(x=1, y=2, class_id=0, name="nose")
        entry = KeypointEntry("nose", "Nose", point, 2)
        annotation = Annotation(
            ann_id=3,
            bbox=BoundingBox(0, 0, 10, 10, 0),
            keypoints={"nose": entry},
            order=["nose"],
        )

        self.assertIs(annotation.keypoints["nose"], entry)
        self.assertEqual(annotation.order, ["nose"])


if __name__ == "__main__":
    unittest.main()
