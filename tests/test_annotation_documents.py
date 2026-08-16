import unittest

from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    PoseAnnotationValue,
    PoseKeypointValue,
    SegmentationAnnotationDocument,
    SegmentationAnnotationValue,
)
from squeakpose.annotation.models import BoundingBox
from squeakpose.annotation.pose import PoseEditState
from squeakpose.annotation.segmentation import SegmentationEditState


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

    def test_named_serialized_operations_avoid_mapping_mutation(self):
        document = PoseAnnotationDocument()
        source = {"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "keypoints": []}

        stored = document.upsert_entry(2, source)
        source["bbox"]["w"] = 99
        stored["bbox"]["h"] = 99

        self.assertEqual(document.export_entries()[2]["bbox"], {"x": 1, "y": 2, "w": 3, "h": 4})
        self.assertTrue(document.delete_entry(2))
        self.assertFalse(document.delete_entry(2))
        document.load_entries({3: source})
        self.assertEqual(list(document), [3])
        document.replace_entries({4: {"bbox": {"w": 1, "h": 1}, "keypoints": []}})
        self.assertEqual(list(document), [4])

    def test_pose_typed_replace_upsert_delete_and_snapshot_restore(self):
        first = PoseAnnotationValue(
            class_id=2,
            box=(1.0, 2.0, 30.0, 40.0),
            keypoints=(
                PoseKeypointValue("nose", 5.0, 6.0, 2, 0, 1),
                PoseKeypointValue("tail", 0.0, 0.0, 0, 1, 0),
            ),
        )
        replacement = PoseAnnotationValue(class_id=4, box=(0.0, 0.0, 5.0, 6.0))
        document = PoseAnnotationDocument()

        document.replace_annotations([first])
        snapshot = document.typed_snapshot()
        stored = document.upsert_annotation(replacement)

        self.assertEqual(stored, replacement)
        self.assertEqual(document.annotation(2), first)
        self.assertEqual(document.export_annotations(), (first, replacement))
        self.assertTrue(document.delete_annotation(2))
        self.assertIsNone(document.annotation(2))
        document.restore_typed_snapshot(snapshot)
        self.assertEqual(document.export_annotations(), (first,))
        self.assertEqual(
            document.load_annotations({2: first.as_dict()}),
            (first,),
        )
        self.assertEqual(
            document.export_entries()[2],
            {
                "class_id": 2,
                "bbox": {"x": 1.0, "y": 2.0, "w": 30.0, "h": 40.0},
                "keypoints": [
                    {
                        "name": "nose",
                        "x": 5.0,
                        "y": 6.0,
                        "vis": 2,
                        "idx": 0,
                        "canon_idx": 1,
                    },
                    {
                        "name": "tail",
                        "x": 0.0,
                        "y": 0.0,
                        "vis": 0,
                        "idx": 1,
                        "canon_idx": 0,
                    },
                ],
            },
        )

    def test_pose_edit_state_round_trip_uses_typed_document_transition(self):
        state = PoseEditState()
        state.select_class(1, ["nose"], canonical_names=["nose"])
        state.set_box(BoundingBox(10, 20, 30, 40, 1))
        state.add_next_keypoint(15, 25)
        document = PoseAnnotationDocument()

        stored = document.apply_edit_state(state)
        restored = document.to_edit_state(1, ["nose"], canonical_names=["nose"])

        self.assertEqual(stored.class_id, 1)
        self.assertTrue(restored.is_complete)
        self.assertEqual(restored.box, BoundingBox(10.0, 20.0, 30.0, 40.0, 1))
        self.assertEqual(restored.keypoints["nose"].kp.x, 15.0)

    def test_incomplete_pose_edit_is_not_applied_by_default(self):
        state = PoseEditState()
        state.select_class(1, ["nose", "tail"])
        state.set_box(BoundingBox(0, 0, 10, 10, 1))
        state.add_next_keypoint(2, 3)
        document = PoseAnnotationDocument()

        self.assertIsNone(document.apply_edit_state(state))
        self.assertEqual(len(document), 0)
        self.assertIsNotNone(document.apply_edit_state(state, require_complete=False))

    def test_segmentation_typed_operations_and_edit_state_round_trip(self):
        first = SegmentationAnnotationValue(
            class_id=3,
            segments=((0.0, 0.0), (4.0, 0.0), (0.0, 4.0)),
            score=0.75,
        )
        second = SegmentationAnnotationValue(
            class_id=1,
            segments=((1.0, 1.0), (3.0, 1.0), (1.0, 3.0)),
            score=0.5,
        )
        document = SegmentationAnnotationDocument()
        document.replace_annotations([first])
        snapshot = document.typed_snapshot()
        self.assertEqual(document.upsert_annotation(second), second)
        self.assertEqual(document.export_annotations(), (second, first))

        state = document.to_edit_state(selected_target=3)
        self.assertEqual(state.selected_target, 3)
        self.assertEqual(state.accepted_masks[1]["score"], 0.5)
        state.clear_accepted_mask(3)
        document.apply_edit_state(state)
        self.assertEqual(document.export_annotations(), (second,))

        document.restore_typed_snapshot(snapshot)
        self.assertEqual(document.export_annotations(), (first,))
        self.assertEqual(
            document.export_entries()[3],
            {
                "class_id": 3,
                "segments": [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)],
                "score": 0.75,
            },
        )

    def test_segmentation_edit_state_replacement_is_detached(self):
        state = SegmentationEditState(
            accepted_masks={
                2: {
                    "class_id": 2,
                    "segments": [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)],
                    "score": 0.2,
                }
            }
        )
        document = SegmentationAnnotationDocument()

        document.apply_edit_state(state)
        state.accepted_masks[2]["score"] = 0.9

        self.assertEqual(document.annotation(2).score, 0.2)


if __name__ == "__main__":
    unittest.main()
