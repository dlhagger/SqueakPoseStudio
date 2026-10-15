import unittest

from squeakpose.annotation.documents import PoseAnnotationDocument, SegmentationAnnotationDocument
from squeakpose.services.frame_annotations import (
    SegmentationBoxUnavailableError,
    plan_segmentation_box_transfers,
)
from squeakpose.services.segmentation_transfer import (
    _assignment_margin,
    _maximum_assignment,
    create_pose_instances,
    incomplete_pose_instances,
    plan_box_assignments,
    replace_pose_boxes,
    suggest_box_assignments,
)


class SegmentationTransferTests(unittest.TestCase):
    def plans(self):
        masks = SegmentationAnnotationDocument()
        masks.replace_instance_entries(
            [
                {"class_id": 1, "segments": [(50, 2), (60, 2), (60, 12), (50, 12)]},
                {"class_id": 1, "segments": [(5, 2), (15, 2), (15, 12), (5, 12)]},
            ]
        )
        return plan_segmentation_box_transfers(
            pose_class_id=0,
            pose_classes=["mouse", "rat"],
            segmentation_classes=["other", "Mouse"],
            segmentation_document=masks,
        )

    def poses(self):
        poses = PoseAnnotationDocument()
        poses.replace_instance_entries(
            [
                {
                    "class_id": 0,
                    "bbox": {"x": 4, "y": 1, "w": 12, "h": 13},
                    "keypoints": [
                        {"name": "nose", "x": 6.25, "y": 3.5, "vis": 1, "idx": 0, "canon_idx": 0}
                    ],
                },
                {
                    "class_id": 0,
                    "bbox": {"x": 49, "y": 1, "w": 12, "h": 13},
                    "keypoints": [
                        {"name": "nose", "x": 51, "y": 3, "vis": 0, "idx": 0, "canon_idx": 0}
                    ],
                },
                {"class_id": 1, "bbox": {"x": 80, "y": 2, "w": 10, "h": 10}, "keypoints": []},
            ]
        )
        return poses

    def test_create_uses_all_mask_bounds_and_preserves_other_classes_without_mutating_input(self):
        original = self.poses()
        before = original.typed_snapshot()
        result = create_pose_instances(original, self.plans())
        entries = [entry for entry in result.all_entries() if entry["class_id"] == 0]
        self.assertEqual([entry["bbox"]["x"] for entry in entries], [50, 5])
        self.assertTrue(all(entry["keypoints"] == [] for entry in entries))
        self.assertEqual(len(incomplete_pose_instances(result, [["nose"], []])), 2)
        self.assertEqual(result.annotation(1), original.annotation(1))
        self.assertEqual(original.typed_snapshot(), before)

    def test_replace_uses_spatial_suggestions_not_row_order_and_preserves_every_point_field(self):
        original = self.poses()
        before = original.typed_snapshot()
        targets = [entry for entry in original.all_entries() if entry["class_id"] == 0]
        assignments = suggest_box_assignments(targets, self.plans())
        self.assertEqual(assignments, (1, 0))
        result = replace_pose_boxes(original, self.plans(), assignments)
        for old, new in zip(original.all_entries(), result.all_entries()):
            self.assertEqual(new["keypoints"], old["keypoints"])
            self.assertEqual(result.entry_id(new), original.entry_id(old))
        self.assertEqual(result.all_entries()[0]["bbox"], {"x": 5, "y": 2, "w": 10, "h": 10})
        self.assertEqual(result.annotation(1), original.annotation(1))
        self.assertEqual(original.typed_snapshot(), before)

    def test_invalid_counts_and_duplicate_assignments_leave_input_unchanged(self):
        poses = self.poses()
        before = poses.typed_snapshot()
        with self.assertRaisesRegex(SegmentationBoxUnavailableError, "equal counts"):
            suggest_box_assignments([poses.all_entries()[0]], self.plans())
        with self.assertRaises(SegmentationBoxUnavailableError):
            replace_pose_boxes(poses, self.plans(), (0, 0))
        self.assertEqual(poses.typed_snapshot(), before)

    def test_overlapping_identical_boxes_resolve_using_mask_polygons_and_visible_keypoints(self):
        masks = SegmentationAnnotationDocument()
        masks.replace_instance_entries(
            [
                {"class_id": 0, "segments": [(10, 10), (0, 10), (10, 0)]},
                {"class_id": 0, "segments": [(0, 0), (10, 0), (0, 10)]},
            ]
        )
        plans = plan_segmentation_box_transfers(
            pose_class_id=0,
            pose_classes=["mouse"],
            segmentation_classes=["mouse"],
            segmentation_document=masks,
        )
        entries = [
            {
                "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
                "keypoints": [{"x": 2, "y": 2, "vis": 2}, {"x": 8, "y": 8, "vis": 0}],
            },
            {"bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "keypoints": [{"x": 8, "y": 8, "vis": 2}]},
        ]
        plan = plan_box_assignments(entries, plans)
        self.assertEqual(plan.assignments, (1, 0))
        self.assertFalse(plan.requires_review)
        # One object with clear point evidence resolves the other's empty draft.
        entries[1]["keypoints"] = []
        self.assertFalse(plan_box_assignments(entries, plans).requires_review)
        # No distinguishing geometry at all requires review, rather than row-order guessing.
        entries[0]["keypoints"] = []
        self.assertTrue(plan_box_assignments(entries, plans).requires_review)

    def test_global_assignment_and_confidence_agree_with_brute_force(self):
        import itertools
        import random

        random_source = random.Random(7)
        for count in range(2, 6):
            for _case in range(10):
                scores = [[random_source.random() for _ in range(count)] for _ in range(count)]
                candidates = sorted(
                    (sum(scores[row][column] for row, column in enumerate(order)), order)
                    for order in itertools.permutations(range(count))
                )
                chosen = _maximum_assignment(scores)
                self.assertEqual(chosen, candidates[-1][1])
                self.assertAlmostEqual(
                    _assignment_margin(scores, chosen), candidates[-1][0] - candidates[-2][0]
                )
