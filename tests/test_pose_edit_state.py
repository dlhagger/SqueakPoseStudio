import unittest

from squeakpose.annotation.models import BoundingBox
from squeakpose.annotation.pose import PoseEditState


class PoseEditStateTests(unittest.TestCase):
    def make_state(self) -> PoseEditState:
        state = PoseEditState()
        state.select_class(
            2,
            ["nose", "tail"],
            canonical_names=["tail", "nose", "ear"],
        )
        return state

    def test_select_class_sets_order_and_starts_empty(self):
        state = self.make_state()

        self.assertEqual(state.active_class_id, 2)
        self.assertEqual(state.keypoint_order, ["nose", "tail"])
        self.assertEqual(state.next_keypoint_name, "nose")
        self.assertEqual(state.current_keypoint_index, 0)
        self.assertFalse(state.is_complete)

    def test_set_box_normalizes_class_and_clears_keypoints(self):
        state = self.make_state()
        state.box = BoundingBox(1, 2, 3, 4, 2)
        state.add_next_keypoint(5, 6)

        box = state.set_box(BoundingBox(10, 20, 30, 40, 99))

        self.assertEqual(box, BoundingBox(10.0, 20.0, 30.0, 40.0, 2))
        self.assertEqual(state.keypoints, {})

    def test_add_keypoints_follows_declared_order_and_completion(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 20, 10, 2))

        nose = state.add_next_keypoint(4, 5)
        tail = state.add_next_keypoint(15, 6, visibility=1)

        self.assertEqual(nose.name, "nose")
        self.assertEqual(tail.name, "tail")
        self.assertEqual(tail.visibility, 1)
        self.assertEqual(state.current_keypoint_index, 2)
        self.assertTrue(state.is_complete)
        self.assertIsNone(state.add_next_keypoint(1, 1))

    def test_keypoint_requires_a_box(self):
        state = self.make_state()

        self.assertIsNone(state.add_next_keypoint(1, 2))
        self.assertEqual(state.keypoints, {})

    def test_mark_next_invisible_uses_zero_coordinates(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 20, 10, 2))

        entry = state.mark_next_invisible()

        self.assertEqual(entry.visibility, 0)
        self.assertEqual((entry.kp.x, entry.kp.y), (0.0, 0.0))

    def test_visibility_change_preserves_coordinates(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 20, 10, 2))
        state.add_next_keypoint(4, 5)

        self.assertTrue(state.set_visibility("nose", 0))
        self.assertEqual(state.keypoints["nose"].visibility, 0)
        self.assertEqual((state.keypoints["nose"].kp.x, state.keypoints["nose"].kp.y), (4.0, 5.0))
        self.assertFalse(state.set_visibility("ear", 1))

    def test_annotation_entry_matches_cache_shape_and_order(self):
        state = self.make_state()
        state.set_box(BoundingBox(10, 20, 30, 40, 2))
        state.add_next_keypoint(12, 23, visibility=2)
        state.add_next_keypoint(35, 50, visibility=0)

        entry = state.to_annotation_entry()

        self.assertEqual(
            entry,
            {
                "class_id": 2,
                "bbox": {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
                "keypoints": [
                    {
                        "name": "nose",
                        "x": 12.0,
                        "y": 23.0,
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

    def test_incomplete_annotation_is_not_cacheable(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 10, 10, 2))
        state.add_next_keypoint(1, 2)

        self.assertIsNone(state.to_annotation_entry())
        partial = state.to_annotation_entry(require_complete=False)
        self.assertEqual([point["name"] for point in partial["keypoints"]], ["nose"])

    def test_load_annotation_uses_names_then_index_fallback(self):
        state = self.make_state()

        loaded = state.load_annotation(
            {
                "class_id": 99,
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "keypoints": [
                    {"name": "tail", "idx": 0, "x": 8, "y": 9, "vis": 1},
                    {"name": "renamed", "idx": 0, "x": 5, "y": 6, "vis": 2},
                ],
            }
        )

        self.assertTrue(loaded)
        self.assertEqual(state.box.class_id, 2)
        self.assertEqual((state.keypoints["nose"].kp.x, state.keypoints["nose"].kp.y), (5.0, 6.0))
        self.assertEqual((state.keypoints["tail"].kp.x, state.keypoints["tail"].kp.y), (8.0, 9.0))

    def test_select_class_can_restore_cached_entry(self):
        state = PoseEditState()

        state.select_class(
            1,
            ["nose"],
            canonical_names=["nose"],
            entry={
                "class_id": 1,
                "bbox": {"x": 0, "y": 0, "w": 10, "h": 5},
                "keypoints": [{"name": "nose", "idx": 0, "x": 3, "y": 2, "vis": 2}],
            },
        )

        self.assertTrue(state.is_complete)
        self.assertEqual(state.keypoints["nose"].kp.class_id, 1)

    def test_apply_template_scales_box_and_keypoints(self):
        state = self.make_state()

        applied = state.apply_template(
            {
                "bbox": {"xc": 0.5, "yc": 0.5, "w": 0.4, "h": 0.5},
                "keypoints": [
                    {"idx": 0, "x": 0.25, "y": 0.75, "vis": 2},
                    {"idx": 1, "x": 0.9, "y": 0.8, "vis": 0},
                ],
            },
            image_width=100,
            image_height=80,
        )

        self.assertTrue(applied)
        self.assertEqual(state.box, BoundingBox(30.0, 20.0, 40.0, 40.0, 2))
        self.assertEqual((state.keypoints["nose"].kp.x, state.keypoints["nose"].kp.y), (25.0, 60.0))
        self.assertEqual((state.keypoints["tail"].kp.x, state.keypoints["tail"].kp.y), (0.0, 0.0))
        self.assertTrue(state.is_complete)

    def test_template_missing_keypoint_creates_invisible_placeholder(self):
        state = self.make_state()

        state.apply_template({}, image_width=100, image_height=50)

        self.assertEqual(state.box, BoundingBox(0.0, 0.0, 100.0, 50.0, 2))
        self.assertEqual([entry.visibility for entry in state.keypoints.values()], [0, 0])

    def test_template_round_trip_preserves_normalized_shape(self):
        state = self.make_state()
        state.set_box(BoundingBox(20, 10, 40, 20, 2))
        state.add_next_keypoint(30, 15, visibility=2)
        state.mark_next_invisible()

        template = state.to_template("mouse", image_width=100, image_height=50)

        self.assertEqual(
            template,
            {
                "class": "mouse",
                "bbox": {"xc": 0.4, "yc": 0.4, "w": 0.4, "h": 0.4},
                "keypoints": [
                    {
                        "name": "nose",
                        "idx": 0,
                        "canon_idx": 1,
                        "x": 0.3,
                        "y": 0.3,
                        "vis": 2,
                    },
                    {
                        "name": "tail",
                        "idx": 1,
                        "canon_idx": 0,
                        "x": 0.0,
                        "y": 0.0,
                        "vis": 0,
                    },
                ],
            },
        )

    def test_delete_transitions_update_completion(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 10, 10, 2))
        state.add_next_keypoint(1, 1)
        state.add_next_keypoint(2, 2)

        self.assertTrue(state.delete_keypoint("nose"))
        self.assertEqual(state.next_keypoint_name, "nose")
        self.assertFalse(state.delete_keypoint("ear"))
        self.assertTrue(state.delete_box())
        self.assertEqual(state.keypoints, {})
        self.assertFalse(state.delete_box())

    def test_box_only_class_is_complete(self):
        state = PoseEditState(active_class_id=0)

        state.set_box(BoundingBox(0, 0, 5, 5, 0))

        self.assertTrue(state.is_complete)
        self.assertEqual(state.to_annotation_entry()["keypoints"], [])

    def test_snapshot_is_detached_and_undo_restores_state(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 10, 10, 2))
        state.add_next_keypoint(1, 2)
        snapshot = state.push_undo_snapshot()

        state.add_next_keypoint(3, 4)
        snapshot.keypoints["nose"].kp.x = 99

        self.assertTrue(state.can_undo)
        self.assertTrue(state.undo())
        self.assertEqual(state.keypoints["nose"].kp.x, 1.0)
        self.assertNotIn("tail", state.keypoints)
        self.assertFalse(state.undo())

    def test_selecting_new_class_clears_state_and_undo(self):
        state = self.make_state()
        state.set_box(BoundingBox(0, 0, 10, 10, 2))
        state.push_undo_snapshot()

        state.select_class(None)

        self.assertIsNone(state.active_class_id)
        self.assertIsNone(state.box)
        self.assertEqual(state.keypoints, {})
        self.assertFalse(state.can_undo)


if __name__ == "__main__":
    unittest.main()
