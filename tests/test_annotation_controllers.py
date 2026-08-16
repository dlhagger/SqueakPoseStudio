import unittest

from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    SegmentationAnnotationDocument,
)
from squeakpose.annotation.models import BoundingBox
from squeakpose.ui.depth_controller import (
    DepthAssistantController,
    DepthControllerCallbacks,
)
from squeakpose.ui.pose_controller import (
    PoseAnnotationController,
    PoseControllerCallbacks,
)
from squeakpose.ui.segmentation_controller import (
    SegmentationAnnotationController,
    SegmentationControllerCallbacks,
    SegmentationPromptResult,
)


class PoseAnnotationControllerTests(unittest.TestCase):
    def test_edit_commit_callbacks_and_undo_stay_in_controller_boundary(self):
        document = PoseAnnotationDocument()
        state_events = []
        document_events = []
        controller = PoseAnnotationController(
            document,
            keypoint_order_for=lambda _class_id: ("nose", "tail"),
            canonical_names=("tail", "nose"),
            callbacks=PoseControllerCallbacks(state_events.append, document_events.append),
        )

        controller.select_class(2)
        controller.set_box(BoundingBox(1, 2, 20, 10, 99))
        controller.add_next_keypoint(4, 5)
        self.assertNotIn(2, document)
        controller.mark_next_invisible()

        self.assertTrue(document.is_complete(2, required_keypoints=["nose", "tail"]))
        self.assertEqual(document[2]["bbox"]["w"], 20.0)
        self.assertEqual(document[2]["keypoints"][1]["canon_idx"], 0)
        self.assertGreaterEqual(len(state_events), 4)
        self.assertGreaterEqual(len(document_events), 3)

        self.assertTrue(controller.undo())
        self.assertNotIn(2, document)
        self.assertEqual(controller.state.next_keypoint_name, "tail")

    def test_segmentation_box_replacement_preserves_keypoints_and_is_undoable(self):
        document = PoseAnnotationDocument()
        controller = PoseAnnotationController(
            document,
            keypoint_order_for=lambda _class_id: ("nose",),
            canonical_names=("nose",),
        )
        controller.select_class(0)
        controller.set_box(BoundingBox(1, 2, 20, 10, 0))
        controller.add_next_keypoint(4, 5)

        controller.replace_box_preserving_keypoints(BoundingBox(3, 4, 30, 15, 99))

        self.assertEqual(controller.state.box, BoundingBox(3, 4, 30, 15, 0))
        self.assertEqual(controller.state.keypoints["nose"].kp.x, 4)
        annotation = document.annotation(0)
        self.assertIsNotNone(annotation)
        self.assertEqual(annotation.box, (3.0, 4.0, 30.0, 15.0))
        self.assertTrue(controller.undo())
        self.assertEqual(controller.state.box, BoundingBox(1, 2, 20, 10, 0))
        self.assertIn("nose", controller.state.keypoints)

    def test_class_switch_and_document_replacement_do_not_leak_edits(self):
        document = PoseAnnotationDocument()
        controller = PoseAnnotationController(
            document,
            keypoint_order_for=lambda class_id: ("nose",) if class_id == 0 else (),
            canonical_names=("nose",),
        )
        controller.select_class(0)
        controller.set_box(BoundingBox(0, 0, 10, 10, 0))
        controller.add_next_keypoint(3, 4)

        controller.select_class(1)
        self.assertIsNone(controller.state.box)
        controller.replace_document({})

        self.assertEqual(document.snapshot(), {})
        self.assertEqual(controller.state.active_class_id, 1)
        self.assertIsNone(controller.state.box)
        self.assertFalse(controller.state.can_undo)


class SegmentationAnnotationControllerTests(unittest.TestCase):
    def test_prompt_predict_accept_and_undo_use_explicit_predictor(self):
        requests = []
        state_events = []
        document_events = []

        def predict(request):
            requests.append(request)
            return SegmentationPromptResult(((1.0, 1.0), (8.0, 1.0), (4.0, 7.0)), 0.85)

        document = SegmentationAnnotationDocument()
        controller = SegmentationAnnotationController(
            document,
            predict=predict,
            callbacks=SegmentationControllerCallbacks(
                state_events.append,
                document_events.append,
            ),
        )
        controller.select_target(3)
        controller.add_prompt(2, 4, positive=True)
        controller.add_prompt(7, 6, positive=False)

        result = controller.request_preview()
        accepted = controller.accept_preview()

        self.assertEqual(requests[0].class_id, 3)
        self.assertEqual(requests[0].prompts, ((2.0, 4.0, 1), (7.0, 6.0, 0)))
        self.assertEqual(result.score, 0.85)
        self.assertEqual(accepted["score"], 0.85)
        self.assertTrue(document.is_complete(3))
        self.assertEqual(controller.state.prompt_points, [])
        self.assertTrue(state_events)
        self.assertTrue(document_events)

        self.assertTrue(controller.undo())
        self.assertNotIn(3, document)
        self.assertTrue(controller.state.has_preview)

    def test_polygon_replacement_removal_and_undo_are_atomic(self):
        document = SegmentationAnnotationDocument()
        controller = SegmentationAnnotationController(document)
        controller.select_target(0)

        with self.assertRaisesRegex(ValueError, "at least three"):
            controller.upsert_polygon(0, [(0, 0), (1, 1)])

        controller.upsert_polygon(0, [(0, 0), (4, 0), (0, 4)])
        self.assertTrue(controller.remove_mask())
        self.assertNotIn(0, document)
        self.assertTrue(controller.undo())
        self.assertIn(0, document)

    def test_predict_requires_narrow_preconditions(self):
        controller = SegmentationAnnotationController(SegmentationAnnotationDocument())
        with self.assertRaisesRegex(RuntimeError, "predictor"):
            controller.request_preview()


class DepthAssistantControllerTests(unittest.TestCase):
    def test_image_probe_view_and_clear_emit_detached_state(self):
        events = []

        def sample(depth_map, *, x, y):
            px, py = int(x), int(y)
            if px < 0 or py < 0 or py >= len(depth_map) or px >= len(depth_map[0]):
                raise ValueError("pixel outside map")
            value = depth_map[py][px]
            return {"x": px, "y": py, "depth": value, "valid": value > 0}

        controller = DepthAssistantController(
            sampler=sample,
            callbacks=DepthControllerCallbacks(events.append),
        )
        controller.load_image(
            "frame.png",
            depth_map=[[1.0, 2.0], [3.0, 4.0]],
            metadata={"p02_depth": 1, "p98_depth": 4, "median_depth": 2.5},
        )
        self.assertEqual(controller.set_view_mode("overlay"), "overlay")

        attempt = controller.probe(1, 0)
        self.assertTrue(attempt.accepted)
        self.assertEqual(attempt.probe.depth, 2.0)
        self.assertIn("2.000 m", controller.state.probe_text())
        self.assertFalse(controller.probe(9, 9).accepted)
        self.assertEqual(controller.probe(9, 9).error, "pixel outside map")

        self.assertTrue(controller.clear_probes())
        self.assertFalse(controller.clear_probes())
        self.assertEqual(events[-1].probes, [])

    def test_image_change_drops_map_and_missing_map_returns_error(self):
        controller = DepthAssistantController(
            sampler=lambda _depth_map, *, x, y: {
                "x": x,
                "y": y,
                "depth": 1,
                "valid": True,
            }
        )
        controller.load_image("one.png", depth_map=object())
        self.assertTrue(controller.probe(0, 0).accepted)

        controller.load_image("two.png", probe_error="No raw map for this frame.")

        self.assertIsNone(controller.depth_map)
        attempt = controller.probe(0, 0)
        self.assertFalse(attempt.accepted)
        self.assertEqual(attempt.error, "No raw map for this frame.")


if __name__ == "__main__":
    unittest.main()
