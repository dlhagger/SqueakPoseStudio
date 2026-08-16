import os
import tempfile
import unittest

from squeakpose.annotation.documents import SegmentationAnnotationDocument
from squeakpose.annotation.segmentation_assistant import (
    SamPromptRequest,
    discover_sam_weight_candidates,
    select_existing_sam_weight,
    select_sam_contour,
)
from squeakpose.ui.segmentation_controller import SegmentationAnnotationController


class _Values:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class _Masks:
    def __init__(self, contours):
        self.xy = contours


class _Boxes:
    def __init__(self, scores):
        self.conf = _Values(scores)


class _Result:
    def __init__(self, contours, scores):
        self.masks = _Masks(contours)
        self.boxes = _Boxes(scores)


class SegmentationAssistantServiceTests(unittest.TestCase):
    def test_weight_discovery_prioritizes_default_then_prefix_then_other(self):
        with tempfile.TemporaryDirectory() as root:
            names = [
                "project-sam3-last.pth",
                "sam3-z.pt",
                "sam3.pt",
                "sam2.pt",
                "sam3-not-a-weight.txt",
            ]
            for name in names:
                open(os.path.join(root, name), "wb").close()

            candidates = discover_sam_weight_candidates(
                root,
                default_filename="sam3.pt",
            )

            self.assertEqual(
                [os.path.basename(path) for path in candidates],
                ["sam3.pt", "sam3-z.pt", "project-sam3-last.pth"],
            )
            self.assertEqual(
                select_existing_sam_weight(["missing.pt", *candidates]),
                candidates[0],
            )

    def test_request_kwargs_and_best_result_contour_are_model_agnostic(self):
        request = SamPromptRequest(
            source="frame.png",
            class_id=4,
            prompts=((1.5, 2.5, 1), (8.0, 9.0, 0)),
        )
        self.assertEqual(
            request.predict_kwargs(),
            {
                "source": "frame.png",
                "points": [[1.5, 2.5], [8.0, 9.0]],
                "labels": [1, 0],
                "verbose": False,
            },
        )
        result = select_sam_contour(
            [
                _Result(
                    [
                        [(0, 0), (1, 0), (0, 1)],
                        [(2, 2), (7, 2), (4, 8)],
                    ],
                    [0.2, 0.91],
                )
            ]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.points, ((2.0, 2.0), (7.0, 2.0), (4.0, 8.0)))
        self.assertEqual(result.score, 0.91)

    def test_controller_builds_request_and_applies_selected_preview(self):
        controller = SegmentationAnnotationController(SegmentationAnnotationDocument())
        controller.select_target(2)
        controller.add_prompt(3, 4)

        request = controller.build_prompt_request("frame.jpg")
        applied = controller.apply_prediction_results([_Result([[(1, 1), (8, 1), (4, 7)]], [0.75])])

        self.assertEqual(request.class_id, 2)
        self.assertEqual(request.labels, [1])
        self.assertIsNotNone(applied)
        self.assertEqual(controller.state.preview_score, 0.75)
        self.assertTrue(controller.state.has_preview)

    def test_unusable_model_output_does_not_replace_preview(self):
        controller = SegmentationAnnotationController(SegmentationAnnotationDocument())
        controller.select_target(0)
        controller.set_preview([(0, 0), (2, 0), (1, 1)], score=0.4)

        self.assertIsNone(controller.apply_prediction_results([]))
        self.assertEqual(controller.state.preview_score, 0.4)


if __name__ == "__main__":
    unittest.main()
