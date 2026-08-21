import math
import unittest

from squeakpose.services.analysis_state import (
    AnalysisAnnotationState,
    AnalysisROI,
    FrameDimensions,
)


class AnalysisAnnotationStateTests(unittest.TestCase):
    def test_two_point_scale_and_real_world_conversion(self):
        state = AnalysisAnnotationState(real_world_distance_mm=25)

        points = state.set_scale_points([(10, 20), (40, 60), (100, 100)])

        self.assertEqual(points, ((10.0, 20.0), (40.0, 60.0)))
        self.assertEqual(state.pixel_distance, 50.0)
        self.assertEqual(state.mm_per_pixel, 0.5)
        state.set_real_world_distance(10)
        self.assertEqual(state.mm_per_pixel, 0.2)

    def test_incomplete_and_cleared_scale_have_no_conversion(self):
        state = AnalysisAnnotationState()
        state.set_scale_points([(0, 0), (3, 4)])
        self.assertEqual(state.pixel_distance, 5.0)

        state.set_scale_points([(8, 9)])
        self.assertEqual(state.pixel_distance, 0.0)
        self.assertIsNone(state.mm_per_pixel)
        state.clear_scale()
        self.assertEqual(state.scale_points, ())

    def test_roi_is_named_normalized_and_clamped_to_frame(self):
        state = AnalysisAnnotationState(frame_width=100, frame_height=80)

        roi = state.add_roi({"type": "rect", "x1": 120, "y1": 70, "x2": -5, "y2": 90})

        self.assertEqual(
            roi,
            AnalysisROI(name="ROI 1", x1=0.0, y1=70.0, x2=100.0, y2=80.0),
        )
        self.assertEqual(roi.width, 100.0)
        self.assertEqual(roi.height, 10.0)
        named = state.add_roi({"x1": 20, "y1": 30, "x2": 10, "y2": 5}, name=" Nest ")
        self.assertEqual(named.name, "Nest")
        self.assertEqual((named.x1, named.y1, named.x2, named.y2), (10, 5, 20, 30))

    def test_roi_replace_export_and_delete_are_detached(self):
        source = [{"name": "Center", "x1": 1, "y1": 2, "x2": 3, "y2": 4}]
        state = AnalysisAnnotationState()
        state.replace_rois(source)
        source[0]["name"] = "Changed"
        exported = state.worker_rois()
        exported[0]["name"] = "Also changed"

        self.assertEqual(state.rois[0].name, "Center")
        self.assertFalse(state.delete_roi(-1))
        self.assertFalse(state.delete_roi(2))
        self.assertTrue(state.delete_roi(0))
        self.assertEqual(state.worker_rois(), [])

    def test_polygon_roi_is_clamped_exported_and_renamed(self):
        state = AnalysisAnnotationState(frame_width=100, frame_height=80)

        roi = state.add_roi(
            {
                "type": "polygon",
                "points": [[-5, 10], [60, -4], [120, 70], [20, 90]],
            }
        )

        self.assertEqual(roi.type, "polygon")
        self.assertEqual(roi.points, ((0.0, 10.0), (60.0, 0.0), (100.0, 70.0), (20.0, 80.0)))
        self.assertGreater(roi.area, 0)
        self.assertTrue(state.rename_roi(0, "Nest"))
        exported = state.worker_rois()
        self.assertEqual(exported[0]["name"], "Nest")
        self.assertEqual(exported[0]["type"], "polygon")
        exported[0]["points"][0][0] = 999
        self.assertEqual(state.rois[0].points[0], (0.0, 10.0))

    def test_polygon_roi_rejects_degenerate_vertices(self):
        state = AnalysisAnnotationState()
        with self.assertRaisesRegex(ValueError, "three unique"):
            state.add_roi({"type": "polygon", "points": [[0, 0], [1, 1], [0, 0]]})
        with self.assertRaisesRegex(ValueError, "enclose an area"):
            state.add_roi({"type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]})

    def test_roi_priority_can_be_reordered_independently_of_creation(self):
        state = AnalysisAnnotationState()
        shape = {"type": "polygon", "points": [[0, 0], [10, 0], [0, 10]]}
        state.add_roi(shape, name="First")
        state.add_roi(shape, name="Second")
        state.add_roi(shape, name="Third")

        self.assertEqual(state.move_roi(2, -1), 1)
        self.assertEqual(state.move_roi(1, -1), 0)
        self.assertEqual(
            [roi["name"] for roi in state.worker_rois()],
            ["Third", "First", "Second"],
        )
        self.assertEqual(state.move_roi(0, -1), 0)
        self.assertEqual(state.move_roi(99, 1), -1)

    def test_snapshot_restore_captures_all_annotation_state(self):
        state = AnalysisAnnotationState(
            frame_width=640,
            frame_height=480,
            real_world_distance_mm=12,
        )
        state.set_scale_points([(1, 2), (4, 6)])
        state.add_roi({"name": "Arena", "x1": 10, "y1": 20, "x2": 30, "y2": 40})
        snapshot = state.snapshot()

        state.clear()
        state.set_frame_dimensions(10, 20)
        state.set_real_world_distance(99)
        state.restore(snapshot)

        self.assertEqual(state.frame, FrameDimensions(640, 480))
        self.assertEqual(state.scale_points, ((1.0, 2.0), (4.0, 6.0)))
        self.assertEqual(state.pixel_distance, 5.0)
        self.assertTrue(math.isclose(state.mm_per_pixel or 0.0, 2.4))
        self.assertEqual(
            state.worker_rois(),
            [
                {
                    "type": "rect",
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 30.0,
                    "y2": 40.0,
                    "name": "Arena",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
