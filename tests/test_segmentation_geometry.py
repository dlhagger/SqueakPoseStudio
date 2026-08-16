import unittest

import numpy as np

from squeakpose.annotation.segmentation import (
    SegmentationEditState,
    apply_brush_stroke,
    clamp_point_to_image,
    downsample_polygon_points,
    mask_to_polygon,
    normalize_polygon_points,
    polygon_bounds,
    polygon_signed_area,
    polygon_to_mask,
    rotate_polygon_to_anchor,
    segmentation_mask_shape,
)


class _RasterBackend:
    RETR_EXTERNAL = 0
    CHAIN_APPROX_NONE = 1

    def __init__(self, contours=None):
        self.contours = contours
        self.fill_calls = []
        self.circle_calls = []
        self.line_calls = []

    def fillPoly(self, mask, polygons, value):
        self.fill_calls.append((polygons, value))
        polygon = np.asarray(polygons[0]).reshape((-1, 2))
        min_x, min_y = polygon.min(axis=0)
        max_x, max_y = polygon.max(axis=0)
        mask[max(0, min_y) : max_y + 1, max(0, min_x) : max_x + 1] = value

    def findContours(self, mask, _retrieval, _approximation):
        if self.contours is not None:
            return list(self.contours), None
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return [], None
        contour = np.asarray(
            [
                [[xs.min(), ys.min()]],
                [[xs.max(), ys.min()]],
                [[xs.max(), ys.max()]],
                [[xs.min(), ys.max()]],
            ],
            dtype=np.int32,
        )
        return [contour], None

    @staticmethod
    def contourArea(contour):
        points = np.asarray(contour).reshape((-1, 2))
        return abs(polygon_signed_area([(float(x), float(y)) for x, y in points]))

    @staticmethod
    def pointPolygonTest(contour, point, _measure_distance):
        points = np.asarray(contour).reshape((-1, 2))
        x, y = point
        return (
            0
            if points[:, 0].min() <= x <= points[:, 0].max()
            and points[:, 1].min() <= y <= points[:, 1].max()
            else -1
        )

    def circle(self, mask, center, radius, value, *, thickness):
        self.circle_calls.append((center, radius, value, thickness))
        center_x, center_y = center
        yy, xx = np.indices(mask.shape)
        mask[(xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2] = value

    def line(self, mask, start, end, value, *, thickness):
        self.line_calls.append((start, end, value, thickness))
        start_x, start_y = start
        end_x, end_y = end
        yy, xx = np.indices(mask.shape)
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        length_squared = max(1, delta_x**2 + delta_y**2)
        position = np.clip(
            ((xx - start_x) * delta_x + (yy - start_y) * delta_y) / length_squared,
            0.0,
            1.0,
        )
        nearest_x = start_x + position * delta_x
        nearest_y = start_y + position * delta_y
        mask[(xx - nearest_x) ** 2 + (yy - nearest_y) ** 2 <= (thickness / 2) ** 2] = value

    @staticmethod
    def countNonZero(mask):
        return int(np.count_nonzero(mask))


class SegmentationGeometryTests(unittest.TestCase):
    def test_normalize_polygon_points_preserves_order_and_skips_malformed_entries(self):
        points = [(1, "2.5"), None, (3,), {0: 4, 1: 5}, (6, 7, 8)]

        self.assertEqual(
            normalize_polygon_points(points),
            [(1.0, 2.5), (4.0, 5.0), (6.0, 7.0)],
        )

    def test_polygon_bounds_are_tight_and_reject_degenerate_geometry(self):
        self.assertEqual(
            polygon_bounds([(8, 4), (2, 10), (12, 7), (4, 1)]),
            (2.0, 1.0, 10.0, 9.0),
        )
        self.assertIsNone(polygon_bounds([(1, 1), (1, 5), (1, 9)]))
        self.assertIsNone(polygon_bounds([(1, 1), (2, 2)]))

    def test_mask_shape_and_polygon_rasterization_match_image_convention(self):
        backend = _RasterBackend()

        mask = polygon_to_mask(
            [(1.9, 1.2), (4.8, 1.2), (4.8, 3.9), (1.9, 3.9)],
            image_width=6.2,
            image_height=5.4,
            numpy_module=np,
            cv2_module=backend,
        )

        self.assertEqual(segmentation_mask_shape(6.2, 5.4), (5, 6))
        self.assertEqual(mask.shape, (5, 6))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(np.count_nonzero(mask)), 12)
        raster_points = backend.fill_calls[0][0][0].reshape((-1, 2)).tolist()
        self.assertEqual(raster_points, [[1, 1], [4, 1], [4, 3], [1, 3]])

    def test_polygon_rasterization_cleanly_gates_missing_dependencies_and_bad_geometry(self):
        self.assertIsNone(
            polygon_to_mask(
                [(0, 0), (1, 0), (0, 1)],
                image_width=4,
                image_height=4,
                numpy_module=None,
                cv2_module=_RasterBackend(),
            )
        )
        self.assertIsNone(
            polygon_to_mask(
                [(0, 0), (1, 0)],
                image_width=4,
                image_height=4,
                numpy_module=np,
                cv2_module=_RasterBackend(),
            )
        )

    def test_mask_to_polygon_prefers_contour_containing_original_anchor(self):
        anchored = np.asarray(
            [[[1, 1]], [[3, 1]], [[3, 3]], [[1, 3]]],
            dtype=np.int32,
        )
        larger = np.asarray(
            [[[5, 1]], [[9, 1]], [[9, 5]], [[5, 5]]],
            dtype=np.int32,
        )
        backend = _RasterBackend(contours=[larger, anchored])
        anchor = [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]

        points = mask_to_polygon(
            np.ones((8, 12), dtype=np.uint8),
            cv2_module=backend,
            anchor_points=anchor,
        )

        self.assertEqual(points[0], anchor[0])
        self.assertEqual(set(points), set(anchor))
        self.assertLess(polygon_signed_area(points), 0)

    def test_mask_to_polygon_uses_largest_contour_without_matching_anchor(self):
        small = np.asarray(
            [[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]],
            dtype=np.int32,
        )
        large = np.asarray(
            [[[3, 2]], [[7, 2]], [[7, 6]], [[3, 6]]],
            dtype=np.int32,
        )

        points = mask_to_polygon(
            np.ones((8, 8), dtype=np.uint8),
            cv2_module=_RasterBackend(contours=[small, large]),
            anchor_points=[(20, 20)],
        )

        self.assertEqual(set(points), {(3.0, 2.0), (7.0, 2.0), (7.0, 6.0), (3.0, 6.0)})

    def test_add_brush_clamps_edge_and_connects_the_complete_stroke(self):
        backend = _RasterBackend()
        mask = np.zeros((7, 10), dtype=np.uint8)

        result = apply_brush_stroke(
            mask,
            start=(-5.0, 3.0),
            end=(20.0, 3.0),
            radius=1,
            add=True,
            image_width=10,
            image_height=7,
            cv2_module=backend,
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.erased)
        self.assertGreater(result.pixel_count, 0)
        self.assertEqual(backend.circle_calls, [((9, 3), 2, 255, -1)])
        self.assertEqual(backend.line_calls, [((0, 3), (9, 3), 255, 4)])
        self.assertTrue(np.all(mask[3, :] == 255))
        self.assertGreaterEqual(len(result.points), 3)

    def test_erase_brush_reports_when_the_mask_is_completely_removed(self):
        backend = _RasterBackend()
        mask = np.full((5, 5), 255, dtype=np.uint8)

        result = apply_brush_stroke(
            mask,
            end=(2.0, 2.0),
            radius=20,
            add=False,
            image_width=5,
            image_height=5,
            cv2_module=backend,
            anchor_points=[(0, 0), (4, 0), (4, 4), (0, 4)],
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.erased)
        self.assertEqual(result.points, [])
        self.assertEqual(result.pixel_count, 0)
        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_brush_cleanly_gates_missing_backend_mask_and_dimensions(self):
        common = {
            "end": (1.0, 1.0),
            "add": True,
            "image_width": 4,
            "image_height": 4,
        }
        self.assertIsNone(apply_brush_stroke(None, cv2_module=_RasterBackend(), **common))
        self.assertIsNone(
            apply_brush_stroke(
                np.zeros((4, 4), dtype=np.uint8),
                cv2_module=None,
                **common,
            )
        )
        self.assertIsNone(
            apply_brush_stroke(
                np.zeros((4, 4), dtype=np.uint8),
                cv2_module=_RasterBackend(),
                end=(1.0, 1.0),
                add=True,
                image_width=0,
                image_height=4,
            )
        )

    def test_clamp_point_rounds_and_constrains_coordinates(self):
        self.assertEqual(clamp_point_to_image(4.6, 2.4, 10, 8), (5, 2))
        self.assertEqual(clamp_point_to_image(-3, 20, 10, 8), (0, 7))

    def test_clamp_point_preserves_existing_minimum_upper_bound(self):
        self.assertEqual(clamp_point_to_image(5, 5, 1, 0), (1, 1))

    def test_downsample_returns_original_when_within_limit(self):
        points = [(float(index), 0.0) for index in range(4)]

        result = downsample_polygon_points(points, max_points=4)

        self.assertIs(result, points)

    def test_downsample_uses_ceiling_step_and_keeps_three_points(self):
        points = [(float(index), 0.0) for index in range(10)]

        self.assertEqual(
            downsample_polygon_points(points, max_points=4),
            [points[0], points[3], points[6], points[9]],
        )
        self.assertEqual(downsample_polygon_points(points, max_points=2), points[:3])

    def test_signed_area_reports_orientation_and_degenerate_polygons(self):
        polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

        self.assertEqual(polygon_signed_area(polygon), 12.0)
        self.assertEqual(polygon_signed_area(list(reversed(polygon))), -12.0)
        self.assertEqual(polygon_signed_area(polygon[:2]), 0.0)

    def test_rotate_polygon_starts_at_vertex_nearest_anchor(self):
        points = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]

        self.assertEqual(
            rotate_polygon_to_anchor(points, (4.5, 4.0)),
            [(5.0, 5.0), (0.0, 5.0), (0.0, 0.0), (5.0, 0.0)],
        )

    def test_rotate_polygon_keeps_original_for_empty_or_first_nearest(self):
        empty: list[tuple[float, float]] = []
        points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]

        self.assertIs(rotate_polygon_to_anchor(empty, (1.0, 1.0)), empty)
        self.assertIs(rotate_polygon_to_anchor(points, (0.0, 0.0)), points)


class SegmentationEditStateTests(unittest.TestCase):
    def test_tracks_prompt_labels_and_preview_values(self):
        state = SegmentationEditState()

        positive = state.add_prompt(1, 2, positive=True)
        negative = state.add_prompt(3.5, 4.5, positive=False)
        state.set_preview([(0, 0), (4, 0), (0, 4)], score=0.75)

        self.assertEqual(positive, (1.0, 2.0, 1))
        self.assertEqual(negative, (3.5, 4.5, 0))
        self.assertEqual(state.prompt_points, [positive, negative])
        self.assertEqual(state.preview_points, [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)])
        self.assertEqual(state.preview_score, 0.75)
        self.assertTrue(state.has_preview)

    def test_clear_preview_preserves_prompts_target_and_accepted_masks(self):
        state = SegmentationEditState(
            prompt_points=[(1.0, 2.0, 1)],
            preview_points=[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)],
            preview_score=0.6,
            accepted_masks={1: {"class_id": 1, "segments": [(1.0, 1.0)] * 3, "score": 0.9}},
            selected_target=2,
        )

        state.clear_preview()

        self.assertEqual(state.prompt_points, [(1.0, 2.0, 1)])
        self.assertEqual(state.preview_points, [])
        self.assertEqual(state.preview_score, 0.0)
        self.assertIn(1, state.accepted_masks)
        self.assertEqual(state.selected_target, 2)

    def test_clear_prompt_state_also_clears_preview(self):
        state = SegmentationEditState(
            prompt_points=[(1.0, 2.0, 1)],
            preview_points=[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)],
            preview_score=0.6,
            selected_target=2,
        )

        state.clear_prompt_state()

        self.assertEqual(state.prompt_points, [])
        self.assertEqual(state.preview_points, [])
        self.assertEqual(state.preview_score, 0.0)
        self.assertEqual(state.selected_target, 2)

    def test_accept_preview_uses_annotation_cache_representation(self):
        state = SegmentationEditState(selected_target=3)
        state.add_prompt(1, 1)
        state.set_preview([(0, 0), (5, 0), (0, 5)], score=0.8)

        accepted = state.accept_preview()

        expected = {
            "class_id": 3,
            "segments": [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)],
            "score": 0.8,
        }
        self.assertEqual(accepted, expected)
        self.assertEqual(state.accepted_masks[3], expected)
        self.assertEqual(state.prompt_points, [])
        self.assertEqual(state.preview_points, [])
        self.assertEqual(state.preview_score, 0.0)
        self.assertEqual(state.selected_target, 3)

    def test_accept_preview_requires_target_and_three_points(self):
        state = SegmentationEditState()
        state.set_preview([(0, 0), (1, 0), (0, 1)])
        self.assertIsNone(state.accept_preview())
        self.assertTrue(state.has_preview)

        state.select_target(0)
        state.set_preview([(0, 0), (1, 0)])
        self.assertIsNone(state.accept_preview())
        self.assertNotIn(0, state.accepted_masks)

    def test_clear_accepted_mask_defaults_to_selected_target(self):
        state = SegmentationEditState(
            accepted_masks={
                1: {"class_id": 1, "segments": [(0.0, 0.0)] * 3, "score": 0.0},
                2: {"class_id": 2, "segments": [(1.0, 1.0)] * 3, "score": 0.0},
            },
            selected_target=2,
        )

        self.assertTrue(state.clear_accepted_mask())
        self.assertEqual(list(state.accepted_masks), [1])
        self.assertFalse(state.clear_accepted_mask())

    def test_snapshot_is_detached_and_restore_copies_it(self):
        state = SegmentationEditState(selected_target=1)
        state.add_prompt(1, 2)
        snapshot = state.snapshot()

        state.prompt_points.append((3.0, 4.0, 0))
        state.restore(snapshot)
        snapshot.prompt_points.append((9.0, 9.0, 1))

        self.assertEqual(state.prompt_points, [(1.0, 2.0, 1)])
        self.assertEqual(state.selected_target, 1)

    def test_undo_restores_explicit_snapshot(self):
        state = SegmentationEditState(selected_target=4)
        state.add_prompt(2, 3)
        state.set_preview([(0, 0), (3, 0), (0, 3)], score=0.5)
        state.push_undo_snapshot()

        state.accept_preview()

        self.assertTrue(state.can_undo)
        self.assertTrue(state.undo())
        self.assertFalse(state.can_undo)
        self.assertEqual(state.prompt_points, [(2.0, 3.0, 1)])
        self.assertTrue(state.has_preview)
        self.assertEqual(state.accepted_masks, {})
        self.assertFalse(state.undo())

    def test_set_and_remove_accepted_entry_keep_cache_shape(self):
        state = SegmentationEditState()

        stored = state.set_accepted_entry(
            2,
            {"class_id": 99, "segments": [(0.0, 0.0)] * 3, "score": 0.4},
        )
        stored["score"] = 1.0

        self.assertEqual(state.accepted_masks[2]["class_id"], 2)
        self.assertEqual(state.accepted_masks[2]["score"], 0.4)
        self.assertTrue(state.clear_accepted_mask(2))

    def test_remove_last_prompt_preserves_existing_undo_behavior(self):
        state = SegmentationEditState(prompt_points=[(1.0, 2.0, 1), (3.0, 4.0, 0)])

        self.assertEqual(state.remove_last_prompt(), (3.0, 4.0, 0))
        self.assertEqual(state.prompt_points, [(1.0, 2.0, 1)])
        self.assertEqual(state.remove_last_prompt(), (1.0, 2.0, 1))
        self.assertIsNone(state.remove_last_prompt())

    def test_reset_replaces_image_state_and_clears_undo_history(self):
        state = SegmentationEditState(selected_target=1)
        state.add_prompt(1, 2)
        state.push_undo_snapshot()

        state.reset(
            accepted_masks={3: {"class_id": 3, "segments": [(0.0, 0.0)] * 3, "score": 0.2}},
            selected_target=3,
        )

        self.assertEqual(state.prompt_points, [])
        self.assertEqual(state.preview_points, [])
        self.assertEqual(list(state.accepted_masks), [3])
        self.assertEqual(state.selected_target, 3)
        self.assertFalse(state.can_undo)


if __name__ == "__main__":
    unittest.main()
