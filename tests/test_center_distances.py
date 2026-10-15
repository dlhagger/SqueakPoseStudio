import json
import unittest

import pandas as pd

from squeakpose.services.center_distances import add_center_distances


class CenterDistanceTests(unittest.TestCase):
    def test_distances_cover_all_pairs_and_same_class_instances(self):
        frame = pd.DataFrame(
            [
                dict(
                    frame_index=0,
                    class_name="mouse",
                    track_id=1,
                    detection_index=0,
                    mask_centroid_x=0,
                    mask_centroid_y=0,
                ),
                dict(
                    frame_index=0,
                    class_name="object",
                    track_id=2,
                    detection_index=1,
                    mask_centroid_x=3,
                    mask_centroid_y=4,
                ),
                dict(
                    frame_index=0,
                    class_name="object",
                    track_id=3,
                    detection_index=2,
                    mask_centroid_x=0,
                    mask_centroid_y=4,
                ),
                dict(
                    frame_index=1,
                    class_name="mouse",
                    track_id=1,
                    detection_index=0,
                    mask_centroid_x=0,
                    mask_centroid_y=0,
                ),
            ]
        )
        result = add_center_distances(frame, 0.5)
        entries = [json.loads(value) for value in result.center_distances]
        self.assertEqual([len(items) for items in entries], [2, 2, 2, 0])
        self.assertEqual(
            entries[0][0],
            dict(
                class_name="object", track_id=2, detection_index=1, distance_px=5.0, distance_mm=2.5
            ),
        )
        self.assertEqual(entries[1][1]["distance_px"], 3.0)
        self.assertEqual(entries[2][1]["distance_px"], 3.0)
        self.assertEqual(entries[1][0]["distance_px"], entries[0][0]["distance_px"])
        self.assertNotIn("center_distances", frame)
        pd.testing.assert_frame_equal(
            result.drop(columns=["center_distances", "center_distances_smoothed"]), frame
        )

    def test_mask_centroids_take_precedence_over_boxes_and_smoothed_centers(self):
        frame = pd.DataFrame(
            [
                dict(
                    frame_index=0,
                    mask_centroid_x=0,
                    mask_centroid_y=0,
                    bbox_x1=20,
                    bbox_y1=20,
                    bbox_x2=30,
                    bbox_y2=30,
                    bbox_center_x_euro=100,
                    bbox_center_y_euro=100,
                ),
                dict(
                    frame_index=0,
                    mask_centroid_x=3,
                    mask_centroid_y=4,
                    bbox_x1=50,
                    bbox_y1=50,
                    bbox_x2=60,
                    bbox_y2=60,
                    bbox_center_x_euro=200,
                    bbox_center_y_euro=200,
                ),
            ]
        )
        result = add_center_distances(frame, 1)
        self.assertEqual(json.loads(result.center_distances.iloc[0])[0]["distance_px"], 5)
        self.assertAlmostEqual(
            json.loads(result.center_distances_smoothed.iloc[0])[0]["distance_px"], 200**0.5 * 10
        )

    def test_missing_centers_are_not_imputed_and_boxes_are_supported(self):
        frame = pd.DataFrame(
            [
                dict(frame_index=0, bbox_x1=0, bbox_y1=0, bbox_x2=2, bbox_y2=2),
                dict(frame_index=0, bbox_x1=3, bbox_y1=4, bbox_x2=5, bbox_y2=6),
                dict(frame_index=0),
                dict(frame_index=1, bbox_center_x=1, bbox_center_y=1),
            ]
        )
        result = add_center_distances(frame, None)
        target = json.loads(result.center_distances.iloc[0])[0]
        self.assertEqual(target["distance_px"], 5)
        self.assertIsNone(target["distance_mm"])
        self.assertTrue(pd.isna(result.center_distances.iloc[2]))
        self.assertEqual(result.center_distances.iloc[3], "[]")

    def test_repeated_track_ids_and_duplicate_indices_do_not_merge_rows(self):
        frame = pd.DataFrame(
            [
                dict(frame_index=0, class_name="123", track_id=7, bbox_center_x=0, bbox_center_y=0),
                dict(
                    frame_index=0, class_name="object", track_id=7, bbox_center_x=0, bbox_center_y=0
                ),
            ],
            index=[2, 2],
        )
        result = add_center_distances(frame, 1)
        self.assertEqual(json.loads(result.center_distances.iloc[1])[0]["class_name"], "123")
        self.assertEqual(json.loads(result.center_distances.iloc[0])[0]["distance_px"], 0)
        self.assertEqual(result.index.tolist(), [2, 2])

    def test_missing_smoothed_centers_do_not_fall_back_to_raw(self):
        frame = pd.DataFrame(
            [
                dict(
                    frame_index=0,
                    mask_centroid_x=0,
                    mask_centroid_y=0,
                    bbox_center_x_euro=0,
                    bbox_center_y_euro=0,
                ),
                dict(frame_index=0, mask_centroid_x=3, mask_centroid_y=4),
                dict(frame_index=0, bbox_center_x_euro=3, bbox_center_y_euro=4),
            ]
        )
        result = add_center_distances(frame, 1)
        self.assertEqual(result.center_distances_smoothed.iloc[0], "[]")
        self.assertTrue(pd.isna(result.center_distances_smoothed.iloc[1]))
        self.assertTrue(pd.isna(result.center_distances_smoothed.iloc[2]))
        self.assertEqual(len(json.loads(result.center_distances.iloc[0])), 1)

    def test_smoothing_disabled_centers_give_equal_distances(self):
        frame = pd.DataFrame(
            [
                dict(
                    frame_index=0,
                    mask_centroid_x=0,
                    mask_centroid_y=0,
                    bbox_center_x_euro=0,
                    bbox_center_y_euro=0,
                ),
                dict(
                    frame_index=0,
                    mask_centroid_x=3,
                    mask_centroid_y=4,
                    bbox_center_x_euro=3,
                    bbox_center_y_euro=4,
                ),
            ]
        )
        result = add_center_distances(frame, 0.5)
        self.assertEqual(
            result.center_distances.tolist(), result.center_distances_smoothed.tolist()
        )

    def test_invalid_calibration_is_rejected(self):
        for scale in [0, -1, float("nan"), float("inf")]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                add_center_distances(pd.DataFrame({"frame_index": []}), scale)
