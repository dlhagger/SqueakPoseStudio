import json
import os
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from analysis_ops import AnalysisConfig, AnalysisError, run_analysis_workflow
from analysis_worker import run_analysis_worker


def _write_demo_detections(path: str) -> None:
    rows = []
    for frame in range(6):
        rows.append(
            {
                "frame_index": frame,
                "time_seconds": frame / 10.0,
                "detections_in_frame": 1,
                "detection_index": 0,
                "class_id": 0,
                "class_name": "mouse",
                "confidence": 0.9 - frame * 0.01,
                "bbox_x1": 10 + frame,
                "bbox_y1": 20,
                "bbox_x2": 30 + frame,
                "bbox_y2": 45,
                "bbox_width": 20,
                "bbox_height": 25,
                "bbox_center_x": 20 + frame,
                "bbox_center_y": 32,
                "image_width": 100,
                "image_height": 80,
                "speed_preprocess_ms": 1.0,
                "speed_inference_ms": 2.0,
                "speed_postprocess_ms": 3.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_demo_segmentation(path: str) -> None:
    rows = [
        {
            "frame": 0,
            "det": 0,
            "class_id": 0,
            "class_name": "mouse",
            "conf": 0.90,
            "x1": 10,
            "y1": 10,
            "x2": 30,
            "y2": 30,
            "mask_polygon": json.dumps([[10, 10], [30, 10], [30, 30], [10, 30]]),
            "binary_mask": "",
        },
        {
            "frame": 1,
            "det": 0,
            "class_id": 0,
            "class_name": "mouse",
            "conf": 0.91,
            "x1": 12,
            "y1": 10,
            "x2": 32,
            "y2": 30,
            "mask_polygon": json.dumps([[12, 10], [32, 10], [32, 30], [12, 30]]),
            "binary_mask": "",
        },
        {
            "frame": 2,
            "det": -1,
            "class_id": "",
            "class_name": "",
            "conf": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "mask_polygon": "",
            "binary_mask": "",
        },
        {
            "frame": 3,
            "det": 0,
            "class_id": 0,
            "class_name": "mouse",
            "conf": 0.80,
            "x1": 16,
            "y1": 10,
            "x2": 36,
            "y2": 30,
            "mask_polygon": json.dumps([[16, 10], [36, 10], [36, 30], [16, 30]]),
            "binary_mask": "",
        },
        {
            "frame": 3,
            "det": 1,
            "class_id": 0,
            "class_name": "mouse",
            "conf": 0.95,
            "x1": 18,
            "y1": 10,
            "x2": 38,
            "y2": 30,
            "mask_polygon": json.dumps([[18, 10], [38, 10], [38, 30], [18, 30]]),
            "binary_mask": "",
        },
        {
            "frame": 4,
            "det": 0,
            "class_id": 0,
            "class_name": "mouse",
            "conf": 0.92,
            "x1": 20,
            "y1": 10,
            "x2": 40,
            "y2": 30,
            "mask_polygon": json.dumps([[20, 10], [40, 10], [40, 30], [20, 30]]),
            "binary_mask": "",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class AnalysisOpsTests(unittest.TestCase):
    def test_analysis_rejects_results_from_a_different_layer(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "segmentation.csv")
            _write_demo_segmentation(csv_path)

            with self.assertRaisesRegex(
                AnalysisError, "contains segmentation layer results"
            ):
                run_analysis_workflow(
                    AnalysisConfig(
                        detections_csv=csv_path,
                        output_dir=os.path.join(tmp, "analysis"),
                        layer_id="keypoints",
                        make_plots=False,
                    )
                )

    def test_run_analysis_workflow_writes_features_and_summary(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "detections.csv")
            out_dir = os.path.join(tmp, "analysis")
            _write_demo_detections(csv_path)

            result = run_analysis_workflow(
                AnalysisConfig(
                    detections_csv=csv_path,
                    output_dir=out_dir,
                    fps=10.0,
                    pixel_distance=2.0,
                    real_world_distance_mm=4.0,
                    smooth=False,
                    make_plots=True,
                    rois=[
                        {
                            "name": "Start Zone",
                            "type": "rect",
                            "x1": 18,
                            "y1": 28,
                            "x2": 24,
                            "y2": 36,
                        }
                    ],
                )
            )

            self.assertTrue(os.path.isfile(result["feature_csv"]))
            self.assertTrue(os.path.isfile(result["summary_json"]))
            self.assertTrue(os.path.isfile(result["roi_summary_csv"]))
            self.assertTrue(result["plot_paths"])
            for plot_path in result["plot_paths"]:
                self.assertTrue(os.path.isfile(plot_path))
            features = pd.read_csv(result["feature_csv"])
            self.assertIn("speed_mm_per_sec", features.columns)
            self.assertIn("cumulative_distance_mm", features.columns)
            self.assertIn("roi_label", features.columns)
            self.assertIn("Start Zone", set(features["roi_label"]))
            self.assertAlmostEqual(result["summary"]["mm_per_pixel"], 2.0)
            self.assertEqual(result["summary"]["roi_count"], 1)
            self.assertGreater(result["summary"]["total_distance_mm"], 0.0)

            with open(result["summary_json"], "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertEqual(summary["frames"], 6)

    def test_analysis_worker_emits_progress_and_result(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "detections.csv")
            out_dir = os.path.join(tmp, "analysis")
            _write_demo_detections(csv_path)
            events = []

            code = run_analysis_worker(
                {
                    "detections_csv": csv_path,
                    "output_dir": out_dir,
                    "fps": 10.0,
                    "smooth": True,
                    "make_plots": False,
                },
                event_writer=events.append,
            )

            self.assertEqual(code, 0)
            event_names = [event["event"] for event in events]
            self.assertIn("started", event_names)
            self.assertIn("progress", event_names)
            self.assertIn("result", event_names)
            result = next(event for event in events if event["event"] == "result")
            self.assertTrue(os.path.isfile(result["feature_csv"]))

    def test_run_analysis_workflow_routes_segmentation_csv(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "segmentation.csv")
            out_dir = os.path.join(tmp, "analysis")
            _write_demo_segmentation(csv_path)

            result = run_analysis_workflow(
                AnalysisConfig(
                    detections_csv=csv_path,
                    output_dir=out_dir,
                    fps=10.0,
                    pixel_distance=2.0,
                    real_world_distance_mm=4.0,
                    smooth=False,
                    make_plots=True,
                    rois=[
                        {
                            "name": "Center",
                            "type": "rect",
                            "x1": 15,
                            "y1": 5,
                            "x2": 45,
                            "y2": 35,
                        }
                    ],
                )
            )

            self.assertEqual(result["summary"]["analysis_kind"], "segmentation")
            self.assertEqual(result["summary"]["total_video_frames"], 5)
            self.assertEqual(result["summary"]["no_detection_frames"], 1)
            self.assertEqual(result["summary"]["multi_detection_frames"], 1)
            self.assertTrue(os.path.isfile(result["feature_csv"]))
            self.assertTrue(os.path.isfile(result["segmentation_detections_csv"]))
            self.assertTrue(os.path.isfile(result["roi_summary_csv"]))
            self.assertTrue(result["plot_paths"])
            self.assertTrue(
                any(os.path.basename(path) == "segmentation_confidence.png" for path in result["plot_paths"])
            )

            features = pd.read_csv(result["feature_csv"])
            detections = pd.read_csv(result["segmentation_detections_csv"])
            self.assertIn("mask_area_px2", features.columns)
            self.assertIn("bbox_center_x_euro", features.columns)
            self.assertIn("speed_mm_per_sec", features.columns)
            self.assertIn("roi_label", features.columns)
            self.assertIn("Center", set(features["roi_label"]))
            self.assertEqual(len(features), 4)
            self.assertEqual(len(detections), 5)


if __name__ == "__main__":
    unittest.main()
