import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from analysis_ops import (
    AnalysisConfig,
    AnalysisError,
    _open_h264_video_writer,
    assign_roi_labels,
    create_roi_outputs,
    normalize_rois,
    render_annotated_video,
    run_analysis_workflow,
)
from analysis_worker import run_analysis_worker
from segmentation_analysis_ops import (
    _mask_area_overlay_text,
    render_segmentation_annotated_video,
)


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


class PolygonRoiTests(unittest.TestCase):
    def test_polygon_normalization_and_center_containment_include_boundary(self):
        rois = normalize_rois(
            [
                {
                    "name": "Triangle",
                    "type": "polygon",
                    "points": [[0, 0], [10, 0], [0, 10]],
                }
            ]
        )
        detections = pd.DataFrame(
            {
                "x": [2.0, 8.0, 5.0, np.nan],
                "y": [2.0, 8.0, 5.0, 2.0],
            }
        )

        labeled = assign_roi_labels(detections, rois, x_col="x", y_col="y")

        self.assertEqual(rois[0]["type"], "polygon")
        self.assertEqual(
            labeled["roi_label"].tolist(),
            ["Triangle", "Outside", "Triangle", "Outside"],
        )

    def test_polygon_normalization_rejects_degenerate_shapes(self):
        self.assertEqual(
            normalize_rois(
                [{"type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]}]
            ),
            [],
        )

    def test_first_roi_has_explicit_precedence_when_polygons_overlap(self):
        detections = pd.DataFrame({"x": [5.0], "y": [5.0]})
        shared_shape = [[0, 0], [10, 0], [10, 10], [0, 10]]
        rois = [
            {"name": "Highest", "type": "polygon", "points": shared_shape},
            {"name": "Lower", "type": "polygon", "points": shared_shape},
        ]

        labeled = assign_roi_labels(detections, rois, x_col="x", y_col="y")

        self.assertEqual(labeled.loc[0, "roi_label"], "Highest")


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
    def test_segmentation_video_formats_calibrated_mask_area(self):
        row = pd.Series({"mask_area_px2": 200, "mask_area_mm2": 12.54})

        self.assertEqual(_mask_area_overlay_text(row), "Mask area: 12.5 mm^2")

    def test_analysis_config_uses_low_latency_one_euro_defaults(self):
        direct = AnalysisConfig(detections_csv="detections.csv", output_dir="analysis")
        loaded = AnalysisConfig.from_dict(
            {"detections_csv": "detections.csv", "output_dir": "analysis"}
        )

        for config in (direct, loaded):
            self.assertEqual(config.min_cutoff, 1.0)
            self.assertEqual(config.beta, 0.1)

    def test_roi_transition_matrix_counts_adjacent_label_changes(self):
        with TemporaryDirectory() as tmp:
            features = pd.DataFrame(
                {
                    "frame_index": range(7),
                    "roi_label": ["Center", "Left", "Left", "Center", "Right", "Right", "Center"],
                    "dt_seconds": [0.1] * 7,
                    "distance_mm": [0.0] * 7,
                    "speed_mm_per_sec": [0.0] * 7,
                }
            )

            outputs = create_roi_outputs(features, Path(tmp), 10.0)
            transition = pd.read_csv(outputs["roi_transition_csv"], index_col=0)

            self.assertEqual(transition.loc["Center", "Left"], 1)
            self.assertEqual(transition.loc["Left", "Left"], 1)
            self.assertEqual(transition.loc["Left", "Center"], 1)
            self.assertEqual(transition.loc["Center", "Right"], 1)
            self.assertEqual(transition.loc["Right", "Right"], 1)
            self.assertEqual(transition.loc["Right", "Center"], 1)

    def test_pyav_h264_writer_encodes_mp4_and_pads_odd_dimensions(self):
        with TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "output.mp4")
            frame = np.zeros((81, 101, 3), dtype=np.uint8)
            frame[:, :, 1] = 180

            with _open_h264_video_writer(output_path, 8.0, 101, 81) as writer:
                writer.write(frame)
                writer.write(frame)

            capture = cv2.VideoCapture(output_path)
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(
                    (
                        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    ),
                    (102, 82),
                )
                decoded = 0
                while capture.read()[0]:
                    decoded += 1
                self.assertEqual(decoded, 2)
            finally:
                capture.release()

    def test_pyav_h264_writer_preserves_existing_file_after_failure(self):
        with TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "output.mp4")
            with open(output_path, "wb") as fh:
                fh.write(b"previous export")

            with self.assertRaisesRegex(RuntimeError, "stop export"):
                with _open_h264_video_writer(output_path, 8.0, 100, 80) as writer:
                    writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
                    raise RuntimeError("stop export")

            with open(output_path, "rb") as fh:
                self.assertEqual(fh.read(), b"previous export")
            self.assertEqual(os.listdir(tmp), ["output.mp4"])

    def test_pyav_h264_writer_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(AnalysisError, "invalid frame size"):
            _open_h264_video_writer("output.mp4", 8.0, 0, 80)
        with self.assertRaisesRegex(AnalysisError, "invalid frame rate"):
            _open_h264_video_writer("output.mp4", 0.0, 100, 80)

    def test_annotated_video_renderers_export_decodable_h264(self):
        with TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "source.mp4")
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            with _open_h264_video_writer(source_path, 10.0, 100, 80) as writer:
                for _ in range(3):
                    writer.write(frame)

            pose_path = os.path.join(tmp, "pose.mp4")
            pose_rows = pd.DataFrame(
                [
                    {
                        "frame_index": 0,
                        "bbox_x1": 10,
                        "bbox_y1": 10,
                        "bbox_x2": 30,
                        "bbox_y2": 30,
                        "bbox_center_x_euro": 20,
                        "bbox_center_y_euro": 20,
                        "cumulative_distance_mm": 0.0,
                        "speed_mm_per_sec": 0.0,
                        "roi_label": "",
                    }
                ]
            )
            self.assertEqual(
                render_annotated_video(
                    pose_rows,
                    source_path,
                    pose_path,
                    10.0,
                    rois=[
                        {
                            "name": "Arena",
                            "type": "polygon",
                            "points": [[5, 5], [90, 5], [90, 70], [5, 70]],
                        }
                    ],
                ),
                pose_path,
            )

            segmentation_path = os.path.join(tmp, "segmentation.mp4")
            segmentation_rows = pd.DataFrame(
                [
                    {
                        "frame_index": 0,
                        "bbox_x1": 10,
                        "bbox_y1": 10,
                        "bbox_x2": 30,
                        "bbox_y2": 30,
                        "bbox_center_x_euro": 20,
                        "bbox_center_y_euro": 20,
                        "mask_polygon": json.dumps([[10, 10], [30, 10], [30, 30]]),
                        "mask_area_px2": 200,
                        "mask_area_mm2": 12.5,
                        "speed_mm_per_sec": 0.0,
                        "roi_label": "",
                    }
                ]
            )
            self.assertEqual(
                render_segmentation_annotated_video(
                    segmentation_rows,
                    source_path,
                    segmentation_path,
                    10.0,
                    rois=[
                        {
                            "name": "Arena",
                            "type": "polygon",
                            "points": [[5, 5], [90, 5], [90, 70], [5, 70]],
                        }
                    ],
                ),
                segmentation_path,
            )

            for output_path in (pose_path, segmentation_path):
                capture = cv2.VideoCapture(output_path)
                try:
                    self.assertTrue(capture.isOpened())
                    decoded = 0
                    while capture.read()[0]:
                        decoded += 1
                    self.assertEqual(decoded, 3)
                finally:
                    capture.release()

    def test_analysis_rejects_results_from_a_different_layer(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "segmentation.csv")
            _write_demo_segmentation(csv_path)

            with self.assertRaisesRegex(AnalysisError, "contains segmentation layer results"):
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
                            "type": "polygon",
                            "points": [[18, 28], [28, 28], [28, 36], [18, 36]],
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
                any(
                    os.path.basename(path) == "segmentation_confidence.png"
                    for path in result["plot_paths"]
                )
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
