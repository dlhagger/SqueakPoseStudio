import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from analysis_ops import (
    AnalysisConfig,
    AnalysisError,
    _open_h264_video_writer,
    assign_roi_labels,
    build_combined_analysis_outputs,
    create_roi_outputs,
    draw_supersampled_polygon_overlay,
    normalize_rois,
    prepare_analysis_output_dir,
    render_annotated_video,
    run_analysis_workflow,
)
from analysis_worker import run_analysis_worker
from segmentation_analysis_ops import (
    _mask_area_overlay_text,
    compute_segmentation_detection_features,
    render_segmentation_annotated_video,
)
from unified_analysis_ops import render_unified_annotated_video


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
            normalize_rois([{"type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]}]),
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
    def test_supersampled_polygon_overlay_has_antialiased_edges(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        draw_supersampled_polygon_overlay(
            frame,
            [(10.25, 10.25), (52.5, 18.75), (20.5, 53.25)],
            (255, 255, 255),
            cv2,
            alpha=1.0,
            supersample=2,
        )

        values = np.unique(frame)
        self.assertTrue(np.any((values > 0) & (values < 255)))
        self.assertTrue(np.any(values == 255))
        self.assertEqual(frame[0, 0].tolist(), [0, 0, 0])

    def test_stable_output_cleanup_removes_only_generated_artifacts(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp, "analysis outputs", "session", "combined")
            plots = output / "plots"
            plots.mkdir(parents=True)
            (plots / "stale.png").write_bytes(b"old")
            (output / "analysis.csv").write_text("old", encoding="utf-8")
            (output / "research_notes.txt").write_text("keep", encoding="utf-8")

            prepare_analysis_output_dir(
                output,
                generated_files=("analysis.csv",),
                generated_directories=("plots",),
            )

            self.assertFalse((output / "analysis.csv").exists())
            self.assertFalse(plots.exists())
            self.assertEqual((output / "research_notes.txt").read_text(encoding="utf-8"), "keep")

    def test_segmentation_geometry_uses_mask_bounds_and_retains_inference_box(self):
        raw = pd.DataFrame(
            [
                {
                    "frame": 0,
                    "det": 0,
                    "class_id": 0,
                    "class_name": "mouse",
                    "conf": 0.9,
                    "x1": 12,
                    "y1": 12,
                    "x2": 18,
                    "y2": 18,
                    "mask_polygon": json.dumps([[10, 10], [20, 10], [20, 20], [10, 20]]),
                }
            ]
        )

        features = compute_segmentation_detection_features(raw, 0.5)
        row = features.iloc[0]

        self.assertEqual(
            [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
            [10.0, 10.0, 20.0, 20.0],
        )
        self.assertEqual(
            [
                row["inference_bbox_x1"],
                row["inference_bbox_y1"],
                row["inference_bbox_x2"],
                row["inference_bbox_y2"],
            ],
            [12.0, 12.0, 18.0, 18.0],
        )
        self.assertEqual(row["bbox_source"], "segmentation_mask_bounds")
        self.assertEqual(row["mask_fill_ratio"], 1.0)

    def test_combined_outputs_prefer_mask_centroid_and_retain_keypoint_rois(self):
        with TemporaryDirectory() as tmp:
            pose_csv = Path(tmp, "pose_features.csv")
            segment_csv = Path(tmp, "segmentation_features.csv")
            output_dir = Path(tmp, "combined")
            pd.DataFrame(
                {
                    "frame_index": [0, 1, 2],
                    "bbox_center_x": [10.0, 11.0, 12.0],
                    "bbox_center_y": [10.0, 10.0, 10.0],
                    "bbox_x1": [8.0, 9.0, 10.0],
                    "bbox_y1": [8.0, 8.0, 8.0],
                    "bbox_x2": [12.0, 13.0, 14.0],
                    "bbox_y2": [12.0, 12.0, 12.0],
                    "bbox_center_x_euro": [10.0, 11.0, 12.0],
                    "bbox_center_y_euro": [10.0, 10.0, 10.0],
                    "kp_nose_x": [5.0, 6.0, 7.0],
                    "kp_nose_y": [5.0, 5.0, 5.0],
                    "kp_tail_base_x": [25.0, 26.0, 27.0],
                    "kp_tail_base_y": [25.0, 25.0, 25.0],
                }
            ).to_csv(pose_csv, index=False)
            pd.DataFrame(
                {
                    "frame_index": [0, 2],
                    "bbox_center_x": [5.0, 25.0],
                    "bbox_center_y": [5.0, 25.0],
                    "bbox_x1": [1.0, 21.0],
                    "bbox_y1": [1.0, 21.0],
                    "bbox_x2": [9.0, 29.0],
                    "bbox_y2": [9.0, 29.0],
                    "bbox_center_x_euro": [5.0, 25.0],
                    "bbox_center_y_euro": [5.0, 25.0],
                    "mask_centroid_x": [5.0, 25.0],
                    "mask_centroid_y": [5.0, 25.0],
                }
            ).to_csv(segment_csv, index=False)

            result = build_combined_analysis_outputs(
                pose_feature_csv=str(pose_csv),
                segmentation_feature_csv=str(segment_csv),
                output_dir=str(output_dir),
                fps=10.0,
                mm_per_pixel=0.5,
                rois=[
                    {
                        "name": "Center",
                        "type": "rect",
                        "x1": 0,
                        "y1": 0,
                        "x2": 20,
                        "y2": 20,
                    }
                ],
            )

            combined = pd.read_csv(result["feature_csv"])
            self.assertEqual(len(combined), 3)
            self.assertEqual(
                combined["centroid_source"].tolist(),
                ["segmentation_mask", "pose_bbox", "segmentation_mask"],
            )
            self.assertEqual(combined["centroid_x"].tolist(), [5.0, 11.0, 25.0])
            self.assertEqual(
                combined["bbox_source"].tolist(),
                ["segmentation_bbox", "pose_bbox", "segmentation_bbox"],
            )
            self.assertEqual(combined["bbox_x1"].tolist(), [1.0, 9.0, 21.0])
            self.assertEqual(combined["roi_label"].tolist(), ["Center", "Center", "Outside"])
            self.assertEqual(combined["roi_nose"].tolist(), ["Center"] * 3)
            self.assertEqual(combined["roi_tail_base"].tolist(), ["Outside"] * 3)
            self.assertIn("movement_heading_deg", combined.columns)
            self.assertIn("heading_deg", combined.columns)
            self.assertIn("acceleration_mm_per_sec2", combined.columns)
            self.assertEqual(result["summary"]["pose_valid_frames"], 3)
            self.assertEqual(result["summary"]["segmentation_valid_frames"], 2)
            self.assertTrue(Path(result["summary_json"]).is_file())
            self.assertTrue(Path(result["keypoint_roi_summary_csv"]).is_file())

    def test_video_writer_prefers_nvenc_and_falls_back_to_software(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp, "output.mp4")
            sentinel = object()
            with (
                patch("analysis_ops._nvenc_available", return_value=True),
                patch("analysis_ops._FFmpegNVENCH264VideoWriter", return_value=sentinel),
            ):
                self.assertIs(_open_h264_video_writer(output, 8.0, 640, 480), sentinel)

            with (
                patch("analysis_ops._nvenc_available", return_value=True),
                patch(
                    "analysis_ops._FFmpegNVENCH264VideoWriter",
                    side_effect=AnalysisError("NVENC unavailable"),
                ),
                patch("analysis_ops._PyAVH264VideoWriter", return_value=sentinel),
            ):
                self.assertIs(_open_h264_video_writer(output, 8.0, 640, 480), sentinel)

            with (
                patch("analysis_ops._nvenc_available", return_value=True),
                patch("analysis_ops._PyAVH264VideoWriter", return_value=sentinel),
            ):
                self.assertIs(_open_h264_video_writer(output, 8.0, 100, 80), sentinel)

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

            unified_path = os.path.join(tmp, "unified.mp4")
            unified_rows = segmentation_rows.assign(
                centroid_x_smooth=20.0,
                centroid_y_smooth=20.0,
                cumulative_distance_mm=0.0,
                kp_nose_x=18.0,
                kp_nose_y=14.0,
                kp_head_x=20.0,
                kp_head_y=18.0,
                kp_back_x=22.0,
                kp_back_y=23.0,
                kp_tail_base_x=24.0,
                kp_tail_base_y=27.0,
            )
            self.assertEqual(
                render_unified_annotated_video(
                    unified_rows,
                    source_path,
                    Path(unified_path),
                    10.0,
                    normalize_rois(
                        [
                            {
                                "name": "Arena",
                                "type": "polygon",
                                "points": [[5, 5], [90, 5], [90, 70], [5, 70]],
                            }
                        ]
                    ),
                ),
                unified_path,
            )

            for output_path in (pose_path, segmentation_path, unified_path):
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

    def test_analysis_worker_runs_both_layers_as_one_authoritative_analysis(self):
        with TemporaryDirectory() as tmp:
            pose_csv = os.path.join(tmp, "pose.csv")
            segment_csv = os.path.join(tmp, "segmentation.csv")
            out_dir = os.path.join(tmp, "analysis")
            _write_demo_detections(pose_csv)
            pose = pd.read_csv(pose_csv)
            pose["kp_nose_x"] = pose["bbox_center_x"]
            pose["kp_nose_y"] = pose["bbox_center_y"]
            pose.to_csv(pose_csv, index=False)
            _write_demo_segmentation(segment_csv)
            events = []

            code = run_analysis_worker(
                {
                    "analysis_mode": "both",
                    "analysis_inputs": {
                        "keypoints": pose_csv,
                        "segmentation": segment_csv,
                    },
                    "selected_layers": ["keypoints", "segmentation"],
                    "output_dir": out_dir,
                    "fps": 10.0,
                    "pixel_distance": 2.0,
                    "real_world_distance_mm": 4.0,
                    "smooth": False,
                    "make_plots": False,
                    "make_annotated_video": False,
                    "run_clustering": False,
                    "export_cluster_clips": False,
                    "rois": [
                        {
                            "name": "Arena",
                            "type": "rect",
                            "x1": 0,
                            "y1": 0,
                            "x2": 100,
                            "y2": 80,
                        }
                    ],
                },
                event_writer=events.append,
            )

            self.assertEqual(code, 0)
            result = next(event for event in events if event["event"] == "result")
            self.assertEqual(result["analysis_mode"], "both")
            self.assertEqual(Path(result["feature_csv"]).name, "analysis.csv")
            self.assertTrue(Path(result["feature_csv"]).is_file())
            self.assertTrue(Path(result["manifest_path"]).is_file())
            self.assertFalse(Path(out_dir, "keypoints").exists())
            self.assertFalse(Path(out_dir, "segmentation").exists())
            combined = pd.read_csv(result["feature_csv"])
            self.assertIn("roi_nose", combined.columns)
            self.assertIn("mask_polygon", combined.columns)
            self.assertEqual(len(combined), 6)
            self.assertEqual(
                combined["bbox_source"].value_counts().to_dict(),
                {"segmentation_bbox": 4, "pose_bbox": 2},
            )
            self.assertEqual(
                combined["centroid_source"].value_counts().to_dict(),
                {"segmentation_mask": 4, "pose_bbox": 2},
            )
            transitions = pd.read_csv(result["roi_transition_csv"])
            self.assertTrue(transitions.empty)

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
