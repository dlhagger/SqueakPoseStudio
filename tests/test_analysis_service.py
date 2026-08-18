import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.services.analysis import (
    AnalysisConfigError,
    analysis_csv_matches_layer,
    build_analysis_run_config,
    default_analysis_output_dir,
    inspect_analysis_csv,
    latest_analysis_csv,
    load_segmentation_preview,
)


class AnalysisServiceTests(unittest.TestCase):
    def test_segmentation_preview_uses_first_frame_with_valid_masks(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "segmentation.csv")
            csv_path.write_text(
                "frame,det,mask_polygon\n"
                '0,-1,""\n'
                '2,0,"[[1, 2], [11, 2], [11, 12]]"\n'
                '2,1,"[[20, 20], [30, 20], [30, 30]]"\n'
                '3,0,"[[40, 40], [50, 40], [50, 50]]"\n',
                encoding="utf-8",
            )

            preview = load_segmentation_preview(str(csv_path))

            self.assertEqual(preview.frame_index, 2)
            self.assertEqual(len(preview.polygons), 2)
            self.assertEqual(preview.polygons[0][0], (1.0, 2.0))

    def test_csv_layer_detection_and_output_path(self):
        with TemporaryDirectory() as tmp:
            pose = Path(tmp, "pose results.csv")
            pose.write_text("frame,det,x,y\n", encoding="utf-8")
            segment = Path(tmp, "segment.csv")
            segment.write_text("frame,det,mask_polygon\n", encoding="utf-8")

            self.assertTrue(analysis_csv_matches_layer(str(pose), "keypoints"))
            self.assertFalse(analysis_csv_matches_layer(str(pose), "segmentation"))
            self.assertTrue(analysis_csv_matches_layer(str(segment), "segmentation"))
            self.assertEqual(
                default_analysis_output_dir(
                    tmp,
                    "keypoints",
                    str(pose),
                    timestamp="20260815-120000",
                ),
                os.path.join(
                    tmp,
                    "analysis outputs",
                    "keypoints",
                    "pose_results_20260815-120000",
                ),
            )

    def test_worker_payload_is_exact_and_detached(self):
        with TemporaryDirectory() as tmp:
            detections = Path(tmp, "detections.csv")
            detections.write_text("frame,det,x,y\n", encoding="utf-8")
            rois = [{"name": "Nest", "x1": 1, "y1": 2, "x2": 3, "y2": 4}]

            config = build_analysis_run_config(
                layer_id="keypoints",
                detections_csv=str(detections),
                video_path="",
                output_dir=os.path.join(tmp, "output"),
                pixel_distance=50,
                real_world_distance_mm=25,
                smooth=True,
                min_cutoff=1.5,
                beta=0.2,
                make_plots=True,
                make_annotated_video=True,
                run_clustering=True,
                export_cluster_clips=True,
                umap_neighbors=12,
                umap_min_dist=0.1,
                hdbscan_min_cluster_size=5,
                cluster_clip_length_sec=4,
                samples_per_cluster=3,
                rois=rois,
            )
            rois[0]["name"] = "Changed"
            payload = config.as_dict()
            payload["rois"][0]["name"] = "Also changed"

            self.assertTrue(config.video_fallback_notice)
            self.assertEqual(config.as_dict()["rois"][0]["name"], "Nest")
            self.assertEqual(config.as_dict()["fps"], 0.0)
            self.assertEqual(config.as_dict()["d_cutoff"], 1.0)
            self.assertEqual(config.as_dict()["pixel_distance"], 50.0)

    def test_validation_errors_have_stable_codes(self):
        common = dict(
            layer_id="keypoints",
            detections_csv="missing.csv",
            video_path="",
            output_dir="output",
            pixel_distance=1,
            real_world_distance_mm=1,
            smooth=False,
            min_cutoff=1,
            beta=0,
            make_plots=False,
            make_annotated_video=False,
            run_clustering=False,
            export_cluster_clips=False,
            umap_neighbors=5,
            umap_min_dist=0.1,
            hdbscan_min_cluster_size=2,
            cluster_clip_length_sec=1,
            samples_per_cluster=1,
            rois=[],
        )
        with self.assertRaises(AnalysisConfigError) as missing:
            build_analysis_run_config(**common)
        self.assertEqual(missing.exception.code, "csv_required")

        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp, "detections.csv")
            csv_path.write_text("frame,det,x,y\n", encoding="utf-8")
            common["detections_csv"] = str(csv_path)
            common["pixel_distance"] = 0
            with self.assertRaises(AnalysisConfigError) as scale:
                build_analysis_run_config(**common)
            self.assertEqual(scale.exception.code, "scale_required")

            common["pixel_distance"] = 1
            common["export_cluster_clips"] = True
            with self.assertRaises(AnalysisConfigError) as clustering:
                build_analysis_run_config(**common)
            self.assertEqual(clustering.exception.code, "clustering_required")

    def test_csv_context_and_latest_matching_file_are_service_decisions(self):
        with TemporaryDirectory() as tmp:
            video = Path(tmp, "source.mp4")
            video.write_bytes(b"video")
            older = Path(tmp, "older.csv")
            older.write_text("frame,det,x,y\n0,0,1,2\n", encoding="utf-8")
            newer = Path(tmp, "newer.csv")
            newer.write_text(
                "frame,det,x,y,video_path,image_width,image_height\n"
                f"0,0,1,2,{video},640,480\n"
                "1,0,1,2,/missing.mp4,1280,720\n",
                encoding="utf-8",
            )
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            context = inspect_analysis_csv(str(newer))
            self.assertEqual(context.video_path, str(video))
            self.assertEqual((context.width, context.height), (1280, 720))
            self.assertEqual(latest_analysis_csv([tmp], "keypoints"), str(newer))
            self.assertEqual(inspect_analysis_csv("missing.csv").width, 1280)

    def test_csv_context_falls_back_to_matching_inference_manifest(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            output_root = project / "inference outputs"
            segmentation_dir = output_root / "segmentation"
            runs_dir = output_root / "runs"
            segmentation_dir.mkdir(parents=True)
            runs_dir.mkdir()
            video = project / "videos" / "source.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            run_id = "source_20260818-122205_abcdef123456"
            csv_path = segmentation_dir / f"{run_id}_segmentation.csv"
            csv_path.write_text(
                "frame,det,x1,y1,x2,y2,mask_polygon\n0,0,1,2,3,4,[]\n",
                encoding="utf-8",
            )
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "video_path": str(video),
                "passes": [{"layer_id": "segmentation", "csv_path": str(csv_path)}],
            }
            (runs_dir / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")

            context = inspect_analysis_csv(str(csv_path))

            self.assertEqual(context.video_path, str(video))

    def test_csv_context_ignores_manifest_for_a_different_csv(self):
        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "inference outputs"
            segmentation_dir = output_root / "segmentation"
            runs_dir = output_root / "runs"
            segmentation_dir.mkdir(parents=True)
            runs_dir.mkdir()
            run_id = "source_20260818-122205_abcdef123456"
            csv_path = segmentation_dir / f"{run_id}_segmentation.csv"
            csv_path.write_text("frame,det,mask_polygon\n", encoding="utf-8")
            manifest = {
                "video_path": str(Path(tmp) / "video.mp4"),
                "passes": [{"csv_path": str(segmentation_dir / "other.csv")}],
            }
            (runs_dir / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(inspect_analysis_csv(str(csv_path)).video_path, "")


if __name__ == "__main__":
    unittest.main()
