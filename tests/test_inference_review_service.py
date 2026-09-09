import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.services.inference_review import (
    RANK_KEYPOINT_CONFIDENCE,
    RANK_OVERALL,
    discover_model_inference_sources,
    rank_inference_review_candidates,
    scan_project_inference_quality,
)

POSE_FIELDS = (
    "video_path",
    "model_path",
    "frame_index",
    "time_seconds",
    "detections_in_frame",
    "detection_index",
    "track_id",
    "tracks_in_frame",
    "expected_animal_count",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "kp_nose_x",
    "kp_nose_y",
    "kp_nose_conf",
    "kp_head_x",
    "kp_head_y",
    "kp_head_conf",
)
SEGMENT_FIELDS = (
    "video_path",
    "model_path",
    "frame",
    "det",
    "track_id",
    "tracks_in_frame",
    "expected_animal_count",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "mask_polygon",
)


def _write_output(
    project: Path,
    video_name: str,
    layer_id: str,
    rows: list[dict],
    *,
    run_id: str = "run-1",
    created_at: str = "",
    recorded_video_path: str = "",
) -> Path:
    output_dir = project / "inference outputs" / layer_id
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "keypoints" if layer_id == LAYER_KEYPOINTS else "segmentation"
    csv_path = output_dir / f"{run_id}_{suffix}.csv"
    fields = POSE_FIELDS if layer_id == LAYER_KEYPOINTS else SEGMENT_FIELDS
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    runs_dir = project / "inference outputs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "video_path": recorded_video_path or f"/old/moved/project/{video_name}",
        "created_at": created_at,
        "passes": [],
    }
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["passes"].append(
        {
            "layer_id": layer_id,
            "csv_path": f"/old/project/inference outputs/{layer_id}/{csv_path.name}",
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return csv_path


def _pose_row(frame: int, **updates) -> dict:
    row = {
        "video_path": "/old/session.mp4",
        "model_path": "/models/pose/best.pt",
        "frame_index": frame,
        "time_seconds": frame / 10,
        "detections_in_frame": 1,
        "detection_index": 0,
        "track_id": 1,
        "tracks_in_frame": 1,
        "expected_animal_count": 1,
        "confidence": 0.9,
        "bbox_x1": 10,
        "bbox_y1": 20,
        "bbox_x2": 30,
        "bbox_y2": 50,
        "kp_nose_x": 20,
        "kp_nose_y": 23,
        "kp_nose_conf": 0.9,
        "kp_head_x": 20,
        "kp_head_y": 30,
        "kp_head_conf": 0.9,
    }
    row.update(updates)
    return row


def _segment_row(frame: int, **updates) -> dict:
    row = {
        "video_path": "/old/session.mp4",
        "model_path": "/models/segment/best.pt",
        "frame": frame,
        "det": 0,
        "track_id": 1,
        "tracks_in_frame": 1,
        "expected_animal_count": 1,
        "conf": 0.85,
        "x1": 10,
        "y1": 20,
        "x2": 30,
        "y2": 50,
        "mask_polygon": "[[10, 20], [30, 20], [30, 50], [10, 50]]",
    }
    row.update(updates)
    return row


class InferenceReviewServiceTests(unittest.TestCase):
    def test_discovers_newest_output_for_each_model_on_same_video(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "session.mp4").write_bytes(b"video")
            old_a = _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [_pose_row(0, model_path="/models/a.pt")],
                run_id="a-old",
                created_at="2026-01-01T00:00:00",
            )
            model_b = _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [_pose_row(1, model_path="/models/b.pt")],
                run_id="b-new",
                created_at="2026-01-03T00:00:00",
            )
            new_a = _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [_pose_row(2, model_path="/models/a.pt")],
                run_id="a-new",
                created_at="2026-01-02T00:00:00",
            )

            sources = discover_model_inference_sources(str(project), LAYER_KEYPOINTS)

            self.assertEqual(len(sources), 2)
            by_model = {source.model_path: source for source in sources}
            self.assertEqual(by_model["/models/a.pt"].inference_csv, str(new_a))
            self.assertEqual(by_model["/models/a.pt"].run_id, "a-new")
            self.assertEqual(by_model["/models/b.pt"].inference_csv, str(model_b))
            self.assertNotIn(str(old_a), {source.inference_csv for source in sources})

    def test_candidate_keys_separate_models_and_runs(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "session.mp4").write_bytes(b"video")
            _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [_pose_row(0, model_path="/models/a.pt")],
                run_id="a-run",
                created_at="2026-01-01T00:00:00",
            )
            _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [_pose_row(0, model_path="/models/b.pt")],
                run_id="b-run",
                created_at="2026-01-02T00:00:00",
            )

            result = scan_project_inference_quality(
                str(project), layer_id=LAYER_KEYPOINTS, pool_per_mode_per_video=1
            )

            self.assertEqual(result.frames_scanned, 2)
            self.assertEqual(len(result.candidates), 2)
            self.assertEqual(len({candidate.key for candidate in result.candidates}), 2)
            self.assertEqual(
                {candidate.run_id for candidate in result.candidates}, {"a-run", "b-run"}
            )

    def test_discovers_relocated_pose_output_associated_with_project_video(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            video = project / "videos" / "session.mp4"
            video.write_bytes(b"video")
            missing = {
                "detection_index": -1,
                "detections_in_frame": 0,
                "confidence": "",
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
            }
            inference_csv = _write_output(
                project,
                video.name,
                LAYER_KEYPOINTS,
                [_pose_row(0), _pose_row(1, confidence=0.2), _pose_row(2, **missing)],
            )

            sources = discover_model_inference_sources(str(project), LAYER_KEYPOINTS)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].video_path, str(video))
            self.assertEqual(sources[0].inference_csv, str(inference_csv))
            self.assertEqual(sources[0].model_path, "/models/pose/best.pt")

            result = scan_project_inference_quality(
                str(project), layer_id=LAYER_KEYPOINTS, pool_per_mode_per_video=20
            )
            self.assertEqual(result.videos_with_inference, 1)
            self.assertEqual(result.frames_scanned, 3)
            self.assertEqual(result.warning_frames, 1)
            self.assertEqual(result.bad_frames, 1)
            ranked = rank_inference_review_candidates(
                result.candidates,
                mode=RANK_OVERALL,
                limit=3,
                per_video_limit=3,
                min_frame_spacing=0,
                model_path="/models/pose/best.pt",
            )
            self.assertEqual([candidate.frame_index for candidate in ranked], [2, 1, 0])
            self.assertEqual(ranked[0].reasons, ("missing_detection",))
            self.assertEqual(ranked[1].overlays[0].keypoints[0][0], "head")

    def test_scans_segmentation_output_and_reconstructs_mask(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "session.mp4").write_bytes(b"video")
            _write_output(
                project,
                "session.mp4",
                LAYER_SEGMENTATION,
                [_segment_row(0), _segment_row(1, conf=0.1)],
            )
            result = scan_project_inference_quality(str(project), layer_id=LAYER_SEGMENTATION)
            self.assertEqual(result.layer_id, LAYER_SEGMENTATION)
            self.assertEqual(result.model_paths, ("/models/segment/best.pt",))
            self.assertEqual(len(result.candidates[0].overlays[0].mask_polygon), 4)
            self.assertTrue(any(item.detection_confidence == 0.1 for item in result.candidates))

    def test_ranking_temporally_suppresses_adjacent_pose_frames(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "session.mp4").write_bytes(b"video")
            _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [
                    _pose_row(10, kp_nose_conf=0.1),
                    _pose_row(11, kp_nose_conf=0.05),
                    _pose_row(30, kp_nose_conf=0.3),
                ],
            )
            candidates = scan_project_inference_quality(
                str(project), layer_id=LAYER_KEYPOINTS
            ).candidates
            ranked = rank_inference_review_candidates(
                candidates,
                mode=RANK_KEYPOINT_CONFIDENCE,
                limit=3,
                per_video_limit=3,
                min_frame_spacing=2,
            )
            self.assertEqual([candidate.frame_index for candidate in ranked], [11, 30])

    def test_streaming_scan_flags_track_id_transitions(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "session.mp4").write_bytes(b"video")
            _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [_pose_row(0, track_id=1), _pose_row(1, track_id=2)],
            )

            result = scan_project_inference_quality(
                str(project), layer_id=LAYER_KEYPOINTS, pool_per_mode_per_video=2
            )

            changed = next(
                candidate for candidate in result.candidates if candidate.frame_index == 1
            )
            self.assertIn("track_id_changed", changed.reasons)
            self.assertEqual(changed.status, "warning")
            self.assertEqual(result.warning_frames, 1)

    def test_reports_project_video_without_selected_layer_output(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "missing.mp4").write_bytes(b"video")
            result = scan_project_inference_quality(str(project), layer_id=LAYER_SEGMENTATION)
            self.assertEqual(result.videos_with_inference, 0)
            self.assertIn("segmentation inference output not found", result.issues[0])

    def test_large_scan_retains_only_bounded_ranking_pools(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "videos").mkdir()
            (project / "videos" / "session.mp4").write_bytes(b"video")
            frame_count = 2500
            _write_output(
                project,
                "session.mp4",
                LAYER_KEYPOINTS,
                [
                    _pose_row(
                        frame,
                        confidence=(frame % 100) / 100,
                        kp_nose_conf=((frame * 7) % 100) / 100,
                    )
                    for frame in range(frame_count)
                ],
            )

            result = scan_project_inference_quality(
                str(project), layer_id=LAYER_KEYPOINTS, pool_per_mode_per_video=4
            )

            self.assertEqual(result.frames_scanned, frame_count)
            self.assertLessEqual(len(result.candidates), 12)
            lowest_detection = rank_inference_review_candidates(
                result.candidates,
                mode="pose_confidence",
                limit=1,
                per_video_limit=1,
                min_frame_spacing=0,
            )
            self.assertEqual(lowest_detection[0].detection_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
