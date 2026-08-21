import datetime
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.project.layers import LAYER_DEPTH, LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.project.safety import ProjectPathError
from squeakpose.services.inference import (
    InferenceRunAccumulator,
    aggregate_inference_result,
    configured_inference_layers,
    create_inference_run_id,
    finalize_inference_run,
    plan_inference_run,
    prepare_inference_run,
    project_video_inference_statuses,
    video_identity,
)


class InferenceServiceTests(unittest.TestCase):
    def test_project_video_statuses_combine_successful_layers_and_ignore_bad_manifests(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            videos = root / "videos"
            videos.mkdir()
            link = videos / "session.mp4"
            link.symlink_to(source)
            runs = root / "inference outputs" / "runs"
            runs.mkdir(parents=True)
            (runs / "first.json").write_text(
                json.dumps(
                    {
                        "video_path": str(link),
                        "created_at": "2026-08-19T10:00:00",
                        "passes": [
                            {"layer_id": "keypoints", "had_error": False, "canceled": False},
                            {"layer_id": "depth", "had_error": True, "canceled": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (runs / "second.json").write_text(
                json.dumps(
                    {
                        "video_path": str(source),
                        "created_at": "2026-08-20T12:00:00",
                        "passes": [
                            {"layer_id": "segmentation", "had_error": False, "canceled": False}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (runs / "damaged.json").write_text("not json", encoding="utf-8")

            statuses = project_video_inference_statuses(tmp)

            status = statuses[video_identity(str(link))]
            self.assertEqual(status.successful_layers, ("keypoints", "segmentation"))
            self.assertEqual(status.latest_created_at, "2026-08-20T12:00:00")
            self.assertEqual(status.run_count, 2)

    def test_configured_layers_are_active_first_and_deduplicated(self):
        layers = configured_inference_layers(
            "segment",
            {
                "keypoints": "pose.pt",
                "segmentation": "segment.pt",
                "depth": "",
                "unknown": "ignored.pt",
            },
        )

        self.assertEqual(layers, (LAYER_SEGMENTATION, LAYER_KEYPOINTS))

    def test_plan_builds_contained_per_layer_worker_payloads(self):
        with TemporaryDirectory() as tmp:
            created_at = datetime.datetime(2026, 8, 15, 12, 34, 56)
            plan = plan_inference_run(
                project_root=tmp,
                video_path="/videos/mouse session.mp4",
                active_layer=LAYER_SEGMENTATION,
                model_paths={
                    LAYER_KEYPOINTS: "pose.pt",
                    LAYER_SEGMENTATION: "segment.pt",
                    LAYER_DEPTH: "depth.pt",
                },
                pose_classes=["mouse"],
                segmentation_classes=["animal", "object"],
                keypoint_names=["nose", "tail"],
                device="mps",
                batch_size=8,
                total_frames=120,
                fps=30.0,
                created_at=created_at,
                run_id="mouse_session_20260815-123456_test",
            )

            self.assertEqual(
                tuple(job.layer_id for job in plan.jobs),
                (LAYER_SEGMENTATION, LAYER_KEYPOINTS, LAYER_DEPTH),
            )
            self.assertEqual([job.job_index for job in plan.jobs], [1, 2, 3])
            self.assertTrue(plan.manifest_path.endswith(f"runs{os.sep}{plan.run_id}.json"))
            for job in plan.jobs:
                self.assertEqual(os.path.commonpath((tmp, job.csv_path)), os.path.abspath(tmp))
                self.assertEqual(job.worker_config()["kp_names"], list(job.keypoint_names))
                self.assertEqual(job.worker_config()["mode"], job.workflow)
            self.assertEqual(plan.jobs[0].classes, ("animal", "object"))
            self.assertEqual(plan.jobs[1].classes, ("mouse",))
            self.assertEqual(plan.jobs[1].keypoint_names, ("nose", "tail"))
            self.assertEqual(plan.jobs[2].classes, ())
            self.assertTrue(plan.jobs[2].preview_path.endswith("_depth_preview.mp4"))
            self.assertFalse(os.path.exists(os.path.join(tmp, "inference outputs")))

            prepare_inference_run(plan)

            self.assertTrue(all(os.path.isdir(path) for path in plan.output_directories))

    def test_run_ids_are_safe_and_invalid_explicit_ids_are_rejected(self):
        run_id = create_inference_run_id(
            "/videos/mouse session!.mp4",
            created_at=datetime.datetime(2026, 8, 15, 1, 2, 3),
            token="test-token",
        )
        self.assertEqual(run_id, "mouse_session_20260815-010203_testtoken")

        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plan_inference_run(
                    project_root=tmp,
                    video_path="video.mp4",
                    active_layer=LAYER_KEYPOINTS,
                    model_paths={LAYER_KEYPOINTS: "pose.pt"},
                    run_id="../escape",
                )

    def test_planning_rejects_symlinked_output_root_that_escapes_project(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            output_root = Path(tmp) / "inference outputs"
            try:
                output_root.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaises(ProjectPathError):
                self._plan(tmp, layers=(LAYER_KEYPOINTS,))

    def test_partial_canceled_result_retains_rows_and_planned_paths(self):
        with TemporaryDirectory() as tmp:
            plan = self._plan(tmp, layers=(LAYER_KEYPOINTS,))
            job = plan.jobs[0]

            result = aggregate_inference_result(
                job,
                {
                    "event": "result",
                    "rows_written": 17,
                    "processed_frames": 9,
                    "canceled": True,
                },
                project_root=tmp,
                exit_code=130,
                crashed=True,
                cancel_requested=True,
            )

            self.assertTrue(result.canceled)
            self.assertFalse(result.had_error)
            self.assertEqual(result.rows_written, 17)
            self.assertEqual(result.processed_frames, 9)
            self.assertEqual(result.csv_path, job.csv_path)
            self.assertEqual(result.discard_paths, ())

    def test_failed_empty_result_identifies_discardable_outputs(self):
        with TemporaryDirectory() as tmp:
            plan = self._plan(tmp, layers=(LAYER_DEPTH,))
            job = plan.jobs[0]

            result = aggregate_inference_result(
                job,
                None,
                project_root=tmp,
                exit_code=1,
                crashed=True,
                stderr="model failed",
            )

            self.assertTrue(result.had_error)
            self.assertEqual(result.error_message, "model failed")
            self.assertEqual(result.discard_paths, (job.csv_path, job.preview_path))

    def test_unsafe_worker_output_is_rejected_in_favor_of_plan(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            plan = self._plan(tmp, layers=(LAYER_KEYPOINTS,))
            job = plan.jobs[0]

            result = aggregate_inference_result(
                job,
                {
                    "event": "result",
                    "csv_path": os.path.join(outside, "stolen.csv"),
                    "rows_written": 1,
                },
                project_root=tmp,
            )

            self.assertEqual(result.csv_path, job.csv_path)
            self.assertTrue(result.had_error)
            self.assertIn("escapes the project root", result.error_message)

    def test_accumulator_writes_manifest_and_returns_structured_summary(self):
        with TemporaryDirectory() as tmp:
            plan = self._plan(tmp, layers=(LAYER_KEYPOINTS, LAYER_SEGMENTATION))
            accumulator = InferenceRunAccumulator(plan)
            first = accumulator.record(
                plan.jobs[0],
                {"event": "result", "rows_written": 12, "processed_frames": 6},
            )
            second = accumulator.record(
                plan.jobs[1],
                {"event": "error", "error_message": "bad model"},
                exit_code=1,
            )

            self.assertFalse(first.had_error)
            self.assertTrue(second.had_error)
            self.assertEqual(accumulator.pending_jobs, ())
            with self.assertRaises(ValueError):
                accumulator.record(plan.jobs[0], {"event": "result"})

            summary = accumulator.finalize()

            self.assertEqual(summary.successful_count, 1)
            self.assertEqual(summary.failed_count, 1)
            self.assertFalse(summary.canceled)
            self.assertTrue(any("bad model" in detail for detail in summary.details))
            manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["run_id"], plan.run_id)
            self.assertEqual(len(manifest["passes"]), 2)

    def test_manifest_write_failure_is_returned_not_raised(self):
        with TemporaryDirectory() as tmp:
            plan = self._plan(tmp, layers=(LAYER_KEYPOINTS,))

            def fail_writer(_path, _text):
                raise OSError("disk full")

            summary = finalize_inference_run(plan, (), writer=fail_writer)

            self.assertEqual(summary.manifest_path, "")
            self.assertEqual(summary.manifest_error, "disk full")
            self.assertIn("Run manifest failed", summary.details[-1])

    @staticmethod
    def _plan(tmp: str, *, layers: tuple[str, ...]):
        models = {layer_id: f"{layer_id}.pt" for layer_id in layers}
        return plan_inference_run(
            project_root=tmp,
            video_path="/videos/mouse.mp4",
            active_layer=layers[0],
            model_paths=models,
            run_id="mouse_20260815-120000_test",
        )


if __name__ == "__main__":
    unittest.main()
