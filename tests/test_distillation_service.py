import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.services.distillation import (
    DistillationPlanError,
    build_distillation_corpus,
    build_distillation_run_plan,
    count_distillation_images,
    distillation_frame_filename,
    distillation_sample_count,
    plan_distillation_corpus,
    student_task_mismatch,
)


class FakeVideoReader:
    def __init__(self, frames):
        self.frames = frames
        self.requested = []
        self.closed = False

    def read_frame(self, frame_index):
        self.requested.append(frame_index)
        return self.frames.get(frame_index)

    def close(self):
        self.closed = True


class DistillationServiceTests(unittest.TestCase):
    def test_sample_count_normalizes_stride_and_cap(self):
        self.assertEqual(distillation_sample_count(61, 30), 3)
        self.assertEqual(distillation_sample_count(100, 0), 100)
        self.assertEqual(distillation_sample_count(1000, 30, 10), 10)

    def test_corpus_plan_normalizes_stride_cap_and_paths(self):
        plan = plan_distillation_corpus(
            [("relative.mp4", 61), ("empty.mp4", -4)],
            stride=0,
            maximum_per_video=10,
        )

        self.assertEqual(plan.stride, 1)
        self.assertEqual(plan.maximum_per_video, 10)
        self.assertEqual(plan.estimated_samples, 10)
        self.assertEqual(plan.videos[0][1:], (61, 10))
        self.assertTrue(os.path.isabs(plan.videos[0][0]))
        self.assertEqual(plan.videos[1][1:], (0, 0))

    def test_image_count_is_recursive_and_extension_tolerant(self):
        with TemporaryDirectory() as tmp:
            nested = Path(tmp, "nested")
            nested.mkdir()
            Path(tmp, "one.JPG").write_bytes(b"x")
            Path(nested, "two.png").write_bytes(b"x")
            Path(nested, "note.txt").write_text("no", encoding="utf-8")

            self.assertEqual(count_distillation_images(tmp), 2)
            self.assertEqual(count_distillation_images(os.path.join(tmp, "missing")), 0)

    def test_corpus_builder_preserves_names_skips_and_atomic_writes(self):
        with TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source video.mp4")
            plan = plan_distillation_corpus([(source, 7)], stride=2)
            existing_name = distillation_frame_filename(source, 2)
            Path(tmp, existing_name).write_bytes(b"existing")
            reader = FakeVideoReader({0: b"zero", 4: None, 6: b"six"})
            qualities = []
            progress = []

            def write_image(path, frame, quality):
                qualities.append(quality)
                Path(path).write_bytes(frame)
                return True

            result = build_distillation_corpus(
                plan,
                data_dir=tmp,
                jpeg_quality=91,
                open_video=lambda _path: reader,
                write_image=write_image,
                on_progress=progress.append,
            )

            self.assertEqual((result.saved, result.skipped, result.failed), (2, 1, 1))
            self.assertFalse(result.canceled)
            self.assertEqual(result.handled, 4)
            self.assertEqual(reader.requested, [0, 4, 6])
            self.assertTrue(reader.closed)
            self.assertEqual(qualities, [91, 91])
            self.assertEqual([item.frame_index for item in progress], [0, 2, 4, 6])
            self.assertEqual(
                Path(tmp, distillation_frame_filename(source, 0)).read_bytes(),
                b"zero",
            )
            self.assertEqual(
                Path(tmp, distillation_frame_filename(source, 6)).read_bytes(),
                b"six",
            )
            self.assertIn("source video.mp4 frame 4: read failed", result.failures)

    def test_corpus_builder_partial_cancel_keeps_completed_images(self):
        with TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "video.mp4")
            plan = plan_distillation_corpus([(source, 10)], stride=1, maximum_per_video=5)
            reader = FakeVideoReader({index: str(index).encode() for index in range(5)})
            progress = []

            def write_image(path, frame, _quality):
                Path(path).write_bytes(frame)
                return True

            result = build_distillation_corpus(
                plan,
                data_dir=tmp,
                jpeg_quality=80,
                open_video=lambda _path: reader,
                write_image=write_image,
                is_canceled=lambda: len(progress) >= 2,
                on_progress=progress.append,
            )

            self.assertTrue(result.canceled)
            self.assertEqual((result.saved, result.handled), (2, 2))
            self.assertEqual(reader.requested, [0, 1])
            self.assertTrue(reader.closed)
            self.assertTrue(Path(tmp, distillation_frame_filename(source, 0)).is_file())
            self.assertTrue(Path(tmp, distillation_frame_filename(source, 1)).is_file())

    def test_failed_writer_cleans_staged_image(self):
        with TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "video.mp4")
            plan = plan_distillation_corpus([(source, 1)], stride=1)
            reader = FakeVideoReader({0: b"frame"})

            def fail_after_write(path, frame, _quality):
                Path(path).write_bytes(frame)
                return False

            result = build_distillation_corpus(
                plan,
                data_dir=tmp,
                jpeg_quality=95,
                open_video=lambda _path: reader,
                write_image=fail_after_write,
            )

            self.assertEqual((result.saved, result.failed), (0, 1))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_model_task_check_preserves_conventional_filename_rules(self):
        self.assertTrue(student_task_mismatch("yolo26s-pose.pt", "segment"))
        self.assertTrue(student_task_mismatch("yolo26s-seg.pt", "pose"))
        self.assertFalse(student_task_mismatch("custom.pt", "pose"))

    def test_run_plan_preserves_exact_worker_cli(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "images"
            data_dir.mkdir()
            (data_dir / "image.jpg").write_bytes(b"x")
            script = root / "distiller.py"
            script.write_text("", encoding="utf-8")
            runs = root / "runs"

            plan = build_distillation_run_plan(
                program=sys.executable,
                script_path=str(script),
                app_base_dir=tmp,
                project_root=tmp,
                runs_root=str(runs),
                data_dir=str(data_dir),
                run_name="my-run",
                student="ultralytics/yolo26s-seg.pt",
                teacher="dinov3/vitb16",
                task="segmentation",
                epochs=12,
                batch_size=8,
                precision="bf16-mixed",
                overwrite=True,
            )

            self.assertEqual(plan.task, "segment")
            self.assertEqual(plan.task_label, "Segmentation")
            self.assertEqual(plan.output_dir, str(runs / "my-run"))
            self.assertEqual(plan.image_count, 1)
            self.assertEqual(
                list(plan.arguments),
                [
                    "-u",
                    str(script),
                    "--project-root",
                    tmp,
                    "--data-dir",
                    str(data_dir),
                    "--run-name",
                    "my-run",
                    "--model",
                    "ultralytics/yolo26s-seg.pt",
                    "--task",
                    "segment",
                    "--teacher",
                    "dinov3/vitb16",
                    "--epochs",
                    "12",
                    "--batch-size",
                    "8",
                    "--precision",
                    "bf16-mixed",
                    "--overwrite",
                ],
            )

    def test_validation_reports_stable_errors(self):
        with TemporaryDirectory() as tmp:
            script = Path(tmp, "distiller.py")
            script.write_text("", encoding="utf-8")
            common = dict(
                program=sys.executable,
                script_path=str(script),
                app_base_dir=tmp,
                project_root=tmp,
                runs_root=os.path.join(tmp, "runs"),
                data_dir=os.path.join(tmp, "missing"),
                run_name="run",
                student="model.pt",
                teacher="teacher",
                task="pose",
                epochs=1,
                batch_size=1,
                precision="32-true",
            )
            with self.assertRaises(DistillationPlanError) as missing:
                build_distillation_run_plan(**common)
            self.assertEqual(missing.exception.code, "corpus_required")

            data_dir = Path(tmp, "images")
            data_dir.mkdir()
            Path(data_dir, "image.jpg").write_bytes(b"x")
            common["data_dir"] = str(data_dir)
            common["run_name"] = "bad name"
            with self.assertRaises(DistillationPlanError) as invalid:
                build_distillation_run_plan(**common)
            self.assertEqual(invalid.exception.code, "invalid_run_name")

            common["run_name"] = "run"
            common["student"] = "model-pose.pt"
            common["task"] = "segment"
            with self.assertRaises(DistillationPlanError) as mismatch:
                build_distillation_run_plan(**common)
            self.assertEqual(mismatch.exception.code, "student_task_mismatch")


if __name__ == "__main__":
    unittest.main()
