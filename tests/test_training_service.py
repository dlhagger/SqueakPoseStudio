import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.services.training import (
    TrainingConfigError,
    TrainingConsoleBuffer,
    build_training_run_plan,
    build_training_worker_config,
    infer_training_task_from_yaml,
    normalize_training_run_label,
    resolve_dataset_yaml,
    resolve_model_config,
    training_run_name,
)


class TrainingServiceTests(unittest.TestCase):
    def test_console_buffer_collapses_carriage_return_progress_updates(self):
        buffer = TrainingConsoleBuffer()

        lines = buffer.feed(
            "Loading model\n\r\x1b[K1/10: 10% ━─────────── 1/10\r"
            "\x1b[K1/10: 100% ━━━━━━━━━━━━ 10/10\nValidation complete\n"
        )

        self.assertEqual(
            lines,
            ["Loading model", "1/10: 100% ━━━━━━━━━━━━ 10/10", "Validation complete"],
        )
        self.assertEqual(buffer.finish(), [])

    def test_dataset_root_and_yaml_file_resolve(self):
        with TemporaryDirectory() as tmp:
            dataset_yaml = Path(tmp, "dataset.yaml")
            dataset_yaml.write_text("task: pose\n", encoding="utf-8")

            self.assertEqual(resolve_dataset_yaml(tmp), str(dataset_yaml))
            self.assertEqual(resolve_dataset_yaml(str(dataset_yaml)), str(dataset_yaml))

    def test_dataset_resolution_reports_stable_error_codes(self):
        with self.assertRaises(TrainingConfigError) as required:
            resolve_dataset_yaml("")
        self.assertEqual(required.exception.code, "required")

        with TemporaryDirectory() as tmp:
            with self.assertRaises(TrainingConfigError) as missing:
                resolve_dataset_yaml(tmp)
            self.assertEqual(missing.exception.code, "yaml_missing")

            absent = os.path.join(tmp, "absent.yaml")
            with self.assertRaises(TrainingConfigError) as not_found:
                resolve_dataset_yaml(absent)
            self.assertEqual(not_found.exception.code, "not_found")

    def test_task_inference_is_tolerant_of_invalid_yaml(self):
        with TemporaryDirectory() as tmp:
            valid = Path(tmp, "valid.yaml")
            valid.write_text("task: segment\n", encoding="utf-8")
            invalid = Path(tmp, "invalid.yaml")
            invalid.write_text("[broken", encoding="utf-8")

            self.assertEqual(infer_training_task_from_yaml(str(valid)), "segment")
            self.assertIsNone(infer_training_task_from_yaml(str(invalid)))
            self.assertIsNone(infer_training_task_from_yaml(str(Path(tmp, "missing.yaml"))))

    def test_model_variants_preserve_existing_resolution_rules(self):
        self.assertEqual(resolve_model_config("yolo26n.yaml", "pose")[0], "yolo26n-pose.yaml")
        self.assertEqual(resolve_model_config("yolo26n.yaml", "segment")[0], "yolo26n-seg.yaml")
        self.assertEqual(resolve_model_config("yolo26n-pose.yaml", "detect")[0], "yolo26n.yaml")
        self.assertEqual(resolve_model_config("custom.pt", None), ("custom.pt", None))

    def test_worker_config_normalizes_layer_and_copies_params(self):
        params = {"epochs": 5}
        config = build_training_worker_config(
            layer_id="pose",
            model_cfg="model.pt",
            params=params,
        )
        params["epochs"] = 10

        self.assertEqual(
            config.as_dict(),
            {"layer_id": "keypoints", "model_cfg": "model.pt", "params": {"epochs": 5}},
        )

    def test_run_plan_resolves_auto_task_and_scratch_model(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp, "dataset.yaml")
            dataset.write_text("task: segment\n", encoding="utf-8")
            plan = build_training_run_plan(
                source_mode="scratch",
                dataset_path=str(dataset),
                base_model_cfg="yolo26n.yaml",
                selected_task="auto",
                default_task="pose",
                device="cuda",
                epochs=12,
                batch=0,
                project_runs_dir=os.path.join(tmp, "runs"),
            )

            self.assertEqual(plan.task, "segment")
            self.assertEqual(plan.model_cfg, "yolo26n-seg.yaml")
            self.assertEqual(plan.params["batch"], -1)
            self.assertEqual(plan.params["data"], str(dataset))
            self.assertTrue(plan.params["project"].endswith(os.path.join("train", "segment")))

    def test_run_plan_preserves_checkpoint_and_exact_resume_modes(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp, "dataset.yaml")
            dataset.write_text("task: pose\n", encoding="utf-8")
            checkpoint = Path(tmp, "runs", "train", "mouse", "weights", "best.pt")
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            continued = build_training_run_plan(
                source_mode="checkpoint",
                dataset_path=str(dataset),
                base_model_cfg="ignored.yaml",
                checkpoint_path=str(checkpoint),
                selected_task="pose",
                device="cpu",
                epochs=2,
                batch=4,
                project_runs_dir=os.path.join(tmp, "project-runs"),
            )
            self.assertEqual(continued.model_cfg, str(checkpoint))
            self.assertEqual(continued.params["name"], "mouse_continue")

            last = checkpoint.with_name("last.pt")
            last.write_bytes(b"checkpoint")
            resumed = build_training_run_plan(
                source_mode="resume",
                dataset_path="",
                base_model_cfg="ignored.yaml",
                checkpoint_path=str(last),
                device="mps",
                epochs=99,
                batch=0,
                project_runs_dir=tmp,
            )
            self.assertIsNone(resumed.dataset_yaml)
            self.assertEqual(resumed.params, {"resume": True, "device": "mps"})

    def test_run_plan_passes_a_safe_custom_ultralytics_name(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp, "dataset.yaml")
            dataset.write_text("task: segment\n", encoding="utf-8")

            plan = build_training_run_plan(
                source_mode="scratch",
                dataset_path=str(dataset),
                base_model_cfg="yolo26n.yaml",
                selected_task="segment",
                device="mps",
                epochs=20,
                batch=4,
                project_runs_dir=os.path.join(tmp, "runs"),
                run_name="Baseline masks / August #1",
            )

            self.assertEqual(plan.params["name"], "Baseline_masks_August_1")
            self.assertFalse(plan.params["exist_ok"])

    def test_run_plan_reports_task_resume_and_mps_errors(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp, "dataset.yaml")
            dataset.write_text("task: pose\n", encoding="utf-8")
            checkpoint = Path(tmp, "best.pt")
            checkpoint.write_bytes(b"checkpoint")
            common = dict(
                dataset_path=str(dataset),
                base_model_cfg="yolo26n.yaml",
                device="cpu",
                epochs=1,
                batch=1,
                project_runs_dir=tmp,
            )
            with self.assertRaises(TrainingConfigError) as mismatch:
                build_training_run_plan(source_mode="scratch", selected_task="segment", **common)
            self.assertEqual(mismatch.exception.code, "task_mismatch")

            with self.assertRaises(TrainingConfigError) as resume:
                build_training_run_plan(
                    source_mode="resume", checkpoint_path=str(checkpoint), **common
                )
            self.assertEqual(resume.exception.code, "resume_checkpoint")

            with self.assertRaises(TrainingConfigError) as mps:
                build_training_run_plan(
                    source_mode="scratch",
                    selected_task="pose",
                    **{**common, "device": "mps", "batch": 0},
                )
            self.assertEqual(mps.exception.code, "mps_batch")

    def test_training_run_name_sanitizes_worker_output_name(self):
        self.assertEqual(training_run_name("/tmp/My model.yaml"), "My_model")
        self.assertEqual(training_run_name(""), "model")
        self.assertEqual(normalize_training_run_label("Study / cohort #2"), "Study_cohort_2")


if __name__ == "__main__":
    unittest.main()
