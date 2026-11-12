import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from squeakpose.project.distillation import discover_distillation_exports, distillation_run_task

DISTILLER_PATH = Path(__file__).resolve().parents[1] / "distillation" / "distiller.py"


def _load_distiller_module():
    spec = importlib.util.spec_from_file_location("test_distiller_module", DISTILLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DistillerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distiller = _load_distiller_module()

    def test_build_run_config_uses_project_default_paths(self):
        with TemporaryDirectory() as tmp:
            config = self.distiller.build_run_config(project_root=tmp)

            self.assertEqual(
                config["data"],
                os.path.join(tmp, "distillation", "unlabeled_images"),
            )
            self.assertEqual(
                config["out"],
                os.path.join(tmp, "runs", "distillation", self.distiller.DEFAULT_RUN_NAME),
            )
            self.assertFalse(config["overwrite"])

    def test_run_distillation_passes_project_aware_paths_to_lightly_train(self):
        with TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "external_images")
            out_dir = os.path.join(tmp, "runs", "distillation", "custom-run")
            os.makedirs(data_dir, exist_ok=True)

            calls = []

            class FakeLightlyTrain:
                @staticmethod
                def pretrain(**kwargs):
                    calls.append(kwargs)

            with patch.dict(sys.modules, {"lightly_train": FakeLightlyTrain}):
                config = self.distiller.run_distillation(
                    project_root=tmp,
                    data_dir=data_dir,
                    out_dir=out_dir,
                    teacher="dinov3/vitl16",
                    epochs=12,
                    batch_size=8,
                    precision="16-mixed",
                    overwrite=True,
                )

            self.assertEqual(config["out"], out_dir)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["out"], out_dir)
            self.assertEqual(calls[0]["data"], data_dir)
            self.assertEqual(calls[0]["method"], "distillation")
            self.assertEqual(calls[0]["method_args"], {"teacher": "dinov3/vitl16"})
            self.assertEqual(calls[0]["epochs"], 12)
            self.assertEqual(calls[0]["batch_size"], 8)
            self.assertEqual(calls[0]["precision"], "16-mixed")
            self.assertFalse(calls[0]["overwrite"])
            with open(
                os.path.join(out_dir, self.distiller.DISTILLATION_MANIFEST_FILENAME),
                "r",
                encoding="utf-8",
            ) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["task"], "pose")

    def test_segmentation_task_selects_segmentation_student_and_run_name(self):
        with TemporaryDirectory() as tmp:
            config = self.distiller.build_run_config(
                project_root=tmp,
                task="segmentation",
            )

            self.assertEqual(config["task"], "segment")
            self.assertEqual(config["model"], "ultralytics/yolo26n-seg.pt")
            self.assertEqual(config["run_name"], "dinov3-segmentation")
            self.assertEqual(
                config["out"],
                os.path.join(tmp, "runs", "distillation", "dinov3-segmentation"),
            )

    def test_run_distillation_requires_existing_image_directory(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.distiller.run_distillation(project_root=tmp)

    def test_interrupted_export_is_discoverable_and_resume_reuses_settings(self):
        with TemporaryDirectory() as tmp:
            data = Path(tmp, "images")
            data.mkdir()
            out = Path(tmp, "custom-run")

            def interrupted(**kwargs):
                self.assertFalse(out.exists())  # Metadata must not block Lightly startup.
                (out / "exported_models").mkdir(parents=True)
                (out / "exported_models" / "exported_last.pt").write_bytes(b"export")
                (out / "checkpoints").mkdir()
                (out / "checkpoints" / "last.ckpt").write_bytes(b"checkpoint")
                raise KeyboardInterrupt()

            fake = Mock()
            fake.pretrain.side_effect = interrupted
            with patch.dict(sys.modules, {"lightly_train": fake}):
                with self.assertRaises(KeyboardInterrupt):
                    self.distiller.run_distillation(
                        project_root=tmp,
                        data_dir=str(data),
                        out_dir=str(out),
                        task="segment",
                        epochs=17,
                        batch_size=4,
                    )
                self.assertEqual(distillation_run_task(str(out)), "segment")
                self.assertEqual(
                    len(discover_distillation_exports([("Runs", tmp)], task="segment")), 1
                )
                fake.pretrain.side_effect = None
                self.distiller.run_distillation(
                    project_root=tmp,
                    out_dir=str(out),
                    resume_interrupted=True,
                    task="pose",
                    epochs=99,
                    batch_size=99,
                )
            args = fake.pretrain.call_args.kwargs
            self.assertTrue(args["resume_interrupted"])
            self.assertEqual(args["epochs"], 17)
            self.assertEqual(args["batch_size"], 4)
            self.assertEqual(args["data"], str(data))
            self.assertEqual(args["model"], "ultralytics/yolo26n-seg.pt")
            self.assertFalse(Path(str(out) + ".squeakpose.json").exists())

    def test_overwrite_preserves_old_export_and_never_mislabels_it(self):
        with TemporaryDirectory() as tmp:
            data = Path(tmp, "images")
            data.mkdir()
            out = Path(tmp, "custom-run")
            old_config = self.distiller.build_run_config(project_root=tmp, out_dir=str(out))
            self.distiller.write_run_manifest(old_config)
            exported = out / "exported_models"
            exported.mkdir()
            (exported / "exported_last.pt").write_bytes(b"old pose")
            fake = Mock()
            fake.pretrain.side_effect = RuntimeError("failed before first export")
            with patch.dict(sys.modules, {"lightly_train": fake}):
                with self.assertRaises(RuntimeError):
                    self.distiller.run_distillation(
                        project_root=tmp,
                        data_dir=str(data),
                        out_dir=str(out),
                        task="segment",
                        overwrite=True,
                    )
            self.assertEqual(discover_distillation_exports([("Runs", tmp)], task="segment"), [])
            backups = discover_distillation_exports([("Runs", tmp)], task="pose")
            self.assertEqual(len(backups), 1)
            self.assertEqual(Path(backups[0][1]).read_bytes(), b"old pose")

    def test_local_checkpoint_is_loaded_as_yolo_and_task_checked(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp, "custom.pt")
            checkpoint.write_bytes(b"test")
            model = Mock(task="pose")
            yolo = Mock(return_value=model)
            fake = Mock()
            with patch.dict(sys.modules, {"lightly_train": fake, "ultralytics": Mock(YOLO=yolo)}):
                self.distiller.run_distillation(
                    project_root=tmp, data_dir=tmp, model=str(checkpoint)
                )
            yolo.assert_called_once_with(os.path.abspath(checkpoint))
            self.assertIs(fake.pretrain.call_args.kwargs["model"], model)
            self.assertEqual(fake.pretrain.call_args.kwargs["precision"], "auto")
            self.assertEqual(fake.pretrain.call_args.kwargs["gradient_accumulation_steps"], 2)

    def test_distributed_child_leaves_metadata_and_exports_owned_by_parent(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp, "run")
            out.mkdir()
            config = self.distiller.build_run_config(
                project_root=tmp, data_dir=tmp, out_dir=str(out)
            )
            manifest = Path(self.distiller.write_run_manifest(config, pending=True))
            original = manifest.read_bytes()
            fake = Mock()
            with (
                patch.dict(sys.modules, {"lightly_train": fake}),
                patch.dict(os.environ, {"LOCAL_RANK": "1"}),
            ):
                self.distiller.run_distillation(project_root=tmp, data_dir=tmp, out_dir=str(out))
            self.assertTrue(fake.pretrain.call_args.kwargs["overwrite"])
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
