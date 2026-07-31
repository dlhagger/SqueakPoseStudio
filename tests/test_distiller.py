import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


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
            self.assertTrue(calls[0]["overwrite"])
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
            self.assertEqual(config["model"], "ultralytics/yolo26s-seg.pt")
            self.assertEqual(config["run_name"], "dinov3-segmentation")
            self.assertEqual(
                config["out"],
                os.path.join(tmp, "runs", "distillation", "dinov3-segmentation"),
            )

    def test_run_distillation_requires_existing_image_directory(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.distiller.run_distillation(project_root=tmp)


if __name__ == "__main__":
    unittest.main()
