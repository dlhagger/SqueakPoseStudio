import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset_builder import create_dataset_yaml
import yaml


class DatasetBuilderTests(unittest.TestCase):
    def test_create_dataset_yaml_writes_expected_pose_metadata(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for rel in (
                ("images", "train"),
                ("images", "val"),
                ("labels", "train"),
                ("labels", "val"),
            ):
                (base.joinpath(*rel)).mkdir(parents=True, exist_ok=True)

            out_path = create_dataset_yaml(
                base_dir=str(base),
                class_names=["mouse"],
                kp_names=["nose", "left_ear", "right_ear"],
                verbose=False,
            )

            self.assertEqual(out_path, str(base / "dataset.yaml"))
            with open(out_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)

            self.assertEqual(data["names"], ["mouse"])
            self.assertEqual(data["kpt_shape"], [3, 3])
            self.assertEqual(data["kp_names"], ["nose", "left_ear", "right_ear"])
            self.assertEqual(data["flip_idx"], [0, 2, 1])

    def test_create_dataset_yaml_requires_all_split_directories(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "images" / "train").mkdir(parents=True, exist_ok=True)
            (base / "images" / "val").mkdir(parents=True, exist_ok=True)
            # Intentionally omit labels/train and labels/val.

            with self.assertRaises(FileNotFoundError):
                create_dataset_yaml(
                    base_dir=str(base),
                    class_names=["mouse"],
                    kp_names=["nose"],
                    verbose=False,
                )

    def test_create_dataset_yaml_can_reference_final_path_from_staging_directory(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for rel in (
                ("images", "train"),
                ("images", "val"),
                ("labels", "train"),
                ("labels", "val"),
            ):
                (base.joinpath(*rel)).mkdir(parents=True, exist_ok=True)

            out_path = create_dataset_yaml(
                base_dir=str(base),
                class_names=["mouse"],
                kp_names=["nose"],
                verbose=False,
                dataset_path="/final/project/datasets/pose",
            )

            data = yaml.safe_load(Path(out_path).read_text(encoding="utf-8"))
            self.assertEqual(data["path"], "/final/project/datasets/pose")


if __name__ == "__main__":
    unittest.main()
