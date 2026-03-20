import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset_builder import create_dataset_yaml


class DatasetBuilderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
