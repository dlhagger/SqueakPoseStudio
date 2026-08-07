import os
import unittest
from tempfile import TemporaryDirectory

from dataset_ops import DATASET_DETECT, dataset_export_paths
from squeakpose.services.dataset import export_dataset_transaction


class DatasetServiceTests(unittest.TestCase):
    def _source_project(self, root: str) -> tuple[str, str]:
        images = os.path.join(root, "images_all")
        labels = os.path.join(root, "labels_all")
        os.makedirs(images)
        os.makedirs(labels)
        with open(os.path.join(images, "frame.jpg"), "wb") as fh:
            fh.write(b"image")
        with open(os.path.join(labels, "frame.txt"), "w", encoding="utf-8") as fh:
            fh.write("0 0.5 0.5 0.2 0.2 0.1 0.1 2\n")
        return images, labels

    def test_transaction_installs_complete_detection_dataset(self):
        with TemporaryDirectory() as tmp:
            images, labels = self._source_project(tmp)
            paths = dataset_export_paths(tmp, DATASET_DETECT)

            result = export_dataset_transaction(
                images_all_dir=images,
                labels_all_dir=labels,
                final_paths=paths,
                train_images=["frame.jpg"],
                val_images=[],
                mode=DATASET_DETECT,
                classes=["mouse"],
                keypoint_names=["nose"],
                split_seed=17,
            )

            self.assertEqual(result.split_seed, 17)
            self.assertTrue(os.path.isfile(paths.dataset_yaml_path))
            self.assertTrue(os.path.isfile(os.path.join(paths.images_train_dir, "frame.jpg")))
            with open(
                os.path.join(paths.labels_train_dir, "frame.txt"), "r", encoding="utf-8"
            ) as fh:
                self.assertEqual(fh.read(), "0 0.5 0.5 0.2 0.2\n")

    def test_failed_install_leaves_existing_dataset_and_removes_staging(self):
        with TemporaryDirectory() as tmp:
            images, labels = self._source_project(tmp)
            paths = dataset_export_paths(tmp, DATASET_DETECT)
            os.makedirs(paths.base_dir, exist_ok=True)
            with open(paths.dataset_yaml_path, "w", encoding="utf-8") as fh:
                fh.write("old yaml")

            def fail(_replacements):
                raise OSError("injected install failure")

            with self.assertRaises(OSError):
                export_dataset_transaction(
                    images_all_dir=images,
                    labels_all_dir=labels,
                    final_paths=paths,
                    train_images=["frame.jpg"],
                    val_images=[],
                    mode=DATASET_DETECT,
                    classes=["mouse"],
                    keypoint_names=["nose"],
                    split_seed=0,
                    committer=fail,
                )

            with open(paths.dataset_yaml_path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "old yaml")
            staging = [
                name
                for name in os.listdir(os.path.dirname(paths.base_dir))
                if name.startswith(".detect-export-")
            ]
            self.assertEqual(staging, [])


if __name__ == "__main__":
    unittest.main()
