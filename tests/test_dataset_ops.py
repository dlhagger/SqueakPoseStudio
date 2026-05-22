import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from dataset_ops import (
    DATASET_DETECT,
    DATASET_POSE,
    DATASET_SEGMENT,
    dataset_export_paths,
    export_dataset_files,
    format_dataset_export_summary,
    normalize_label_directory,
    split_train_val_images,
    write_dataset_yaml_for_mode,
)


class DatasetOpsTests(unittest.TestCase):
    def test_split_train_val_keeps_nonempty_val_when_possible(self):
        train, val = split_train_val_images(["a.jpg", "b.jpg"], 0.95)
        self.assertEqual(train, ["a.jpg"])
        self.assertEqual(val, ["b.jpg"])

        train, val = split_train_val_images(["a.jpg"], 0.1)
        self.assertEqual(train, ["a.jpg"])
        self.assertEqual(val, [])

    def test_export_detection_dataset_converts_pose_rows_to_bbox_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_all = root / "images_all"
            labels_all = root / "labels_all"
            images_all.mkdir()
            labels_all.mkdir()
            (images_all / "frame1.jpg").write_bytes(b"fake-image")
            (images_all / "frame2.jpg").write_bytes(b"fake-image")
            (labels_all / "frame1.txt").write_text(
                "0 0.5 0.5 0.2 0.2 0.1 0.1 2\n"
                "bad row\n",
                encoding="utf-8",
            )

            paths = dataset_export_paths(str(root), DATASET_DETECT)
            result = export_dataset_files(
                images_all_dir=str(images_all),
                labels_all_dir=str(labels_all),
                paths=paths,
                train_images=["frame1.jpg"],
                val_images=["frame2.jpg"],
                mode=DATASET_DETECT,
            )
            yaml_path = write_dataset_yaml_for_mode(
                paths.base_dir,
                DATASET_DETECT,
                ["mouse"],
                ["nose"],
                verbose=False,
            )

            self.assertFalse(result.canceled)
            self.assertEqual(result.processed, 2)
            self.assertTrue(any("insufficient columns" in msg for msg in result.warnings))
            self.assertTrue(any("frame2.txt: missing" in msg for msg in result.warnings))
            label_out = Path(paths.labels_train_dir) / "frame1.txt"
            self.assertEqual(label_out.read_text(encoding="utf-8"), "0 0.5 0.5 0.2 0.2\n")
            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
            self.assertEqual(data["names"], ["mouse"])
            self.assertNotIn("task", data)

    def test_dataset_export_summary_includes_split_seed(self):
        with TemporaryDirectory() as tmp:
            paths = dataset_export_paths(tmp, DATASET_POSE)
            result = export_dataset_files(
                images_all_dir=os.path.join(tmp, "missing-images"),
                labels_all_dir=os.path.join(tmp, "missing-labels"),
                paths=paths,
                train_images=[],
                val_images=[],
                mode=DATASET_POSE,
            )
            result.split_seed = 42

            summary = format_dataset_export_summary(result)

        self.assertIn("Split seed: 42", summary)

    def test_write_segmentation_dataset_yaml_sets_task(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = dataset_export_paths(str(root), DATASET_SEGMENT)
            for directory in paths.split_dirs:
                Path(directory).mkdir(parents=True, exist_ok=True)

            yaml_path = write_dataset_yaml_for_mode(
                paths.base_dir,
                DATASET_SEGMENT,
                ["mouse"],
                [],
                verbose=False,
            )

            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
            self.assertEqual(data["task"], "segment")
            self.assertEqual(data["nc"], 1)

    def test_normalize_pose_directory_backs_up_labels_and_copies_missing_image(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels_all"
            images_all = root / "images_all"
            images_to_label = root / "images_to_label"
            labels.mkdir()
            images_all.mkdir()
            images_to_label.mkdir()
            original = "0 1.2 -0.1 0.2 0.2 0.1 0.1 2\n"
            (labels / "frame1.txt").write_text(original, encoding="utf-8")
            (images_to_label / "frame1.jpg").write_bytes(b"fake-image")

            result = normalize_label_directory(
                labels_dir=str(labels),
                images_all_dir=str(images_all),
                images_to_label_dir=str(images_to_label),
                mode=DATASET_POSE,
                class_count=1,
                keypoint_count=2,
            )

            self.assertFalse(result.canceled)
            self.assertEqual(result.normalized, 1)
            self.assertEqual(result.copied_images, 1)
            self.assertIsNotNone(result.backup_dir)
            self.assertTrue((images_all / "frame1.jpg").exists())
            self.assertEqual(
                (labels / "frame1.txt").read_text(encoding="utf-8"),
                "0 1.000000 0.000000 0.200000 0.200000 "
                "0.100000 0.100000 2 0.000000 0.000000 0\n",
            )
            backup_label = Path(result.backup_dir) / "frame1.txt"
            self.assertEqual(backup_label.read_text(encoding="utf-8"), original)

    def test_normalize_segmentation_directory_leaves_valid_labels_untouched(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels_seg_all"
            images_all = root / "images_all"
            images_to_label = root / "images_to_label"
            labels.mkdir()
            images_all.mkdir()
            images_to_label.mkdir()
            label_line = "0 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000\n"
            (labels / "frame1.txt").write_text(label_line, encoding="utf-8")
            (images_all / "frame1.png").write_bytes(b"fake-image")

            result = normalize_label_directory(
                labels_dir=str(labels),
                images_all_dir=str(images_all),
                images_to_label_dir=str(images_to_label),
                mode=DATASET_SEGMENT,
                class_count=1,
            )

            self.assertEqual(result.normalized, 0)
            self.assertEqual(result.untouched, 1)
            self.assertIsNone(result.backup_dir)
            self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
