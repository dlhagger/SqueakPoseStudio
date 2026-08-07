import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from dataset_ops import (
    DATASET_DETECT,
    DATASET_POSE,
    DATASET_SEGMENT,
    dataset_export_paths,
    export_dataset_files,
    format_dataset_export_summary,
    format_project_health_summary,
    label_file_has_usable_rows,
    list_image_files,
    list_label_files,
    normalize_label_directory,
    partition_images_by_usable_labels,
    scan_project_health,
    split_train_val_images,
    write_dataset_yaml_for_mode,
)


class DatasetOpsTests(unittest.TestCase):
    def test_list_image_files_ignores_hidden_transaction_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frame.png").write_bytes(b"image")
            (root / ".frame.abc.tmp.png").write_bytes(b"staged")

            self.assertEqual(list_image_files(str(root)), ["frame.png"])

    def test_list_label_files_ignores_hidden_transaction_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frame.txt").write_text("label", encoding="utf-8")
            (root / ".frame.tmp.txt").write_text("staged", encoding="utf-8")

            self.assertEqual(list_label_files(str(root)), ["frame.txt"])

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
                "0 0.5 0.5 0.2 0.2 0.1 0.1 2\nbad row\n",
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
            self.assertTrue(any("frame2.txt: missing" in msg for msg in result.errors))
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
            result.skipped_images = ["unlabeled.jpg", "invalid.jpg"]

            summary = format_dataset_export_summary(result)

        self.assertIn("Split seed: 42", summary)
        self.assertIn("Skipped without usable labels: 2", summary)

    def test_partition_images_skips_missing_and_invalid_active_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels_all"
            labels.mkdir()
            (labels / "pose_ok.txt").write_text(
                "0 0.5 0.5 0.2 0.2 0.1 0.1 2\n",
                encoding="utf-8",
            )
            (labels / "invalid.txt").write_text("invalid row\n", encoding="utf-8")

            exportable, skipped = partition_images_by_usable_labels(
                ["pose_ok.jpg", "missing.jpg", "invalid.jpg"],
                labels_dir=str(labels),
                mode=DATASET_POSE,
                class_count=1,
                keypoint_count=1,
            )

            self.assertEqual(exportable, ["pose_ok.jpg"])
            self.assertEqual(skipped, ["missing.jpg", "invalid.jpg"])

    def test_partition_segmentation_images_uses_segmentation_label_rules(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels_seg_all"
            labels.mkdir()
            (labels / "seg_ok.txt").write_text(
                "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n",
                encoding="utf-8",
            )
            (labels / "too_short.txt").write_text(
                "0 0.1 0.1 0.8 0.1\n",
                encoding="utf-8",
            )

            exportable, skipped = partition_images_by_usable_labels(
                ["seg_ok.png", "too_short.png", "missing.png"],
                labels_dir=str(labels),
                mode=DATASET_SEGMENT,
                class_count=1,
            )

            self.assertEqual(exportable, ["seg_ok.png"])
            self.assertEqual(skipped, ["too_short.png", "missing.png"])

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
                dataset_path="/final/project/datasets/segment",
            )

            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
            self.assertEqual(data["task"], "segment")
            self.assertEqual(data["nc"], 1)
            self.assertEqual(data["path"], "/final/project/datasets/segment")

    def test_segmentation_export_normalizes_rows_and_drops_invalid_polygons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_all = root / "images_all"
            labels_all = root / "labels_seg_all"
            images_all.mkdir()
            labels_all.mkdir()
            (images_all / "frame1.png").write_bytes(b"image")
            (labels_all / "frame1.txt").write_text(
                "0 -0.1 0.1 0.8 0.1 0.8 1.2 0.1 0.8\n0 0.2 0.2 0.3 0.3 0.4 0.4\n",
                encoding="utf-8",
            )
            paths = dataset_export_paths(str(root), DATASET_SEGMENT)

            result = export_dataset_files(
                images_all_dir=str(images_all),
                labels_all_dir=str(labels_all),
                paths=paths,
                train_images=["frame1.png"],
                val_images=[],
                mode=DATASET_SEGMENT,
                class_count=1,
            )

            self.assertEqual(result.errors, [])
            self.assertTrue(any("clamped" in warning for warning in result.warnings))
            self.assertTrue(any("zero-area" in warning for warning in result.warnings))
            exported = (Path(paths.labels_train_dir) / "frame1.txt").read_text(encoding="utf-8")
            self.assertEqual(
                exported,
                "0 0.000000 0.100000 0.800000 0.100000 0.800000 1.000000 0.100000 0.800000\n",
            )

    def test_pose_export_normalizes_rows_to_current_keypoint_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_all = root / "images_all"
            labels_all = root / "labels_all"
            images_all.mkdir()
            labels_all.mkdir()
            (images_all / "frame1.png").write_bytes(b"image")
            (labels_all / "frame1.txt").write_text(
                "0 1.2 0.5 0.2 0.2 0.1 0.1 2\n",
                encoding="utf-8",
            )
            paths = dataset_export_paths(str(root), DATASET_POSE)

            result = export_dataset_files(
                images_all_dir=str(images_all),
                labels_all_dir=str(labels_all),
                paths=paths,
                train_images=["frame1.png"],
                val_images=[],
                mode=DATASET_POSE,
                class_count=1,
                keypoint_count=2,
            )

            self.assertEqual(result.errors, [])
            self.assertTrue(any("clamped" in warning for warning in result.warnings))
            self.assertTrue(any("missing keypoint" in warning for warning in result.warnings))
            self.assertEqual(
                (Path(paths.labels_train_dir) / "frame1.txt").read_text(encoding="utf-8"),
                "0 1.000000 0.500000 0.200000 0.200000 0.100000 0.100000 2 0.000000 0.000000 0\n",
            )

    def test_project_health_reports_orphans_copies_and_temp_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in (
                "images_to_label",
                "images_all",
                "labels_all",
                "labels_seg_all",
                "datasets",
            ):
                (root / directory).mkdir()
            (root / "images_all" / "mouse.png").write_bytes(b"image")
            (root / "images_all" / "mouse 2.png").write_bytes(b"copy")
            (root / "labels_all" / "mouse.txt").write_text(
                "0 0.5 0.5 0.2 0.2 0.1 0.1 2\n",
                encoding="utf-8",
            )
            (root / "labels_seg_all" / "orphan.txt").write_text(
                "0 0.1 0.1 0.8 0.1 0.8 0.8\n",
                encoding="utf-8",
            )
            temp_path = root / "images_all" / ".mouse.abcdefgh.tmp.png"
            temp_path.write_bytes(b"staged")

            report = scan_project_health(
                str(root),
                pose_class_count=1,
                pose_keypoint_count=1,
                segmentation_class_count=1,
            )

            self.assertEqual(report.stored_images, 2)
            self.assertEqual(report.usable_pose_labels, 1)
            self.assertEqual(report.orphan_segmentation_labels, ["orphan.txt"])
            self.assertEqual(report.likely_duplicate_images, [("mouse.png", "mouse 2.png")])
            self.assertEqual(report.temporary_paths, [str(temp_path)])
            self.assertIn("Likely numbered image copies: 1", format_project_health_summary(report))

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
                "0 1.000000 0.000000 0.200000 0.200000 0.100000 0.100000 2 0.000000 0.000000 0\n",
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

    def test_normalize_quarantines_completely_invalid_label_after_backup(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels_all"
            images_all = root / "images_all"
            images_to_label = root / "images_to_label"
            labels.mkdir()
            images_all.mkdir()
            images_to_label.mkdir()
            original = "invalid row\n9 0.5 0.5 0.2 0.2\n"
            label_path = labels / "frame1.txt"
            label_path.write_text(original, encoding="utf-8")

            result = normalize_label_directory(
                labels_dir=str(labels),
                images_all_dir=str(images_all),
                images_to_label_dir=str(images_to_label),
                mode=DATASET_POSE,
                class_count=1,
                keypoint_count=1,
            )

            self.assertEqual(result.quarantined, 1)
            self.assertFalse(label_path.exists())
            self.assertIsNotNone(result.backup_dir)
            self.assertIsNotNone(result.quarantine_dir)
            self.assertEqual(
                (Path(result.backup_dir) / "frame1.txt").read_text(encoding="utf-8"), original
            )
            self.assertEqual(
                (Path(result.quarantine_dir) / "frame1.txt").read_text(encoding="utf-8"), original
            )

    def test_empty_and_invalid_labels_are_not_usable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.txt"
            invalid = root / "invalid.txt"
            valid = root / "valid.txt"
            empty.write_text("", encoding="utf-8")
            invalid.write_text("bad row\n", encoding="utf-8")
            valid.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            self.assertFalse(
                label_file_has_usable_rows(
                    str(empty), mode=DATASET_POSE, class_count=1, keypoint_count=0
                )
            )
            self.assertFalse(
                label_file_has_usable_rows(
                    str(invalid), mode=DATASET_POSE, class_count=1, keypoint_count=0
                )
            )
            self.assertTrue(
                label_file_has_usable_rows(
                    str(valid), mode=DATASET_POSE, class_count=1, keypoint_count=0
                )
            )

    def test_dataset_copy_failure_is_reported_as_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_all = root / "images_all"
            labels_all = root / "labels_all"
            images_all.mkdir()
            labels_all.mkdir()
            (images_all / "frame1.jpg").write_bytes(b"image")
            paths = dataset_export_paths(str(root), DATASET_POSE)

            with patch("dataset_ops.shutil.copy2", side_effect=OSError("disk full")):
                result = export_dataset_files(
                    images_all_dir=str(images_all),
                    labels_all_dir=str(labels_all),
                    paths=paths,
                    train_images=["frame1.jpg"],
                    val_images=[],
                    mode=DATASET_POSE,
                )

            self.assertEqual(result.warnings, [])
            self.assertEqual(len(result.errors), 1)
            self.assertIn("copy image failed", result.errors[0])


if __name__ == "__main__":
    unittest.main()
