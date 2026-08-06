import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose_core import (
    CURRENT_PROJECT_SCHEMA_VERSION,
    InferenceCsvWriter,
    atomic_write_text_files,
    atomic_write_text,
    build_segmentation_inference_rows,
    commit_staged_paths,
    effective_prediction_batch,
    filter_image_stem_collisions,
    find_duplicate_names,
    infer_dataset_task,
    migrate_project_metadata,
    model_task_mismatch_message,
    normalize_pose_label_lines,
    normalize_segmentation_label_lines,
    normalize_yolo_task,
    parse_yolo_pose_label_line,
    resolve_default_training_dataset_path,
    stable_path_id,
)


class _FakeWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


class CoreHelpersTests(unittest.TestCase):
    def test_task_helpers_normalize_and_detect_mismatch(self):
        self.assertEqual(normalize_yolo_task("segmentation"), "segment")
        self.assertEqual(normalize_yolo_task("keypoints"), "pose")
        self.assertEqual(normalize_yolo_task("depth"), "depth")
        self.assertEqual(infer_dataset_task({"task": "seg", "train": "images/train"}), "segment")
        self.assertEqual(infer_dataset_task({"kpt_shape": [6, 3]}), "pose")
        self.assertEqual(infer_dataset_task({"train": "images/train"}), "detect")
        self.assertIsNone(model_task_mismatch_message("pose", "keypoints"))
        self.assertIn("requires 'segment'", model_task_mismatch_message("detect", "segment"))
        self.assertIn("requires 'depth'", model_task_mismatch_message("pose", "depth"))

    def test_project_metadata_migration_preserves_unknown_fields(self):
        migrated, changed = migrate_project_metadata(
            {
                "schema_version": 1,
                "workflow": "seg",
                "sam_path": "models/sam.pt",
                "custom": {"keep": True},
            },
            created_at="2026-06-09T12:00:00",
        )

        self.assertTrue(changed)
        self.assertEqual(migrated["schema_version"], CURRENT_PROJECT_SCHEMA_VERSION)
        self.assertEqual(migrated["active_workflow"], "segmentation")
        self.assertEqual(migrated["active_layer"], "segmentation")
        self.assertIn("keypoints", migrated["layers"])
        self.assertIn("segmentation", migrated["layers"])
        self.assertEqual(migrated["sam_model_path"], "models/sam.pt")
        self.assertEqual(migrated["custom"], {"keep": True})
        self.assertEqual(migrated["created_at"], "2026-06-09T12:00:00")

    def test_project_metadata_does_not_downgrade_newer_schema(self):
        payload = {"schema_version": CURRENT_PROJECT_SCHEMA_VERSION + 1, "future": True}

        migrated, changed = migrate_project_metadata(payload)

        self.assertFalse(changed)
        self.assertEqual(migrated, payload)

    def test_find_duplicate_names_preserves_first_seen_order(self):
        dupes = find_duplicate_names(["nose", "tail", "nose", "ear", "tail", "tail"])
        self.assertEqual(dupes, ["nose", "tail"])

    def test_effective_prediction_batch_auto_and_explicit(self):
        self.assertEqual(effective_prediction_batch(12, "cpu"), 12)
        self.assertEqual(effective_prediction_batch(-1, "cuda"), 8)
        self.assertEqual(effective_prediction_batch(0, "cuda:0"), 8)
        self.assertEqual(effective_prediction_batch(0, "mps"), 8)
        self.assertEqual(effective_prediction_batch(-1, "cpu"), 1)

    def test_parse_label_line_ignores_extra_keypoints_without_schema_mutation(self):
        canonical = ["nose", "tail"]
        line = "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.8 0.7 1 0.9 0.1 2"
        entry, had_extra = parse_yolo_pose_label_line(
            line,
            classes_count=1,
            canonical_names=canonical,
            class_keypoint_lookup=[["nose", "tail"]],
            img_w=100,
            img_h=200,
        )

        self.assertTrue(had_extra)
        self.assertIsNotNone(entry)
        self.assertEqual(canonical, ["nose", "tail"])
        self.assertEqual(len(entry["keypoints"]), 2)
        self.assertEqual(entry["keypoints"][0]["name"], "nose")
        self.assertEqual(entry["keypoints"][1]["name"], "tail")

    def test_parse_label_line_filters_to_class_keypoint_order(self):
        line = "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.8 0.7 1"
        entry, had_extra = parse_yolo_pose_label_line(
            line,
            classes_count=1,
            canonical_names=["nose", "tail"],
            class_keypoint_lookup=[["tail"]],
            img_w=100,
            img_h=100,
        )

        self.assertFalse(had_extra)
        self.assertIsNotNone(entry)
        self.assertEqual(len(entry["keypoints"]), 1)
        self.assertEqual(entry["keypoints"][0]["name"], "tail")
        self.assertEqual(entry["keypoints"][0]["idx"], 0)

    def test_inference_csv_writer_streams_incrementally(self):
        fake = _FakeWriter()
        stream = InferenceCsvWriter(fake)

        stream.write_row({"frame_index": 0, "confidence": 0.8})
        self.assertEqual(stream.rows_written, 1)
        self.assertEqual(len(fake.rows), 1)

        stream.write_row({"frame_index": 1, "confidence": 0.9})
        self.assertEqual(stream.rows_written, 2)
        self.assertEqual(len(fake.rows), 2)

    def test_atomic_write_text_replaces_existing_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            path.write_text("old\n", encoding="utf-8")

            atomic_write_text(str(path), "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
            self.assertEqual([item.name for item in Path(tmp).iterdir()], ["labels.txt"])

    def test_filter_image_stem_collisions_blocks_case_and_extension_variants(self):
        accepted, collisions = filter_image_stem_collisions(
            ["frame.jpg", "frame.png", "Mouse.JPG", "mouse.jpeg", "unique.webp"]
        )

        self.assertEqual(accepted, ["unique.webp"])
        self.assertEqual(collisions["frame"], ["frame.jpg", "frame.png"])
        self.assertEqual(collisions["mouse"], ["mouse.jpeg", "Mouse.JPG"])

    def test_stable_path_id_distinguishes_same_named_files_in_different_folders(self):
        first = stable_path_id("/tmp/session-a/video.mp4")
        second = stable_path_id("/tmp/session-b/video.mp4")

        self.assertNotEqual(first, second)
        self.assertEqual(first, stable_path_id("/tmp/session-a/video.mp4"))

    def test_atomic_write_text_files_rolls_back_all_targets_on_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = root / "classes.txt"
            keypoints = root / "keypoints.txt"
            classes.write_text("old-class\n", encoding="utf-8")
            keypoints.write_text("old-keypoint\n", encoding="utf-8")
            real_replace = __import__("os").replace

            def fail_second_staged_install(src, dst):
                if Path(dst) == keypoints and ".tmp" in Path(src).name:
                    raise OSError("injected install failure")
                return real_replace(src, dst)

            with patch("squeakpose_core.os.replace", side_effect=fail_second_staged_install):
                with self.assertRaises(OSError):
                    atomic_write_text_files(
                        {
                            str(classes): "new-class\n",
                            str(keypoints): "new-keypoint\n",
                        }
                    )

            self.assertEqual(classes.read_text(encoding="utf-8"), "old-class\n")
            self.assertEqual(keypoints.read_text(encoding="utf-8"), "old-keypoint\n")
            self.assertFalse(any(".backup-" in item.name or ".tmp" in item.name for item in root.iterdir()))

    def test_commit_staged_paths_restores_directories_when_final_install_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_images = root / "images"
            old_labels = root / "labels"
            old_yaml = root / "dataset.yaml"
            old_images.mkdir()
            old_labels.mkdir()
            (old_images / "old.jpg").write_text("old-image", encoding="utf-8")
            (old_labels / "old.txt").write_text("old-label", encoding="utf-8")
            old_yaml.write_text("old-yaml", encoding="utf-8")

            staged_images = root / "staged-images"
            staged_labels = root / "staged-labels"
            staged_yaml = root / "staged.yaml"
            staged_images.mkdir()
            staged_labels.mkdir()
            (staged_images / "new.jpg").write_text("new-image", encoding="utf-8")
            (staged_labels / "new.txt").write_text("new-label", encoding="utf-8")
            staged_yaml.write_text("new-yaml", encoding="utf-8")
            real_replace = __import__("os").replace

            def fail_yaml_install(src, dst):
                if Path(src) == staged_yaml and Path(dst) == old_yaml:
                    raise OSError("injected yaml install failure")
                return real_replace(src, dst)

            with patch("squeakpose_core.os.replace", side_effect=fail_yaml_install):
                with self.assertRaises(OSError):
                    commit_staged_paths(
                        [
                            (str(staged_images), str(old_images)),
                            (str(staged_labels), str(old_labels)),
                            (str(staged_yaml), str(old_yaml)),
                        ]
                    )

            self.assertEqual((old_images / "old.jpg").read_text(encoding="utf-8"), "old-image")
            self.assertEqual((old_labels / "old.txt").read_text(encoding="utf-8"), "old-label")
            self.assertEqual(old_yaml.read_text(encoding="utf-8"), "old-yaml")

    def test_normalize_pose_label_lines_clamps_pads_and_drops_invalid_rows(self):
        normalized, warnings, changed = normalize_pose_label_lines(
            [
                "0 1.2 -0.1 2.0 0.2 1.5 -0.5 3 0.4 0.6 2 0.9 0.9 2",
                "9 0.5 0.5 0.2 0.2 0.1 0.1 2",
                "0 0.5 0.5 -0.2 0.2 0.1 0.1 2",
            ],
            class_count=1,
            keypoint_count=2,
        )

        self.assertTrue(changed)
        self.assertEqual(
            normalized,
            [
                "0 1.000000 0.000000 1.000000 0.200000 "
                "1.000000 0.000000 2 0.400000 0.600000 2"
            ],
        )
        self.assertTrue(any("extra keypoint" in warning for warning in warnings))
        self.assertTrue(any("parse error" in warning for warning in warnings))
        self.assertTrue(any("non-positive bbox" in warning for warning in warnings))

    def test_normalize_pose_label_lines_pads_missing_keypoints(self):
        normalized, warnings, changed = normalize_pose_label_lines(
            ["0 0.5 0.5 0.2 0.2 0.1 0.1 2"],
            class_count=1,
            keypoint_count=2,
        )

        self.assertTrue(changed)
        self.assertEqual(
            normalized,
            ["0 0.500000 0.500000 0.200000 0.200000 0.100000 0.100000 2 0.000000 0.000000 0"],
        )
        self.assertTrue(any("missing keypoint" in warning for warning in warnings))

    def test_normalize_segmentation_label_lines_clamps_and_drops_invalid_rows(self):
        normalized, warnings, changed = normalize_segmentation_label_lines(
            [
                "0 1.2 -0.1 0.5 0.5 0.1 1.1 99",
                "2 0.1 0.1 0.2 0.2 0.3 0.3",
            ],
            class_count=1,
        )

        self.assertTrue(changed)
        self.assertEqual(
            normalized,
            ["0 1.000000 0.000000 0.500000 0.500000 0.100000 1.000000"],
        )
        self.assertTrue(any("odd coordinate count" in warning for warning in warnings))
        self.assertTrue(any("invalid class id" in warning for warning in warnings))

    def test_normalize_segmentation_label_lines_drops_zero_area_polygons(self):
        normalized, warnings, changed = normalize_segmentation_label_lines(
            [
                "0 0.1 0.1 0.2 0.2 0.3 0.3",
                "0 0.5 0.5 0.5 0.5 0.5 0.5",
            ],
            class_count=1,
        )

        self.assertEqual(normalized, [])
        self.assertTrue(changed)
        self.assertTrue(any("zero-area" in warning for warning in warnings))
        self.assertTrue(any("<3 unique" in warning for warning in warnings))

    def test_build_segmentation_inference_rows_formats_detection_schema(self):
        rows = build_segmentation_inference_rows(
            frame_index=7,
            detections=[
                {
                    "class_id": 1,
                    "conf": 0.91,
                    "box": [1, 2, 30, 40],
                    "mask_polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "binary_mask": [[1, 0], [0, 1]],
                }
            ],
            class_names={1: "mouse"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], 7)
        self.assertEqual(rows[0]["det"], 0)
        self.assertEqual(rows[0]["class_id"], 1)
        self.assertEqual(rows[0]["class_name"], "mouse")
        self.assertAlmostEqual(rows[0]["conf"], 0.91, places=5)
        self.assertEqual(rows[0]["x1"], 1.0)
        self.assertEqual(rows[0]["y2"], 40.0)
        self.assertEqual(rows[0]["mask_polygon"][2], [1.0, 1.0])
        self.assertEqual(rows[0]["binary_mask"], [[1, 0], [0, 1]])

    def test_build_segmentation_inference_rows_emits_no_detection_row(self):
        rows = build_segmentation_inference_rows(
            frame_index=8,
            detections=[],
            class_names={0: "mouse"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], 8)
        self.assertEqual(rows[0]["det"], -1)
        self.assertEqual(rows[0]["class_id"], "")
        self.assertEqual(rows[0]["class_name"], "")
        self.assertEqual(rows[0]["conf"], "")
        self.assertIsNone(rows[0]["mask_polygon"])
        self.assertIsNone(rows[0]["binary_mask"])

    def test_build_segmentation_inference_rows_can_omit_binary_masks(self):
        rows = build_segmentation_inference_rows(
            frame_index=9,
            detections=[
                {
                    "class_id": 0,
                    "conf": 0.8,
                    "box": [1, 2, 3, 4],
                    "binary_mask": [[1, 1], [0, 0]],
                }
            ],
            include_binary_mask=False,
        )

        self.assertEqual(rows[0]["frame"], 9)
        self.assertIsNone(rows[0]["binary_mask"])

    def test_resolve_default_training_dataset_path_prefers_pose_then_segment_then_detect(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = root / "datasets"
            pose = datasets / "pose"
            segment = datasets / "segment"
            detect = datasets / "detect"
            pose.mkdir(parents=True, exist_ok=True)
            segment.mkdir(parents=True, exist_ok=True)
            detect.mkdir(parents=True, exist_ok=True)

            (detect / "dataset.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(detect),
            )

            (segment / "dataset.yaml").write_text("task: segment\nnames: [mouse]\n", encoding="utf-8")
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(segment),
            )

            (pose / "dataset.yaml").write_text("names: [mouse]\n", encoding="utf-8")
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(pose),
            )

    def test_resolve_default_training_dataset_path_falls_back_to_datasets_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "datasets"
            self.assertEqual(
                resolve_default_training_dataset_path(str(root)),
                str(expected),
            )


if __name__ == "__main__":
    unittest.main()
