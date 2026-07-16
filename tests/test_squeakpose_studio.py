import json
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

_OPTIONAL_STUDIO_IMPORT_MODULES = {"PyQt6", "cv2", "numpy", "torch", "ultralytics", "yaml"}


def _is_optional_studio_import_error(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        name = (getattr(exc, "name", "") or "").split(".", 1)[0]
        return name in _OPTIONAL_STUDIO_IMPORT_MODULES
    if isinstance(exc, ImportError):
        msg = str(exc)
        return any(mod in msg for mod in _OPTIONAL_STUDIO_IMPORT_MODULES)
    return False


try:
    from squeakpose_studio import (
        LabelingApp,
        VideoReviewDialog,
        WORKFLOW_POSE,
        WORKFLOW_SEG,
        _discover_distillation_exports,
        _distillation_export_search_roots,
        _distillation_sample_count,
        _ensure_project_structure,
        _project_paths,
        _retain_main_window,
    )
    _STUDIO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent import gate
    if not _is_optional_studio_import_error(exc):
        raise
    LabelingApp = None
    VideoReviewDialog = None
    _discover_distillation_exports = None
    _distillation_export_search_roots = None
    _distillation_sample_count = None
    _ensure_project_structure = None
    _project_paths = None
    _retain_main_window = None
    WORKFLOW_POSE = "pose"
    WORKFLOW_SEG = "segmentation"
    _STUDIO_IMPORT_ERROR = exc


class _EmptyBoxes:
    def __len__(self):
        return 0


class _Results:
    def __init__(self, boxes):
        self.boxes = boxes


class _CacheDummy:
    def __init__(self, video_path: str, model_path: str, workflow: str):
        self.path = video_path
        self.model_path = model_path
        self.workflow = workflow
        self.preds = {}

    def _cache_path(self):
        return os.path.abspath(self.path) + ".sqp_preds.json"

    def _video_signature(self):
        return {
            "path": os.path.abspath(self.path),
            "size": int(os.path.getsize(self.path)),
            "mtime": float(os.path.getmtime(self.path)),
            "total": 10,
            "fps": 30.0,
        }


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text: str):
        self.text = text


class _FakeCombo:
    def __init__(self):
        self.minimum_contents_length = None

    def setMinimumContentsLength(self, value: int):
        self.minimum_contents_length = value


@unittest.skipIf(VideoReviewDialog is None, f"squeakpose_studio import unavailable: {_STUDIO_IMPORT_ERROR}")
class StudioVideoReviewTests(unittest.TestCase):
    def test_retain_main_window_stores_reference_on_qapplication(self):
        fake_app = type("FakeApp", (), {})()
        marker = object()

        with patch("squeakpose_studio._qt_app_instance", return_value=fake_app):
            _retain_main_window(marker)

        self.assertIs(fake_app._squeakpose_main_window, marker)

    def test_corrupt_project_metadata_is_preserved_before_recovery(self):
        with TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "squeakpose_project.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            app = LabelingApp.__new__(LabelingApp)
            app.project_root = tmp
            app._project_meta_recovery = None

            payload = LabelingApp._read_project_meta(app)

            self.assertEqual(payload, {})
            self.assertFalse(os.path.exists(meta_path))
            backups = [
                name
                for name in os.listdir(tmp)
                if name.startswith("squeakpose_project.corrupt-")
            ]
            self.assertEqual(len(backups), 1)
            self.assertIsNotNone(app._project_meta_recovery)

    def test_sync_canonical_keypoints_appends_class_map_names(self):
        with TemporaryDirectory() as tmp:
            keypoint_file = os.path.join(tmp, "keypoints.txt")
            app = LabelingApp.__new__(LabelingApp)
            app.kp_names = ["nose", "head", "left_ear", "right_ear", "back", "tail_base"]
            app.classes = ["mouse"]
            app.class_keypoints = {
                "mouse": [
                    "nose",
                    "head",
                    "left_ear",
                    "right_ear",
                    "back",
                    "tail_base",
                    "mid_back",
                    "hip",
                    "tail_mid",
                    "tail_tip",
                ]
            }
            app.keypoint_file = keypoint_file

            changed = LabelingApp._sync_canonical_keypoints_from_class_map(app)

            self.assertTrue(changed)
            self.assertEqual(app.kp_names, app.class_keypoints["mouse"])
            with open(keypoint_file, "r", encoding="utf-8") as f:
                self.assertEqual([line.strip() for line in f if line.strip()], app.class_keypoints["mouse"])

    def test_cache_validation_rejects_workflow_mismatch(self):
        with TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "sample.mp4")
            with open(video_path, "wb") as f:
                f.write(b"fake-video-bytes")

            dummy = _CacheDummy(video_path=video_path, model_path="model.pt", workflow=WORKFLOW_SEG)
            sig = dummy._video_signature()
            payload = {
                "meta": {
                    "video": sig,
                    "model_path": "model.pt",
                    "workflow": WORKFLOW_POSE,
                },
                "preds": {"4": {"ok": True, "conf": 0.7}},
            }
            with open(dummy._cache_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f)

            ok = VideoReviewDialog._load_cache_if_valid(dummy)
            self.assertFalse(ok)
            self.assertEqual(dummy.preds, {})

    def test_cache_validation_accepts_matching_workflow(self):
        with TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "sample.mp4")
            with open(video_path, "wb") as f:
                f.write(b"fake-video-bytes")

            dummy = _CacheDummy(video_path=video_path, model_path="model.pt", workflow=WORKFLOW_SEG)
            sig = dummy._video_signature()
            payload = {
                "meta": {
                    "video": sig,
                    "model_path": "model.pt",
                    "workflow": WORKFLOW_SEG,
                },
                "preds": {
                    "2": {"ok": True, "conf": 0.8},
                    "5": {"ok": True, "conf": 0.9},
                },
            }
            with open(dummy._cache_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f)

            ok = VideoReviewDialog._load_cache_if_valid(dummy)
            self.assertTrue(ok)
            self.assertEqual(sorted(dummy.preds.keys()), [2, 5])

    def test_segmentation_rows_include_no_detection_frames(self):
        app = LabelingApp.__new__(LabelingApp)
        app.classes = ["mouse"]
        results = _Results(boxes=_EmptyBoxes())

        rows = LabelingApp._segmentation_rows_from_result(
            app,
            results,
            frame_idx=12,
            include_binary_mask=False,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], 12)
        self.assertEqual(rows[0]["det"], -1)
        self.assertEqual(rows[0]["class_id"], "")

    def test_backup_label_dir_copies_existing_labels(self):
        with TemporaryDirectory() as tmp:
            labels_dir = os.path.join(tmp, "labels_all")
            os.makedirs(labels_dir, exist_ok=True)
            label_path = os.path.join(labels_dir, "frame001.txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

            app = LabelingApp.__new__(LabelingApp)
            backup_dir = LabelingApp._backup_label_dir(app, labels_dir)

            self.assertTrue(os.path.isdir(backup_dir))
            self.assertTrue(os.path.basename(backup_dir).startswith("labels_all_backup_"))
            copied_path = os.path.join(backup_dir, "frame001.txt")
            with open(copied_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "0 0.5 0.5 0.2 0.2\n")

    def test_delete_image_files_removes_queue_source_and_exported_copies(self):
        with TemporaryDirectory() as tmp:
            frame = "frame001.jpg"
            app = LabelingApp.__new__(LabelingApp)
            app.project_root = tmp
            app.image_dir_queue = os.path.join(tmp, "images_to_label")
            app.image_dir_all = os.path.join(tmp, "images_all")
            app.pose_label_dir = os.path.join(tmp, "labels_all")
            app.seg_label_dir = os.path.join(tmp, "labels_seg_all")
            app.active_image_dir = app.image_dir_queue

            paths = [
                os.path.join(app.image_dir_queue, frame),
                os.path.join(app.image_dir_all, frame),
                os.path.join(app.pose_label_dir, "frame001.txt"),
                os.path.join(app.seg_label_dir, "frame001.txt"),
                os.path.join(tmp, "annotations", "frame001_annotated.png"),
                os.path.join(tmp, "datasets", "pose", "images", "train", frame),
                os.path.join(tmp, "datasets", "pose", "labels", "train", "frame001.txt"),
                os.path.join(tmp, "datasets", "segment", "images", "val", frame),
                os.path.join(tmp, "datasets", "detect", "labels", "val", "frame001.txt"),
            ]
            for path in paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"data")

            removed, errors = LabelingApp._delete_image_files(app, frame)

            self.assertEqual(errors, [])
            for path in paths:
                self.assertFalse(os.path.exists(path), path)
            self.assertEqual(set(removed), set(paths))

    def test_delete_image_blocks_when_another_extension_shares_the_stem(self):
        with TemporaryDirectory() as tmp:
            queue = os.path.join(tmp, "images_to_label")
            images_all = os.path.join(tmp, "images_all")
            os.makedirs(queue)
            os.makedirs(images_all)
            with open(os.path.join(queue, "frame001.jpg"), "wb") as fh:
                fh.write(b"queue")
            with open(os.path.join(images_all, "frame001.png"), "wb") as fh:
                fh.write(b"stored")

            class DeleteDummy:
                images = ["frame001.jpg"]
                current_idx = 0
                image_dir_queue = queue
                image_dir_all = images_all

            app = DeleteDummy()
            with (
                patch("squeakpose_studio.QMessageBox.warning") as warning,
                patch("squeakpose_studio.QMessageBox.question") as question,
            ):
                LabelingApp.delete_current_image(app)

            warning.assert_called_once()
            question.assert_not_called()
            self.assertTrue(os.path.exists(os.path.join(queue, "frame001.jpg")))
            self.assertTrue(os.path.exists(os.path.join(images_all, "frame001.png")))

    def test_save_labels_keeps_existing_files_when_transaction_fails(self):
        with TemporaryDirectory() as tmp:
            queue = os.path.join(tmp, "images_to_label")
            images_all = os.path.join(tmp, "images_all")
            labels = os.path.join(tmp, "labels_all")
            annotations = os.path.join(tmp, "annotations")
            for directory in (queue, images_all, labels, annotations):
                os.makedirs(directory)

            source_path = os.path.join(queue, "frame001.jpg")
            image_path = os.path.join(images_all, "frame001.jpg")
            label_path = os.path.join(labels, "frame001.txt")
            overlay_path = os.path.join(annotations, "frame001_annotated.png")
            with open(source_path, "wb") as fh:
                fh.write(b"new-image")
            with open(image_path, "wb") as fh:
                fh.write(b"old-image")
            with open(label_path, "w", encoding="utf-8") as fh:
                fh.write("old-label\n")
            with open(overlay_path, "wb") as fh:
                fh.write(b"old-overlay")

            class SaveDummy:
                def _is_seg_workflow(self):
                    return False

                def _is_pose_workflow(self):
                    return True

                def _cache_active_annotation(self):
                    return True

                def _annotation_entry_to_line(self, _entry):
                    return "0 0.5 0.5 0.2 0.2"

                def _render_overlay_from_cache(self, path):
                    with open(path, "wb") as fh:
                        fh.write(b"new-overlay")
                    return True

                def _update_progress_label(self):
                    raise AssertionError("failed save must not update progress")

            app = SaveDummy()
            app.images = ["frame001.jpg"]
            app.current_idx = 0
            app.project_root = tmp
            app.label_dir = labels
            app.classes = ["mouse"]
            app.annotation_cache = {0: {"class_id": 0}}
            app.current_image_path = source_path
            app.active_image_dir = queue
            app.image_dir_queue = queue
            app.image_dir_all = images_all

            with (
                patch("squeakpose_studio.commit_staged_paths", side_effect=OSError("injected failure")),
                patch("squeakpose_studio.QMessageBox.warning"),
            ):
                saved = LabelingApp.save_labels(app)

            self.assertFalse(saved)
            with open(image_path, "rb") as fh:
                self.assertEqual(fh.read(), b"old-image")
            with open(label_path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "old-label\n")
            with open(overlay_path, "rb") as fh:
                self.assertEqual(fh.read(), b"old-overlay")

    def test_complete_does_not_advance_after_failed_save(self):
        class CompleteDummy:
            images = ["frame001.jpg", "frame002.jpg"]
            current_idx = 0

            def _is_seg_workflow(self):
                return False

            def _is_fully_labeled(self):
                return True

            def save_labels(self):
                return False

            def _find_next_unlabeled(self, _start):
                raise AssertionError("failed save must not search for the next image")

        app = CompleteDummy()
        LabelingApp.complete_and_next_unlabeled(app)
        self.assertEqual(app.current_idx, 0)

    def test_schema_state_is_unchanged_when_transaction_fails(self):
        with TemporaryDirectory() as tmp:
            class SchemaDummy:
                def _schema_is_locked(self):
                    return False

            app = SchemaDummy()
            app.classes = ["mouse"]
            app.kp_names = ["nose"]
            app.class_keypoints = {"mouse": ["nose"]}
            app.class_file = os.path.join(tmp, "classes.txt")
            app.keypoint_file = os.path.join(tmp, "keypoints.txt")
            app.class_keypoints_path = os.path.join(tmp, "class_keypoints.json")

            with (
                patch("squeakpose_studio.atomic_write_text_files", side_effect=OSError("disk full")),
                patch("squeakpose_studio.QMessageBox.warning"),
            ):
                changed = LabelingApp._apply_class_manager_results(
                    app,
                    ["mouse", "rat"],
                    ["nose", "tail"],
                    {"mouse": ["nose"], "rat": ["tail"]},
                )

            self.assertFalse(changed)
            self.assertEqual(app.classes, ["mouse"])
            self.assertEqual(app.kp_names, ["nose"])
            self.assertEqual(app.class_keypoints, {"mouse": ["nose"]})

    def test_video_export_dedupe_is_scoped_to_source_id(self):
        with TemporaryDirectory() as tmp:
            for name in (
                "session_sourcea_f000003.png",
                "session_sourceb_f000007.png",
            ):
                with open(os.path.join(tmp, name), "wb") as fh:
                    fh.write(b"frame")

            class ReviewDummy:
                base = "session"
                video_source_id = "sourcea"

                def _labeler_image_dir(self):
                    return tmp

            indices = VideoReviewDialog._existing_export_indices(ReviewDummy())

            self.assertEqual(indices, {3})

    def test_progress_label_shows_queue_count(self):
        with TemporaryDirectory() as tmp:
            labels_dir = os.path.join(tmp, "labels_all")
            os.makedirs(labels_dir, exist_ok=True)
            for name in ("queue_labeled", "shared"):
                with open(os.path.join(labels_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                    f.write("0 0.5 0.5 0.2 0.2\n")

            app = LabelingApp.__new__(LabelingApp)
            app.label_dir = labels_dir
            app.images_queue = ["queue_labeled.jpg", "queue_unlabeled.jpg", "shared.jpg"]
            app.progress_label = _FakeLabel()

            LabelingApp._update_progress_label(app)

            self.assertEqual(app.progress_label.text, "Queue: 2/3 labeled")

    def test_fit_class_selector_uses_longest_class_name(self):
        app = LabelingApp.__new__(LabelingApp)
        app.classes = ["mouse", "very_long_behavioral_state"]
        app.class_selector = _FakeCombo()

        LabelingApp._fit_class_selector_to_items(app)

        self.assertEqual(app.class_selector.minimum_contents_length, len("very_long_behavioral_state"))

    def test_distillation_sample_count_applies_stride_and_cap(self):
        self.assertEqual(_distillation_sample_count(0, 30), 0)
        self.assertEqual(_distillation_sample_count(1, 30), 1)
        self.assertEqual(_distillation_sample_count(61, 30), 3)
        self.assertEqual(_distillation_sample_count(1000, 30, 10), 10)
        self.assertEqual(_distillation_sample_count(100, 0), 100)

    def test_ensure_project_structure_creates_distillation_directories(self):
        with TemporaryDirectory() as tmp:
            paths = _ensure_project_structure(tmp)

            for key in (
                "videos",
                "distillation",
                "distillation_unlabeled_images",
                "distillation_runs",
            ):
                self.assertTrue(os.path.isdir(paths[key]), key)

            expected = _project_paths(tmp)
            self.assertEqual(paths["videos"], expected["videos"])
            self.assertEqual(paths["distillation_runs"], expected["distillation_runs"])

    def test_distillation_export_discovery_uses_project_runs(self):
        with TemporaryDirectory() as tmp:
            export_path = os.path.join(
                tmp,
                "runs",
                "distillation",
                "dinov3-pose",
                "exported_models",
                "exported_last.pt",
            )
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "wb") as fh:
                fh.write(b"weights")

            roots = _distillation_export_search_roots(tmp)
            exports = _discover_distillation_exports(roots)

            self.assertEqual(
                roots,
                [("Project runs", os.path.join(tmp, "runs", "distillation"))],
            )
            self.assertEqual(exports, [
                ("Project runs: dinov3-pose", os.path.abspath(export_path)),
            ])



if __name__ == "__main__":
    unittest.main()
