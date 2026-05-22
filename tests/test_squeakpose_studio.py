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
    from squeakpose_studio import LabelingApp, VideoReviewDialog, WORKFLOW_POSE, WORKFLOW_SEG, _retain_main_window
    _STUDIO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent import gate
    if not _is_optional_studio_import_error(exc):
        raise
    LabelingApp = None
    VideoReviewDialog = None
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


if __name__ == "__main__":
    unittest.main()
