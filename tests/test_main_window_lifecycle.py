import importlib
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
from unittest.mock import Mock, patch

# This must be selected before importing PyQt or creating QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

_OPTIONAL_UI_MODULES = {"PyQt6", "cv2", "numpy", "torch", "ultralytics", "yaml"}


def _is_optional_ui_import_error(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        name = (getattr(exc, "name", "") or "").split(".", 1)[0]
        return name in _OPTIONAL_UI_MODULES
    if isinstance(exc, ImportError):
        message = str(exc)
        return any(module in message for module in _OPTIONAL_UI_MODULES)
    return False


try:
    QApplication = importlib.import_module("PyQt6.QtWidgets").QApplication
    QImage = importlib.import_module("PyQt6.QtGui").QImage
    studio = importlib.import_module("squeakpose_studio")
    project_safety = importlib.import_module("squeakpose.project.safety")
    InferenceController = importlib.import_module(
        "squeakpose.ui.inference_controller"
    ).InferenceController
    PredictionController = importlib.import_module(
        "squeakpose.ui.prediction_controller"
    ).PredictionController
    SamAssistantController = importlib.import_module(
        "squeakpose.ui.sam_assistant_controller"
    ).SamAssistantController
    _UI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent import gate
    if not _is_optional_ui_import_error(exc):
        raise
    QApplication = None
    QImage = None
    studio = None
    project_safety = None
    InferenceController = None
    PredictionController = None
    SamAssistantController = None
    _UI_IMPORT_ERROR = exc


@unittest.skipIf(studio is None, f"UI imports unavailable: {_UI_IMPORT_ERROR}")
class MainWindowLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication(["squeakpose-lifecycle-test"])
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def test_complete_main_window_constructs_and_closes_offscreen(self):
        self.assertEqual(os.environ.get("QT_QPA_PLATFORM"), "offscreen")

        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            paths = studio._ensure_project_structure(str(project_root))
            Path(paths["classes_file"]).write_text("mouse\n", encoding="utf-8")
            Path(paths["keypoints_file"]).write_text("nose\ntail_base\n", encoding="utf-8")
            Path(paths["class_keypoints_file"]).write_text(
                '{"mouse": ["nose", "tail_base"]}\n',
                encoding="utf-8",
            )

            image_path = Path(paths["images_to_label"]) / "frame001.png"
            image = QImage(32, 24, QImage.Format.Format_RGB32)
            image.fill(0xFFE7EBEF)
            self.assertTrue(image.save(str(image_path)))

            lock_path = Path(project_safety.project_lock_path(str(project_root)))
            window = None
            modal_patches = [
                patch(
                    f"squeakpose.ui.main_window.QMessageBox.{method}",
                    side_effect=AssertionError(f"unexpected modal QMessageBox.{method}"),
                )
                for method in ("critical", "information", "question", "warning")
            ]

            with ExitStack() as stack:
                for modal_patch in modal_patches:
                    stack.enter_context(modal_patch)
                stack.enter_context(
                    patch(
                        "squeakpose.ui.main_window.ClassManagerDialog.exec",
                        side_effect=AssertionError("unexpected initial class manager"),
                    )
                )
                stack.enter_context(
                    patch("squeakpose.ui.main_window._auto_device", return_value="cpu")
                )
                try:
                    window = studio.LabelingApp(
                        paths["images_to_label"],
                        paths["labels_all"],
                        paths["classes_file"],
                        paths["keypoints_file"],
                        project_root=paths["root"],
                        force_initial_setup=False,
                    )
                    window.show()
                    self.qt_app.processEvents()

                    self.assertTrue(window.isVisible())
                    self.assertEqual(window.project_root, str(project_root))
                    self.assertEqual(window.images, ["frame001.png"])
                    self.assertEqual(window.current_image_path, str(image_path))
                    self.assertEqual((window.img_w, window.img_h), (32, 24))
                    self.assertEqual(window.classes, ["mouse"])
                    self.assertEqual(window.kp_names, ["nose", "tail_base"])
                    for panel_name in (
                        "top_left_frame",
                        "top_right_frame",
                        "analysis_frame",
                        "bottom_left_frame",
                        "bottom_right_frame",
                    ):
                        self.assertTrue(
                            getattr(window, panel_name).isVisibleTo(window),
                            panel_name,
                        )
                    self.assertEqual(window.workflow_selector.count(), 3)
                    self.assertEqual(
                        [
                            window.workflow_selector.itemData(index)
                            for index in range(window.workflow_selector.count())
                        ],
                        ["keypoints", "segmentation", "depth"],
                    )
                    self.assertEqual(window.workflow_selector.currentData(), "keypoints")
                    self.assertEqual(window.class_selector.currentText(), "mouse")
                    self.assertEqual(window.filter_combo.count(), 3)
                    for control_name in (
                        "save_btn",
                        "delete_image_btn",
                        "predict_btn",
                        "analysis_btn",
                        "export_dataset_btn",
                        "train_btn",
                        "inference_btn",
                    ):
                        control = getattr(window, control_name)
                        self.assertTrue(control.isVisibleTo(window), control_name)
                        self.assertTrue(control.isEnabled(), control_name)
                    self.assertTrue(window.bbox_btn.isVisibleTo(window))
                    self.assertTrue(window.keypoint_btn.isVisibleTo(window))
                    self.assertTrue(window.seg_tools_frame.isHidden())
                    self.assertTrue(window.depth_display_frame.isHidden())
                    self.assertTrue(window.depth_assistant_frame.isHidden())
                    self.assertIn("Keypoints", window.dataset_training_title.text())
                    self.assertEqual(
                        window.model_inference_title.text(),
                        "Project Models & Inference",
                    )
                    self.assertFalse(hasattr(window, "sam_model"))
                    self.assertIsInstance(
                        window._sam_assistant_controller,
                        SamAssistantController,
                    )
                    self.assertIsInstance(window._prediction_coordinator, PredictionController)
                    self.assertIsInstance(window._inference_coordinator, InferenceController)
                    self.assertFalse(window._prediction_coordinator.is_busy)
                    self.assertFalse(window._inference_coordinator.is_busy)
                    window.predict_model_path = "pose.pt"
                    with patch.object(
                        window._prediction_coordinator,
                        "submit_prediction",
                        return_value=41,
                    ) as submit:
                        window.run_prediction_on_current_image()
                    submit.assert_called_once_with(
                        layer_id="keypoints",
                        model_path="pose.pt",
                        image_path=str(image_path),
                        device="cpu",
                        depth_targets=None,
                    )
                    window.layer_model_paths["keypoints"] = "pose.pt"
                    inference_dialog = SimpleNamespace(
                        exec=lambda: 1,
                        selected_video_paths=(str(project_root / "video.mp4"),),
                        selected_video_settings=(
                            SimpleNamespace(
                                video_path=str(project_root / "video.mp4"),
                                expected_animal_count=2,
                                requested_tracker="botsort",
                            ),
                        ),
                        batch_size=4,
                    )
                    with (
                        patch(
                            "squeakpose.ui.main_window.InferenceVideoDialog",
                            return_value=inference_dialog,
                        ),
                        patch(
                            "squeakpose.ui.main_window.probe_video_metadata",
                            return_value=SimpleNamespace(opened=True, total_frames=12, fps=30.0),
                        ),
                        patch.object(window._inference_coordinator, "start") as start,
                    ):
                        window.run_video_inference()
                    inference_plan = start.call_args.args[0]
                    start.assert_called_once()
                    self.assertEqual(inference_plan.jobs[0].worker_config()["batch_size"], 4)
                    self.assertEqual(
                        inference_plan.jobs[0].worker_config()["expected_animal_count"], 2
                    )
                    self.assertEqual(
                        inference_plan.jobs[0].worker_config()["resolved_tracker"], "botsort"
                    )
                    self.assertTrue(lock_path.is_file())
                finally:
                    if window is not None:
                        window.close()
                        self.qt_app.processEvents()

            self.assertFalse(lock_path.exists())

    def test_finished_progress_dialog_cannot_cancel_remaining_video_batch(self):
        callbacks = []

        class FakeSignal:
            def disconnect(self, callback):
                callbacks.remove(callback)

        class FakeProgress:
            canceled = FakeSignal()

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True
                for callback in list(callbacks):
                    callback()

        cancel = Mock()
        progress = FakeProgress()
        callbacks.append(cancel)
        window = SimpleNamespace(
            _inference_progress=progress,
            _cancel_inference_process=cancel,
        )

        studio.LabelingApp._inference_controller_pass_finished(window, SimpleNamespace())

        self.assertTrue(progress.closed)
        self.assertIsNone(window._inference_progress)
        cancel.assert_not_called()

    def test_intermediate_video_completion_starts_next_without_message(self):
        next_plan = object()
        coordinator = SimpleNamespace(start=Mock())
        window = SimpleNamespace(
            _inference_batch_summaries=[],
            _inference_plan_queue=[next_plan],
            _inference_batch_index=1,
            _inference_batch_total=2,
            _inference_coordinator=coordinator,
        )
        summary = SimpleNamespace(results=(object(),), canceled=False)

        with (
            patch("squeakpose.ui.main_window.QMessageBox.information") as information,
            patch("squeakpose.ui.main_window.QMessageBox.warning") as warning,
        ):
            studio.LabelingApp._inference_controller_completed(window, summary)

        coordinator.start.assert_called_once_with(next_plan)
        information.assert_not_called()
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
