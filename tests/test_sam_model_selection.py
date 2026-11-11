"""Model selection tests use tiny files and a fake persistent SAM controller."""

import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from squeakpose.project.session import ProjectSession
from squeakpose.services.sam_models import SAM_MODELS, sam_download_target
from tests.test_annotation_controller_integration import (
    _SamAssistantController,
    qt_widgets,
    studio,
)


class SamModelPreferenceTests(unittest.TestCase):
    def test_old_new_and_explicit_preferences_round_trip(self):
        with TemporaryDirectory() as root:
            self.assertEqual(
                ProjectSession.from_preferences(root, {}).assistant_model_variant, "sam3.1"
            )
            Path(root, "sam3.pt").touch()
            self.assertEqual(
                ProjectSession.from_preferences(root, {}).assistant_model_variant, "sam3"
            )
            Path(root, "sam3.1_multiplex.pt").touch()
            for filename, key in (("sam3.pt", "sam3"), ("sam3.1_multiplex.pt", "sam3.1")):
                session = ProjectSession.from_preferences(root, {"sam_model_path": filename})
                self.assertEqual(session.assistant_model_variant, key)
                prefs = session.to_preferences()
                self.assertEqual(prefs["sam_model_path"], filename)
                self.assertEqual(prefs["sam_model_variant"], key)
            custom = Path(root, "custom-sam3.pth")
            custom.touch()
            session = ProjectSession.from_preferences(
                root,
                {
                    "sam_model_path": custom.name,
                    "sam_model_variant": "sam3",
                },
            )
            self.assertEqual(session.assistant_model_path, str(custom))
            self.assertEqual(session.to_preferences()["sam_model_variant"], "sam3")
            session = ProjectSession.from_preferences(root, {"sam_model_variant": "unsupported"})
            self.assertEqual(session.assistant_model_variant, "sam3.1")


class SamModelSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["sam-selection"]
        )
        cls.qt_app.setQuitOnLastWindowClosed(False)

    @contextmanager
    def window(self, root):
        paths = studio._ensure_project_structure(root)
        Path(paths["classes_file"]).write_text("mouse\n")
        Path(paths["keypoints_file"]).write_text("nose\n")
        Path(paths["class_keypoints_file"]).write_text('{"mouse": ["nose"]}\n')
        Path(paths["classes_seg_file"]).write_text("body\n")
        with ExitStack() as stack:
            stack.enter_context(patch("squeakpose.ui.main_window._auto_device", return_value="cpu"))
            stack.enter_context(
                patch("squeakpose.ui.main_window.LabelingApp._restart_prediction_worker")
            )
            stack.enter_context(
                patch("squeakpose.ui.main_window.SamAssistantController", _SamAssistantController)
            )
            stack.enter_context(patch("squeakpose.ui.main_window.QMessageBox.warning"))
            stack.enter_context(patch("squeakpose.ui.main_window.QMessageBox.information"))
            window = studio.LabelingApp(
                paths["images_to_label"],
                paths["labels_all"],
                paths["classes_file"],
                paths["keypoints_file"],
                project_root=root,
                force_initial_setup=False,
            )
            try:
                yield window
            finally:
                window.close()
                self.qt_app.processEvents()

    def test_new_project_defaults_to_sam31_and_selector_updates_download_ui(self):
        with TemporaryDirectory() as root, self.window(root) as window:
            window._switch_workflow(studio.WORKFLOW_SEG)
            self.assertEqual(window.sam_model_selector.currentData(), "sam3.1")
            self.assertEqual(window.sam_download_btn.text(), "Download SAM 3.1")
            for key, model in SAM_MODELS.items():
                window.sam_model_selector.setCurrentIndex(window.sam_model_selector.findData(key))
                self.assertEqual(window.sam_model_variant, key)
                self.assertEqual(window.sam_download_btn.text(), f"Download {model.display_name}")
                self.assertIn(model.filename, window.sam_download_btn.toolTip())
                self.assertEqual(window._read_project_meta()["sam_model_variant"], key)
                target = Path(sam_download_target(root, key))
                target.touch()
                window._refresh_sam_controls()
                self.assertEqual(window.sam_download_btn.text(), f"Load {model.display_name}")
                with (
                    patch.object(window, "_warm_sam_model") as warm,
                    patch("squeakpose.ui.main_window.WorkerJobController") as worker,
                ):
                    window._download_sam3_from_hugging_face()
                    warm.assert_called_once_with(str(target))
                    worker.assert_not_called()
                target.unlink()

    def test_persisted_sam31_wins_when_both_checkpoints_exist(self):
        with TemporaryDirectory() as root:
            Path(root, "sam3.pt").touch()
            selected = Path(root, "sam3.1_multiplex.pt")
            selected.touch()
            with self.window(root) as window:
                window._write_project_meta(
                    {"sam_model_path": selected.name, "sam_model_variant": "sam3"}
                )
                window._load_project_preferences()
                self.assertEqual(window.sam_model_variant, "sam3")
                window._try_autoload_sam_model_from_project_root()
                self.assertEqual(
                    window._sam_assistant_controller.restart_requests[-1]["model_path"],
                    str(selected),
                )

    def test_old_sam3_project_selects_and_autoloads_original_checkpoint(self):
        with TemporaryDirectory() as root:
            target = Path(root, "sam3.pt")
            target.touch()
            with self.window(root) as window:
                self.assertEqual(window.sam_model_selector.currentData(), "sam3")
                window._try_autoload_sam_model_from_project_root()
                self.assertEqual(
                    window._sam_assistant_controller.restart_requests[-1]["model_path"], str(target)
                )

    def test_autoload_prefers_selector_then_falls_back_and_preserves_custom_path(self):
        with TemporaryDirectory() as root, self.window(root) as window:
            for model in SAM_MODELS.values():
                Path(root, model.filename).touch()
            window.sam_model_variant = "sam3.1"
            with patch.object(window, "_warm_sam_model") as warm:
                window._try_autoload_sam_model_from_project_root()
                warm.assert_called_once_with(str(Path(root, "sam3.1_multiplex.pt")))
            Path(root, "sam3.1_multiplex.pt").unlink()
            with patch.object(window, "_warm_sam_model") as warm:
                window._try_autoload_sam_model_from_project_root()
                warm.assert_called_once_with(str(Path(root, "sam3.pt")))
            custom = Path(root, "custom-sam3.pth")
            custom.touch()
            window.sam_model_path = str(custom)
            with patch.object(window, "_warm_sam_model") as warm:
                window._try_autoload_sam_model_from_project_root()
                warm.assert_called_once_with(str(custom))

    def test_sam31_download_uses_selected_worker_argument_and_warms_result(self):
        with TemporaryDirectory() as root, self.window(root) as window:
            window._switch_workflow(studio.WORKFLOW_SEG)
            with (
                patch(
                    "squeakpose.ui.main_window.QMessageBox.question",
                    return_value=qt_widgets.QMessageBox.StandardButton.Yes,
                ),
                patch("squeakpose.ui.main_window.WorkerJobController") as worker,
            ):
                worker.return_value.start.return_value = True
                window._download_sam3_from_hugging_face()
                args = worker.return_value.start.call_args.args[1]
                self.assertEqual(args[-2:], ["--model", "sam3.1"])
                target = Path(root, "sam3.1_multiplex.pt")
                target.touch()
                window._sam3_download_event({"event": "result", "model_path": str(target)})
                window._sam3_download_finished(SimpleNamespace(succeeded=True))
                self.assertEqual(
                    window._sam_assistant_controller.restart_requests[-1]["model_path"], str(target)
                )
                self.assertEqual(window._read_project_meta()["sam_model_path"], target.name)

    def test_finished_download_does_not_warm_outside_segmentation_or_while_busy(self):
        with TemporaryDirectory() as root, self.window(root) as window:
            target = Path(sam_download_target(root, window.sam_model_variant))
            target.touch()

            def finish_download():
                window._sam3_download_event({"event": "result", "model_path": str(target)})
                window._sam3_download_finished(SimpleNamespace(succeeded=True))

            # Keypoints layer active: the model is not loaded behind the user's back.
            with patch.object(window, "_warm_sam_model") as warm:
                finish_download()
            warm.assert_not_called()
            self.assertIn("Segmentation layer", window.status.currentMessage())

            # Segmentation layer with a SAM request in flight: no worker restart.
            window._switch_workflow(studio.WORKFLOW_SEG)
            window._sam_assistant_controller.is_busy = True
            with patch.object(window, "_warm_sam_model") as warm:
                finish_download()
            warm.assert_not_called()

            window._sam_assistant_controller.is_busy = False
            with patch.object(window, "_warm_sam_model") as warm:
                finish_download()
            warm.assert_called_once_with(str(target))

    def test_manual_load_switches_checkpoint_and_preserves_custom_path(self):
        with TemporaryDirectory() as root, self.window(root) as window:
            custom = Path(root, "my-sam3.pth")
            custom.touch()
            with patch(
                "squeakpose.ui.main_window.QFileDialog.getOpenFileName",
                return_value=(str(custom), ""),
            ):
                window._load_sam_model_interactive()
            self.assertEqual(window.sam_model_path, str(custom))
            self.assertEqual(
                window._sam_assistant_controller.restart_requests[-1]["model_path"], str(custom)
            )
