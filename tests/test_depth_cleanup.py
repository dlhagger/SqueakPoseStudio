import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QMessageBox

from squeakpose.project.paths import ProjectPaths
from squeakpose.project.safety import ProjectPathError
from squeakpose.services.depth_cleanup import delete_project_depth_data
from squeakpose.ui.main_window import LabelingApp


class DepthCleanupTests(unittest.TestCase):
    def test_removes_only_depth_outputs_and_keeps_required_directories(self):
        with TemporaryDirectory() as tmp:
            paths = ProjectPaths.from_root(tmp)
            outputs = [
                paths.depth_images,
                paths.depth_previews,
                paths.inference_depth,
                paths.analysis_depth,
            ]
            keep = [
                paths.images_all,
                paths.labels_all,
                paths.labels_seg_all,
                paths.inference_keypoints,
                paths.inference_segmentation,
                str(Path(tmp) / "models"),
            ]
            for directory in outputs + keep:
                Path(directory).mkdir(parents=True, exist_ok=True)
                Path(directory, "sample.dat").write_text("data")
            # Nested links must not cause deletion outside the output folders.
            Path(paths.depth_images, "linked-models").symlink_to(keep[-1], target_is_directory=True)
            delete_project_depth_data(tmp)
            for directory in outputs:
                self.assertTrue(Path(directory).is_dir())
                self.assertEqual(list(Path(directory).iterdir()), [])
            for directory in keep:
                self.assertEqual(Path(directory, "sample.dat").read_text(), "data")
            delete_project_depth_data(tmp)  # Already empty is safe.

    def test_rejects_redirected_directory_before_deleting_any_data(self):
        with TemporaryDirectory() as tmp:
            paths = ProjectPaths.from_root(tmp)
            Path(paths.depth_images).mkdir(parents=True)
            original = Path(paths.depth_images, "frame.npy")
            original.write_text("keep")
            Path(paths.inference_keypoints).mkdir(parents=True)
            Path(paths.inference_depth).symlink_to(
                paths.inference_keypoints, target_is_directory=True
            )
            with self.assertRaises(ProjectPathError):
                delete_project_depth_data(tmp)
            self.assertTrue(original.exists())

    def window(self):
        return SimpleNamespace(
            project_root="/project",
            _is_depth_layer=lambda: True,
            _instance_operation_busy=lambda: False,
            _clear_depth_probes=Mock(),
            _depth_controller=Mock(),
            load_image=Mock(),
            _update_progress_label=Mock(),
            update_status_bar=Mock(),
        )

    def test_cancel_never_deletes(self):
        window = self.window()
        with patch("squeakpose.ui.main_window.delete_project_depth_data") as delete:
            with patch.object(
                QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel
            ):
                LabelingApp._delete_all_depth_data(window)
        delete.assert_not_called()
        window.load_image.assert_not_called()

    def test_confirm_deletes_and_refreshes_depth_display(self):
        window = self.window()
        with patch("squeakpose.ui.main_window.delete_project_depth_data") as delete:
            with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
                LabelingApp._delete_all_depth_data(window)
        delete.assert_called_once_with("/project")
        window._depth_controller.clear_image.assert_called_once()
        window.load_image.assert_called_once()
        window._update_progress_label.assert_called_once()

    def test_running_inference_blocks_deletion(self):
        window = self.window()
        window._inference_coordinator = SimpleNamespace(is_busy=True)
        with patch("squeakpose.ui.main_window.delete_project_depth_data") as delete:
            with patch.object(QMessageBox, "information") as info:
                LabelingApp._delete_all_depth_data(window)
        delete.assert_not_called()
        info.assert_called_once()

    def test_job_started_during_confirmation_blocks_deletion(self):
        window = self.window()
        window._instance_operation_busy = Mock(side_effect=[False, True])
        with patch("squeakpose.ui.main_window.delete_project_depth_data") as delete:
            with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
                LabelingApp._delete_all_depth_data(window)
        delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
