import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from squeakpose.services.video_library import VideoLibraryEntry
from squeakpose.ui.distillation_dialog import DistillationDialog
from squeakpose.workers.process import WorkerJobResult


class _VideoCapture:
    def __init__(self, total_frames=90):
        self.total_frames = total_frames
        self.released = False

    def isOpened(self):
        return True

    def get(self, _property):
        return self.total_frames

    def release(self):
        self.released = True


class DistillationDialogProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.parent = QWidget()
        self.parent.project_root = self.temp_dir.name
        self.parent.app_base_dir = os.getcwd()
        self.dialog = DistillationDialog(self.parent)

    def tearDown(self):
        self.dialog.deleteLater()
        self.parent.deleteLater()
        self.temp_dir.cleanup()

    def test_controller_output_preserves_plain_and_json_log_content(self):
        self.dialog._handle_process_output("plain output")
        self.dialog._handle_process_event({"event": "metric", "value": 3})

        log = self.dialog.log_view.toPlainText()
        self.assertIn("plain output", log)
        self.assertIn('"event": "metric"', log)
        self.assertEqual(self.dialog.status_label.text(), "Running")

    def test_process_factory_preserves_merged_terminal_channel(self):
        process = self.dialog._create_distillation_process(self.dialog)
        self.assertEqual(
            process.processChannelMode(),
            QProcess.ProcessChannelMode.MergedChannels,
        )
        process.deleteLater()

    def test_video_probe_resolves_project_alias_before_opening(self):
        selected = os.path.join(self.temp_dir.name, "videos", "session.mp4")
        resolved = os.path.join(self.temp_dir.name, "session", "session.mp4")
        capture = _VideoCapture()
        self.dialog.video_paths = [selected]

        with (
            patch(
                "squeakpose.ui.distillation_dialog.resolve_project_video_paths",
                return_value=[resolved],
            ) as resolve_paths,
            patch(
                "squeakpose.ui.distillation_dialog._cv2.VideoCapture",
                return_value=capture,
            ) as open_video,
        ):
            probes, errors = self.dialog._probe_selected_videos()

        resolve_paths.assert_called_once_with(self.dialog.paths["videos"], [selected])
        open_video.assert_called_once_with(resolved)
        self.assertEqual(probes, [(resolved, 90, 3)])
        self.assertEqual(errors, [])
        self.assertTrue(capture.released)

    def test_project_picker_selection_keeps_library_paths_for_late_resolution(self):
        videos_dir = self.dialog.paths["videos"]
        entry = VideoLibraryEntry(
            name="session.mp4",
            path=os.path.join(self.temp_dir.name, "source", "session.mp4"),
            is_link=True,
            target=os.path.join(self.temp_dir.name, "source", "session.mp4"),
            target_exists=True,
        )
        picker = Mock()
        picker.exec.return_value = QDialog.DialogCode.Accepted
        picker.selected_entries = (entry,)

        with patch(
            "squeakpose.ui.distillation_dialog.ProjectVideoPickerDialog",
            return_value=picker,
        ) as picker_class:
            self.dialog._add_videos()

        picker_class.assert_called_once()
        self.assertEqual(
            self.dialog.video_paths,
            [os.path.join(videos_dir, entry.name)],
        )
        self.assertEqual(self.dialog.video_list.item(0).text(), entry.name)
        self.assertIn(entry.target, self.dialog.video_list.item(0).toolTip())

    def test_success_result_restores_controls_and_preserves_completion_message(self):
        self.dialog.run_btn.setEnabled(False)
        self.dialog.cancel_btn.setEnabled(True)
        self.dialog.create_dataset_btn.setEnabled(False)

        with patch.object(self.dialog, "_output_dir", return_value="/project/run"):
            with patch("squeakpose.ui.distillation_dialog.QMessageBox.information") as information:
                self.dialog._finish_process_result(
                    WorkerJobResult(
                        state="finished",
                        exit_code=0,
                        exit_status=QProcess.ExitStatus.NormalExit,
                    )
                )

        self.assertEqual(self.dialog.status_label.text(), "Complete")
        self.assertTrue(self.dialog.run_btn.isEnabled())
        self.assertFalse(self.dialog.cancel_btn.isEnabled())
        self.assertTrue(self.dialog.create_dataset_btn.isEnabled())
        self.assertIn(
            "Distillation complete. Output: /project/run", self.dialog.log_view.toPlainText()
        )
        information.assert_called_once()

    def test_failure_and_cancellation_keep_distinct_terminal_behavior(self):
        with patch("squeakpose.ui.distillation_dialog.QMessageBox.critical") as critical:
            self.dialog._finish_process_result(
                WorkerJobResult(
                    state="start_failed",
                    error_message="missing interpreter",
                )
            )

        self.assertEqual(self.dialog.status_label.text(), "Failed")
        self.assertIn("Process error: missing interpreter", self.dialog.log_view.toPlainText())
        critical.assert_called_once()

        self.dialog.log_view.clear()
        self.dialog.cancel_requested = True
        with patch("squeakpose.ui.distillation_dialog.QMessageBox.critical") as critical:
            self.dialog._finish_process_result(
                WorkerJobResult(
                    state="cancelled",
                    exit_code=-1,
                    exit_status=QProcess.ExitStatus.CrashExit,
                )
            )

        self.assertEqual(self.dialog.status_label.text(), "Canceled")
        self.assertIn("Distillation canceled", self.dialog.log_view.toPlainText())
        critical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
