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
    def test_model_sizes_follow_task_and_custom_model_survives_task_switch(self):
        for size in ("n", "s", "m", "l", "x"):
            for task, suffix in (("pose", "pose"), ("segment", "seg")):
                self.dialog.model_size_combo.setCurrentIndex(
                    self.dialog.model_size_combo.findData(size)
                )
                self.dialog.task_combo.setCurrentIndex(self.dialog.task_combo.findData(task))
                self.assertEqual(
                    self.dialog.student_edit.text(), f"ultralytics/yolo26{size}-{suffix}.pt"
                )
        self.dialog.model_size_combo.setCurrentIndex(
            self.dialog.model_size_combo.findData("custom")
        )
        self.dialog.student_edit.setText("/models/custom.pt")
        self.dialog.task_combo.setCurrentIndex(0)
        self.assertEqual(self.dialog.student_edit.text(), "/models/custom.pt")
        self.assertFalse(self.dialog.student_edit.isReadOnly())
        self.dialog.resume_check.setChecked(True)
        self.assertFalse(self.dialog.model_size_combo.isEnabled())
        self.assertFalse(self.dialog.student_browse_btn.isEnabled())

    def test_initial_task_follows_active_layer(self):
        for layer, task, suffix in (
            ("keypoints", "pose", "pose"),
            ("segmentation", "segment", "seg"),
        ):
            with self.subTest(layer=layer):
                self.parent.active_layer = layer
                dialog = DistillationDialog(self.parent)
                try:
                    self.assertEqual(dialog._selected_task(), task)
                    self.assertEqual(dialog._current_task, task)
                    self.assertEqual(dialog.student_edit.text(), f"ultralytics/yolo26n-{suffix}.pt")
                    self.assertEqual(
                        dialog.run_name_edit.text(), dialog.TASK_DEFAULTS[task]["run_name"]
                    )
                finally:
                    dialog.deleteLater()

    def test_prepare_step_leads_to_run_settings(self):
        self.assertFalse(self.dialog.next_btn.isHidden())
        self.assertTrue(self.dialog.run_btn.isHidden())
        self.dialog.next_btn.click()
        self.assertEqual(self.dialog.tabs.currentIndex(), 1)
        self.assertTrue(self.dialog.next_btn.isHidden())
        self.assertFalse(self.dialog.run_btn.isHidden())
        self.dialog.tabs.setCurrentIndex(0)
        self.assertFalse(self.dialog.next_btn.isHidden())
        self.assertTrue(self.dialog.run_btn.isHidden())

    def test_video_empty_state_tracks_selection_and_clear(self):
        self.assertFalse(self.dialog.video_empty_label.isHidden())
        self.assertFalse(self.dialog.clear_video_btn.isEnabled())
        self.dialog.video_paths = ["example.mp4"]
        self.dialog.video_list.addItem("example.mp4")
        self.assertTrue(self.dialog.video_empty_label.isHidden())
        self.assertTrue(self.dialog.clear_video_btn.isEnabled())
        self.dialog.video_list.setCurrentRow(0)
        self.assertTrue(self.dialog.remove_video_btn.isEnabled())
        self.dialog.clear_video_btn.click()
        self.assertEqual(self.dialog.video_paths, [])
        self.assertFalse(self.dialog.video_empty_label.isHidden())
        self.assertFalse(self.dialog.remove_video_btn.isEnabled())

    def test_effective_batch_hint_tracks_both_controls(self):
        self.dialog.batch_spin.setValue(4)
        self.dialog.accumulation_spin.setValue(8)
        self.assertIn("32 images", self.dialog.batch_hint.text())

    def test_resume_and_restart_are_exclusive_and_precision_is_automatic(self):
        self.assertEqual(self.dialog.precision_combo.currentText(), "auto")
        self.assertEqual(self.dialog.accumulation_spin.value(), 2)
        self.dialog.overwrite_check.setChecked(True)
        self.dialog.resume_check.setChecked(True)
        self.assertFalse(self.dialog.overwrite_check.isChecked())
        self.assertFalse(self.dialog.epochs_spin.isEnabled())
        self.dialog.overwrite_check.setChecked(True)
        self.assertFalse(self.dialog.resume_check.isChecked())
        self.assertTrue(self.dialog.epochs_spin.isEnabled())

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

    def test_monitor_parses_lightning_progress_and_resumed_epoch_count(self):
        self.dialog._handle_process_output('"epochs": 300,')
        self.dialog._handle_process_output(
            "\x1b[32mEpoch 5:  50%|█████     | 125/249 [00:46<00:46, "
            "2.66it/s, train_loss=1.230, data_wait=8.2%]\x1b[0m"
        )
        self.assertEqual(self.dialog.epoch_label.text(), "Epoch 6 / 300")
        self.assertEqual(self.dialog.epoch_progress.value(), 125)
        self.assertEqual(self.dialog.epoch_progress.maximum(), 249)
        self.assertEqual(self.dialog.loss_label.text(), "Batch loss 1.230")
        self.assertEqual(self.dialog.overall_progress.value(), 18)
        # Lightning can reset the old epoch's bar before announcing the next epoch.
        self.dialog._handle_process_output("Epoch 5: 0%| | 0/249 [00:00<?, ?it/s]")
        self.assertEqual(self.dialog.epoch_progress.value(), 125)
        self.dialog._handle_process_output("unrecognized diagnostic line")
        self.assertEqual(self.dialog.epoch_progress.value(), 125)
        self.assertIn("unrecognized diagnostic line", self.dialog.log_view.toPlainText())
        self.dialog._handle_process_output("Epoch 6: 0%| | 0/0 [00:00<?, ?it/s]")
        self.assertEqual(self.dialog.epoch_progress.value(), 125)

    def test_monitor_elapsed_freezes_when_run_finishes(self):
        self.dialog._run_started_at = 100.0
        self.dialog._elapsed_timer.start()
        with patch("squeakpose.ui.distillation_dialog.time.monotonic", return_value=3761.0):
            with patch("squeakpose.ui.distillation_dialog.QMessageBox.information"):
                self.dialog._finish_process_result(WorkerJobResult(state="finished", exit_code=0))
        self.assertEqual(self.dialog.elapsed_label.text(), "Elapsed 1:01:01")
        self.assertFalse(self.dialog._elapsed_timer.isActive())
        self.assertEqual(self.dialog.overall_progress.value(), 1000)

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

    def test_completion_message_uses_output_dir_captured_at_start(self):
        started_output = os.path.join(self.temp_dir.name, "runs", "started_run")
        self.dialog._active_output_dir = started_output
        # The run-name field stays editable while the run is in progress.
        self.dialog.run_name_edit.setText("edited_after_start")

        with patch("squeakpose.ui.distillation_dialog.QMessageBox.information"):
            self.dialog._finish_process_result(
                WorkerJobResult(
                    state="finished",
                    exit_code=0,
                    exit_status=QProcess.ExitStatus.NormalExit,
                )
            )

        log = self.dialog.log_view.toPlainText()
        self.assertIn(f"Distillation complete. Output: {started_output}", log)
        self.assertNotIn("edited_after_start", log)
        self.assertEqual(self.dialog._active_output_dir, "")

    def test_output_after_cancel_keeps_canceling_status(self):
        controller = Mock()
        controller.is_running = True
        self.dialog.process_controller = controller

        self.dialog._cancel_process()
        self.dialog._handle_process_output("epoch 3 loss 0.1")
        self.dialog._handle_process_event({"event": "metric", "value": 3})

        self.assertEqual(self.dialog.status_label.text(), "Canceling")
        self.assertIn("epoch 3 loss 0.1", self.dialog.log_view.toPlainText())
        controller.cancel.assert_called_once_with(kill_after_ms=5000)
        self.dialog.process_controller = None

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
