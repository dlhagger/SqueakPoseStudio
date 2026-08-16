import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QWidget

from squeakpose.ui.distillation_dialog import DistillationDialog
from squeakpose.workers.process import WorkerJobResult


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
