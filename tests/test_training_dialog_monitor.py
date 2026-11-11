import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from squeakpose.ui.training_dialog import TRAINING_CANCEL_KILL_AFTER_MS, TrainDialog
from squeakpose.workers.process import WorkerJobResult


class _FakeTerminalSignal:
    def disconnect(self, _slot):
        return None


class _FakeTrainingController:
    def __init__(self):
        self.terminal = _FakeTerminalSignal()
        self.shutdown_calls = 0
        self.delete_later_calls = 0
        self.cancel_calls = []
        self.force_stop_calls = 0
        self.is_running = True

    def cancel(self, *, kill_after_ms=5000):
        self.cancel_calls.append(kill_after_ms)
        return True

    def force_stop(self):
        self.force_stop_calls += 1
        return True

    def shutdown(self):
        self.shutdown_calls += 1

    def deleteLater(self):
        self.delete_later_calls += 1


class TrainingDialogMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["training-monitor-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def _make_dialog(self, tmp):
        parent = QWidget()
        parent.project_root = tmp
        parent.app_base_dir = str(Path(__file__).resolve().parents[1])
        dialog = TrainDialog(
            parent,
            str(Path(tmp) / "datasets" / "segment"),
            default_task="segment",
            layer_id="segmentation",
        )
        return parent, dialog

    def test_terminal_output_lines_are_committed_individually(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)

            dialog._write_training_terminal_output("Ultralytics 8.3")
            dialog._write_training_terminal_output("Epoch GPU_mem box_loss")

            self.assertEqual(
                dialog.log_view.toPlainText().splitlines(),
                ["Ultralytics 8.3", "Epoch GPU_mem box_loss"],
            )
            self.assertEqual(dialog.training_console.pending, "")
            dialog.close()
            parent.close()

    def test_reject_keeps_dialog_open_when_training_cancel_is_declined(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            controller = _FakeTrainingController()
            dialog.training_controller = controller
            dialog.training_running = True
            dialog.show()

            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
                dialog.reject()

            self.assertTrue(dialog.isVisible())
            self.assertTrue(dialog.training_running)
            self.assertIs(dialog.training_controller, controller)
            self.assertEqual(controller.shutdown_calls, 0)

            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                dialog.reject()

            self.assertFalse(dialog.isVisible())
            self.assertFalse(dialog.training_running)
            self.assertIsNone(dialog.training_controller)
            self.assertEqual(controller.shutdown_calls, 1)
            parent.close()

    def test_cancel_is_graceful_first_and_force_stop_second(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            controller = _FakeTrainingController()
            dialog.training_controller = controller
            dialog.training_running = True
            dialog.cancel_train_btn.setEnabled(True)

            dialog._cancel_training_process()

            self.assertEqual(controller.cancel_calls, [TRAINING_CANCEL_KILL_AFTER_MS])
            self.assertEqual(controller.force_stop_calls, 0)
            self.assertEqual(dialog.train_status_label.text(), "Canceling")
            self.assertEqual(dialog.cancel_train_btn.text(), "Force Stop")

            # Progress that arrives while Ultralytics validates/saves must not
            # flip the status back to "Running".
            dialog._handle_training_event(
                {"event": "batch_progress", "epoch": 1, "epochs": 3, "batch": 2, "batches": 4}
            )
            dialog._handle_training_event(
                {"event": "training_stopping", "message": "Stopping after the current batch"}
            )
            self.assertEqual(dialog.train_status_label.text(), "Canceling")

            dialog._cancel_training_process()

            self.assertEqual(controller.force_stop_calls, 1)
            self.assertEqual(len(controller.cancel_calls), 1)
            parent.close()

    def test_canceled_result_with_checkpoints_reports_save_dir(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            controller = _FakeTrainingController()
            dialog.training_controller = controller
            dialog.training_running = True
            dialog.train_cancel_requested = True
            dialog.cancel_train_btn.setText("Force Stop")
            dialog.train_result_event = {
                "event": "result",
                "canceled": True,
                "had_error": False,
                "error_message": "",
                "save_dir": "/runs/pose/train7",
            }

            with patch.object(QMessageBox, "information") as info:
                dialog._finish_training_job(WorkerJobResult(state="cancelled", exit_code=0))

            info.assert_called_once()
            self.assertIn("/runs/pose/train7", info.call_args.args[2])
            self.assertEqual(dialog.train_status_label.text(), "Canceled")
            self.assertEqual(dialog.cancel_train_btn.text(), "Cancel Training")
            self.assertFalse(dialog.cancel_train_btn.isEnabled())
            self.assertIn("Checkpoints saved to: /runs/pose/train7", dialog.log_view.toPlainText())
            parent.close()

    def test_structured_events_update_progress_cards_and_epoch_history(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            dialog._reset_training_monitor(10)

            dialog._handle_training_event(
                {"event": "epoch_start", "epoch": 3, "epochs": 10, "batches": 20}
            )
            dialog._handle_training_event(
                {
                    "event": "batch_progress",
                    "epoch": 3,
                    "epochs": 10,
                    "batch": 5,
                    "batches": 20,
                    "losses": {"box_loss": 1.0, "seg_loss": 2.0},
                    "eta_seconds": 15,
                }
            )
            dialog._handle_training_event(
                {
                    "event": "epoch_end",
                    "epoch": 3,
                    "epochs": 10,
                    "epoch_seconds": 12,
                    "eta_seconds": 84,
                    "losses": {"box_loss": 0.6, "seg_loss": 0.9},
                    "metrics": {
                        "metrics/precision(B)": 0.5,
                        "metrics/precision(M)": 0.8,
                        "metrics/recall(M)": 0.7,
                        "metrics/mAP50(M)": 0.65,
                        "metrics/mAP50-95(M)": 0.45,
                    },
                    "best_fitness": 0.45,
                }
            )

            self.assertEqual(dialog.epoch_label.text(), "Epoch 3 / 10")
            self.assertEqual(dialog.overall_progress.value(), 300)
            self.assertEqual(dialog.epoch_progress.value(), 1000)
            self.assertEqual(dialog.metric_values["primary_map"].text(), "0.4500")
            self.assertEqual(dialog.metric_values["precision"].text(), "0.8000")
            self.assertEqual(dialog.metric_values["loss"].text(), "1.5000")
            self.assertEqual(dialog.history_table.rowCount(), 1)
            self.assertEqual(dialog.history_table.item(0, 6).text(), "0.4500")
            dialog.close()
            parent.close()


if __name__ == "__main__":
    unittest.main()
