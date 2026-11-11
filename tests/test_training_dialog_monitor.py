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

    def test_stale_dataset_yaml_requires_confirmation_before_training(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            moved = Path(tmp) / "moved" / "datasets" / "pose"
            moved.mkdir(parents=True)
            original = Path(tmp) / "original" / "datasets" / "pose"
            original.mkdir(parents=True)
            stale_yaml = moved / "dataset.yaml"
            stale_yaml.write_text(
                f"path: {original}\ntrain: images/train\nval: images/val\n", encoding="utf-8"
            )
            current_yaml = original / "dataset.yaml"
            current_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")

            with patch.object(QMessageBox, "question") as question:
                self.assertTrue(dialog._confirm_dataset_location(str(current_yaml)))
            question.assert_not_called()

            no = QMessageBox.StandardButton.No
            with patch.object(QMessageBox, "question", return_value=no) as question:
                self.assertFalse(dialog._confirm_dataset_location(str(stale_yaml)))
            question.assert_called_once()
            self.assertIn(str(original), question.call_args.args[2])

            yes = QMessageBox.StandardButton.Yes
            with patch.object(QMessageBox, "question", return_value=yes):
                self.assertTrue(dialog._confirm_dataset_location(str(stale_yaml)))
            parent.close()

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

    def test_final_validation_updates_metrics_without_new_history_row(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            dialog._reset_training_monitor(100)
            dialog._handle_training_event(
                {
                    "event": "epoch_end",
                    "epoch": 100,
                    "epochs": 100,
                    "metrics": {"metrics/mAP50-95(M)": 0.40},
                    "best_fitness": 0.45,
                }
            )

            dialog._handle_training_event(
                {
                    "event": "final_validation",
                    "epochs_completed": 100,
                    "epochs": 100,
                    "metrics": {
                        "metrics/precision(M)": 0.9,
                        "metrics/mAP50-95(M)": 0.47,
                    },
                    "model_path": "/runs/segment/train/weights/best.pt",
                }
            )

            self.assertEqual(dialog.history_table.rowCount(), 1)
            self.assertEqual(dialog.epoch_label.text(), "Epoch 100 / 100")
            self.assertNotIn("101", dialog.epoch_label.text())
            self.assertEqual(dialog.phase_label.text(), "Final validation (best.pt)")
            self.assertEqual(dialog.metric_values["primary_map"].text(), "0.4700")
            self.assertIn("Final validation (best.pt)", dialog.log_view.toPlainText())
            dialog.close()
            parent.close()

    def test_canceled_result_points_to_resumable_checkpoint(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            dialog.training_controller = _FakeTrainingController()
            dialog.training_running = True
            dialog.train_cancel_requested = True
            resumable = "/runs/pose/train7/weights/last_resumable.pt"
            dialog.train_result_event = {
                "event": "result",
                "canceled": True,
                "had_error": False,
                "error_message": "",
                "save_dir": "/runs/pose/train7",
                "resume_checkpoint": resumable,
            }

            with patch.object(QMessageBox, "information") as info:
                dialog._finish_training_job(WorkerJobResult(state="cancelled", exit_code=0))

            message = info.call_args.args[2]
            self.assertIn(resumable, message)
            self.assertIn("Resume YOLO run (exact)", message)
            self.assertIn(f"Resumable checkpoint: {resumable}", dialog.log_view.toPlainText())
            parent.close()

    def test_cancel_during_final_validation_of_finished_run_reports_completion(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            dialog.training_controller = _FakeTrainingController()
            dialog.training_running = True
            dialog.train_cancel_requested = True
            dialog.train_result_event = {
                "event": "result",
                "canceled": False,
                "had_error": False,
                "error_message": "",
                "save_dir": "/runs/pose/train8",
            }

            with patch.object(QMessageBox, "information") as info:
                dialog._finish_training_job(WorkerJobResult(state="cancelled", exit_code=0))

            self.assertEqual(info.call_args.args[1], "Training complete")
            self.assertEqual(dialog.train_status_label.text(), "Complete")
            self.assertIn("completed normally", dialog.log_view.toPlainText())
            parent.close()

    def test_resume_list_offers_resumable_checkpoint(self):
        with TemporaryDirectory() as tmp:
            parent, dialog = self._make_dialog(tmp)
            weights = Path(tmp) / "runs" / "train" / "pose" / "mouse" / "weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"stripped")
            (weights / "last_resumable.pt").write_bytes(b"full")
            dialog.project_runs_dir = str(Path(tmp) / "runs")

            dialog._refresh_resume_list()

            paths = [path for _label, path in dialog.resume_exports]
            self.assertIn(str(weights / "last_resumable.pt"), paths)
            dialog.close()
            parent.close()


if __name__ == "__main__":
    unittest.main()
