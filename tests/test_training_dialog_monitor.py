import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

from PyQt6.QtWidgets import QApplication, QWidget

from squeakpose.ui.training_dialog import TrainDialog


class TrainingDialogMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["training-monitor-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_structured_events_update_progress_cards_and_epoch_history(self):
        with TemporaryDirectory() as tmp:
            parent = QWidget()
            parent.project_root = tmp
            parent.app_base_dir = str(Path(__file__).resolve().parents[1])
            dialog = TrainDialog(
                parent,
                str(Path(tmp) / "datasets" / "segment"),
                default_task="segment",
                layer_id="segmentation",
            )
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
