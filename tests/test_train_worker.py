import time
import unittest
from collections import defaultdict
from types import SimpleNamespace

from train_worker import run_training_worker


class _TrainResults:
    save_dir = "/tmp/squeakpose-run"


class _TrainModel:
    def __init__(self):
        self.calls = []

    def train(self, **params):
        self.calls.append(params)
        return _TrainResults()


class TrainWorkerTests(unittest.TestCase):
    def test_training_worker_emits_structured_ultralytics_progress_callbacks(self):
        class CallbackModel:
            def __init__(self):
                self.callbacks = defaultdict(list)

            def add_callback(self, event, callback):
                self.callbacks[event].append(callback)

            def emit(self, event, trainer):
                for callback in self.callbacks[event]:
                    callback(trainer)

            def train(self, **_params):
                trainer = SimpleNamespace(
                    epoch=0,
                    epochs=2,
                    start_epoch=0,
                    train_loader=[1, 2],
                    device="mps",
                    save_dir="/tmp/squeakpose-run",
                    tloss={"box_loss": 1.2, "seg_loss": 0.8},
                    metrics={
                        "metrics/precision(M)": 0.7,
                        "metrics/recall(M)": 0.6,
                        "metrics/mAP50(M)": 0.65,
                        "metrics/mAP50-95(M)": 0.4,
                    },
                    lr={"lr/pg0": 0.001},
                    fitness=0.4,
                    best_fitness=0.4,
                    epoch_time=3.5,
                    train_time_start=time.time() - 4.0,
                )
                self.emit("on_train_start", trainer)
                self.emit("on_train_epoch_start", trainer)
                self.emit("on_train_batch_end", trainer)
                self.emit("on_train_batch_end", trainer)
                self.emit("on_fit_epoch_end", trainer)
                return _TrainResults()

        events = []
        exit_code = run_training_worker(
            {"model_cfg": "yolo26n-seg.yaml", "params": {"epochs": 2}},
            model_factory=lambda _cfg: CallbackModel(),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        event_names = [event["event"] for event in events]
        self.assertIn("training_setup", event_names)
        self.assertIn("epoch_start", event_names)
        self.assertEqual(event_names.count("batch_progress"), 2)
        epoch_end = next(event for event in events if event["event"] == "epoch_end")
        self.assertEqual(epoch_end["losses"], {"box_loss": 1.2, "seg_loss": 0.8})
        self.assertEqual(epoch_end["metrics"]["metrics/mAP50-95(M)"], 0.4)

    def test_training_worker_runs_model_train_and_emits_result(self):
        model = _TrainModel()
        events = []

        exit_code = run_training_worker(
            {
                "model_cfg": "yolo26n-seg.yaml",
                "params": {
                    "data": "dataset.yaml",
                    "epochs": 2,
                    "batch": 1,
                    "task": "segment",
                },
            },
            model_factory=lambda _cfg: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            model.calls, [{"data": "dataset.yaml", "epochs": 2, "batch": 1, "task": "segment"}]
        )
        self.assertEqual([event["event"] for event in events], ["started", "training", "result"])
        self.assertFalse(events[-1]["had_error"])
        self.assertEqual(events[-1]["save_dir"], "/tmp/squeakpose-run")

    def test_training_worker_reports_missing_model_config(self):
        events = []

        exit_code = run_training_worker(
            {"params": {"epochs": 1}},
            model_factory=lambda _cfg: _TrainModel(),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertIn("model_cfg", events[0]["error_message"])

    def test_training_worker_reports_train_error(self):
        class BrokenModel:
            def train(self, **_params):
                raise RuntimeError("boom")

        events = []

        exit_code = run_training_worker(
            {"model_cfg": "bad.yaml", "params": {"epochs": 1}},
            model_factory=lambda _cfg: BrokenModel(),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1]["event"], "result")
        self.assertTrue(events[-1]["had_error"])
        self.assertEqual(events[-1]["error_message"], "boom")

    def test_training_worker_rejects_checkpoint_task_mismatch(self):
        model = _TrainModel()
        model.task = "detect"
        events = []

        exit_code = run_training_worker(
            {
                "model_cfg": "checkpoint.pt",
                "params": {"data": "dataset.yaml", "task": "pose"},
            },
            model_factory=lambda _cfg: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual([event["event"] for event in events], ["started", "error"])
        self.assertIn("task mismatch", events[-1]["error_message"])
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
