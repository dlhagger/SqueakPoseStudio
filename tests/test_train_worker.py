import time
import unittest
from collections import defaultdict
from types import SimpleNamespace

import train_worker
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


class _StoppableTrainModel:
    """Mimics Ultralytics: callbacks fire per batch and ``trainer.stop`` ends the loop."""

    def __init__(self, *, epochs=3, batches=4, cancel_at_batch=None, cancel_at=None):
        self.callbacks = defaultdict(list)
        self.epochs = epochs
        self.batches = batches
        self.cancel_at_batch = cancel_at_batch
        self.cancel_at = cancel_at
        self.batches_run = 0
        self.saved = False

    def add_callback(self, event, callback):
        self.callbacks[event].append(callback)

    def _emit(self, event, trainer):
        for callback in self.callbacks[event]:
            callback(trainer)

    def train(self, **_params):
        trainer = SimpleNamespace(
            epoch=0,
            epochs=self.epochs,
            start_epoch=0,
            train_loader=list(range(self.batches)),
            device="cpu",
            save_dir="/tmp/squeakpose-run",
            stop=False,
            train_time_start=time.time(),
        )
        if self.cancel_at == "pretrain":
            train_worker._request_cancel()
        self._emit("on_pretrain_routine_end", trainer)
        self._emit("on_train_start", trainer)
        for epoch in range(self.epochs):
            trainer.epoch = epoch
            self._emit("on_train_epoch_start", trainer)
            for _batch in range(self.batches):
                self.batches_run += 1
                if self.batches_run == self.cancel_at_batch:
                    train_worker._request_cancel()
                self._emit("on_train_batch_end", trainer)
                if trainer.stop:
                    break
            self._emit("on_fit_epoch_end", trainer)
            self.saved = True  # Ultralytics validates and saves last.pt/best.pt here.
            if trainer.stop:
                break
        return SimpleNamespace(save_dir=trainer.save_dir)


class TrainWorkerCancelTests(unittest.TestCase):
    def tearDown(self):
        train_worker._CANCEL_REQUESTED = False

    def test_cancel_mid_training_stops_at_batch_boundary_and_keeps_checkpoint(self):
        model = _StoppableTrainModel(epochs=3, batches=4, cancel_at_batch=6)
        events = []

        exit_code = run_training_worker(
            {"model_cfg": "yolo26n-pose.yaml", "params": {"epochs": 3}},
            model_factory=lambda _cfg: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        # Stopped right after the batch where cancel arrived, not after all 12.
        self.assertEqual(model.batches_run, 6)
        self.assertTrue(model.saved)
        names = [event["event"] for event in events]
        self.assertEqual(names.count("training_stopping"), 1)
        result = next(event for event in events if event["event"] == "result")
        self.assertTrue(result["canceled"])
        self.assertFalse(result["had_error"])
        self.assertEqual(result["save_dir"], "/tmp/squeakpose-run")

    def test_cancel_before_first_batch_aborts_without_training(self):
        model = _StoppableTrainModel(cancel_at="pretrain")
        events = []

        exit_code = run_training_worker(
            {"model_cfg": "yolo26n-pose.yaml", "params": {"epochs": 3}},
            model_factory=lambda _cfg: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(model.batches_run, 0)
        self.assertFalse(model.saved)
        result = next(event for event in events if event["event"] == "result")
        self.assertTrue(result["canceled"])
        self.assertFalse(result["had_error"])
        self.assertEqual(result["error_message"], "")

    def test_signal_handler_only_sets_flag(self):
        train_worker._CANCEL_REQUESTED = False
        train_worker._handle_cancel_signal(None, None)  # must not raise SystemExit
        self.assertTrue(train_worker._CANCEL_REQUESTED)

    def test_uncanceled_run_reports_not_canceled(self):
        model = _StoppableTrainModel(epochs=1, batches=2)
        events = []

        exit_code = run_training_worker(
            {"model_cfg": "yolo26n-pose.yaml", "params": {"epochs": 1}},
            model_factory=lambda _cfg: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(model.batches_run, 2)
        result = next(event for event in events if event["event"] == "result")
        self.assertFalse(result["canceled"])


if __name__ == "__main__":
    unittest.main()
