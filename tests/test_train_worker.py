import json
import os
import time
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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
        train_worker._CANCEL_SIGNAL_COUNT = 0

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
        train_worker._CANCEL_SIGNAL_COUNT = 0
        train_worker._handle_cancel_signal(None, None)  # must not raise SystemExit
        self.assertTrue(train_worker._CANCEL_REQUESTED)

    def test_second_signal_forces_exit(self):
        train_worker._CANCEL_REQUESTED = False
        train_worker._CANCEL_SIGNAL_COUNT = 0
        train_worker._handle_cancel_signal(None, None)
        with self.assertRaises(SystemExit) as raised:
            train_worker._handle_cancel_signal(None, None)
        self.assertEqual(raised.exception.code, 130)
        train_worker._CANCEL_SIGNAL_COUNT = 0

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


class _UltralyticsLikeModel:
    """Mirror ``BaseTrainer._do_train`` + ``final_eval`` callback ordering.

    Per epoch: on_train_epoch_start, batches (on_train_batch_end, ``if stop:
    break``), validation, ``stop |= final_epoch``, save_model (last.pt with
    optimizer state) + on_model_save, on_fit_epoch_end, ``if stop: break``.
    Then final_eval strips last.pt and calls on_fit_epoch_end with epoch + 1.
    """

    def __init__(
        self,
        save_dir,
        *,
        epochs=3,
        batches=4,
        cancel_at_batch=None,
        cancel_during_validation_of=None,
        cancel_during_final_eval=False,
        ckpt=None,
    ):
        self.callbacks = defaultdict(list)
        self.save_dir = Path(save_dir)
        self.epochs = epochs
        self.batches = batches
        self.cancel_at_batch = cancel_at_batch
        self.cancel_during_validation_of = cancel_during_validation_of
        self.cancel_during_final_eval = cancel_during_final_eval
        self.batches_run = 0
        self.validations = 0
        self.epochs_completed = 0
        if ckpt is not None:
            self.ckpt = ckpt
        self.train_calls = []

    def add_callback(self, event, callback):
        self.callbacks[event].append(callback)

    def _emit(self, event, trainer):
        for callback in self.callbacks[event]:
            callback(trainer)

    def _save(self, trainer):
        payload = {"epoch": trainer.epoch, "optimizer": {"state": 1}}
        trainer.last.write_text(json.dumps(payload), encoding="utf-8")
        trainer.best.write_text(json.dumps(payload), encoding="utf-8")

    def train(self, **params):
        self.train_calls.append(params)
        wdir = self.save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        trainer = SimpleNamespace(
            epoch=0,
            epochs=self.epochs,
            start_epoch=0,
            train_loader=list(range(self.batches)),
            device="cpu",
            save_dir=self.save_dir,
            last=wdir / "last.pt",
            best=wdir / "best.pt",
            stop=False,
            metrics={},
            train_time_start=time.time(),
        )
        self._emit("on_pretrain_routine_end", trainer)
        self._emit("on_train_start", trainer)
        epoch = 0
        while True:
            trainer.epoch = epoch
            self._emit("on_train_epoch_start", trainer)
            for _batch in range(self.batches):
                self.batches_run += 1
                if self.batches_run == self.cancel_at_batch:
                    train_worker._request_cancel()
                self._emit("on_train_batch_end", trainer)
                if trainer.stop:
                    break
            final_epoch = epoch + 1 >= self.epochs
            self.validations += 1
            if self.cancel_during_validation_of == epoch:
                train_worker._request_cancel()
            trainer.metrics = {"metrics/mAP50-95(P)": 0.1 * (epoch + 1)}
            trainer.stop |= final_epoch
            self._save(trainer)
            self._emit("on_model_save", trainer)
            self._emit("on_fit_epoch_end", trainer)
            self.epochs_completed += 1
            if trainer.stop:
                break
            epoch += 1
        # final_eval(): strip_optimizer(last.pt), validate best.pt, log at epoch + 1.
        trainer.last.write_text(json.dumps({"epoch": -1, "optimizer": None}), encoding="utf-8")
        if self.cancel_during_final_eval:
            train_worker._request_cancel()
        trainer.metrics = {"metrics/mAP50-95(P)": 0.99}
        trainer.epoch += 1
        self._emit("on_fit_epoch_end", trainer)
        trainer.epoch -= 1
        return SimpleNamespace(save_dir=str(self.save_dir))


class TrainWorkerUltralyticsLifecycleTests(unittest.TestCase):
    def tearDown(self):
        train_worker._CANCEL_REQUESTED = False
        train_worker._CANCEL_SIGNAL_COUNT = 0

    def _run(self, model, params=None):
        events = []
        exit_code = run_training_worker(
            {"model_cfg": "yolo26n-pose.yaml", "params": params or {"epochs": model.epochs}},
            model_factory=lambda _cfg: model,
            event_writer=events.append,
        )
        result = next(event for event in events if event["event"] == "result")
        return exit_code, events, result

    def test_final_eval_is_reported_as_final_validation_not_extra_epoch(self):
        with TemporaryDirectory() as tmp:
            model = _UltralyticsLikeModel(tmp, epochs=2, batches=2)

            exit_code, events, result = self._run(model)

            self.assertEqual(exit_code, 0)
            epoch_ends = [event for event in events if event["event"] == "epoch_end"]
            self.assertEqual([event["epoch"] for event in epoch_ends], [1, 2])
            self.assertTrue(all(event["epoch"] <= event["epochs"] for event in epoch_ends))
            finals = [event for event in events if event["event"] == "final_validation"]
            self.assertEqual(len(finals), 1)
            self.assertEqual(finals[0]["epochs_completed"], 2)
            self.assertEqual(finals[0]["metrics"], {"metrics/mAP50-95(P)": 0.99})
            self.assertFalse(result["canceled"])

    def test_cancel_during_validation_stops_at_epoch_end_without_another_epoch(self):
        with TemporaryDirectory() as tmp:
            model = _UltralyticsLikeModel(tmp, epochs=5, batches=3, cancel_during_validation_of=1)

            exit_code, events, result = self._run(model)

            self.assertEqual(exit_code, 0)
            # Epochs 0 and 1 ran; no extra batches or validation after the cancel.
            self.assertEqual(model.batches_run, 6)
            self.assertEqual(model.validations, 2)
            self.assertTrue(result["canceled"])
            names = [event["event"] for event in events]
            self.assertEqual(names.count("training_stopping"), 1)

    def test_cancel_saves_resumable_checkpoint_before_final_eval_strips_last(self):
        with TemporaryDirectory() as tmp:
            model = _UltralyticsLikeModel(tmp, epochs=5, batches=4, cancel_at_batch=6)

            exit_code, _events, result = self._run(model)

            self.assertEqual(exit_code, 0)
            self.assertTrue(result["canceled"])
            resumable = Path(tmp) / "weights" / "last_resumable.pt"
            self.assertEqual(result["resume_checkpoint"], str(resumable))
            saved = json.loads(resumable.read_text(encoding="utf-8"))
            self.assertEqual(saved["epoch"], 1)
            self.assertIsNotNone(saved["optimizer"])
            stripped = json.loads((Path(tmp) / "weights" / "last.pt").read_text("utf-8"))
            self.assertEqual(stripped["epoch"], -1)

    def test_cancel_during_validation_also_saves_resumable_checkpoint(self):
        with TemporaryDirectory() as tmp:
            model = _UltralyticsLikeModel(tmp, epochs=5, batches=2, cancel_during_validation_of=0)

            _exit_code, _events, result = self._run(model)

            resumable = Path(tmp) / "weights" / "last_resumable.pt"
            self.assertEqual(result["resume_checkpoint"], str(resumable))
            self.assertEqual(json.loads(resumable.read_text("utf-8"))["epoch"], 0)

    def test_cancel_during_final_eval_of_completed_run_reports_completion(self):
        with TemporaryDirectory() as tmp:
            stale = Path(tmp) / "weights" / "last_resumable.pt"
            stale.parent.mkdir(parents=True)
            stale.write_text("{}", encoding="utf-8")
            model = _UltralyticsLikeModel(tmp, epochs=2, batches=2, cancel_during_final_eval=True)

            exit_code, events, result = self._run(model)

            self.assertEqual(exit_code, 0)
            self.assertFalse(result["canceled"])
            self.assertNotIn("resume_checkpoint", result)
            self.assertNotIn("training_stopping", [event["event"] for event in events])
            self.assertEqual(model.epochs_completed, 2)
            # A finished run drops the older snapshot so it cannot rewind the run.
            self.assertFalse(stale.exists())

    def test_cancel_during_final_epoch_validation_reports_completion(self):
        with TemporaryDirectory() as tmp:
            model = _UltralyticsLikeModel(tmp, epochs=2, batches=2, cancel_during_validation_of=1)

            _exit_code, _events, result = self._run(model)

            self.assertFalse(result["canceled"])
            self.assertFalse((Path(tmp) / "weights" / "last_resumable.pt").exists())

    def test_exact_resume_of_finalized_checkpoint_is_rejected(self):
        with TemporaryDirectory() as tmp:
            for ckpt in ({"epoch": -1, "optimizer": None}, {"epoch": 3, "optimizer": None}, {}):
                model = _UltralyticsLikeModel(tmp, ckpt=ckpt)
                events = []

                exit_code = run_training_worker(
                    {"model_cfg": "last.pt", "params": {"resume": True, "device": "cpu"}},
                    model_factory=lambda _cfg, model=model: model,
                    event_writer=events.append,
                )

                self.assertEqual(exit_code, 1)
                self.assertEqual(events[-1]["event"], "error")
                self.assertIn("cannot be resumed exactly", events[-1]["error_message"])
                self.assertIn("last_resumable.pt", events[-1]["error_message"])
                self.assertEqual(model.train_calls, [])

    def test_exact_resume_of_full_checkpoint_trains(self):
        with TemporaryDirectory() as tmp:
            model = _UltralyticsLikeModel(
                tmp, epochs=1, batches=1, ckpt={"epoch": 4, "optimizer": {"state": 1}}
            )

            exit_code, _events, result = self._run(model, {"resume": True, "device": "cpu"})

            self.assertEqual(exit_code, 0)
            self.assertEqual(model.train_calls, [{"resume": True, "device": "cpu"}])
            self.assertFalse(result["had_error"])


class TrainWorkerMainTests(unittest.TestCase):
    def test_train_main_watches_cancel_marker_next_to_config(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, ".train_config_x.json")
            Path(config_path).write_text("{}", encoding="utf-8")
            with (
                patch("train_worker.signal.signal"),
                patch("train_worker.start_cancel_file_watcher") as watcher,
                patch("train_worker.run_training_worker", return_value=0) as run,
            ):
                self.assertEqual(train_worker.main(["--config", config_path]), 0)

            watcher.assert_called_once_with(f"{config_path}.cancel", train_worker._request_cancel)
            self.assertEqual(run.call_args.kwargs, {"reset_cancel": False})


if __name__ == "__main__":
    unittest.main()
