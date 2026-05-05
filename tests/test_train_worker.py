import unittest

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
        self.assertEqual(model.calls, [{"data": "dataset.yaml", "epochs": 2, "batch": 1, "task": "segment"}])
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


if __name__ == "__main__":
    unittest.main()
