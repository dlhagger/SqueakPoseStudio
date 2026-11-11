import json
import unittest

from predict_worker import run_predict_server
from squeakpose.services.prediction import correlate_prediction_event


class _IdleModel:
    task = "pose"

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return []


class PredictWorkerServerTests(unittest.TestCase):
    def test_load_failure_is_reported_as_error_event_with_request_id(self):
        events = []
        requests = [
            json.dumps(
                {"command": "load", "request_id": 7, "model_path": "broken.pt", "workflow": "pose"}
            ),
        ]

        def factory(path):
            raise RuntimeError(f"cannot open {path}")

        exit_code = run_predict_server(
            requests,
            model_factory=factory,
            cv2_module=object(),
            numpy_module=object(),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual([event["event"] for event in events], ["ready", "loading", "error"])
        error = events[-1]
        self.assertEqual(error["request_id"], 7)
        self.assertEqual(error["model_path"], "broken.pt")
        self.assertIn("cannot open broken.pt", error["error_message"])
        self.assertNotIn("had_error", error)

        # The warm-up request id is never the current prediction request id,
        # so the service must surface the failure as a background load error
        # (clearing the "Loading prediction model..." status) rather than
        # ignoring a stale ``result``.
        decision = correlate_prediction_event(error, current_request_id=None)
        self.assertEqual(decision.action, "background_error")
        self.assertIn("cannot open broken.pt", decision.error_message)

    def test_load_failure_does_not_poison_following_requests(self):
        attempts = []
        model = _IdleModel()
        events = []
        requests = [
            json.dumps(
                {"command": "load", "request_id": 1, "model_path": "flaky.pt", "workflow": "pose"}
            ),
            json.dumps(
                {"command": "load", "request_id": 2, "model_path": "flaky.pt", "workflow": "pose"}
            ),
        ]

        def factory(path):
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError("transient")
            return model

        exit_code = run_predict_server(
            requests,
            model_factory=factory,
            cv2_module=object(),
            numpy_module=object(),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(attempts, ["flaky.pt", "flaky.pt"])
        self.assertEqual(
            [(event["event"], event.get("request_id")) for event in events[1:]],
            [("loading", 1), ("error", 1), ("loading", 2), ("loaded", 2)],
        )

    def test_prediction_failure_remains_a_result_with_had_error(self):
        class _FailingModel(_IdleModel):
            def predict(self, **kwargs):
                raise RuntimeError("inference exploded")

        events = []
        requests = [
            json.dumps(
                {
                    "command": "predict",
                    "request_id": 3,
                    "model_path": "model.pt",
                    "image_path": "img.png",
                    "workflow": "pose",
                }
            ),
        ]

        exit_code = run_predict_server(
            requests,
            model_factory=lambda _path: _FailingModel(),
            cv2_module=object(),
            numpy_module=object(),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        result = events[-1]
        self.assertEqual(result["event"], "result")
        self.assertEqual(result["request_id"], 3)
        self.assertTrue(result["had_error"])
        self.assertEqual(result["error_message"], "inference exploded")


if __name__ == "__main__":
    unittest.main()
