import json
import unittest

from sam_worker import run_sam_server
from squeakpose.annotation.segmentation_assistant import SamPromptRequest
from squeakpose.services.sam_assistant import (
    build_sam_prediction_request,
    correlate_sam_event,
    deserialize_sam_selection,
)


class _Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _Result:
    def __init__(self):
        self.masks = type(
            "Masks",
            (),
            {
                "xy": [
                    [(0, 0), (2, 0), (1, 1)],
                    [(4, 3), (10, 3), (7, 9)],
                ]
            },
        )()
        self.boxes = type("Boxes", (), {"conf": _Values([0.2, 0.93])})()


class _Model:
    def __init__(self, results=None):
        self.results = [_Result()] if results is None else results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class SamAssistantServiceTests(unittest.TestCase):
    def test_request_payload_is_json_safe_and_round_trips_prompt_values(self):
        request = build_sam_prediction_request(
            request_id="sam-4",
            model_path="sam3.pt",
            prompt=SamPromptRequest(
                source="frame.png",
                class_id=2,
                prompts=((1.5, 2.5, 1), (7.0, 8.0, 0)),
            ),
            device="mps",
        )
        payload = json.loads(json.dumps(request.as_worker_payload()))

        self.assertEqual(payload["command"], "predict")
        self.assertEqual(payload["request_id"], "sam-4")
        self.assertEqual(payload["points"], [[1.5, 2.5], [7.0, 8.0]])
        self.assertEqual(payload["labels"], [1, 0])

    def test_worker_reuses_model_and_serializes_highest_confidence_contour(self):
        model = _Model()
        factory_calls = []
        events = []
        requests = [
            json.dumps(
                {
                    "command": "load",
                    "request_id": 1,
                    "model_path": "sam3.pt",
                    "device": "cpu",
                }
            ),
            json.dumps(
                {
                    "command": "predict",
                    "request_id": 2,
                    "model_path": "sam3.pt",
                    "image_path": "frame.png",
                    "points": [[2, 3]],
                    "labels": [1],
                    "device": "cpu",
                }
            ),
            json.dumps({"command": "shutdown", "request_id": 3}),
        ]

        exit_code = run_sam_server(
            requests,
            model_factory=lambda path: factory_calls.append(path) or model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(factory_calls, ["sam3.pt"])
        self.assertEqual(events[0], {"event": "ready"})
        result = next(event for event in events if event["event"] == "result")
        self.assertEqual(result["request_id"], 2)
        selection = deserialize_sam_selection(result["prediction"])
        self.assertEqual(selection.result.score, 0.93)
        self.assertEqual(
            selection.result.points,
            ((4.0, 3.0), (10.0, 3.0), (7.0, 9.0)),
        )
        self.assertEqual(
            model.calls,
            [
                {
                    "source": "frame.png",
                    "points": [[2.0, 3.0]],
                    "labels": [1],
                    "verbose": False,
                    "device": "cpu",
                }
            ],
        )

    def test_worker_returns_json_safe_no_mask_and_validation_errors(self):
        model = _Model(results=[])
        events = []
        run_sam_server(
            [
                json.dumps(
                    {
                        "command": "predict",
                        "request_id": 8,
                        "model_path": "sam3.pt",
                        "image_path": "frame.png",
                        "points": [[2, 3]],
                        "labels": [1],
                    }
                ),
                json.dumps(
                    {
                        "command": "predict",
                        "request_id": 9,
                        "model_path": "sam3.pt",
                        "image_path": "frame.png",
                        "points": [],
                        "labels": [],
                    }
                ),
            ],
            model_factory=lambda _path: model,
            event_writer=events.append,
        )

        results = [event for event in events if event["event"] == "result"]
        self.assertEqual(results[0]["prediction"]["failure"], "no_masks")
        self.assertFalse(results[0]["had_error"])
        self.assertTrue(results[1]["had_error"])
        json.dumps(events)

    def test_worker_reports_model_warm_failure_as_correlated_error(self):
        events = []
        run_sam_server(
            [
                json.dumps(
                    {
                        "command": "load",
                        "request_id": 11,
                        "model_path": "broken.pt",
                    }
                )
            ],
            model_factory=lambda _path: (_ for _ in ()).throw(RuntimeError("bad weights")),
            event_writer=events.append,
        )

        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["request_id"], 11)
        self.assertIn("bad weights", events[-1]["error_message"])

    def test_event_correlation_ignores_stale_and_discards_changed_image(self):
        stale = correlate_sam_event(
            {"event": "result", "request_id": 2, "prediction": {}},
            current_request_id=3,
            requested_image_path="one.png",
            displayed_image_path="one.png",
        )
        changed = correlate_sam_event(
            {"event": "result", "request_id": 3, "prediction": {}},
            current_request_id=3,
            requested_image_path="one.png",
            displayed_image_path="two.png",
        )

        self.assertEqual(stale.action, "ignore")
        self.assertEqual(changed.action, "discard")


if __name__ == "__main__":
    unittest.main()
