import unittest

from squeakpose.ui.prediction_controller import PredictionController
from squeakpose.workers.process import PersistentWorkerResult


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class FakeSession:
    def __init__(self, _parent):
        self.ready = FakeSignal()
        self.event_received = FakeSignal()
        self.output_received = FakeSignal()
        self.stderr_received = FakeSignal()
        self.terminal = FakeSignal()
        self.is_running = False
        self.is_ready = False
        self.starts = []
        self.requests = []
        self.shutdowns = []

    def start(self, program, arguments, **kwargs):
        self.starts.append((program, arguments, kwargs))
        self.is_running = True
        return True

    def send_request(self, request, **_kwargs):
        if not self.is_running or not self.is_ready:
            return False
        self.requests.append(dict(request))
        return True

    def become_ready(self):
        self.is_ready = True
        self.ready.emit()

    def shutdown(self, **kwargs):
        self.shutdowns.append(kwargs)
        return True


class FailingSession(FakeSession):
    def start(self, program, arguments, **kwargs):
        self.starts.append((program, arguments, kwargs))
        self.terminal.emit(
            PersistentWorkerResult(state="start_failed", error_message="cannot start")
        )
        return False


class PredictionControllerTests(unittest.TestCase):
    def setUp(self):
        self.displayed = "/images/frame.png"
        self.sessions = []

        def factory(parent):
            session = FakeSession(parent)
            self.sessions.append(session)
            return session

        self.controller = PredictionController(
            displayed_image_path=lambda: self.displayed,
            session_factory=factory,
            program="python",
            working_directory="/app",
        )

    def test_queues_until_ready_and_preserves_worker_payload(self):
        request_id = self.controller.submit_prediction(
            layer_id="pose",
            model_path="pose.pt",
            image_path=self.displayed,
            device="mps",
        )
        session = self.sessions[0]

        self.assertTrue(self.controller.is_busy)
        self.assertEqual(session.requests, [])
        self.assertEqual(session.starts[0][0:2], ("python", ["-m", "predict_worker", "--server"]))

        session.become_ready()
        self.assertEqual(
            session.requests,
            [
                {
                    "command": "predict",
                    "request_id": request_id,
                    "layer_id": "keypoints",
                    "model_path": "pose.pt",
                    "workflow": "pose",
                    "device": "mps",
                    "image_path": self.displayed,
                }
            ],
        )

    def test_correlates_result_and_discards_when_display_changed(self):
        decisions = []
        self.controller.decision_ready.connect(decisions.append)
        request_id = self.controller.submit_prediction(
            layer_id="keypoints",
            model_path="pose.pt",
            image_path=self.displayed,
        )
        session = self.sessions[0]
        session.become_ready()
        session.event_received.emit(
            {"event": "result", "request_id": request_id + 1, "prediction": {"ok": True}}
        )
        self.assertTrue(self.controller.is_busy)
        self.assertEqual(decisions, [])

        self.displayed = "/images/other.png"
        session.event_received.emit(
            {"event": "result", "request_id": request_id, "prediction": {"ok": True}}
        )
        self.assertEqual(decisions[-1].action, "discard")
        self.assertFalse(self.controller.is_busy)

    def test_unexpected_terminal_fails_active_request_and_cancel_is_distinct(self):
        decisions = []
        self.controller.decision_ready.connect(decisions.append)
        self.controller.submit_prediction(
            layer_id="keypoints", model_path="pose.pt", image_path=self.displayed
        )
        session = self.sessions[0]
        session.terminal.emit(
            PersistentWorkerResult(state="failed", exit_code=1, stderr="worker crashed")
        )
        self.assertEqual(decisions[-1].action, "error")
        self.assertEqual(decisions[-1].error_message, "worker crashed")

        self.controller.submit_prediction(
            layer_id="keypoints", model_path="pose.pt", image_path=self.displayed
        )
        session.is_running = True
        self.assertTrue(self.controller.cancel())
        session.terminal.emit(PersistentWorkerResult(state="stopped", exit_code=0))
        self.assertEqual(decisions[-1].action, "cancel")
        self.assertFalse(self.controller.is_busy)

    def test_warm_request_uses_separate_id_without_marking_busy(self):
        request_id = self.controller.warm_model(
            layer_id="segmentation", model_path="seg.pt", device="cuda"
        )
        session = self.sessions[0]
        session.become_ready()

        self.assertFalse(self.controller.is_busy)
        self.assertEqual(session.requests[0]["command"], "load")
        self.assertEqual(session.requests[0]["request_id"], request_id)
        self.assertNotIn("image_path", session.requests[0])

    def test_start_failure_clears_busy_and_does_not_leave_request_queued(self):
        controller = PredictionController(
            displayed_image_path=lambda: self.displayed,
            session_factory=FailingSession,
        )
        decisions = []
        controller.decision_ready.connect(decisions.append)

        controller.submit_prediction(
            layer_id="keypoints", model_path="pose.pt", image_path=self.displayed
        )

        self.assertFalse(controller.is_busy)
        self.assertEqual(decisions[-1].action, "error")
        self.assertEqual(decisions[-1].error_message, "cannot start")
        # A subsequent request is accepted, proving the failed request was not retained.
        controller.warm_model(layer_id="keypoints", model_path="pose.pt")

    def test_restart_owns_stop_then_warm_sequence(self):
        controller = PredictionController(
            displayed_image_path=lambda: self.displayed,
            session_factory=lambda parent: FakeSession(parent),
            schedule=lambda callback: callback(),
        )
        controller.warm_model(layer_id="keypoints", model_path="old.pt")
        session = controller.session
        session.become_ready()
        self.assertTrue(
            controller.restart_model(layer_id="segmentation", model_path="new.pt", device="mps")
        )
        session.is_running = False
        session.is_ready = False
        session.terminal.emit(PersistentWorkerResult(state="stopped", exit_code=0))
        session.become_ready()

        self.assertEqual(len(session.starts), 2)
        self.assertEqual(session.requests[-1]["command"], "load")
        self.assertEqual(session.requests[-1]["layer_id"], "segmentation")
        self.assertEqual(session.requests[-1]["model_path"], "new.pt")


if __name__ == "__main__":
    unittest.main()
