import unittest

from squeakpose.annotation.segmentation_assistant import SamPromptRequest
from squeakpose.ui.sam_assistant_controller import SamAssistantController
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


class SamAssistantControllerTests(unittest.TestCase):
    def setUp(self):
        self.displayed = "/images/frame.png"
        self.sessions = []

        def factory(parent):
            session = FakeSession(parent)
            self.sessions.append(session)
            return session

        self.controller = SamAssistantController(
            displayed_image_path=lambda: self.displayed,
            session_factory=factory,
            program="python",
            working_directory="/app",
        )
        self.prompt = SamPromptRequest(
            source=self.displayed,
            class_id=0,
            prompts=((3.0, 4.0, 1),),
        )

    def test_queues_until_ready_with_monotonic_request_id(self):
        request_id = self.controller.submit_prompt(
            model_path="sam3.pt",
            prompt=self.prompt,
        )
        session = self.sessions[0]

        self.assertTrue(self.controller.is_busy)
        self.assertEqual(session.requests, [])
        self.assertEqual(
            session.starts[0][0:2],
            ("python", ["-m", "sam_worker", "--server"]),
        )
        session.become_ready()
        self.assertEqual(session.requests[0]["request_id"], request_id)
        self.assertEqual(session.requests[0]["points"], [[3.0, 4.0]])

    def test_stale_event_is_ignored_and_matching_result_applies(self):
        decisions = []
        self.controller.decision_ready.connect(decisions.append)
        request_id = self.controller.submit_prompt(model_path="sam3.pt", prompt=self.prompt)
        session = self.sessions[0]
        session.become_ready()
        session.event_received.emit(
            {"event": "result", "request_id": request_id + 1, "prediction": {}}
        )
        self.assertEqual(decisions, [])
        self.assertTrue(self.controller.is_busy)

        session.event_received.emit(
            {
                "event": "result",
                "request_id": request_id,
                "prediction": {
                    "points": [[1, 1], [8, 1], [4, 7]],
                    "score": 0.84,
                    "failure": "",
                },
            }
        )
        self.assertEqual(decisions[-1].action, "apply")
        self.assertEqual(decisions[-1].result.score, 0.84)
        self.assertFalse(self.controller.is_busy)

    def test_cancel_shutdown_and_unexpected_terminal_are_distinct(self):
        decisions = []
        self.controller.decision_ready.connect(decisions.append)
        self.controller.submit_prompt(model_path="sam3.pt", prompt=self.prompt)
        session = self.sessions[0]
        self.assertTrue(self.controller.cancel())
        self.assertEqual(
            session.shutdowns[-1],
            {"terminate_after_ms": 250, "kill_after_ms": 3000},
        )
        session.terminal.emit(PersistentWorkerResult(state="stopped", exit_code=0))
        self.assertEqual(decisions[-1].action, "cancel")

        self.controller.submit_prompt(model_path="sam3.pt", prompt=self.prompt)
        session.terminal.emit(PersistentWorkerResult(state="failed", exit_code=1, stderr="crashed"))
        self.assertEqual(decisions[-1].action, "error")
        self.assertEqual(decisions[-1].error_message, "crashed")

    def test_restart_stops_generation_then_warms_after_terminal(self):
        scheduled = []
        controller = SamAssistantController(
            displayed_image_path=lambda: self.displayed,
            session_factory=FakeSession,
            schedule=scheduled.append,
        )
        controller.warm_model(model_path="old.pt")
        session = controller.session
        session.become_ready()

        self.assertTrue(controller.restart_model(model_path="new.pt"))
        session.is_running = False
        session.is_ready = False
        session.terminal.emit(PersistentWorkerResult(state="stopped", exit_code=0))
        self.assertEqual(len(scheduled), 1)
        scheduled.pop()()
        session.become_ready()

        self.assertEqual(len(session.starts), 2)
        self.assertEqual(session.requests[-1]["command"], "load")
        self.assertEqual(session.requests[-1]["model_path"], "new.pt")

    def test_shutdown_requests_bounded_escalation_without_cancel_decision(self):
        decisions = []
        self.controller.decision_ready.connect(decisions.append)
        self.controller.warm_model(model_path="sam3.pt")
        session = self.sessions[0]

        self.assertTrue(self.controller.shutdown())
        self.assertEqual(
            session.shutdowns[-1],
            {"terminate_after_ms": 250, "kill_after_ms": 3000},
        )
        session.terminal.emit(PersistentWorkerResult(state="stopped", exit_code=0))
        self.assertEqual(decisions, [])

    def test_start_failure_clears_active_request_exactly_once(self):
        controller = SamAssistantController(
            displayed_image_path=lambda: self.displayed,
            session_factory=FailingSession,
        )
        decisions = []
        controller.decision_ready.connect(decisions.append)

        controller.submit_prompt(model_path="sam3.pt", prompt=self.prompt)

        self.assertFalse(controller.is_busy)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "error")
        self.assertEqual(decisions[0].error_message, "cannot start")


if __name__ == "__main__":
    unittest.main()
