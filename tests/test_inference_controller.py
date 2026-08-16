import datetime
import os
import unittest
from tempfile import TemporaryDirectory

from squeakpose.services.inference import plan_inference_run
from squeakpose.ui.inference_controller import InferenceController
from squeakpose.workers.process import WorkerJobResult


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class FakeJobController:
    def __init__(self, _parent):
        self.event_received = FakeSignal()
        self.output_received = FakeSignal()
        self.stderr_received = FakeSignal()
        self.terminal = FakeSignal()
        self.starts = []
        self.cancels = []
        self.is_running = False

    def start(self, program, arguments, **kwargs):
        self.starts.append((program, arguments, kwargs))
        self.is_running = True
        return True

    def cancel(self, **kwargs):
        self.cancels.append(kwargs)
        return True

    def shutdown(self):
        self.is_running = False
        return True


class InferenceControllerTests(unittest.TestCase):
    def _plan(self, root):
        return plan_inference_run(
            project_root=root,
            video_path=os.path.join(root, "video.mp4"),
            active_layer="keypoints",
            model_paths={"keypoints": "pose.pt", "segmentation": "seg.pt"},
            pose_classes=["mouse"],
            segmentation_classes=["mouse-seg"],
            keypoint_names=["nose"],
            device="cpu",
            batch_size=4,
            total_frames=10,
            fps=30,
            created_at=datetime.datetime(2026, 8, 15, 12, 0),
            run_id="test-run",
        )

    def test_runs_jobs_in_order_with_exact_configs_and_summary(self):
        with TemporaryDirectory() as tmp:
            controllers = []
            configs = []
            manifests = []

            def controller_factory(parent):
                controller = FakeJobController(parent)
                controllers.append(controller)
                return controller

            def config_writer(project_root, directory, kind, payload):
                configs.append((project_root, directory, kind, payload))
                return os.path.join(directory, f"config-{len(configs)}.json")

            controller = InferenceController(
                controller_factory=controller_factory,
                config_writer=config_writer,
                manifest_writer=lambda path, text: manifests.append((path, text)),
                schedule=lambda callback: callback(),
                program="python",
                working_directory="/app",
            )
            passes = []
            summaries = []
            progress = []
            controller.pass_finished.connect(passes.append)
            controller.completed.connect(summaries.append)
            controller.progress.connect(lambda job, event: progress.append((job, event)))
            plan = self._plan(tmp)
            controller.start(plan)

            first = controllers[0]
            self.assertEqual(configs[0][3], plan.jobs[0].worker_config())
            self.assertEqual(
                first.starts[0][1],
                ["-m", "inference_worker", "--config", configs[0][1] + "/config-1.json"],
            )
            first.event_received.emit(
                {"event": "progress", "processed_frames": 4, "total_frames": 10}
            )
            first.event_received.emit(
                {
                    "event": "result",
                    "csv_path": plan.jobs[0].csv_path,
                    "rows_written": 8,
                    "processed_frames": 10,
                }
            )
            first.terminal.emit(WorkerJobResult(state="finished", exit_code=0))

            second = controllers[1]
            second.event_received.emit(
                {
                    "event": "result",
                    "csv_path": plan.jobs[1].csv_path,
                    "rows_written": 5,
                    "processed_frames": 10,
                }
            )
            second.terminal.emit(WorkerJobResult(state="finished", exit_code=0))

            self.assertEqual([item.layer_id for item in passes], ["keypoints", "segmentation"])
            self.assertEqual(progress[0][0], plan.jobs[0])
            self.assertFalse(controller.is_busy)
            self.assertEqual(summaries[0].successful_count, 2)
            self.assertEqual(manifests[0][0], plan.manifest_path)

    def test_config_failure_is_recorded_and_next_job_starts(self):
        with TemporaryDirectory() as tmp:
            controllers = []
            calls = 0

            def config_writer(_root, directory, _kind, _payload):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("disk full")
                return os.path.join(directory, "config.json")

            controller = InferenceController(
                controller_factory=lambda parent: (
                    controllers.append(FakeJobController(parent)) or controllers[-1]
                ),
                config_writer=config_writer,
                manifest_writer=lambda _path, _text: None,
                schedule=lambda callback: callback(),
            )
            passes = []
            controller.pass_finished.connect(passes.append)
            controller.start(self._plan(tmp))

            self.assertTrue(passes[0].had_error)
            self.assertIn("disk full", passes[0].error_message)
            self.assertEqual(controller.active_job.layer_id, "segmentation")

    def test_cancel_stops_remaining_jobs_and_finalizes_canceled_run(self):
        with TemporaryDirectory() as tmp:
            controllers = []
            controller = InferenceController(
                controller_factory=lambda parent: (
                    controllers.append(FakeJobController(parent)) or controllers[-1]
                ),
                config_writer=lambda _root, directory, _kind, _payload: os.path.join(
                    directory, "config.json"
                ),
                manifest_writer=lambda _path, _text: None,
                schedule=lambda callback: callback(),
            )
            summaries = []
            controller.completed.connect(summaries.append)
            controller.start(self._plan(tmp))

            self.assertTrue(controller.cancel())
            controllers[0].terminal.emit(WorkerJobResult(state="cancelled", exit_code=-1))

            self.assertEqual(len(controllers), 1)
            self.assertFalse(controller.is_busy)
            self.assertTrue(summaries[0].canceled)
            self.assertEqual(summaries[0].canceled_count, 1)

    def test_stale_events_and_terminal_results_are_ignored(self):
        with TemporaryDirectory() as tmp:
            controllers = []
            controller = InferenceController(
                controller_factory=lambda parent: (
                    controllers.append(FakeJobController(parent)) or controllers[-1]
                ),
                config_writer=lambda _root, directory, _kind, _payload: os.path.join(
                    directory, "config.json"
                ),
                manifest_writer=lambda _path, _text: None,
                schedule=lambda callback: callback(),
            )
            passes = []
            controller.pass_finished.connect(passes.append)
            output = []
            controller.output_received.connect(output.append)
            plan = self._plan(tmp)
            controller.start(plan)
            first = controllers[0]
            first.terminal.emit(WorkerJobResult(state="finished", exit_code=0))
            self.assertEqual(controller.active_job, plan.jobs[1])

            first.event_received.emit({"event": "result", "rows_written": 999})
            first.output_received.emit("stale output")
            first.terminal.emit(WorkerJobResult(state="finished", exit_code=0))
            self.assertEqual(len(passes), 1)
            self.assertEqual(output, [])


if __name__ == "__main__":
    unittest.main()
