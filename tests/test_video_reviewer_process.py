import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.ui.video_reviewer import APP_BASE_DIR, VideoReviewDialog
from squeakpose.workers.process import WorkerJobResult


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        if callback not in self.callbacks:
            raise TypeError("disconnect() failed: not connected")
        self.callbacks.remove(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class _Process:
    def __init__(self):
        self.running = True
        self.kill_calls = 0

    def state(self):
        return QProcess.ProcessState.Running if self.running else QProcess.ProcessState.NotRunning

    def kill(self):
        self.kill_calls += 1
        self.running = False

    def errorString(self):
        return "fake process error"


class _Controller:
    def __init__(self):
        self.event_received = _Signal()
        self.output_received = _Signal()
        self.stderr_received = _Signal()
        self.terminal = _Signal()
        self.process = _Process()
        self.start_call = None
        self.cancel_calls = []
        self.shutdown_calls = 0

    @property
    def is_running(self):
        return self.process.running

    def start(self, program, arguments, **kwargs):
        self.start_call = (program, list(arguments), dict(kwargs))
        return True

    def cancel(self, *, kill_after_ms):
        self.cancel_calls.append(kill_after_ms)
        return True

    def finish(self, result):
        self.process.running = False
        self.terminal.emit(result)

    def shutdown(self):
        self.shutdown_calls += 1
        self.finish(WorkerJobResult(state="cancelled"))
        return True


class _Progress:
    def __init__(self):
        self.label = ""
        self.maximum = 0
        self.value = 0
        self.closed = False
        self.shown = False
        self.canceled = _Signal()

    def setWindowTitle(self, _title):
        pass

    def setWindowModality(self, _modality):
        pass

    def setMinimumDuration(self, _duration):
        pass

    def setLabelText(self, text):
        self.label = text

    def setMaximum(self, value):
        self.maximum = value

    def setValue(self, value):
        self.value = value

    def close(self):
        self.closed = True
        # Mirror QProgressDialog: closing the dialog emits canceled().
        self.canceled.emit()

    def show(self):
        self.shown = True


class _Capture:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FakeVideoCapture:
    def __init__(self, _path):
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            _FakeCv2.CAP_PROP_FRAME_COUNT: 10,
            _FakeCv2.CAP_PROP_FPS: 25.0,
            _FakeCv2.CAP_PROP_FRAME_WIDTH: 64,
            _FakeCv2.CAP_PROP_FRAME_HEIGHT: 48,
        }.get(prop, 0)

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    VideoCapture = _FakeVideoCapture


class VideoReviewerProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["video-reviewer-process-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def make_dialog(self, project_root):
        dialog = VideoReviewDialog(
            None,
            "cpu",
            ["nose"],
            ["mouse"],
            layer_id=LAYER_KEYPOINTS,
            model_paths={LAYER_KEYPOINTS: "pose.pt", LAYER_SEGMENTATION: "segment.pt"},
        )
        dialog.project_root = project_root
        dialog.path = "/tmp/review.mp4"
        dialog._review_settings = {"start": 0, "end": 4, "stride": 1}
        dialog._review_job_queue = [LAYER_KEYPOINTS, LAYER_SEGMENTATION]
        dialog._review_pass_total = 2
        dialog._review_pass_index = 0
        dialog._review_steps_per_pass = 5
        dialog._review_progress = _Progress()
        dialog._review_run_meta = {}
        dialog._review_run_errors = []
        dialog._review_run_canceled = False
        return dialog

    def test_project_video_button_selects_available_library_link(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as sources:
            dialog = self.make_dialog(tmp)
            videos_dir = Path(tmp) / "videos"
            videos_dir.mkdir()
            source = Path(sources) / "session.mp4"
            source.write_bytes(b"video")
            link = videos_dir / source.name
            link.symlink_to(source)
            (videos_dir / "missing.mov").symlink_to(Path(sources) / "missing.mov")

            with (
                patch(
                    "squeakpose.ui.video_reviewer.QInputDialog.getItem",
                    return_value=(source.name, True),
                ) as choose,
                patch.object(dialog, "_open_video") as open_video,
            ):
                dialog._choose_project_video()

            self.assertEqual(choose.call_args.args[3], [source.name])
            self.assertIn("1 missing", choose.call_args.args[2])
            open_video.assert_called_once_with(str(link))

    def test_controller_runs_passes_sequentially_and_merges_streamed_results(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            first = _Controller()
            second = _Controller()
            finished = Mock()
            dialog._finish_project_review_prediction = finished

            with (
                patch.object(
                    dialog,
                    "_create_review_job_controller",
                    side_effect=[first, second],
                ),
                patch(
                    "squeakpose.ui.video_reviewer.create_worker_config",
                    side_effect=[f"{tmp}/pose.json", f"{tmp}/segment.json"],
                ),
                patch(
                    "squeakpose.ui.video_reviewer.QTimer.singleShot", side_effect=lambda _, cb: cb()
                ),
            ):
                dialog._start_next_review_prediction_pass()
                self.assertIs(dialog._review_job, first)
                self.assertEqual(first.start_call[0], os.sys.executable)
                self.assertEqual(
                    first.start_call[1],
                    ["-m", "video_review_worker", "--config", f"{tmp}/pose.json"],
                )
                self.assertEqual(first.start_call[2]["working_directory"], APP_BASE_DIR)
                self.assertEqual(first.start_call[2]["start_timeout_ms"], 1000)

                first.event_received.emit(
                    {
                        "event": "progress",
                        "processed": 2,
                        "total": 5,
                        "predictions": {"1": {"ok": True, "confidence": 0.3}},
                    }
                )
                first.event_received.emit(
                    {"event": "result", "preds": {"3": {"ok": True, "confidence": 0.8}}}
                )
                first.finish(
                    WorkerJobResult(
                        state="finished",
                        exit_code=0,
                        exit_status=QProcess.ExitStatus.NormalExit,
                    )
                )

                self.assertEqual(set(dialog.preds_by_layer[LAYER_KEYPOINTS]), {1, 3})
                self.assertIs(dialog._review_job, second)
                self.assertEqual(dialog._review_pass_index, 2)
                self.assertEqual(dialog._review_progress.value, 2)
                self.assertFalse(finished.called)

                second.event_received.emit(
                    {"event": "result", "preds": {"4": {"ok": True, "confidence": 0.6}}}
                )
                second.finish(
                    WorkerJobResult(
                        state="finished",
                        exit_code=0,
                        exit_status=QProcess.ExitStatus.NormalExit,
                    )
                )

            self.assertEqual(set(dialog.preds_by_layer[LAYER_SEGMENTATION]), {4})
            finished.assert_called_once_with()
            dialog.close()

    def test_prediction_menu_can_run_keypoints_without_clearing_segmentation(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            dialog.cap = _Capture()
            dialog.total = 1
            dialog.fps = 30.0
            dialog.preds_by_layer[LAYER_KEYPOINTS] = {0: {"old": "pose"}}
            existing_segmentation = {0: {"old": "segmentation"}}
            dialog.preds_by_layer[LAYER_SEGMENTATION] = existing_segmentation
            progress = _Progress()

            self.assertEqual(
                [action.text() for action in dialog.predict_layer_actions.values()],
                ["Predict Keypoints", "Predict Segmentation", "Predict Both Layers"],
            )
            with patch.object(dialog, "_start_range_prediction") as selected_prediction:
                dialog.predict_layer_actions["keypoints"].trigger()
            selected_prediction.assert_called_once_with((LAYER_KEYPOINTS,))

            with (
                patch(
                    "squeakpose.ui.video_reviewer.QProgressDialog",
                    return_value=progress,
                ),
                patch.object(dialog, "_start_next_review_prediction_pass") as start_pass,
            ):
                dialog._start_range_prediction([LAYER_KEYPOINTS])

            self.assertEqual(dialog._review_job_queue, [LAYER_KEYPOINTS])
            self.assertEqual(dialog._review_pass_total, 1)
            self.assertEqual(dialog._review_run_meta["layers"], [LAYER_KEYPOINTS])
            self.assertEqual(dialog.preds_by_layer[LAYER_KEYPOINTS], {})
            self.assertEqual(dialog.preds_by_layer[LAYER_SEGMENTATION], existing_segmentation)
            self.assertTrue(progress.shown)
            start_pass.assert_called_once_with()
            dialog.cap = None
            dialog.close()

    def test_cancel_delegates_escalation_and_stops_remaining_passes(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            controller = _Controller()
            dialog._review_job = controller
            dialog._review_process = controller.process
            controller.terminal.connect(dialog._finish_review_prediction_job)
            dialog._review_current_layer = LAYER_KEYPOINTS
            dialog._review_partial_preds = {2: {"ok": True}}
            finished = Mock()
            dialog._finish_project_review_prediction = finished

            dialog._cancel_review_prediction_process()

            self.assertEqual(controller.cancel_calls, [5000])
            self.assertTrue(dialog._review_cancel_requested)
            self.assertEqual(dialog._review_progress.label, "Canceling prediction process…")

            controller.finish(
                WorkerJobResult(
                    state="cancelled",
                    exit_code=-1,
                    exit_status=QProcess.ExitStatus.CrashExit,
                )
            )

            self.assertEqual(dialog._review_job_queue, [])
            self.assertTrue(dialog._review_run_canceled)
            self.assertEqual(dialog.preds_by_layer[LAYER_KEYPOINTS], {2: {"ok": True}})
            finished.assert_called_once_with()
            dialog.close()

    def test_successful_completion_does_not_treat_progress_close_as_cancel(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            progress = QProgressDialog("Predicting…", "Cancel", 0, 1, dialog)
            progress.canceled.connect(dialog._cancel_review_prediction_process)
            dialog._review_progress = progress
            dialog._review_job_queue = []
            dialog.preds_by_layer[LAYER_KEYPOINTS] = {0: {"ok": True, "confidence": 0.8}}

            with (
                patch.object(dialog, "_save_cache"),
                patch.object(QMessageBox, "information") as information,
                patch.object(QMessageBox, "warning") as warning,
            ):
                dialog._finish_project_review_prediction()

            self.assertIsNone(dialog._review_progress)
            self.assertFalse(dialog._review_run_canceled)
            information.assert_not_called()
            warning.assert_not_called()
            dialog.close()

    def test_reject_uses_controller_shutdown_and_suppresses_completion_ui(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            controller = _Controller()
            capture = _Capture()
            progress = dialog._review_progress
            progress.canceled.connect(dialog._cancel_review_prediction_process)
            dialog._review_job = controller
            dialog._review_process = controller.process
            controller.terminal.connect(dialog._finish_review_prediction_job)
            dialog.cap = capture
            finished = Mock()
            dialog._finish_project_review_prediction = finished

            with (
                patch.object(
                    QMessageBox,
                    "question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                patch.object(QMessageBox, "information") as information,
            ):
                dialog.reject()

            self.assertEqual(controller.shutdown_calls, 1)
            self.assertFalse(finished.called)
            information.assert_not_called()
            self.assertTrue(progress.closed)
            self.assertEqual(progress.canceled.callbacks, [])
            self.assertTrue(capture.released)
            self.assertIsNone(dialog._review_job)
            self.assertIsNone(dialog._review_process)
            self.assertIsNone(dialog._review_progress)

    def test_finish_is_a_no_op_while_reviewer_is_closing(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            dialog._review_closing = True
            dialog._review_run_canceled = True
            dialog.preds_by_layer[LAYER_KEYPOINTS] = {0: {"ok": True, "confidence": 0.8}}

            with (
                patch.object(dialog, "_save_cache") as save_cache,
                patch.object(QMessageBox, "information") as information,
            ):
                dialog._finish_project_review_prediction()

            save_cache.assert_not_called()
            information.assert_not_called()
            dialog.close()

    def test_open_video_discards_previous_video_predictions(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)
            old_capture = _Capture()
            dialog.cap = old_capture
            dialog.preds_by_layer[LAYER_KEYPOINTS] = {0: {"ok": True, "confidence": 0.8}}
            dialog.preds_by_layer[LAYER_SEGMENTATION] = {1: {"ok": True, "confidence": 0.4}}
            dialog.preds = dialog.preds_by_layer[LAYER_KEYPOINTS]

            with (
                patch("squeakpose.ui.video_reviewer._cv2", _FakeCv2),
                patch.object(dialog, "_load_cache_if_valid", return_value=False) as load_cache,
                patch.object(dialog, "_seek"),
            ):
                dialog._open_video(os.path.join(tmp, "second.mp4"))

            load_cache.assert_called_once_with()
            self.assertTrue(old_capture.released)
            self.assertEqual(dialog.preds_by_layer, {LAYER_KEYPOINTS: {}, LAYER_SEGMENTATION: {}})
            self.assertIs(dialog.preds, dialog.preds_by_layer[LAYER_KEYPOINTS])
            self.assertEqual(dialog._overlay_items, [])
            self.assertEqual(dialog.base, "second")
            dialog.cap = None
            dialog.close()

    def test_legacy_event_line_helper_keeps_protocol_and_plain_output_behavior(self):
        with TemporaryDirectory() as tmp:
            dialog = self.make_dialog(tmp)

            dialog._handle_review_prediction_event_line(
                '{"event":"result","preds":{"5":{"ok":true}}}'
            )
            dialog._handle_review_prediction_event_line("worker diagnostic")

            self.assertEqual(dialog._review_result_event["preds"]["5"], {"ok": True})
            self.assertIn("worker diagnostic", dialog._review_stderr)
            dialog.close()


if __name__ == "__main__":
    unittest.main()
