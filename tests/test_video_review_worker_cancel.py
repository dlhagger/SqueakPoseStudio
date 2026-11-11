import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import video_review_worker
from tests.test_prediction_ops import _Boxes, _FakeCv2, _Result, _VideoReviewModel
from video_review_worker import run_video_review_worker


def _config(end):
    return {
        "model_path": "model.pt",
        "video_path": "video.mp4",
        "workflow": "pose",
        "device": "cpu",
        "start": 0,
        "end": end,
        "stride": 1,
        "batch": 1,
        "effective_batch": 1,
    }


def _model(count):
    return _VideoReviewModel(
        [_Result(boxes=_Boxes(xyxy=[[1, 2, 11, 12]], conf=[0.5], cls=[0])) for _ in range(count)]
    )


class VideoReviewWorkerCancelTests(unittest.TestCase):
    def tearDown(self):
        video_review_worker._CANCEL_REQUESTED = False
        video_review_worker._CANCEL_SIGNAL_COUNT = 0

    def test_cancel_that_arrived_during_startup_is_kept(self):
        model = _model(3)
        cv2 = _FakeCv2(["frame0", "frame1", "frame2"])
        events = []
        video_review_worker._request_cancel()

        exit_code = run_video_review_worker(
            _config(2),
            model_factory=lambda _path: model,
            cv2_module=cv2,
            event_writer=events.append,
            reset_cancel=False,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(events[-1]["canceled"])
        self.assertEqual(model.calls, [])
        self.assertTrue(cv2.capture.released)

    def test_first_signal_stops_cooperatively_second_forces_exit_and_releases_capture(self):
        model = _model(3)
        cv2 = _FakeCv2(["frame0", "frame1", "frame2"])
        original_predict = model.predict

        def predict_then_force(**kwargs):
            out = original_predict(**kwargs)
            video_review_worker._handle_cancel_signal(None, None)
            video_review_worker._handle_cancel_signal(None, None)
            return out

        model.predict = predict_then_force

        with self.assertRaises(SystemExit) as raised:
            run_video_review_worker(
                _config(2),
                model_factory=lambda _path: model,
                cv2_module=cv2,
                event_writer=[].append,
            )

        self.assertEqual(raised.exception.code, 130)
        self.assertTrue(cv2.capture.released)

    def test_main_watches_cancel_marker_and_keeps_startup_cancel(self):
        with TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, ".video_review_config_x.json")
            Path(config_path).write_text("{}", encoding="utf-8")
            with (
                patch("video_review_worker.signal.signal"),
                patch("video_review_worker.start_cancel_file_watcher") as watcher,
                patch("video_review_worker.run_video_review_worker", return_value=0) as run,
            ):
                self.assertEqual(video_review_worker.main(["--config", config_path]), 0)

            watcher.assert_called_once_with(
                f"{config_path}.cancel", video_review_worker._request_cancel
            )
            self.assertEqual(run.call_args.kwargs, {"reset_cancel": False})


if __name__ == "__main__":
    unittest.main()
