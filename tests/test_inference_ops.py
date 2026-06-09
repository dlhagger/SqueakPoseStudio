import csv
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from inference_ops import (
    probe_video_metadata,
    run_pose_video_inference,
    run_segmentation_video_inference,
    segmentation_rows_from_result,
)
from inference_worker import run_inference_worker


class _Tensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _Boxes:
    def __init__(self, xyxy=None, xywh=None, conf=None, cls=None):
        self.xyxy = _Tensor(xyxy or [])
        self.xywh = _Tensor(xywh or [])
        self.conf = _Tensor(conf or []) if conf is not None else None
        self.cls = _Tensor(cls or []) if cls is not None else None
        self.id = None
        self._n = len(xyxy or [])

    def __len__(self):
        return self._n


class _Keypoints:
    def __init__(self, data, xyn=None):
        self.data = _Tensor(data)
        self.xyn = _Tensor(xyn or [])


class _Masks:
    def __init__(self, xy=None, data=None):
        self.xy = xy or []
        self.data = _Tensor(data or [])


class _Result:
    def __init__(self, *, boxes, keypoints=None, masks=None, orig_shape=(20, 40), names=None):
        self.boxes = boxes
        self.keypoints = keypoints
        self.masks = masks
        self.orig_shape = orig_shape
        self.names = names or {0: "mouse"}
        self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}


class _FakeCapture:
    def __init__(self, frames, *, opened=True, fps=5.0):
        self._frames = list(frames)
        self._opened = opened
        self._fps = fps
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == _FakeCv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        if prop == _FakeCv2.CAP_PROP_FPS:
            return self._fps
        return 0

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FPS = 5

    def __init__(self, frames, *, opened=True, fps=5.0):
        self.frames = frames
        self.opened = opened
        self.fps = fps
        self.captures = []

    def VideoCapture(self, _path):
        cap = _FakeCapture(self.frames, opened=self.opened, fps=self.fps)
        self.captures.append(cap)
        return cap


class _PoseModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["source"])
        out = self.results[:count]
        self.results = self.results[count:]
        return out


class _SegModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return iter(self.results)


class InferenceOpsTests(unittest.TestCase):
    def test_probe_video_metadata_reads_count_and_fps(self):
        cv2 = _FakeCv2(["f1", "f2"], fps=12.5)

        meta = probe_video_metadata("video.mp4", cv2)

        self.assertTrue(meta.opened)
        self.assertEqual(meta.total_frames, 2)
        self.assertEqual(meta.fps, 12.5)
        self.assertTrue(cv2.captures[0].released)

    def test_segmentation_rows_include_explicit_no_detection_row(self):
        rows = segmentation_rows_from_result(
            _Result(boxes=_Boxes()),
            3,
            classes=["mouse"],
            include_binary_mask=False,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], 3)
        self.assertEqual(rows[0]["det"], -1)
        self.assertEqual(rows[0]["class_id"], "")
        self.assertIsNone(rows[0]["binary_mask"])

    def test_run_pose_video_inference_streams_csv_rows(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "pose.csv")
            cv2 = _FakeCv2(["frame0", "frame1"], fps=10.0)
            model = _PoseModel(
                [
                    _Result(boxes=_Boxes(), orig_shape=(20, 40)),
                    _Result(
                        boxes=_Boxes(
                            xyxy=[[1, 2, 11, 12]],
                            xywh=[[6, 7, 10, 10]],
                            conf=[0.9],
                            cls=[0],
                        ),
                        keypoints=_Keypoints([[[3, 4, 0.8]]], [[[0.075, 0.2]]]),
                        orig_shape=(20, 40),
                    ),
                ]
            )
            progress = []

            result = run_pose_video_inference(
                model=model,
                cv2_module=cv2,
                video_path="video.mp4",
                csv_path=csv_path,
                model_path="model.pt",
                classes=["mouse"],
                kp_names=["nose"],
                device="cpu",
                batch_size=1,
                total_frames=2,
                fps=10.0,
                progress_callback=lambda processed, total, message: progress.append((processed, total, message)),
            )

            self.assertFalse(result.had_error)
            self.assertEqual(result.rows_written, 2)
            self.assertEqual(result.processed_frames, 2)
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(model.calls[0]["batch"], 1)
            with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["detection_index"], "-1")
            self.assertEqual(rows[1]["class_name"], "mouse")
            self.assertEqual(rows[1]["kp_nose_x"], "3")
            self.assertEqual(progress[-1], (2, 2, "Inferencing frame 2/2"))

    def test_run_pose_video_inference_rejects_short_result_batch(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "pose.csv")
            cv2 = _FakeCv2(["frame0", "frame1"])
            model = _PoseModel([_Result(boxes=_Boxes())])

            result = run_pose_video_inference(
                model=model,
                cv2_module=cv2,
                video_path="video.mp4",
                csv_path=csv_path,
                model_path="model.pt",
                classes=["mouse"],
                kp_names=["nose"],
                device="cpu",
                batch_size=2,
                total_frames=2,
                fps=10.0,
            )

            self.assertTrue(result.had_error)
            self.assertEqual(result.processed_frames, 0)
            self.assertIn("1 results for 2 input frames", result.error_message)

    def test_pose_inference_cancellation_preserves_completed_rows(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "pose.csv")
            cv2 = _FakeCv2(["frame0", "frame1"])
            model = _PoseModel(
                [
                    _Result(boxes=_Boxes()),
                    _Result(boxes=_Boxes()),
                ]
            )
            cancel_checks = iter([False, False, False, True])

            result = run_pose_video_inference(
                model=model,
                cv2_module=cv2,
                video_path="video.mp4",
                csv_path=csv_path,
                model_path="model.pt",
                classes=["mouse"],
                kp_names=["nose"],
                device="cpu",
                batch_size=2,
                total_frames=2,
                fps=10.0,
                cancel_requested=lambda: next(cancel_checks, True),
            )

            self.assertTrue(result.canceled)
            self.assertFalse(result.had_error)
            self.assertEqual(result.processed_frames, 1)
            with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 1)

    def test_run_segmentation_video_inference_writes_json_polygons_without_binary_masks(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "seg.csv")
            model = _SegModel(
                [
                    _Result(
                        boxes=_Boxes(
                            xyxy=[[1, 2, 11, 12]],
                            conf=[0.95],
                            cls=[0],
                        ),
                        masks=_Masks(xy=[[[1, 2], [11, 2], [11, 12]]], data=[[[1, 0], [0, 1]]]),
                    )
                ]
            )

            result = run_segmentation_video_inference(
                model=model,
                video_path="video.mp4",
                csv_path=csv_path,
                classes=["mouse"],
                device="cpu",
                total_frames=1,
            )

            self.assertFalse(result.had_error)
            self.assertEqual(result.rows_written, 1)
            with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["class_name"], "mouse")
            self.assertEqual(rows[0]["binary_mask"], "")
            self.assertEqual(rows[0]["mask_polygon"], "[[1, 2], [11, 2], [11, 12]]")

    def test_segmentation_inference_cancellation_preserves_completed_rows(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "seg.csv")
            model = _SegModel(
                [
                    _Result(boxes=_Boxes()),
                    _Result(boxes=_Boxes()),
                ]
            )
            cancel_checks = iter([False, True])

            result = run_segmentation_video_inference(
                model=model,
                video_path="video.mp4",
                csv_path=csv_path,
                classes=["mouse"],
                device="cpu",
                total_frames=2,
                cancel_requested=lambda: next(cancel_checks, True),
            )

            self.assertTrue(result.canceled)
            self.assertFalse(result.had_error)
            self.assertEqual(result.processed_frames, 1)
            with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 1)

    def test_inference_worker_runs_pose_config_and_emits_events(self):
        with TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "worker_pose.csv")
            cv2 = _FakeCv2(["frame0"], fps=10.0)
            fake_model = _PoseModel([_Result(boxes=_Boxes(), orig_shape=(20, 40))])
            events = []

            exit_code = run_inference_worker(
                {
                    "mode": "pose",
                    "model_path": "model.pt",
                    "video_path": "video.mp4",
                    "csv_path": csv_path,
                    "classes": ["mouse"],
                    "kp_names": ["nose"],
                    "device": "cpu",
                    "batch_size": 1,
                    "total_frames": 1,
                    "fps": 10.0,
                },
                model_factory=lambda _path: fake_model,
                cv2_module=cv2,
                event_writer=events.append,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(events[0]["event"], "started")
            self.assertTrue(any(event["event"] == "progress" for event in events))
            result_event = events[-1]
            self.assertEqual(result_event["event"], "result")
            self.assertEqual(result_event["rows_written"], 1)
            self.assertFalse(result_event["had_error"])
            self.assertTrue(Path(csv_path).exists())

    def test_inference_worker_rejects_missing_model_path(self):
        events = []

        exit_code = run_inference_worker(
            {
                "mode": "pose",
                "video_path": "video.mp4",
                "csv_path": "out.csv",
            },
            model_factory=lambda _path: None,
            cv2_module=_FakeCv2([]),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertIn("model_path", events[0]["error_message"])

    def test_inference_worker_rejects_model_task_mismatch(self):
        events = []
        model = _SegModel([])
        model.task = "segment"

        exit_code = run_inference_worker(
            {
                "mode": "pose",
                "model_path": "model.pt",
                "video_path": "video.mp4",
                "csv_path": "out.csv",
            },
            model_factory=lambda _path: model,
            cv2_module=_FakeCv2([]),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("task mismatch", events[-1]["error_message"])


if __name__ == "__main__":
    unittest.main()
