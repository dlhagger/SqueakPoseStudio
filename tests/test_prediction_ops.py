import unittest

from prediction_ops import serialize_prediction_result, top_prediction_from_payload
from predict_worker import run_predict_worker
from video_review_worker import run_video_review_worker


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
    def __init__(self, xyxy=None, conf=None, cls=None):
        self.xyxy = _Tensor(xyxy or [])
        self.conf = _Tensor(conf or []) if conf is not None else None
        self.cls = _Tensor(cls or []) if cls is not None else None
        self._n = len(xyxy or [])

    def __len__(self):
        return self._n


class _Keypoints:
    def __init__(self, data):
        self.data = _Tensor(data)


class _Masks:
    def __init__(self, xy=None, data=None):
        self.xy = xy or []
        self.data = _Tensor(data or [])


class _Result:
    def __init__(self, *, boxes, keypoints=None, masks=None):
        self.boxes = boxes
        self.keypoints = keypoints
        self.masks = masks


class _PredictModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class _VideoReviewModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["source"])
        out = self.results[:count]
        self.results = self.results[count:]
        return out


class _FailingVideoReviewModel:
    def predict(self, **kwargs):
        raise RuntimeError("predict boom")


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.pos = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == _FakeCv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)

    def read(self):
        if self.pos < 0 or self.pos >= len(self.frames):
            return False, None
        frame = self.frames[self.pos]
        self.pos += 1
        return True, frame

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, frames):
        self.capture = _FakeCapture(frames)

    def VideoCapture(self, _path):
        return self.capture


class PredictionOpsTests(unittest.TestCase):
    def test_serialize_prediction_result_keeps_all_pose_detections(self):
        result = _Result(
            boxes=_Boxes(
                xyxy=[[1, 2, 11, 12], [20, 21, 40, 41]],
                conf=[0.4, 0.9],
                cls=[1, 0],
            ),
            keypoints=_Keypoints(
                [
                    [[3, 4, 0.5]],
                    [[22, 23, 0.95], [24, 25, 0.8]],
                ]
            ),
        )

        payload = serialize_prediction_result(result, workflow="pose")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["workflow"], "pose")
        self.assertEqual(len(payload["detections"]), 2)
        self.assertEqual(payload["detections"][0]["class_id"], 1)
        self.assertAlmostEqual(payload["detections"][1]["confidence"], 0.9)
        self.assertEqual(payload["detections"][1]["keypoints"][0], [22.0, 23.0, 0.95])

    def test_serialize_prediction_result_uses_segmentation_polygons(self):
        result = _Result(
            boxes=_Boxes(xyxy=[[1, 2, 11, 12]], conf=[0.95], cls=[0]),
            masks=_Masks(xy=[[[1, 2], [11, 2], [11, 12], [1, 12]]]),
        )

        payload = serialize_prediction_result(result, workflow="segmentation")

        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(
            payload["detections"][0]["segments"],
            [[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]],
        )

    def test_top_prediction_from_payload_picks_highest_confidence_detection(self):
        payload = {
            "detections": [
                {"class_id": 0, "confidence": 0.2, "xyxy": [0, 0, 5, 5], "keypoints": [[1, 2, 0.3]]},
                {"class_id": 1, "confidence": 0.8, "xyxy": [10, 11, 20, 21], "keypoints": [[12, 13, 0.9]]},
            ]
        }

        out = top_prediction_from_payload(payload, workflow="pose")

        self.assertTrue(out["ok"])
        self.assertEqual(out["cls"], 1)
        self.assertEqual(out["xyxy"], [10.0, 11.0, 20.0, 21.0])
        self.assertEqual(out["kps"], [[12.0, 13.0, 0.9]])

    def test_predict_worker_runs_model_and_emits_serialized_result(self):
        model = _PredictModel(
            [
                _Result(
                    boxes=_Boxes(xyxy=[[1, 2, 11, 12]], conf=[0.75], cls=[0]),
                    keypoints=_Keypoints([[[3, 4, 0.8]]]),
                )
            ]
        )
        events = []

        exit_code = run_predict_worker(
            {
                "model_path": "model.pt",
                "image_path": "image.png",
                "workflow": "pose",
                "device": "cpu",
            },
            model_factory=lambda _path: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual([event["event"] for event in events], ["started", "result"])
        self.assertEqual(model.calls[0]["source"], "image.png")
        self.assertEqual(model.calls[0]["device"], "cpu")
        self.assertFalse(events[-1]["had_error"])
        self.assertEqual(events[-1]["prediction"]["detections"][0]["xyxy"], [1.0, 2.0, 11.0, 12.0])

    def test_predict_worker_reports_missing_image_path(self):
        events = []

        exit_code = run_predict_worker(
            {"model_path": "model.pt"},
            model_factory=lambda _path: _PredictModel([]),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertIn("image_path", events[0]["error_message"])

    def test_predict_worker_rejects_model_task_mismatch(self):
        model = _PredictModel([])
        model.task = "detect"
        events = []

        exit_code = run_predict_worker(
            {
                "model_path": "model.pt",
                "image_path": "image.png",
                "workflow": "segmentation",
            },
            model_factory=lambda _path: model,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("task mismatch", events[-1]["error_message"])

    def test_video_review_worker_predicts_strided_frames_and_emits_cache_shape(self):
        model = _VideoReviewModel(
            [
                _Result(boxes=_Boxes(xyxy=[[1, 2, 11, 12]], conf=[0.25], cls=[0])),
                _Result(boxes=_Boxes(xyxy=[[5, 6, 15, 16]], conf=[0.85], cls=[1])),
            ]
        )
        cv2 = _FakeCv2(["frame0", "frame1", "frame2"])
        events = []

        exit_code = run_video_review_worker(
            {
                "model_path": "model.pt",
                "video_path": "video.mp4",
                "workflow": "pose",
                "device": "cpu",
                "start": 0,
                "end": 2,
                "stride": 2,
                "batch": 1,
                "effective_batch": 1,
            },
            model_factory=lambda _path: model,
            cv2_module=cv2,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(events[0]["event"], "started")
        self.assertEqual([event["event"] for event in events].count("progress"), 2)
        self.assertTrue(cv2.capture.released)
        self.assertEqual(model.calls[0]["source"], ["frame0"])
        self.assertEqual(model.calls[1]["source"], ["frame2"])
        result = events[-1]
        self.assertEqual(result["event"], "result")
        self.assertFalse(result["had_error"])
        self.assertEqual(sorted(result["preds"].keys()), ["0", "2"])
        self.assertEqual(result["preds"]["2"]["cls"], 1)
        self.assertAlmostEqual(result["preds"]["2"]["conf"], 0.85)

    def test_video_review_worker_reports_batch_prediction_errors(self):
        cv2 = _FakeCv2(["frame0", "frame1"])
        events = []

        exit_code = run_video_review_worker(
            {
                "model_path": "model.pt",
                "video_path": "video.mp4",
                "workflow": "pose",
                "device": "cpu",
                "start": 0,
                "end": 1,
                "stride": 1,
                "batch": 2,
                "effective_batch": 2,
            },
            model_factory=lambda _path: _FailingVideoReviewModel(),
            cv2_module=cv2,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        result = events[-1]
        self.assertEqual(result["event"], "result")
        self.assertTrue(result["had_error"])
        self.assertIn("predict boom", result["error_message"])
        self.assertEqual(result["preds"]["0"]["ok"], False)
        self.assertEqual(result["preds"]["1"]["ok"], False)

    def test_video_review_worker_rejects_model_task_mismatch(self):
        model = _VideoReviewModel([])
        model.task = "pose"
        cv2 = _FakeCv2(["frame0"])
        events = []

        exit_code = run_video_review_worker(
            {
                "model_path": "model.pt",
                "video_path": "video.mp4",
                "workflow": "segmentation",
                "start": 0,
                "end": 0,
            },
            model_factory=lambda _path: model,
            cv2_module=cv2,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("task mismatch", events[-1]["error_message"])
        self.assertTrue(cv2.capture.released)


if __name__ == "__main__":
    unittest.main()
