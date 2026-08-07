import json
import os
import unittest
from tempfile import TemporaryDirectory

from predict_worker import run_predict_server, run_predict_worker
from prediction_ops import (
    best_predictions_by_class_from_payload,
    prediction_confidences_by_class,
    rank_prediction_frames,
    serialize_prediction_result,
    top_prediction_from_payload,
)
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


class _Depth:
    def __init__(self, data):
        self.data = _Tensor(data)


class _DepthResult:
    def __init__(self, data):
        self.depth = _Depth(data)


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
        count = len(kwargs["source"]) if isinstance(kwargs["source"], list) else 1
        out = self.results[:count]
        self.results = self.results[count:]
        return out


class _FailingVideoReviewModel:
    def predict(self, **kwargs):
        raise RuntimeError("predict boom")


class _OomThenVideoReviewModel(_VideoReviewModel):
    def __init__(self, results):
        super().__init__(results)
        self.failed_once = False

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if not self.failed_once and len(kwargs["source"]) > 1:
            self.failed_once = True
            raise RuntimeError("CUDA out of memory")
        count = len(kwargs["source"])
        out = self.results[:count]
        self.results = self.results[count:]
        return out


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.pos = 0
        self.released = False
        self.set_positions = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == _FakeCv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)
            self.set_positions.append(int(value))

    def read(self):
        if self.pos < 0 or self.pos >= len(self.frames):
            return False, None
        frame = self.frames[self.pos]
        self.pos += 1
        return True, frame

    def grab(self):
        if self.pos < 0 or self.pos >= len(self.frames):
            return False
        self.pos += 1
        return True

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, frames):
        self.capture = _FakeCapture(frames)
        self.staged_frames = {}
        self.written_frames = []

    def VideoCapture(self, _path):
        return self.capture

    def imwrite(self, path, frame):
        self.staged_frames[path] = frame
        self.written_frames.append(frame)
        with open(path, "wb") as handle:
            handle.write(b"image")
        return True


def _streamed_predictions(events):
    predictions = {}
    for event in events:
        if event.get("event") == "progress":
            predictions.update(event.get("predictions") or {})
    return predictions


class PredictionOpsTests(unittest.TestCase):
    def test_predict_worker_writes_depth_outputs_without_json_array(self):
        with TemporaryDirectory() as tmp:
            model = _PredictModel([_DepthResult([[1.0, 2.0], [3.0, 4.0]])])
            model.task = "depth"
            events = []
            config = {
                "model_path": "yolo26n-depth.pt",
                "image_path": "image.png",
                "workflow": "depth",
                "layer_id": "depth",
                "depth_map_path": os.path.join(tmp, "depth.npy"),
                "depth_preview_path": os.path.join(tmp, "depth.png"),
                "depth_metadata_path": os.path.join(tmp, "depth.json"),
            }

            exit_code = run_predict_worker(
                config,
                model_factory=lambda _path: model,
                event_writer=events.append,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(model.calls[0]["imgsz"], 768)
            self.assertNotIn("conf", model.calls[0])
            prediction = events[-1]["prediction"]
            self.assertEqual(prediction["workflow"], "depth")
            self.assertNotIn("depth_map", prediction)
            self.assertTrue(os.path.isfile(config["depth_map_path"]))

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
                {
                    "class_id": 0,
                    "confidence": 0.2,
                    "xyxy": [0, 0, 5, 5],
                    "keypoints": [[1, 2, 0.3]],
                },
                {
                    "class_id": 1,
                    "confidence": 0.8,
                    "xyxy": [10, 11, 20, 21],
                    "keypoints": [[12, 13, 0.9]],
                },
            ]
        }

        out = top_prediction_from_payload(payload, workflow="pose")

        self.assertTrue(out["ok"])
        self.assertEqual(out["cls"], 1)
        self.assertEqual(out["xyxy"], [10.0, 11.0, 20.0, 21.0])
        self.assertEqual(out["kps"], [[12.0, 13.0, 0.9]])

    def test_best_predictions_by_class_matches_labeler_selection(self):
        payload = {
            "detections": [
                {
                    "class_id": 0,
                    "confidence": 0.7,
                    "xyxy": [1, 2, 3, 4],
                    "keypoints": [[5, 6, 0.8]],
                },
                {
                    "class_id": 1,
                    "confidence": 0.9,
                    "xyxy": [10, 20, 30, 40],
                    "keypoints": [[50, 60, 0.95]],
                },
                {
                    "class_id": 0,
                    "confidence": 0.8,
                    "xyxy": [7, 8, 9, 10],
                    "keypoints": [[11, 12, 0.85]],
                },
            ]
        }

        outputs = best_predictions_by_class_from_payload(payload, workflow="pose")

        self.assertEqual([output["cls"] for output in outputs], [0, 1])
        self.assertEqual(outputs[0]["xyxy"], [7.0, 8.0, 9.0, 10.0])
        self.assertEqual(outputs[1]["kps"], [[50.0, 60.0, 0.95]])

    def test_prediction_confidences_by_class_reads_multi_class_overlay(self):
        prediction = {
            "ok": True,
            "cls": 1,
            "conf": 0.95,
            "detections": [
                {"ok": True, "cls": 0, "conf": 0.6},
                {"ok": True, "cls": 1, "conf": 0.95},
            ],
        }

        self.assertEqual(prediction_confidences_by_class(prediction), {0: 0.6, 1: 0.95})

    def test_rank_prediction_frames_low_treats_missing_class_as_zero(self):
        predictions = {
            10: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.7}]},
            11: {"ok": True, "detections": [{"ok": True, "cls": 1, "conf": 0.9}]},
            12: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.3}]},
        }

        ranked = rank_prediction_frames(predictions, class_ids=[0], order="low")

        self.assertEqual(ranked, [(11, 0.0, 0), (12, 0.3, 0), (10, 0.7, 0)])

    def test_rank_prediction_frames_high_excludes_missing_class(self):
        predictions = {
            10: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.7}]},
            11: {"ok": True, "detections": [{"ok": True, "cls": 1, "conf": 0.9}]},
            12: {"ok": True, "detections": [{"ok": True, "cls": 0, "conf": 0.3}]},
        }

        ranked = rank_prediction_frames(predictions, class_ids=[0], order="high")

        self.assertEqual(ranked, [(10, 0.7, 0), (12, 0.3, 0)])

    def test_rank_prediction_frames_balances_classes_and_deduplicates(self):
        predictions = {
            1: {
                "ok": True,
                "detections": [
                    {"ok": True, "cls": 0, "conf": 0.1},
                    {"ok": True, "cls": 1, "conf": 0.8},
                ],
            },
            2: {
                "ok": True,
                "detections": [
                    {"ok": True, "cls": 0, "conf": 0.2},
                    {"ok": True, "cls": 1, "conf": 0.1},
                ],
            },
            3: {
                "ok": True,
                "detections": [
                    {"ok": True, "cls": 0, "conf": 0.3},
                    {"ok": True, "cls": 1, "conf": 0.2},
                ],
            },
        }

        ranked = rank_prediction_frames(
            predictions,
            class_ids=[0, 1],
            order="low",
            balanced=True,
        )

        self.assertEqual(ranked, [(1, 0.1, 0), (2, 0.1, 1), (3, 0.3, 0)])
        self.assertEqual(len({frame_idx for frame_idx, _, _ in ranked}), len(ranked))

    def test_rank_prediction_frames_skips_prediction_errors(self):
        predictions = {
            1: {"ok": False, "error": "predict failed"},
            2: {"ok": False, "detections": []},
        }

        ranked = rank_prediction_frames(predictions, class_ids=[0], order="low")

        self.assertEqual(ranked, [(2, 0.0, 0)])

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
        self.assertFalse(model.calls[0]["end2end"])
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

    def test_predict_server_reuses_loaded_model_across_requests(self):
        model = _PredictModel(
            [
                _Result(
                    boxes=_Boxes(xyxy=[[1, 2, 11, 12]], conf=[0.75], cls=[0]),
                    keypoints=_Keypoints([[[3, 4, 0.8]]]),
                )
            ]
        )
        factory_calls = []
        events = []
        requests = [
            json.dumps(
                {
                    "command": "load",
                    "request_id": 1,
                    "model_path": "model.pt",
                    "workflow": "pose",
                    "device": "cuda",
                }
            ),
            json.dumps(
                {
                    "command": "predict",
                    "request_id": 2,
                    "model_path": "model.pt",
                    "image_path": "one.png",
                    "workflow": "pose",
                    "device": "cuda",
                }
            ),
            json.dumps(
                {
                    "command": "predict",
                    "request_id": 3,
                    "model_path": "model.pt",
                    "image_path": "two.png",
                    "workflow": "pose",
                    "device": "cuda",
                }
            ),
            json.dumps({"command": "shutdown", "request_id": 4}),
        ]

        def factory(path):
            factory_calls.append(path)
            return model

        exit_code = run_predict_server(
            requests,
            model_factory=factory,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(factory_calls, ["model.pt"])
        self.assertEqual([call["source"] for call in model.calls], ["one.png", "two.png"])
        self.assertEqual(events[0]["event"], "ready")
        self.assertEqual(
            [
                (event["event"], event.get("request_id"))
                for event in events
                if event["event"] == "result"
            ],
            [("result", 2), ("result", 3)],
        )

    def test_predict_server_reloads_only_when_model_path_changes(self):
        models = {
            "one.pt": _PredictModel([]),
            "two.pt": _PredictModel([]),
        }
        factory_calls = []
        events = []
        requests = [
            json.dumps(
                {"command": "load", "request_id": 1, "model_path": "one.pt", "workflow": "pose"}
            ),
            json.dumps(
                {"command": "load", "request_id": 2, "model_path": "one.pt", "workflow": "pose"}
            ),
            json.dumps(
                {"command": "load", "request_id": 3, "model_path": "two.pt", "workflow": "pose"}
            ),
        ]

        def factory(path):
            factory_calls.append(path)
            return models[path]

        exit_code = run_predict_server(
            requests,
            model_factory=factory,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(factory_calls, ["one.pt", "two.pt"])

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
        progress_events = [event for event in events if event["event"] == "progress"]
        self.assertEqual(progress_events[0]["predictions"]["0"]["xyxy"], [1.0, 2.0, 11.0, 12.0])
        self.assertTrue(cv2.capture.released)
        self.assertEqual(cv2.capture.set_positions, [0])
        self.assertEqual(model.calls[0]["source"], ["frame0"])
        self.assertEqual(model.calls[1]["source"], ["frame2"])
        self.assertEqual(model.calls[0]["batch"], 1)
        self.assertFalse(model.calls[0]["end2end"])
        result = events[-1]
        self.assertEqual(result["event"], "result")
        self.assertFalse(result["had_error"])
        self.assertTrue(result["preds_streamed"])
        self.assertEqual(result["prediction_count"], 2)
        self.assertEqual(result["preds"], {})
        predictions = _streamed_predictions(events)
        self.assertEqual(sorted(predictions), ["0", "2"])
        self.assertEqual(predictions["2"]["cls"], 1)
        self.assertAlmostEqual(predictions["2"]["conf"], 0.85)
        self.assertEqual(predictions["2"]["detections"][0]["cls"], 1)

    def test_video_review_worker_batches_frames_and_preserves_result_order(self):
        model = _VideoReviewModel(
            [
                _Result(boxes=_Boxes(xyxy=[[0, 0, 10, 10]], conf=[0.4], cls=[0])),
                _Result(boxes=_Boxes(xyxy=[[1, 1, 11, 11]], conf=[0.5], cls=[0])),
                _Result(boxes=_Boxes(xyxy=[[2, 2, 12, 12]], conf=[0.6], cls=[0])),
            ]
        )
        cv2 = _FakeCv2(["frame0", "frame1", "frame2"])
        events = []

        exit_code = run_video_review_worker(
            {
                "model_path": "model.pt",
                "video_path": "video.mp4",
                "workflow": "pose",
                "device": "cuda",
                "start": 0,
                "end": 2,
                "stride": 1,
                "batch": 2,
                "effective_batch": 2,
            },
            model_factory=lambda _path: model,
            cv2_module=cv2,
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(model.calls[0]["source"], ["frame0", "frame1"])
        self.assertEqual(model.calls[1]["source"], ["frame2"])
        predictions = _streamed_predictions(events)
        self.assertEqual(predictions["0"]["xyxy"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(predictions["1"]["xyxy"], [1.0, 1.0, 11.0, 11.0])
        self.assertEqual(predictions["2"]["xyxy"], [2.0, 2.0, 12.0, 12.0])

    def test_video_review_worker_auto_batch_retries_after_cuda_oom(self):
        model = _OomThenVideoReviewModel(
            [
                _Result(boxes=_Boxes(xyxy=[[0, 0, 10, 10]], conf=[0.4], cls=[0])),
                _Result(boxes=_Boxes(xyxy=[[1, 1, 11, 11]], conf=[0.5], cls=[0])),
            ]
        )
        events = []

        exit_code = run_video_review_worker(
            {
                "model_path": "model.pt",
                "video_path": "video.mp4",
                "workflow": "pose",
                "device": "cuda",
                "start": 0,
                "end": 1,
                "batch": 0,
                "effective_batch": 2,
            },
            model_factory=lambda _path: model,
            cv2_module=_FakeCv2(["frame0", "frame1"]),
            event_writer=events.append,
        )

        self.assertEqual(exit_code, 0)
        adjustments = [event for event in events if event["event"] == "batch_adjusted"]
        self.assertEqual(adjustments[0]["effective_batch"], 1)
        self.assertEqual([call["batch"] for call in model.calls], [2, 1, 1])
        self.assertEqual(sorted(_streamed_predictions(events)), ["0", "1"])

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
        predictions = _streamed_predictions(events)
        self.assertEqual(predictions["0"]["ok"], False)
        self.assertEqual(predictions["1"]["ok"], False)

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
