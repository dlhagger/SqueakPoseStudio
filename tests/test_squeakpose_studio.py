import json
import os
import unittest
from tempfile import TemporaryDirectory

try:
    from squeakpose_studio import VideoReviewDialog, WORKFLOW_POSE, WORKFLOW_SEG
    _STUDIO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent import gate
    VideoReviewDialog = None
    WORKFLOW_POSE = "pose"
    WORKFLOW_SEG = "segmentation"
    _STUDIO_IMPORT_ERROR = exc


class _ArrayLike:
    def __init__(self, data):
        self._data = data

    @property
    def shape(self):
        dims = []
        cur = self._data
        while isinstance(cur, list):
            dims.append(len(cur))
            cur = cur[0] if cur else []
        return tuple(dims)

    def argmax(self):
        if not isinstance(self._data, list) or not self._data:
            return 0
        best_idx = 0
        best_val = float(self._data[0])
        for idx, value in enumerate(self._data[1:], start=1):
            val = float(value)
            if val > best_val:
                best_val = val
                best_idx = idx
        return best_idx

    def tolist(self):
        return self._data

    def __len__(self):
        return len(self._data) if isinstance(self._data, list) else 0

    def __getitem__(self, idx):
        value = self._data[idx]
        if isinstance(value, list):
            return _ArrayLike(value)
        return value

    def __iter__(self):
        return iter(self._data if isinstance(self._data, list) else [])


class _TensorLike:
    def __init__(self, data):
        self._arr = _ArrayLike(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, idx):
        return self._arr[idx]


class _Boxes:
    def __init__(self, conf, cls, xyxy):
        self.conf = _TensorLike(conf) if conf is not None else None
        self.cls = _TensorLike(cls) if cls is not None else None
        self.xyxy = _TensorLike(xyxy) if xyxy is not None else None
        self._n = len(conf) if conf is not None else 0

    def __len__(self):
        return self._n


class _Keypoints:
    def __init__(self, data):
        self.data = _TensorLike(data)


class _Masks:
    def __init__(self, xy=None, data=None):
        self.xy = xy
        self.data = _TensorLike(data) if data is not None else None


class _Results:
    def __init__(self, boxes, keypoints=None, masks=None):
        self.boxes = boxes
        self.keypoints = keypoints
        self.masks = masks


class _CacheDummy:
    def __init__(self, video_path: str, model_path: str, workflow: str):
        self.path = video_path
        self.model_path = model_path
        self.workflow = workflow
        self.preds = {}

    def _cache_path(self):
        return os.path.abspath(self.path) + ".sqp_preds.json"

    def _video_signature(self):
        return {
            "path": os.path.abspath(self.path),
            "size": int(os.path.getsize(self.path)),
            "mtime": float(os.path.getmtime(self.path)),
            "total": 10,
            "fps": 30.0,
        }


@unittest.skipIf(VideoReviewDialog is None, f"squeakpose_studio import unavailable: {_STUDIO_IMPORT_ERROR}")
class StudioVideoReviewTests(unittest.TestCase):
    def test_extract_top_pose_uses_highest_confidence_detection(self):
        boxes = _Boxes(
            conf=[0.25, 0.91],
            cls=[1, 0],
            xyxy=[
                [10, 11, 50, 51],
                [1, 2, 30, 40],
            ],
        )
        keypoints = _Keypoints(
            [
                [[1, 2, 0.4], [3, 4, 0.6]],
                [[7, 8, 0.9], [9, 10, 0.2]],
            ]
        )
        results = _Results(boxes=boxes, keypoints=keypoints)

        out = VideoReviewDialog._extract_top_pose(results)
        self.assertTrue(out["ok"])
        self.assertAlmostEqual(out["conf"], 0.91, places=5)
        self.assertEqual(out["cls"], 0)
        self.assertEqual([round(v, 2) for v in out["xyxy"]], [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(len(out["kps"]), 2)
        self.assertEqual([round(v, 2) for v in out["kps"][0]], [7.0, 8.0, 0.9])

    def test_extract_top_seg_uses_mask_polygon_for_best_detection(self):
        boxes = _Boxes(
            conf=[0.4, 0.95],
            cls=[0, 1],
            xyxy=[
                [5, 5, 10, 10],
                [11, 12, 21, 22],
            ],
        )
        masks = _Masks(
            xy=[
                [[1, 1], [2, 1], [2, 2]],
                [[10, 10], [20, 10], [20, 20], [10, 20]],
            ]
        )
        results = _Results(boxes=boxes, masks=masks)

        out = VideoReviewDialog._extract_top_seg(results)
        self.assertTrue(out["ok"])
        self.assertAlmostEqual(out["conf"], 0.95, places=5)
        self.assertEqual(out["cls"], 1)
        self.assertEqual(len(out["segments"]), 4)
        self.assertEqual([round(v, 2) for v in out["segments"][0]], [10.0, 10.0])

    def test_extract_prediction_dispatches_by_workflow(self):
        boxes = _Boxes(conf=[0.9], cls=[0], xyxy=[[0, 0, 10, 10]])
        pose_results = _Results(
            boxes=boxes,
            keypoints=_Keypoints([[[1, 2, 0.8]]]),
        )
        seg_results = _Results(
            boxes=boxes,
            masks=_Masks(xy=[[[0, 0], [1, 0], [1, 1]]]),
        )

        pose_dlg = VideoReviewDialog.__new__(VideoReviewDialog)
        pose_dlg.workflow = WORKFLOW_POSE
        pose_out = VideoReviewDialog._extract_prediction(pose_dlg, pose_results)
        self.assertTrue(pose_out["ok"])
        self.assertEqual(len(pose_out["kps"]), 1)
        self.assertEqual(len(pose_out["segments"]), 0)

        seg_dlg = VideoReviewDialog.__new__(VideoReviewDialog)
        seg_dlg.workflow = WORKFLOW_SEG
        seg_out = VideoReviewDialog._extract_prediction(seg_dlg, seg_results)
        self.assertTrue(seg_out["ok"])
        self.assertEqual(len(seg_out["segments"]), 3)

    def test_cache_validation_rejects_workflow_mismatch(self):
        with TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "sample.mp4")
            with open(video_path, "wb") as f:
                f.write(b"fake-video-bytes")

            dummy = _CacheDummy(video_path=video_path, model_path="model.pt", workflow=WORKFLOW_SEG)
            sig = dummy._video_signature()
            payload = {
                "meta": {
                    "video": sig,
                    "model_path": "model.pt",
                    "workflow": WORKFLOW_POSE,
                },
                "preds": {"4": {"ok": True, "conf": 0.7}},
            }
            with open(dummy._cache_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f)

            ok = VideoReviewDialog._load_cache_if_valid(dummy)
            self.assertFalse(ok)
            self.assertEqual(dummy.preds, {})

    def test_cache_validation_accepts_matching_workflow(self):
        with TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "sample.mp4")
            with open(video_path, "wb") as f:
                f.write(b"fake-video-bytes")

            dummy = _CacheDummy(video_path=video_path, model_path="model.pt", workflow=WORKFLOW_SEG)
            sig = dummy._video_signature()
            payload = {
                "meta": {
                    "video": sig,
                    "model_path": "model.pt",
                    "workflow": WORKFLOW_SEG,
                },
                "preds": {
                    "2": {"ok": True, "conf": 0.8},
                    "5": {"ok": True, "conf": 0.9},
                },
            }
            with open(dummy._cache_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f)

            ok = VideoReviewDialog._load_cache_if_valid(dummy)
            self.assertTrue(ok)
            self.assertEqual(sorted(dummy.preds.keys()), [2, 5])


if __name__ == "__main__":
    unittest.main()
