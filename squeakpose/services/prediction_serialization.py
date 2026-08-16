"""Qt-free helpers for serializing and ranking YOLO predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from squeakpose.project.layers import normalize_layer_id


def _to_list(value: Any) -> list:
    if value is None:
        return []
    try:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)
    except Exception:
        return []


def _mask_segments_from_data(
    mask_data: Any, det_idx: int, *, cv2_module: Any = None, numpy_module: Any = None
) -> list[list[float]]:
    if cv2_module is None or numpy_module is None:
        return []
    try:
        if mask_data is None or det_idx >= len(mask_data):
            return []
        mask_arr = mask_data[det_idx]
        if hasattr(mask_arr, "cpu"):
            mask_arr = mask_arr.cpu()
        if hasattr(mask_arr, "numpy"):
            mask_arr = mask_arr.numpy()
        if getattr(mask_arr, "ndim", 0) == 3:
            mask_arr = mask_arr[0]
        mask_u8 = (numpy_module.asarray(mask_arr) > 0.5).astype(numpy_module.uint8) * 255
        contours_info = cv2_module.findContours(
            mask_u8, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_NONE
        )
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return []
        contour = max(contours, key=cv2_module.contourArea)
        return [[float(pt[0][0]), float(pt[0][1])] for pt in contour if len(pt[0]) >= 2]
    except Exception:
        return []


def serialize_prediction_result(
    result: Any,
    *,
    workflow: str,
    layer_id: str = "",
    cv2_module: Any = None,
    numpy_module: Any = None,
) -> dict[str, Any]:
    """Convert one Ultralytics result to JSON-compatible detection data."""
    payload: dict[str, Any] = {
        "ok": True,
        "layer_id": normalize_layer_id(layer_id or workflow),
        "workflow": workflow,
        "detections": [],
    }
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return payload

    xyxy = _to_list(getattr(boxes, "xyxy", None))
    confs = _to_list(getattr(boxes, "conf", None)) or [0.0] * len(xyxy)
    class_ids = _to_list(getattr(boxes, "cls", None)) or [0] * len(xyxy)

    keypoints = []
    if getattr(result, "keypoints", None) is not None:
        keypoints = _to_list(getattr(result.keypoints, "data", None))

    mask_polys: list[Any] = []
    mask_data = None
    if getattr(result, "masks", None) is not None:
        try:
            mask_polys = result.masks.xy or []
        except Exception:
            mask_polys = []
        mask_data = getattr(result.masks, "data", None)

    detections: list[dict[str, Any]] = []
    for det_idx, box in enumerate(xyxy):
        try:
            coords = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        except Exception:
            continue
        det: dict[str, Any] = {
            "det": det_idx,
            "class_id": int(class_ids[det_idx]) if det_idx < len(class_ids) else 0,
            "confidence": float(confs[det_idx]) if det_idx < len(confs) else 0.0,
            "xyxy": coords,
            "keypoints": [],
            "segments": [],
        }

        if det_idx < len(keypoints):
            for raw_kp in keypoints[det_idx] or []:
                try:
                    if len(raw_kp) < 3:
                        continue
                    det["keypoints"].append([float(raw_kp[0]), float(raw_kp[1]), float(raw_kp[2])])
                except Exception:
                    continue

        if det_idx < len(mask_polys):
            points: list[list[float]] = []
            for node in mask_polys[det_idx]:
                try:
                    if len(node) < 2:
                        continue
                    points.append([float(node[0]), float(node[1])])
                except Exception:
                    continue
            if len(points) >= 3:
                det["segments"] = points
        if not det["segments"]:
            fallback = _mask_segments_from_data(
                mask_data, det_idx, cv2_module=cv2_module, numpy_module=numpy_module
            )
            if len(fallback) >= 3:
                det["segments"] = fallback

        detections.append(det)

    payload["detections"] = detections
    return payload


def top_prediction_from_payload(payload: dict[str, Any], *, workflow: str) -> dict[str, Any]:
    """Convert serialized detections to the Video Reviewer single-overlay shape."""
    out: dict[str, Any] = {
        "ok": False,
        "conf": 0.0,
        "cls": 0,
        "xyxy": None,
        "kps": [],
        "segments": [],
    }
    detections = payload.get("detections") or []
    if not isinstance(detections, list):
        return out

    best: dict[str, Any] | None = None
    best_conf = float("-inf")
    for det in detections:
        if not isinstance(det, dict):
            continue
        try:
            conf = float(det.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if best is None or conf >= best_conf:
            best = det
            best_conf = conf
    if best is None:
        return out

    out["ok"] = True
    out["conf"] = 0.0 if best_conf == float("-inf") else float(best_conf)
    try:
        out["cls"] = int(best.get("class_id", 0))
    except Exception:
        out["cls"] = 0

    xyxy = best.get("xyxy") or []
    try:
        if len(xyxy) >= 4:
            out["xyxy"] = [float(v) for v in xyxy[:4]]
    except Exception:
        out["xyxy"] = None

    if str(workflow).strip().lower() == "segmentation":
        segments: list[list[float]] = []
        for pair in best.get("segments") or []:
            try:
                if len(pair) < 2:
                    continue
                segments.append([float(pair[0]), float(pair[1])])
            except Exception:
                continue
        out["segments"] = segments
    else:
        keypoints: list[list[float]] = []
        for raw_kp in best.get("keypoints") or []:
            try:
                if len(raw_kp) < 3:
                    continue
                keypoints.append([float(raw_kp[0]), float(raw_kp[1]), float(raw_kp[2])])
            except Exception:
                continue
        out["kps"] = keypoints
    return out


def best_predictions_by_class_from_payload(
    payload: dict[str, Any], *, workflow: str
) -> list[dict[str, Any]]:
    """Return the highest-confidence serialized detection for each model class."""
    detections = payload.get("detections") or []
    if not isinstance(detections, list):
        return []

    best_by_class: dict[int, dict[str, Any]] = {}
    for det in detections:
        if not isinstance(det, dict):
            continue
        try:
            class_id = int(det.get("class_id", 0))
        except Exception:
            class_id = 0
        try:
            confidence = float(det.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        previous = best_by_class.get(class_id)
        if previous is None or confidence >= float(previous.get("confidence", 0.0) or 0.0):
            best_by_class[class_id] = det

    predictions: list[dict[str, Any]] = []
    for class_id in sorted(best_by_class):
        one_payload = {"detections": [best_by_class[class_id]]}
        predictions.append(top_prediction_from_payload(one_payload, workflow=workflow))
    return predictions


def prediction_confidences_by_class(prediction: dict[str, Any]) -> dict[int, float]:
    """Return the best available detection confidence for each class in one frame."""
    raw_detections = prediction.get("detections")
    detections = raw_detections if isinstance(raw_detections, list) else [prediction]
    confidences: dict[int, float] = {}
    for detection in detections:
        if not isinstance(detection, dict) or not detection.get("ok"):
            continue
        try:
            class_id = int(detection.get("cls", 0))
            confidence = float(detection.get("conf", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if class_id not in confidences or confidence >= confidences[class_id]:
            confidences[class_id] = confidence
    return confidences


def rank_prediction_frames(
    predictions: Mapping[int, dict[str, Any]],
    *,
    class_ids: list[int],
    order: str,
    balanced: bool = False,
) -> list[tuple[int, float, int]]:
    """Rank frames by class confidence, optionally round-robin balancing classes.

    Returned tuples are ``(frame_index, confidence, ranking_class_id)``. For low
    confidence ranking, a successful frame with no detection for the requested
    class receives 0.0. High confidence ranking excludes missing classes.
    """
    order_key = "high" if str(order).lower() == "high" else "low"
    valid_class_ids = list(dict.fromkeys(int(cid) for cid in class_ids))
    if not valid_class_ids:
        return []

    per_class: dict[int, list[tuple[int, float, int]]] = {}
    for class_id in valid_class_ids:
        ranked: list[tuple[int, float, int]] = []
        for frame_idx, prediction in predictions.items():
            if not isinstance(prediction, dict) or prediction.get("error"):
                continue
            confidences = prediction_confidences_by_class(prediction)
            if class_id not in confidences:
                if order_key == "high":
                    continue
                confidence = 0.0
            else:
                confidence = confidences[class_id]
            ranked.append((int(frame_idx), float(confidence), class_id))
        if order_key == "high":
            ranked.sort(key=lambda item: (-item[1], item[0]))
        else:
            ranked.sort(key=lambda item: (item[1], item[0]))
        per_class[class_id] = ranked

    if not balanced or len(valid_class_ids) == 1:
        return per_class[valid_class_ids[0]]

    positions = {class_id: 0 for class_id in valid_class_ids}
    selected: list[tuple[int, float, int]] = []
    selected_frames: set[int] = set()
    while True:
        added_this_round = False
        queues_exhausted = True
        for class_id in valid_class_ids:
            queue = per_class[class_id]
            pos = positions[class_id]
            while pos < len(queue) and queue[pos][0] in selected_frames:
                pos += 1
            positions[class_id] = pos
            if pos >= len(queue):
                continue
            queues_exhausted = False
            item = queue[pos]
            positions[class_id] = pos + 1
            selected.append(item)
            selected_frames.add(item[0])
            added_this_round = True
        if queues_exhausted or not added_this_round:
            break
    return selected
