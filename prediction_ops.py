"""Qt-free helpers for serializing single-image YOLO predictions."""

from __future__ import annotations

from typing import Any


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


def _mask_segments_from_data(mask_data: Any, det_idx: int, *, cv2_module: Any = None, numpy_module: Any = None) -> list[list[float]]:
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
        contours_info = cv2_module.findContours(mask_u8, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_NONE)
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
    cv2_module: Any = None,
    numpy_module: Any = None,
) -> dict[str, Any]:
    """Convert one Ultralytics result to JSON-compatible detection data."""
    payload: dict[str, Any] = {
        "ok": True,
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

    mask_polys = []
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
            fallback = _mask_segments_from_data(mask_data, det_idx, cv2_module=cv2_module, numpy_module=numpy_module)
            if len(fallback) >= 3:
                det["segments"] = fallback

        detections.append(det)

    payload["detections"] = detections
    return payload


def top_prediction_from_payload(payload: dict[str, Any], *, workflow: str) -> dict[str, Any]:
    """Convert serialized detections to the Video Reviewer single-overlay shape."""
    out: dict[str, Any] = {"ok": False, "conf": 0.0, "cls": 0, "xyxy": None, "kps": [], "segments": []}
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
