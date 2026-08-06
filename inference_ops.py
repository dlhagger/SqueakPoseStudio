"""Qt-free synchronous video inference helpers."""

from __future__ import annotations

import csv
import gc
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from depth_ops import colorize_depth_map, depth_array_from_result, depth_map_summary
from squeakpose_core import InferenceCsvWriter, build_segmentation_inference_rows

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class VideoMetadata:
    opened: bool
    total_frames: int = 0
    fps: float = 0.0


@dataclass
class InferenceRunResult:
    csv_path: str
    preview_path: str = ""
    rows_written: int = 0
    processed_frames: int = 0
    canceled: bool = False
    had_error: bool = False
    error_message: str = ""


POSE_BASE_FIELDNAMES = [
    "video_path",
    "model_path",
    "frame_index",
    "time_seconds",
    "detections_in_frame",
    "detection_index",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "image_width",
    "image_height",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "mask_area_px",
    "mask_area_norm",
    "mask_vertices",
]

SEGMENTATION_FIELDNAMES = [
    "frame",
    "det",
    "class_id",
    "class_name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "mask_polygon",
    "binary_mask",
]

DEPTH_FIELDNAMES = [
    "video_path",
    "model_path",
    "frame_index",
    "time_seconds",
    "image_width",
    "image_height",
    "depth_width",
    "depth_height",
    "valid_pixels",
    "min_depth",
    "max_depth",
    "median_depth",
    "p02_depth",
    "p98_depth",
    "units",
    "scale_status",
]


def probe_video_metadata(video_path: str, cv2_module: Any) -> VideoMetadata:
    """Open a video long enough to read frame count and FPS."""
    cap = None
    try:
        cap = cv2_module.VideoCapture(video_path)
        if cap is None or not cap.isOpened():
            return VideoMetadata(opened=False)
        total = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
        return VideoMetadata(opened=True, total_frames=total, fps=fps)
    except Exception:
        return VideoMetadata(opened=False)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def keypoint_column_key(name: str, idx: int) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or f"kp{idx}").strip().lower())
    return safe.strip("_") or f"kp{idx}"


def pose_inference_fieldnames(kp_names: list[str]) -> list[str]:
    kp_columns: list[str] = []
    for idx, kp_name in enumerate(kp_names):
        key = keypoint_column_key(kp_name, idx)
        kp_columns.extend(
            [
                f"kp_{key}_x",
                f"kp_{key}_y",
                f"kp_{key}_conf",
                f"kp_{key}_x_norm",
                f"kp_{key}_y_norm",
            ]
        )
    return POSE_BASE_FIELDNAMES + kp_columns


def _to_list(value: Any) -> list:
    if value is None:
        return []
    try:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "numpy"):
            arr = value.numpy()
            if hasattr(arr, "tolist"):
                return arr.tolist()
            return list(arr)
        return list(value)
    except Exception:
        return []


def _result_shape(result: Any) -> tuple[int, int]:
    if hasattr(result, "orig_shape") and result.orig_shape:
        try:
            return int(result.orig_shape[0]), int(result.orig_shape[1])
        except Exception:
            pass
    if hasattr(result, "orig_img") and getattr(result, "orig_img") is not None:
        try:
            return tuple(int(x) for x in result.orig_img.shape[:2])
        except Exception:
            pass
    return 0, 0


def _class_name(result: Any, cls_id: int, classes: list[str]) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        name = names.get(cls_id, "")
        if name:
            return str(name)
    elif isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    if 0 <= cls_id < len(classes):
        return str(classes[cls_id])
    return ""


def _mask_area(mask: Any) -> Optional[float]:
    try:
        return float(mask.float().sum().item())
    except Exception:
        pass
    try:
        values = _to_list(mask)
        return float(_nested_sum(values))
    except Exception:
        return None


def _nested_sum(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return sum(_nested_sum(item) for item in value)
    return float(value)


def _blank_pose_row(
    *,
    video_path: str,
    model_path: str,
    frame_index: int,
    fps: float,
    detections: int,
    img_w: int,
    img_h: int,
    speed: dict[str, Any],
    kp_columns: list[str],
) -> dict[str, Any]:
    row = {
        "video_path": video_path,
        "model_path": model_path,
        "frame_index": frame_index,
        "time_seconds": (frame_index / fps) if fps > 0 else "",
        "detections_in_frame": detections,
        "detection_index": -1,
        "track_id": "",
        "class_id": "",
        "class_name": "",
        "confidence": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "bbox_width": "",
        "bbox_height": "",
        "bbox_area": "",
        "bbox_center_x": "",
        "bbox_center_y": "",
        "bbox_center_x_norm": "",
        "bbox_center_y_norm": "",
        "bbox_width_norm": "",
        "bbox_height_norm": "",
        "image_width": img_w,
        "image_height": img_h,
        "speed_preprocess_ms": speed.get("preprocess"),
        "speed_inference_ms": speed.get("inference"),
        "speed_postprocess_ms": speed.get("postprocess"),
        "mask_area_px": "",
        "mask_area_norm": "",
        "mask_vertices": "",
    }
    for col in kp_columns:
        row[col] = ""
    return row


def pose_inference_rows_from_result(
    result: Any,
    *,
    frame_index: int,
    video_path: str,
    model_path: str,
    fps: float,
    kp_names: list[str],
    classes: list[str],
) -> list[dict[str, Any]]:
    """Build streamed pose/detection inference CSV rows for one result."""
    img_h, img_w = _result_shape(result)
    detections = int(len(result.boxes) if getattr(result, "boxes", None) is not None else 0)
    speed = getattr(result, "speed", {}) or {}
    fieldnames = pose_inference_fieldnames(kp_names)
    kp_columns = [col for col in fieldnames if col.startswith("kp_")]

    if detections == 0:
        return [
            _blank_pose_row(
                video_path=video_path,
                model_path=model_path,
                frame_index=frame_index,
                fps=fps,
                detections=0,
                img_w=img_w,
                img_h=img_h,
                speed=speed,
                kp_columns=kp_columns,
            )
        ]

    boxes = result.boxes
    xyxy = _to_list(getattr(boxes, "xyxy", None))
    xywh = _to_list(getattr(boxes, "xywh", None))
    confs = _to_list(getattr(boxes, "conf", None)) or [None] * detections
    cls_list = _to_list(getattr(boxes, "cls", None)) or [0] * detections
    ids_list = _to_list(getattr(boxes, "id", None)) or [None] * detections

    kp_abs: list = []
    kp_norm: list = []
    if hasattr(result, "keypoints") and result.keypoints is not None:
        kp_abs = _to_list(getattr(result.keypoints, "data", None))
        kp_norm = _to_list(getattr(result.keypoints, "xyn", None))

    mask_data = _to_list(getattr(getattr(result, "masks", None), "data", None))
    mask_segments = getattr(getattr(result, "masks", None), "xy", []) or []

    rows: list[dict[str, Any]] = []
    for det_idx in range(detections):
        try:
            x1, y1, x2, y2 = xyxy[det_idx]
        except Exception:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        try:
            cx, cy, w, h = xywh[det_idx]
        except Exception:
            w = float(x2) - float(x1)
            h = float(y2) - float(y1)
            cx = float(x1) + w / 2.0
            cy = float(y1) + h / 2.0

        cls_id = int(cls_list[det_idx]) if det_idx < len(cls_list) else 0
        kp_values = kp_abs[det_idx] if det_idx < len(kp_abs) else []
        kp_norm_values = kp_norm[det_idx] if det_idx < len(kp_norm) else []

        mask_area_px: Any = ""
        mask_area_norm: Any = ""
        if det_idx < len(mask_data):
            area = _mask_area(mask_data[det_idx])
            if area is not None:
                mask_area_px = area
                denom = float(img_w * img_h) if img_w and img_h else 0.0
                mask_area_norm = (area / denom) if denom > 0 else ""

        mask_vertices: Any = ""
        if det_idx < len(mask_segments):
            try:
                mask_vertices = int(len(mask_segments[det_idx]))
            except Exception:
                mask_vertices = ""

        row = {
            "video_path": video_path,
            "model_path": model_path,
            "frame_index": frame_index,
            "time_seconds": (frame_index / fps) if fps > 0 else "",
            "detections_in_frame": detections,
            "detection_index": det_idx,
            "track_id": ids_list[det_idx] if det_idx < len(ids_list) and ids_list[det_idx] is not None else "",
            "class_id": cls_id,
            "class_name": _class_name(result, cls_id, classes),
            "confidence": confs[det_idx] if det_idx < len(confs) and confs[det_idx] is not None else "",
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_width": w,
            "bbox_height": h,
            "bbox_area": float(w) * float(h),
            "bbox_center_x": cx,
            "bbox_center_y": cy,
            "bbox_center_x_norm": (float(cx) / img_w) if img_w else "",
            "bbox_center_y_norm": (float(cy) / img_h) if img_h else "",
            "bbox_width_norm": (float(w) / img_w) if img_w else "",
            "bbox_height_norm": (float(h) / img_h) if img_h else "",
            "image_width": img_w,
            "image_height": img_h,
            "speed_preprocess_ms": speed.get("preprocess"),
            "speed_inference_ms": speed.get("inference"),
            "speed_postprocess_ms": speed.get("postprocess"),
            "mask_area_px": mask_area_px,
            "mask_area_norm": mask_area_norm,
            "mask_vertices": mask_vertices,
        }

        for idx_kp, kp_name in enumerate(kp_names):
            key = keypoint_column_key(kp_name, idx_kp)
            abs_val = kp_values[idx_kp] if idx_kp < len(kp_values) else [None, None, None]
            norm_val = kp_norm_values[idx_kp] if idx_kp < len(kp_norm_values) else [None, None]
            row[f"kp_{key}_x"] = abs_val[0] if abs_val and abs_val[0] is not None else ""
            row[f"kp_{key}_y"] = abs_val[1] if abs_val and abs_val[1] is not None else ""
            row[f"kp_{key}_conf"] = abs_val[2] if abs_val and len(abs_val) > 2 else ""
            row[f"kp_{key}_x_norm"] = norm_val[0] if norm_val and norm_val[0] is not None else ""
            row[f"kp_{key}_y_norm"] = norm_val[1] if norm_val and norm_val[1] is not None else ""
        rows.append(row)

    return rows


def segmentation_rows_from_result(
    result: Any,
    frame_idx: int,
    *,
    classes: list[str],
    include_binary_mask: bool = True,
    numpy_module: Any = None,
) -> list[dict[str, Any]]:
    """Build segmentation inference rows for one result without retaining masks."""
    if result.boxes is None or len(result.boxes) == 0:
        return build_segmentation_inference_rows(
            frame_index=frame_idx,
            detections=[],
            class_names=getattr(result, "names", None) or classes,
            include_binary_mask=include_binary_mask,
        )

    try:
        boxes = _to_list(result.boxes.xyxy)
        confs = _to_list(result.boxes.conf) if result.boxes.conf is not None else []
        class_ids = _to_list(result.boxes.cls) if result.boxes.cls is not None else [0] * len(boxes)
    except Exception:
        return []

    mask_polygons = []
    mask_data = None
    if getattr(result, "masks", None) is not None:
        try:
            mask_polygons = result.masks.xy
        except Exception:
            mask_polygons = []
        if include_binary_mask:
            mask_data = _to_list(getattr(result.masks, "data", None))
    if not mask_polygons:
        mask_polygons = [None] * len(boxes)

    detections: list[dict[str, Any]] = []
    for det_idx, box in enumerate(boxes):
        cls_id = int(class_ids[det_idx]) if det_idx < len(class_ids) else 0
        poly = mask_polygons[det_idx] if det_idx < len(mask_polygons) else None
        polygon = poly.tolist() if poly is not None and hasattr(poly, "tolist") else poly
        binary_mask = None
        if include_binary_mask and mask_data is not None and det_idx < len(mask_data):
            if numpy_module is not None:
                try:
                    binary_mask = (numpy_module.asarray(mask_data[det_idx]) > 0.5).astype(numpy_module.uint8)
                except Exception:
                    binary_mask = None
            else:
                binary_mask = mask_data[det_idx]
        detections.append(
            {
                "det": det_idx,
                "class_id": cls_id,
                "class_name": _class_name(result, cls_id, classes),
                "conf": float(confs[det_idx]) if det_idx < len(confs) else 0.0,
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "mask_polygon": polygon,
                "binary_mask": binary_mask,
            }
        )
    return build_segmentation_inference_rows(
        frame_index=frame_idx,
        detections=detections,
        class_names=getattr(result, "names", None) or classes,
        include_binary_mask=include_binary_mask,
    )


def run_pose_video_inference(
    *,
    model: Any,
    cv2_module: Any,
    video_path: str,
    csv_path: str,
    model_path: str,
    classes: list[str],
    kp_names: list[str],
    device: str,
    batch_size: int,
    total_frames: int,
    fps: float,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_requested: Optional[CancelCallback] = None,
) -> InferenceRunResult:
    """Run synchronous batched pose/detection video inference and stream CSV rows."""
    result = InferenceRunResult(csv_path=csv_path)
    csv_handle = None
    cap = None
    try:
        csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = pose_inference_fieldnames(kp_names)
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        stream = InferenceCsvWriter(writer)

        cap = cv2_module.VideoCapture(video_path)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frames: list[Any] = []
        frame_indices: list[int] = []

        def process_batch() -> bool:
            nonlocal frames, frame_indices
            if not frames:
                return True
            batch_frames = frames
            batch_indices = frame_indices
            frames = []
            frame_indices = []

            try:
                predict_args = {
                    "source": batch_frames,
                    "imgsz": 640,
                    "conf": 0.25,
                    "iou": 0.5,
                    # YOLO26 checkpoints default to end-to-end prediction, which
                    # only applies the confidence threshold and bypasses NMS.
                    # This workflow tracks instances, so overlapping duplicate
                    # predictions must pass through standard NMS.
                    "end2end": False,
                    "device": device,
                    "verbose": False,
                }
                if batch_size > 0:
                    predict_args["batch"] = batch_size
                results_list = list(model.predict(**predict_args))
            except Exception as exc:
                result.had_error = True
                result.error_message = str(exc)
                return False
            if len(results_list) != len(batch_indices):
                result.had_error = True
                result.error_message = (
                    "Prediction returned "
                    f"{len(results_list)} results for {len(batch_indices)} input frames."
                )
                return False

            try:
                for fi, inference_result in zip(batch_indices, results_list):
                    if cancel_requested and cancel_requested():
                        result.canceled = True
                        break
                    for row in pose_inference_rows_from_result(
                        inference_result,
                        frame_index=fi,
                        video_path=video_path,
                        model_path=model_path,
                        fps=fps,
                        kp_names=kp_names,
                        classes=classes,
                    ):
                        stream.write_row(row)
                    result.rows_written = stream.rows_written
                    result.processed_frames += 1
                    if progress_callback:
                        progress_callback(
                            result.processed_frames,
                            total_frames,
                            _progress_message(result.processed_frames, total_frames),
                        )
                    if result.processed_frames % 100 == 0:
                        csv_handle.flush()
            finally:
                del batch_frames
                del batch_indices
                try:
                    del results_list
                except Exception:
                    pass
                gc.collect()

            return not result.canceled and not result.had_error

        frame_idx = 0
        while not result.canceled and not result.had_error:
            if cancel_requested and cancel_requested():
                result.canceled = True
                break

            ok, frame = cap.read()
            if not ok:
                break

            frames.append(frame)
            frame_indices.append(frame_idx)
            frame_idx += 1

            if len(frames) >= max(1, int(batch_size)):
                if not process_batch():
                    break

        if not result.canceled and not result.had_error and frames:
            process_batch()
    except Exception as exc:
        result.had_error = True
        result.error_message = str(exc)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if csv_handle is not None:
            try:
                csv_handle.flush()
            except Exception:
                pass
            try:
                csv_handle.close()
            except Exception:
                pass
    return result


def run_segmentation_video_inference(
    *,
    model: Any,
    video_path: str,
    csv_path: str,
    classes: list[str],
    device: str,
    total_frames: int,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_requested: Optional[CancelCallback] = None,
) -> InferenceRunResult:
    """Run synchronous streaming segmentation inference and stream CSV rows."""
    result = InferenceRunResult(csv_path=csv_path)
    csv_handle = None
    try:
        csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=SEGMENTATION_FIELDNAMES)
        writer.writeheader()

        results_iter = model.predict(
            video_path,
            stream=True,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            # YOLO26 end-to-end output bypasses NMS and can emit multiple,
            # strongly overlapping masks for the same animal. Use the
            # one-to-many head plus standard NMS for instance tracking.
            end2end=False,
            device=device,
            verbose=False,
        )
        for frame_idx, inference_result in enumerate(results_iter):
            if cancel_requested and cancel_requested():
                result.canceled = True
                break

            rows = segmentation_rows_from_result(
                inference_result,
                frame_idx,
                classes=classes,
                include_binary_mask=False,
            )
            if not rows:
                raise RuntimeError(
                    f"Could not serialize segmentation result for frame {frame_idx}."
                )
            for row in rows:
                csv_row = dict(row)
                polygon = csv_row.get("mask_polygon")
                csv_row["mask_polygon"] = json.dumps(polygon) if polygon else ""
                csv_row["binary_mask"] = ""
                writer.writerow(csv_row)
                result.rows_written += 1

            result.processed_frames = frame_idx + 1
            if result.processed_frames % 100 == 0:
                csv_handle.flush()
            if progress_callback:
                progress_callback(
                    result.processed_frames,
                    total_frames,
                    _progress_message(result.processed_frames, total_frames),
                )
    except Exception as exc:
        result.had_error = True
        result.error_message = str(exc)
    finally:
        if csv_handle is not None:
            try:
                csv_handle.flush()
            except Exception:
                pass
            try:
                csv_handle.close()
            except Exception:
                pass
        gc.collect()
    return result


def run_depth_video_inference(
    *,
    model: Any,
    cv2_module: Any,
    numpy_module: Any,
    video_path: str,
    csv_path: str,
    preview_path: str,
    model_path: str,
    device: str,
    total_frames: int,
    fps: float,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_requested: Optional[CancelCallback] = None,
) -> InferenceRunResult:
    """Save per-frame depth summaries plus a colorized preview video."""
    result = InferenceRunResult(csv_path=csv_path, preview_path=preview_path)
    csv_handle = None
    cap = None
    video_writer = None
    try:
        if cv2_module is None or numpy_module is None:
            raise RuntimeError("OpenCV and NumPy are required for depth video inference.")
        csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=DEPTH_FIELDNAMES)
        writer.writeheader()

        cap = cv2_module.VideoCapture(video_path)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frame_index = 0
        while True:
            if cancel_requested and cancel_requested():
                result.canceled = True
                break
            ok, frame = cap.read()
            if not ok:
                break

            predictions = list(
                model.predict(
                    source=frame,
                    imgsz=768,
                    device=device,
                    verbose=False,
                )
            )
            if len(predictions) != 1:
                raise RuntimeError(
                    "Depth prediction returned "
                    f"{len(predictions)} results for one input frame."
                )
            depth_map = depth_array_from_result(
                predictions[0], numpy_module=numpy_module
            )
            summary = depth_map_summary(depth_map, numpy_module=numpy_module)
            preview_rgb = colorize_depth_map(
                depth_map, numpy_module=numpy_module, mode="disparity"
            )

            frame_height, frame_width = frame.shape[:2]
            if preview_rgb.shape[:2] != (frame_height, frame_width):
                raise RuntimeError(
                    "Depth map is not aligned to its video frame: "
                    f"map {preview_rgb.shape[:2]}, "
                    f"frame {(frame_height, frame_width)}."
                )
            if video_writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)
                fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2_module.VideoWriter(
                    preview_path,
                    fourcc,
                    float(fps if fps > 0 else 30.0),
                    (int(frame_width), int(frame_height)),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"Unable to create depth preview video: {preview_path}"
                    )
            video_writer.write(preview_rgb[..., ::-1])

            writer.writerow(
                {
                    "video_path": video_path,
                    "model_path": model_path,
                    "frame_index": frame_index,
                    "time_seconds": (frame_index / fps) if fps > 0 else "",
                    "image_width": int(frame_width),
                    "image_height": int(frame_height),
                    "depth_width": summary["width"],
                    "depth_height": summary["height"],
                    "valid_pixels": summary["valid_pixels"],
                    "min_depth": summary["min_depth"],
                    "max_depth": summary["max_depth"],
                    "median_depth": summary["median_depth"],
                    "p02_depth": summary["p02_depth"],
                    "p98_depth": summary["p98_depth"],
                    "units": "estimated_meters",
                    "scale_status": "model_default",
                }
            )
            result.rows_written += 1
            result.processed_frames += 1
            frame_index += 1
            if result.processed_frames % 100 == 0:
                csv_handle.flush()
            if progress_callback:
                progress_callback(
                    result.processed_frames,
                    total_frames,
                    _progress_message(result.processed_frames, total_frames),
                )
    except Exception as exc:
        result.had_error = True
        result.error_message = str(exc)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        if csv_handle is not None:
            try:
                csv_handle.flush()
            except Exception:
                pass
            try:
                csv_handle.close()
            except Exception:
                pass
        gc.collect()
    return result


def _progress_message(processed_frames: int, total_frames: int) -> str:
    if total_frames > 0:
        return f"Inferencing frame {processed_frames}/{total_frames}"
    return f"Inferencing frame {processed_frames}"
