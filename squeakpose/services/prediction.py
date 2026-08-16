"""Qt-free coordination for single-image prediction requests and results."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from squeakpose.core import model_task_mismatch_message
from squeakpose.project.layers import (
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    layer_worker_mode,
    normalize_layer_id,
)


class PredictionValidationError(ValueError):
    """Raised when prediction coordination data violates the layer contract."""


@dataclass(frozen=True, slots=True)
class DepthPredictionTargets:
    final_map: str
    final_preview: str
    final_metadata: str
    staged_map: str
    staged_preview: str
    staged_metadata: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DepthPredictionTargets:
        source = value if isinstance(value, Mapping) else {}
        return cls(
            final_map=str(source.get("final_map") or ""),
            final_preview=str(source.get("final_preview") or ""),
            final_metadata=str(source.get("final_metadata") or ""),
            staged_map=str(source.get("staged_map") or ""),
            staged_preview=str(source.get("staged_preview") or ""),
            staged_metadata=str(source.get("staged_metadata") or ""),
        )

    def worker_paths(self) -> dict[str, str]:
        return {
            "depth_map_path": self.staged_map,
            "depth_preview_path": self.staged_preview,
            "depth_metadata_path": self.staged_metadata,
        }

    def replacements(self) -> tuple[tuple[str, str], ...]:
        replacements = (
            (self.staged_map, self.final_map),
            (self.staged_preview, self.final_preview),
            (self.staged_metadata, self.final_metadata),
        )
        if not all(staged and final for staged, final in replacements):
            raise PredictionValidationError("Depth prediction output transaction is incomplete.")
        return replacements


@dataclass(frozen=True, slots=True)
class PredictionWorkerRequest:
    command: Literal["load", "predict"]
    request_id: Any
    layer_id: str
    model_path: str
    workflow: str
    device: str
    image_path: str = ""
    depth_targets: DepthPredictionTargets | None = None

    def as_worker_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": self.command,
            "request_id": self.request_id,
            "layer_id": self.layer_id,
            "model_path": self.model_path,
            "workflow": self.workflow,
            "device": self.device,
        }
        if self.command == "predict":
            payload["image_path"] = self.image_path
            if self.depth_targets is not None:
                payload.update(self.depth_targets.worker_paths())
        return payload


PredictionEventAction = Literal[
    "ignore",
    "background_error",
    "cancel",
    "error",
    "discard",
    "apply",
]


@dataclass(frozen=True, slots=True)
class PredictionEventDecision:
    action: PredictionEventAction
    request_id: Any = None
    matched: bool = False
    prediction: dict[str, Any] | None = None
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class PredictionKeypointPlan:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PoseDetectionPlan:
    class_id: int
    confidence: float
    x: float
    y: float
    width: float
    height: float
    keypoints: tuple[PredictionKeypointPlan, ...]


@dataclass(frozen=True, slots=True)
class SegmentationDetectionPlan:
    class_id: int
    confidence: float
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class DepthApplicationPlan:
    replacements: tuple[tuple[str, str], ...]
    metadata: dict[str, Any]

    @property
    def median_depth(self) -> float | None:
        raw_value = self.metadata.get("median_depth")
        if raw_value is None:
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None


PredictionApplicationOutcome = Literal[
    "ready",
    "no_detections",
    "no_usable_detections",
    "no_usable_boxes",
    "no_usable_masks",
]


@dataclass(frozen=True, slots=True)
class PredictionApplicationPlan:
    layer_id: str
    outcome: PredictionApplicationOutcome
    detections_seen: int = 0
    selected_classes: tuple[int, ...] = ()
    pose: tuple[PoseDetectionPlan, ...] = ()
    segmentation: tuple[SegmentationDetectionPlan, ...] = ()
    depth: DepthApplicationPlan | None = None
    missing_mask_count: int = 0


def _required_layer_id(value: Any, *, field_name: str = "layer_id") -> str:
    layer_id = normalize_layer_id(value, default="")
    if not layer_id:
        raise PredictionValidationError(f"Unsupported {field_name}: {value!r}")
    return layer_id


def validate_model_task_for_layer(model_task: Any, layer_id: Any) -> str:
    """Validate a model task using the same compatibility rules as workers."""

    normalized_layer = _required_layer_id(layer_id)
    expected_task = layer_definition(normalized_layer).model_task
    mismatch = model_task_mismatch_message(
        model_task,
        expected_task,
        subject="Prediction model",
    )
    if mismatch:
        raise PredictionValidationError(mismatch)
    return expected_task


def validate_prediction_identity(
    prediction: Mapping[str, Any],
    *,
    expected_layer: Any,
) -> str:
    """Ensure declared worker layer/workflow fields match the active layer."""

    if not isinstance(prediction, Mapping):
        raise PredictionValidationError("Prediction payload must be a JSON object.")
    normalized_expected = _required_layer_id(expected_layer, field_name="expected layer")
    for field_name in ("layer_id", "workflow"):
        declared = prediction.get(field_name)
        if declared is None or str(declared).strip() == "":
            continue
        normalized_declared = _required_layer_id(declared, field_name=field_name)
        if normalized_declared != normalized_expected:
            raise PredictionValidationError(
                f"Prediction {field_name} '{declared}' does not match "
                f"the active '{normalized_expected}' layer."
            )
    return normalized_expected


def build_prediction_request(
    *,
    request_id: Any,
    layer_id: Any,
    model_path: str,
    image_path: str,
    device: str = "cpu",
    depth_targets: DepthPredictionTargets | None = None,
) -> PredictionWorkerRequest:
    """Build a validated request matching the existing worker JSON contract."""

    return _build_worker_request(
        command="predict",
        request_id=request_id,
        layer_id=layer_id,
        model_path=model_path,
        image_path=image_path,
        device=device,
        depth_targets=depth_targets,
    )


def build_prediction_load_request(
    *,
    request_id: Any,
    layer_id: Any,
    model_path: str,
    device: str = "cpu",
) -> PredictionWorkerRequest:
    """Build a model warm-up request matching the persistent worker contract."""

    return _build_worker_request(
        command="load",
        request_id=request_id,
        layer_id=layer_id,
        model_path=model_path,
        image_path="",
        device=device,
        depth_targets=None,
    )


def _build_worker_request(
    *,
    command: Literal["load", "predict"],
    request_id: Any,
    layer_id: Any,
    model_path: str,
    image_path: str,
    device: str,
    depth_targets: DepthPredictionTargets | None,
) -> PredictionWorkerRequest:
    normalized_layer = _required_layer_id(layer_id)
    normalized_model_path = str(model_path or "")
    normalized_image_path = str(image_path or "")
    if request_id is None:
        raise PredictionValidationError("request_id is required")
    if not normalized_model_path:
        raise PredictionValidationError("model_path is required")
    if command == "predict" and not normalized_image_path:
        raise PredictionValidationError("image_path is required")
    if command == "predict" and normalized_layer == LAYER_DEPTH:
        if depth_targets is None:
            raise PredictionValidationError("Depth prediction output paths are required.")
        depth_targets.replacements()
    elif depth_targets is not None:
        raise PredictionValidationError("Depth output paths are only valid for the depth layer.")
    return PredictionWorkerRequest(
        command=command,
        request_id=request_id,
        layer_id=normalized_layer,
        model_path=normalized_model_path,
        workflow=layer_worker_mode(normalized_layer),
        device=str(device or "cpu"),
        image_path=normalized_image_path,
        depth_targets=depth_targets,
    )


def correlate_prediction_event(
    event: Mapping[str, Any],
    *,
    current_request_id: Any,
    requested_image_path: str = "",
    displayed_image_path: str = "",
) -> PredictionEventDecision:
    """Reduce a terminal worker event to an explicit UI coordination action."""

    event_type = str(event.get("event") or "")
    request_id = event.get("request_id")
    if event_type == "error":
        error_message = str(event.get("error_message") or "Prediction worker error")
        if request_id is None or request_id == current_request_id:
            return PredictionEventDecision(
                "error",
                request_id=request_id,
                matched=True,
                error_message=error_message,
            )
        return PredictionEventDecision(
            "background_error",
            request_id=request_id,
            error_message=error_message,
        )
    if event_type != "result" or request_id != current_request_id:
        return PredictionEventDecision("ignore", request_id=request_id)
    if bool(event.get("canceled")):
        return PredictionEventDecision("cancel", request_id=request_id, matched=True)
    if bool(event.get("had_error")):
        return PredictionEventDecision(
            "error",
            request_id=request_id,
            matched=True,
            error_message=str(event.get("error_message") or "Unknown prediction error"),
        )
    prediction = event.get("prediction")
    if not isinstance(prediction, Mapping):
        return PredictionEventDecision(
            "error",
            request_id=request_id,
            matched=True,
            error_message="Prediction worker returned no prediction payload.",
        )
    if not requested_image_path or _normalized_path(requested_image_path) != _normalized_path(
        displayed_image_path
    ):
        return PredictionEventDecision("discard", request_id=request_id, matched=True)
    return PredictionEventDecision(
        "apply",
        request_id=request_id,
        matched=True,
        prediction=dict(prediction),
    )


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "")))


def plan_prediction_application(
    prediction: Mapping[str, Any],
    *,
    expected_layer: Any,
    class_names: Sequence[str] = (),
    canonical_keypoints: Sequence[str] = (),
    class_keypoints: Mapping[str, Sequence[str]] | None = None,
    active_class_id: int = 0,
    depth_targets: DepthPredictionTargets | None = None,
) -> PredictionApplicationPlan:
    """Convert a worker payload into immutable pose, mask, or depth actions."""

    layer_id = validate_prediction_identity(prediction, expected_layer=expected_layer)
    if layer_id == LAYER_DEPTH:
        if depth_targets is None:
            raise PredictionValidationError("Depth prediction output transaction is incomplete.")
        raw_metadata = prediction.get("depth_metadata") or {}
        if not isinstance(raw_metadata, Mapping):
            raise PredictionValidationError("Depth prediction metadata must be a JSON object.")
        return PredictionApplicationPlan(
            layer_id=layer_id,
            outcome="ready",
            depth=DepthApplicationPlan(
                replacements=depth_targets.replacements(),
                metadata=dict(raw_metadata),
            ),
        )

    raw_detections = prediction.get("detections") or []
    if not isinstance(raw_detections, list) or not raw_detections:
        return PredictionApplicationPlan(layer_id=layer_id, outcome="no_detections")
    detections = [detection for detection in raw_detections if isinstance(detection, Mapping)]
    if not detections:
        return PredictionApplicationPlan(
            layer_id=layer_id,
            outcome="no_usable_detections",
            detections_seen=len(raw_detections),
        )

    classes = tuple(str(name) for name in class_names)
    if not classes:
        raise PredictionValidationError(f"The {layer_id} layer requires at least one class.")
    if active_class_id < 0 or active_class_id >= len(classes):
        raise PredictionValidationError("active_class_id is outside the class schema")
    selected, confidences = _best_detections_by_class(
        detections,
        class_count=len(classes),
        active_class_id=active_class_id,
    )
    selected_classes = tuple(selected)
    if layer_id == LAYER_SEGMENTATION:
        masks: list[SegmentationDetectionPlan] = []
        missing_masks = 0
        for class_id, detection_index in selected.items():
            points = _segmentation_points(detections[detection_index].get("segments"))
            if len(points) < 3:
                missing_masks += 1
                continue
            masks.append(
                SegmentationDetectionPlan(
                    class_id=class_id,
                    confidence=confidences[detection_index],
                    points=tuple(points),
                )
            )
        return PredictionApplicationPlan(
            layer_id=layer_id,
            outcome="ready" if masks else "no_usable_masks",
            detections_seen=len(detections),
            selected_classes=selected_classes,
            segmentation=tuple(masks),
            missing_mask_count=missing_masks,
        )

    if layer_id != LAYER_KEYPOINTS:
        raise PredictionValidationError(f"Unsupported prediction layer: {layer_id}")
    keypoint_names = tuple(str(name) for name in canonical_keypoints)
    class_keypoint_map = class_keypoints or {}
    boxes: list[PoseDetectionPlan] = []
    for class_id, detection_index in selected.items():
        detection = detections[detection_index]
        box = _pose_box(detection.get("xyxy"))
        if box is None:
            continue
        allowed_names = set(class_keypoint_map.get(classes[class_id], keypoint_names))
        keypoints: list[PredictionKeypointPlan] = []
        raw_keypoints = detection.get("keypoints") or []
        if isinstance(raw_keypoints, Sequence) and not isinstance(raw_keypoints, (str, bytes)):
            for index, raw_keypoint in enumerate(raw_keypoints):
                if index >= len(keypoint_names):
                    break
                parsed = _prediction_keypoint(raw_keypoint, keypoint_names[index])
                if parsed is not None and parsed.name in allowed_names:
                    keypoints.append(parsed)
        boxes.append(
            PoseDetectionPlan(
                class_id=class_id,
                confidence=confidences[detection_index],
                x=box[0],
                y=box[1],
                width=box[2],
                height=box[3],
                keypoints=tuple(keypoints),
            )
        )
    return PredictionApplicationPlan(
        layer_id=layer_id,
        outcome="ready" if boxes else "no_usable_boxes",
        detections_seen=len(detections),
        selected_classes=selected_classes,
        pose=tuple(boxes),
    )


def _best_detections_by_class(
    detections: Sequence[Mapping[str, Any]],
    *,
    class_count: int,
    active_class_id: int,
) -> tuple[dict[int, int], list[float]]:
    best_by_class: dict[int, int] = {}
    confidences: list[float] = []
    for detection_index, detection in enumerate(detections):
        confidence = _float_or_default(detection.get("confidence"), 0.0)
        confidences.append(confidence)
        try:
            class_id = int(detection.get("class_id", active_class_id))
        except (TypeError, ValueError):
            class_id = active_class_id
        if class_id < 0 or class_id >= class_count:
            continue
        previous_index = best_by_class.get(class_id)
        if previous_index is None or confidence >= confidences[previous_index]:
            best_by_class[class_id] = detection_index
    if not best_by_class:
        best_index = max(range(len(detections)), key=lambda index: confidences[index])
        best_by_class[active_class_id] = best_index
    return best_by_class, confidences


def _pose_box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return x1, y1, width, height


def _prediction_keypoint(value: Any, name: str) -> PredictionKeypointPlan | None:
    try:
        if len(value) < 3:
            return None
        x, y, confidence = float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError):
        return None
    return PredictionKeypointPlan(name=name, x=x, y=y, confidence=confidence)


def _segmentation_points(value: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    try:
        pairs = value or []
    except Exception:
        return points
    for pair in pairs:
        try:
            if len(pair) < 2:
                continue
            points.append((float(pair[0]), float(pair[1])))
        except (TypeError, ValueError):
            continue
    return points


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default
