"""Qt-free project layer definitions shared across SqueakPose workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LAYER_KEYPOINTS = "keypoints"
LAYER_SEGMENTATION = "segmentation"
LAYER_DEPTH = "depth"


@dataclass(frozen=True, slots=True)
class LayerDefinition:
    """Stable metadata describing one editable project layer."""

    id: str
    display_name: str
    annotation_name: str
    model_task: str
    worker_mode: str
    label_directory: str
    class_file: str
    keypoint_file: str
    dataset_task: str
    dataset_directory: str
    inference_suffix: str
    editable_annotations: bool = True
    uses_classes: bool = True
    supports_training: bool = True
    dense_output: bool = False

    @property
    def supports_keypoints(self) -> bool:
        return self.id == LAYER_KEYPOINTS

    @property
    def supports_masks(self) -> bool:
        return self.id == LAYER_SEGMENTATION

    @property
    def supports_depth(self) -> bool:
        return self.id == LAYER_DEPTH


LAYER_DEFINITIONS: dict[str, LayerDefinition] = {
    LAYER_KEYPOINTS: LayerDefinition(
        id=LAYER_KEYPOINTS,
        display_name="Keypoints",
        annotation_name="bounding boxes and keypoints",
        model_task="pose",
        worker_mode="pose",
        label_directory="labels_all",
        class_file="classes.txt",
        keypoint_file="keypoints.txt",
        dataset_task="pose",
        dataset_directory="pose",
        inference_suffix="_keypoints.csv",
    ),
    LAYER_SEGMENTATION: LayerDefinition(
        id=LAYER_SEGMENTATION,
        display_name="Segmentation",
        annotation_name="segmentation masks",
        model_task="segment",
        worker_mode="segmentation",
        label_directory="labels_seg_all",
        class_file="classes_seg.txt",
        keypoint_file="",
        dataset_task="segment",
        dataset_directory="segment",
        inference_suffix="_segmentation.csv",
    ),
    LAYER_DEPTH: LayerDefinition(
        id=LAYER_DEPTH,
        display_name="Depth",
        annotation_name="dense depth maps",
        model_task="depth",
        worker_mode="depth",
        label_directory="",
        class_file="",
        keypoint_file="",
        dataset_task="depth",
        dataset_directory="depth",
        inference_suffix="_depth.csv",
        editable_annotations=False,
        uses_classes=False,
        supports_training=False,
        dense_output=True,
    ),
}

_LAYER_ALIASES = {
    LAYER_KEYPOINTS: LAYER_KEYPOINTS,
    "keypoint": LAYER_KEYPOINTS,
    "pose": LAYER_KEYPOINTS,
    "poses": LAYER_KEYPOINTS,
    LAYER_SEGMENTATION: LAYER_SEGMENTATION,
    "segment": LAYER_SEGMENTATION,
    "seg": LAYER_SEGMENTATION,
    "mask": LAYER_SEGMENTATION,
    "masks": LAYER_SEGMENTATION,
    LAYER_DEPTH: LAYER_DEPTH,
    "depths": LAYER_DEPTH,
    "monocular-depth": LAYER_DEPTH,
}


def normalize_layer_id(value: Any, *, default: str = LAYER_KEYPOINTS) -> str:
    """Return a stable layer id, accepting legacy workflow/task names."""

    normalized = _LAYER_ALIASES.get(str(value or "").strip().lower())
    return normalized if normalized is not None else default


def layer_definition(value: Any) -> LayerDefinition:
    """Return the definition for a layer id or legacy workflow name."""

    return LAYER_DEFINITIONS[normalize_layer_id(value)]


def layer_worker_mode(value: Any) -> str:
    return layer_definition(value).worker_mode


def layer_model_task(value: Any) -> str:
    return layer_definition(value).model_task


def normalize_layer_settings(value: Any) -> dict[str, dict[str, Any]]:
    """Normalize persisted per-layer settings without discarding unknown keys."""

    raw_layers = value if isinstance(value, Mapping) else {}
    settings: dict[str, dict[str, Any]] = {}
    for layer_id in LAYER_DEFINITIONS:
        raw = raw_layers.get(layer_id)
        settings[layer_id] = dict(raw) if isinstance(raw, Mapping) else {}
    for raw_id, raw in raw_layers.items():
        normalized_id = normalize_layer_id(raw_id, default="")
        if normalized_id not in LAYER_DEFINITIONS or not isinstance(raw, Mapping):
            continue
        if normalized_id != raw_id:
            settings[normalized_id].update(dict(raw))
    return settings


def layer_model_paths(
    value: Any,
    *,
    resolve_path=lambda path: str(path or ""),
) -> dict[str, str]:
    """Extract model paths from persisted layer settings."""

    settings = normalize_layer_settings(value)
    return {
        layer_id: resolve_path(str(settings[layer_id].get("model_path") or ""))
        for layer_id in LAYER_DEFINITIONS
    }


__all__ = [
    "LayerDefinition",
    "LAYER_DEFINITIONS",
    "LAYER_DEPTH",
    "LAYER_KEYPOINTS",
    "LAYER_SEGMENTATION",
    "layer_definition",
    "layer_model_paths",
    "layer_model_task",
    "layer_worker_mode",
    "normalize_layer_id",
    "normalize_layer_settings",
]
