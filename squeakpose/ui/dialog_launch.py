"""Qt-free launch plans for feature dialogs composed by the main window."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from squeakpose.project.layers import (
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_worker_mode,
    normalize_layer_id,
)


class DialogUnavailableError(ValueError):
    """Describe a feature dialog that is intentionally unavailable for a layer."""

    def __init__(self, *, title: str, message: str) -> None:
        super().__init__(message)
        self.title = title
        self.message = message


@dataclass(frozen=True)
class TrainingDialogPlan:
    default_dataset: str
    default_task: str
    layer_id: str


@dataclass(frozen=True)
class AnalysisDialogPlan:
    project_root: str
    app_base_dir: str
    layer_id: str


@dataclass(frozen=True)
class VideoReviewDialogPlan:
    layer_id: str
    workflow: str
    model_paths: Mapping[str, str]
    layer_schemas: Mapping[str, Mapping[str, object]]

    @property
    def active_schema(self) -> Mapping[str, object]:
        return self.layer_schemas[self.layer_id]


_DEPTH_UNAVAILABLE = {
    "training": "Depth training is not included in the inference-only MVP.",
    "distillation": "Depth distillation is not included in the inference-only MVP.",
    "analysis": "Depth analysis tools are not included in the MVP yet.",
}


def require_dialog_support(feature: str, layer_id: str) -> str:
    """Return the normalized layer or raise the existing user-facing depth notice."""
    normalized = normalize_layer_id(layer_id)
    message = _DEPTH_UNAVAILABLE.get(str(feature)) if normalized == LAYER_DEPTH else None
    if message is not None:
        raise DialogUnavailableError(title="Depth MVP", message=message)
    return normalized


def plan_training_dialog(
    *,
    project_root: str,
    layer_id: str,
    is_directory: Callable[[str], bool] = os.path.isdir,
) -> TrainingDialogPlan:
    normalized = require_dialog_support("training", layer_id)
    task = "segment" if normalized == LAYER_SEGMENTATION else "pose"
    dataset = os.path.join(project_root, "datasets", task)
    if not is_directory(dataset):
        dataset = os.path.join(project_root, "datasets")
    return TrainingDialogPlan(dataset, task, normalized)


def plan_analysis_dialog(
    *,
    project_root: str,
    app_base_dir: str,
    layer_id: str,
) -> AnalysisDialogPlan:
    normalized = require_dialog_support("analysis", layer_id)
    return AnalysisDialogPlan(project_root, app_base_dir, normalized)


def plan_video_review_dialog(
    *,
    active_layer: str,
    layer_model_paths: Mapping[str, str],
    pose_classes: Sequence[str],
    pose_keypoints: Sequence[str],
    pose_class_keypoints: Mapping[str, Sequence[str]],
    segmentation_classes: Sequence[str],
) -> VideoReviewDialogPlan:
    model_paths = {
        layer_id: str(layer_model_paths.get(layer_id) or "")
        for layer_id in (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
    }
    reviewer_layer = normalize_layer_id(active_layer)
    if reviewer_layer == LAYER_DEPTH:
        reviewer_layer = (
            LAYER_KEYPOINTS
            if model_paths[LAYER_KEYPOINTS] or not model_paths[LAYER_SEGMENTATION]
            else LAYER_SEGMENTATION
        )
    schemas: dict[str, Mapping[str, object]] = {
        LAYER_KEYPOINTS: {
            "classes": list(pose_classes),
            "kp_names": list(pose_keypoints),
            "class_keypoints": {
                name: list(pose_class_keypoints.get(name, ())) for name in pose_classes
            },
        },
        LAYER_SEGMENTATION: {
            "classes": list(segmentation_classes),
            "kp_names": [],
            "class_keypoints": {},
        },
    }
    return VideoReviewDialogPlan(
        layer_id=reviewer_layer,
        workflow=layer_worker_mode(reviewer_layer),
        model_paths=model_paths,
        layer_schemas=schemas,
    )


__all__ = [
    "AnalysisDialogPlan",
    "DialogUnavailableError",
    "TrainingDialogPlan",
    "VideoReviewDialogPlan",
    "plan_analysis_dialog",
    "plan_training_dialog",
    "plan_video_review_dialog",
    "require_dialog_support",
]
