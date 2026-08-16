"""Annotation domain models and graphics.

Convenience exports are resolved lazily so Qt-free annotation state can be
imported without initializing the PyQt graphics and video-view modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "DepthAssistantSnapshot": "squeakpose.annotation.depth",
    "DepthAssistantState": "squeakpose.annotation.depth",
    "DepthPredictionTargetPlan": "squeakpose.annotation.depth",
    "DepthProbe": "squeakpose.annotation.depth",
    "DepthRangeSummary": "squeakpose.annotation.depth",
    "AnnotationDocument": "squeakpose.annotation.documents",
    "KeypointAnnotationDocument": "squeakpose.annotation.documents",
    "PoseAnnotationDocument": "squeakpose.annotation.documents",
    "SegmentationAnnotationDocument": "squeakpose.annotation.documents",
    "BoxItem": "squeakpose.annotation.graphics",
    "KeypointItem": "squeakpose.annotation.graphics",
    "LabelView": "squeakpose.annotation.graphics",
    "Annotation": "squeakpose.annotation.models",
    "BoundingBox": "squeakpose.annotation.models",
    "Keypoint": "squeakpose.annotation.models",
    "KeypointEntry": "squeakpose.annotation.models",
    "PoseEditSnapshot": "squeakpose.annotation.pose",
    "PoseEditState": "squeakpose.annotation.pose",
    "SegmentationEditSnapshot": "squeakpose.annotation.segmentation",
    "SegmentationEditState": "squeakpose.annotation.segmentation",
    "load_pose_annotations_from_file": "squeakpose.annotation.serialization",
    "load_segmentation_annotations_from_file": "squeakpose.annotation.serialization",
    "parse_pose_label_line": "squeakpose.annotation.serialization",
    "parse_segmentation_label_line": "squeakpose.annotation.serialization",
    "pose_annotation_to_line": "squeakpose.annotation.serialization",
    "segmentation_annotation_to_line": "squeakpose.annotation.serialization",
    "VideoView": "squeakpose.annotation.video_view",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
