"""Annotation domain models and graphics."""

from squeakpose.annotation.models import (
    Annotation,
    BoundingBox,
    Keypoint,
    KeypointEntry,
)
from squeakpose.annotation.documents import (
    AnnotationDocument,
    KeypointAnnotationDocument,
    PoseAnnotationDocument,
    SegmentationAnnotationDocument,
)
from squeakpose.annotation.video_view import VideoView
from squeakpose.annotation.graphics import BoxItem, KeypointItem, LabelView

__all__ = [
    "Annotation",
    "AnnotationDocument",
    "KeypointAnnotationDocument",
    "BoundingBox",
    "BoxItem",
    "Keypoint",
    "KeypointEntry",
    "KeypointItem",
    "LabelView",
    "PoseAnnotationDocument",
    "SegmentationAnnotationDocument",
    "VideoView",
]
