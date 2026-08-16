"""Compatibility exports for annotation serialization.

New code should import from :mod:`squeakpose.annotation.serialization`.
"""

from squeakpose.annotation.serialization import (
    load_pose_annotations_from_file,
    load_segmentation_annotations_from_file,
    parse_pose_label_line,
    parse_segmentation_label_line,
    pose_annotation_to_line,
    segmentation_annotation_to_line,
)

__all__ = [
    "load_pose_annotations_from_file",
    "load_segmentation_annotations_from_file",
    "parse_pose_label_line",
    "parse_segmentation_label_line",
    "pose_annotation_to_line",
    "segmentation_annotation_to_line",
]
