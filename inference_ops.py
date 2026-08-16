"""Compatibility imports for the video inference runtime.

Package code should import :mod:`squeakpose.services.inference_runtime`.
"""

from squeakpose.services.inference_runtime import (
    DEPTH_FIELDNAMES,
    POSE_BASE_FIELDNAMES,
    SEGMENTATION_FIELDNAMES,
    CancelCallback,
    InferenceRunResult,
    ProgressCallback,
    VideoMetadata,
    keypoint_column_key,
    pose_inference_fieldnames,
    pose_inference_rows_from_result,
    probe_video_metadata,
    run_depth_video_inference,
    run_pose_video_inference,
    run_segmentation_video_inference,
    segmentation_rows_from_result,
)

__all__ = [
    "DEPTH_FIELDNAMES",
    "POSE_BASE_FIELDNAMES",
    "SEGMENTATION_FIELDNAMES",
    "CancelCallback",
    "InferenceRunResult",
    "ProgressCallback",
    "VideoMetadata",
    "keypoint_column_key",
    "pose_inference_fieldnames",
    "pose_inference_rows_from_result",
    "probe_video_metadata",
    "run_depth_video_inference",
    "run_pose_video_inference",
    "run_segmentation_video_inference",
    "segmentation_rows_from_result",
]
