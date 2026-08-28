"""Unified pose + segmentation analysis built around one authoritative frame table."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_ops import (
    AnalysisConfig,
    AnalysisError,
    ProgressCallback,
    _infer_fps,
    _mm_per_pixel,
    _open_h264_video_writer,
    _progress,
    _read_video_metadata,
    _setup_plotting,
    _smooth_centers,
    assign_roi_labels,
    create_plots,
    draw_antialiased_polyline,
    draw_supersampled_polygon_overlay,
    export_cluster_clips,
    normalize_rois,
    prepare_analysis_output_dir,
    run_behavior_clustering,
)
from segmentation_analysis_ops import _parse_polygon, compute_segmentation_detection_features

POSE_SKELETON = (
    ("nose", "head"),
    ("head", "left_ear"),
    ("head", "right_ear"),
    ("head", "back"),
    ("back", "tail_base"),
)

PREDICTION_QC_LOW_CONFIDENCE = 0.35
PREDICTION_QC_MIN_LAYER_IOU = 0.10
PREDICTION_QC_MAX_JUMP_BOX_DIAGONALS = 3.0
PREDICTION_QC_MAX_JUMP_IMAGE_FRACTION = 0.15


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_pose_csv(path: str) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    keep = [
        column
        for column in columns
        if column
        not in {
            "model_path",
            "video_path",
            "bbox_center_x_norm",
            "bbox_center_y_norm",
            "bbox_width_norm",
            "bbox_height_norm",
        }
        and not column.endswith(("_x_norm", "_y_norm"))
    ]
    return pd.read_csv(path, usecols=keep).dropna(axis=1, how="all")


def _read_segmentation_csv(path: str) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    # binary_mask is a large redundant raster serialization. The polygon is
    # sufficient to reproduce geometry, overlays, and every unified feature.
    keep = [column for column in columns if column not in {"binary_mask", "model_path"}]
    return pd.read_csv(path, usecols=keep).dropna(axis=1, how="all")


TRACK_ID_COLUMNS = ("track_id", "tracker_id", "tracking_id")
EXPECTED_ANIMAL_COLUMNS = ("expected_animal_count", "expected_animals", "animal_count")


def _normalized_track_ids(frame: pd.DataFrame) -> pd.Series:
    """Return stable string IDs while treating legacy blanks and sentinels as missing."""
    source = next((column for column in TRACK_ID_COLUMNS if column in frame.columns), None)
    if source is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    values = frame[source].astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    integer = numeric.notna() & np.isclose(numeric, np.round(numeric))
    values.loc[integer] = numeric.loc[integer].round().astype("Int64").astype("string")
    invalid = values.isna() | values.str.lower().isin({"", "nan", "none", "null", "-1"})
    return values.mask(invalid, pd.NA)


def _expected_animal_count(*frames: pd.DataFrame) -> int:
    values: list[int] = []
    for frame in frames:
        for column in EXPECTED_ANIMAL_COLUMNS:
            if column not in frame.columns:
                continue
            parsed = _numeric(frame[column]).dropna()
            values.extend(int(value) for value in parsed if value >= 1)
    return max(values, default=1)


def _pose_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"frame_index", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise AnalysisError(f"Pose CSV is missing required columns: {', '.join(missing)}")
    pose = raw.copy()
    pose["pose_track_id"] = _normalized_track_ids(pose)
    pose["frame_index"] = _numeric(pose["frame_index"])
    pose = pose.dropna(subset=["frame_index"])
    if "detection_index" in pose.columns:
        pose = pose.loc[_numeric(pose["detection_index"]).fillna(-1).ge(0)]
    pose["frame_index"] = pose["frame_index"].astype(int)

    keep = ["frame_index", "pose_track_id"]
    simple = {
        "confidence": "pose_confidence",
        "detection_index": "pose_detection_index",
        "detections_in_frame": "pose_detections_in_frame",
        "tracks_in_frame": "pose_tracks_in_frame",
        "tracker_type": "pose_tracker_type",
        "tracker_profile": "pose_tracker_profile",
        "class_id": "pose_class_id",
        "class_name": "pose_class_name",
        "bbox_x1": "pose_bbox_x1",
        "bbox_y1": "pose_bbox_y1",
        "bbox_x2": "pose_bbox_x2",
        "bbox_y2": "pose_bbox_y2",
        "bbox_center_x": "pose_bbox_center_x",
        "bbox_center_y": "pose_bbox_center_y",
        "image_width": "image_width",
        "image_height": "image_height",
        "speed_preprocess_ms": "pose_preprocess_ms",
        "speed_inference_ms": "pose_inference_ms",
        "speed_postprocess_ms": "pose_postprocess_ms",
    }
    rename: dict[str, str] = {}
    for source, destination in simple.items():
        if source in pose.columns:
            keep.append(source)
            rename[source] = destination
    # Pixel coordinates and confidence are authoritative for analysis. Normalized
    # duplicates are intentionally omitted to keep the single table tractable.
    for column in pose.columns:
        if column.startswith("kp_") and column.endswith(("_x", "_y", "_conf")):
            if column not in keep:
                keep.append(column)
    return pose[keep].rename(columns=rename).reset_index(drop=True)


def _pose_primary(raw: pd.DataFrame) -> pd.DataFrame:
    pose = _pose_candidates(raw)
    sort_columns = ["frame_index"]
    ascending = [True]
    if "pose_confidence" in pose.columns:
        sort_columns.append("pose_confidence")
        ascending.append(False)
    return (
        pose.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates("frame_index", keep="first")
        .reset_index(drop=True)
    )


def _segmentation_candidates(raw: pd.DataFrame, scale: float) -> pd.DataFrame:
    detections = compute_segmentation_detection_features(raw, scale)
    detections["frame_index"] = _numeric(detections["frame_index"]).astype(int)
    # Geometry extraction deliberately has a compact schema. Reattach tracker
    # metadata from the source by the stable frame/detection key.
    raw_keys = raw.copy()
    raw_keys["frame_index"] = _numeric(
        raw_keys.get("frame_index", raw_keys.get("frame", pd.Series(index=raw_keys.index)))
    )
    raw_keys["detection_index"] = _numeric(
        raw_keys.get("detection_index", raw_keys.get("det", pd.Series(index=raw_keys.index)))
    )
    raw_keys["segmentation_track_id"] = _normalized_track_ids(raw_keys)
    tracker_columns = ["frame_index", "detection_index", "segmentation_track_id"]
    for column in (
        "tracks_in_frame",
        "tracker_type",
        "tracker_profile",
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_postprocess_ms",
    ):
        if column in raw_keys.columns:
            tracker_columns.append(column)
    raw_keys = raw_keys[tracker_columns].dropna(subset=["frame_index", "detection_index"])
    raw_keys = raw_keys.drop_duplicates(["frame_index", "detection_index"], keep="first")
    detections = detections.merge(
        raw_keys,
        on=["frame_index", "detection_index"],
        how="left",
        validate="one_to_one",
    )
    # ``compute_segmentation_detection_features`` makes bbox_center_* the mask
    # centroid for tracking. Preserve the geometric box center separately so
    # the documented centroid fallback remains literal.
    detections["segmentation_bbox_center_x"] = (
        _numeric(detections["bbox_x1"]) + _numeric(detections["bbox_x2"])
    ) / 2.0
    detections["segmentation_bbox_center_y"] = (
        _numeric(detections["bbox_y1"]) + _numeric(detections["bbox_y2"])
    ) / 2.0
    rename = {
        "confidence": "segmentation_confidence",
        "detection_index": "segmentation_detection_index",
        "detections_in_frame": "segmentation_detections_in_frame",
        "tracks_in_frame": "segmentation_tracks_in_frame",
        "tracker_type": "segmentation_tracker_type",
        "tracker_profile": "segmentation_tracker_profile",
        "speed_preprocess_ms": "segmentation_preprocess_ms",
        "speed_inference_ms": "segmentation_inference_ms",
        "speed_postprocess_ms": "segmentation_postprocess_ms",
        "class_id": "segmentation_class_id",
        "class_name": "segmentation_class_name",
        "bbox_x1": "segmentation_bbox_x1",
        "bbox_y1": "segmentation_bbox_y1",
        "bbox_x2": "segmentation_bbox_x2",
        "bbox_y2": "segmentation_bbox_y2",
        "bbox_source": "segmentation_bbox_source",
        "inference_bbox_x1": "segmentation_inference_bbox_x1",
        "inference_bbox_y1": "segmentation_inference_bbox_y1",
        "inference_bbox_x2": "segmentation_inference_bbox_x2",
        "inference_bbox_y2": "segmentation_inference_bbox_y2",
        "mask_bbox_x1": "segmentation_mask_bbox_x1",
        "mask_bbox_y1": "segmentation_mask_bbox_y1",
        "mask_bbox_x2": "segmentation_mask_bbox_x2",
        "mask_bbox_y2": "segmentation_mask_bbox_y2",
    }
    return detections.rename(columns=rename).reset_index(drop=True)


def _segmentation_primary(raw: pd.DataFrame, scale: float) -> pd.DataFrame:
    detections = _segmentation_candidates(raw, scale)
    return (
        detections.sort_values(
            ["frame_index", "segmentation_confidence", "mask_area_px2"],
            ascending=[True, False, False],
        )
        .drop_duplicates("frame_index", keep="first")
        .reset_index(drop=True)
    )


def _bbox_iou(left: pd.Series, right: pd.Series, left_prefix: str, right_prefix: str) -> float:
    try:
        lx1, ly1, lx2, ly2 = (
            float(left[f"{left_prefix}_{suffix}"])
            for suffix in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        )
        rx1, ry1, rx2, ry2 = (
            float(right[f"{right_prefix}_{suffix}"])
            for suffix in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        )
    except (KeyError, TypeError, ValueError):
        return math.nan
    if not np.isfinite([lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2]).all():
        return math.nan
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - intersection
    )
    return intersection / union if union > 0 else math.nan


def _track_stats(
    frame: pd.DataFrame, track_column: str, confidence_column: str
) -> list[dict[str, Any]]:
    tracked = frame.dropna(subset=[track_column])
    rows: list[dict[str, Any]] = []
    for track_id, group in tracked.groupby(track_column, sort=False):
        rows.append(
            {
                "track_id": str(track_id),
                "frames": int(group["frame_index"].nunique()),
                "first_frame": int(group["frame_index"].min()),
                "last_frame": int(group["frame_index"].max()),
                "mean_confidence": float(
                    _numeric(
                        group.get(confidence_column, pd.Series(index=group.index, dtype=float))
                    ).mean()
                ),
            }
        )
    return sorted(rows, key=lambda row: (-row["frames"], row["first_frame"], row["track_id"]))


def reconcile_layer_tracks(
    pose: pd.DataFrame,
    segmentation: pd.DataFrame,
    expected_animals: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match independent layer track IDs using temporal overlap and box IoU."""
    pose_stats = _track_stats(pose, "pose_track_id", "pose_confidence")
    segmentation_stats = _track_stats(
        segmentation, "segmentation_track_id", "segmentation_confidence"
    )
    pair_scores: list[dict[str, Any]] = []
    for seg in segmentation_stats:
        seg_rows = segmentation.loc[segmentation["segmentation_track_id"].eq(seg["track_id"])]
        for pose_track in pose_stats:
            pose_rows = pose.loc[pose["pose_track_id"].eq(pose_track["track_id"])]
            overlap = seg_rows.merge(pose_rows, on="frame_index", how="inner")
            ious = [_bbox_iou(row, row, "segmentation", "pose") for _, row in overlap.iterrows()]
            valid_ious = [value for value in ious if np.isfinite(value)]
            overlap_frames = int(overlap["frame_index"].nunique())
            mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
            union_frames = seg["frames"] + pose_track["frames"] - overlap_frames
            temporal_iou = overlap_frames / union_frames if union_frames else 0.0
            pair_scores.append(
                {
                    "segmentation_track_id": seg["track_id"],
                    "pose_track_id": pose_track["track_id"],
                    "overlap_frames": overlap_frames,
                    "mean_bbox_iou": mean_iou,
                    "temporal_iou": temporal_iou,
                    "association_score": mean_iou * math.sqrt(max(overlap_frames, 0)),
                }
            )

    selected_seg_ids = {row["track_id"] for row in segmentation_stats[:expected_animals]}
    matched_seg: set[str] = set()
    matched_pose: set[str] = set()
    matches: dict[str, dict[str, Any]] = {}
    for pair in sorted(
        pair_scores,
        key=lambda row: (-row["association_score"], -row["overlap_frames"]),
    ):
        seg_id = pair["segmentation_track_id"]
        pose_id = pair["pose_track_id"]
        if (
            seg_id not in selected_seg_ids
            or pair["overlap_frames"] <= 0
            or pair["mean_bbox_iou"] <= 0
            or seg_id in matched_seg
            or pose_id in matched_pose
        ):
            continue
        matched_seg.add(seg_id)
        matched_pose.add(pose_id)
        matches[seg_id] = pair

    identities: list[dict[str, Any]] = []
    for seg in segmentation_stats[:expected_animals]:
        pair = matches.get(seg["track_id"], {})
        identities.append(
            {
                "segmentation_track_id": seg["track_id"],
                "pose_track_id": pair.get("pose_track_id"),
                "mean_bbox_iou": pair.get("mean_bbox_iou"),
                "overlap_frames": pair.get("overlap_frames", 0),
                "association_score": pair.get("association_score"),
            }
        )
    for pose_track in pose_stats:
        if len(identities) >= expected_animals:
            break
        if pose_track["track_id"] not in matched_pose:
            identities.append(
                {
                    "segmentation_track_id": None,
                    "pose_track_id": pose_track["track_id"],
                    "mean_bbox_iou": None,
                    "overlap_frames": 0,
                    "association_score": None,
                }
            )
    while len(identities) < expected_animals:
        identities.append(
            {
                "segmentation_track_id": None,
                "pose_track_id": None,
                "mean_bbox_iou": None,
                "overlap_frames": 0,
                "association_score": None,
            }
        )
    for index, identity in enumerate(identities, start=1):
        identity["animal_id"] = f"animal_{index}"
    diagnostics = {
        "expected_animal_count": int(expected_animals),
        "pose_tracks": pose_stats,
        "segmentation_tracks": segmentation_stats,
        "cross_layer_pairs": pair_scores,
        "animal_track_mapping": identities,
        "unmatched_pose_track_ids": [
            row["track_id"] for row in pose_stats if row["track_id"] not in matched_pose
        ],
        "unmatched_segmentation_track_ids": [
            row["track_id"] for row in segmentation_stats if row["track_id"] not in matched_seg
        ],
    }
    return identities, diagnostics


def _frame_count(
    pose_raw: pd.DataFrame,
    segmentation_raw: pd.DataFrame,
    video_path: str,
) -> int:
    candidates: list[int] = []
    metadata_frames = int(_read_video_metadata(video_path).get("frames", 0) or 0)
    if metadata_frames > 0:
        candidates.append(metadata_frames)
    for frame, columns in (
        (pose_raw, ("frame_index", "frame")),
        (segmentation_raw, ("frame_index", "frame")),
    ):
        column = next((name for name in columns if name in frame.columns), None)
        if column is None:
            continue
        values = _numeric(frame[column]).dropna()
        if not values.empty:
            candidates.append(int(values.max()) + 1)
    return max(candidates, default=0)


def _first_complete(
    table: pd.DataFrame,
    candidates: list[tuple[list[str], str]],
) -> tuple[list[pd.Series], pd.Series]:
    width = len(candidates[0][0])
    values = [pd.Series(np.nan, index=table.index, dtype=float) for _ in range(width)]
    source = pd.Series("missing", index=table.index, dtype=object)
    for columns, name in candidates:
        if len(columns) != width or any(column not in table.columns for column in columns):
            continue
        source_values = [_numeric(table[column]) for column in columns]
        complete = source.eq("missing")
        for candidate in source_values:
            complete &= candidate.notna()
        for destination, candidate in zip(values, source_values):
            destination.loc[complete] = candidate.loc[complete]
        source.loc[complete] = name
    return values, source


def _add_prediction_qc(table: pd.DataFrame) -> pd.DataFrame:
    """Add conservative, non-destructive prediction quality diagnostics."""
    expected = (
        _numeric(table.get("expected_animal_count", pd.Series(1, index=table.index)))
        .fillna(1)
        .clip(lower=1)
    )
    pose_count = _numeric(
        table.get("pose_detections_in_frame", table["pose_valid"].astype(int))
    ).fillna(0)
    segmentation_count = _numeric(
        table.get("segmentation_detections_in_frame", table["segmentation_valid"].astype(int))
    ).fillna(0)
    table["extra_pose_detections"] = (pose_count - expected).clip(lower=0).astype(int)
    table["extra_segmentation_detections"] = (
        (segmentation_count - expected).clip(lower=0).astype(int)
    )
    pose_track_count = _numeric(
        table.get("pose_tracks_in_frame", table["pose_valid"].astype(int))
    ).fillna(0)
    segmentation_track_count = _numeric(
        table.get("segmentation_tracks_in_frame", table["segmentation_valid"].astype(int))
    ).fillna(0)
    table["extra_pose_tracks"] = (pose_track_count - expected).clip(lower=0).astype(int)
    table["extra_segmentation_tracks"] = (
        (segmentation_track_count - expected).clip(lower=0).astype(int)
    )

    confidence = pd.concat(
        [
            _numeric(table.get("pose_confidence", pd.Series(np.nan, index=table.index))),
            _numeric(table.get("segmentation_confidence", pd.Series(np.nan, index=table.index))),
        ],
        axis=1,
    )
    # Ultralytics track results expose detector confidence, not a separate
    # tracker confidence. Use the conservative cross-layer minimum and name it
    # accordingly so downstream users do not confuse the two concepts.
    table["primary_detection_confidence"] = confidence.min(axis=1, skipna=True)

    pose_x1 = _numeric(table.get("pose_bbox_x1", pd.Series(np.nan, index=table.index)))
    pose_y1 = _numeric(table.get("pose_bbox_y1", pd.Series(np.nan, index=table.index)))
    pose_x2 = _numeric(table.get("pose_bbox_x2", pd.Series(np.nan, index=table.index)))
    pose_y2 = _numeric(table.get("pose_bbox_y2", pd.Series(np.nan, index=table.index)))
    seg_x1 = _numeric(table.get("segmentation_bbox_x1", pd.Series(np.nan, index=table.index)))
    seg_y1 = _numeric(table.get("segmentation_bbox_y1", pd.Series(np.nan, index=table.index)))
    seg_x2 = _numeric(table.get("segmentation_bbox_x2", pd.Series(np.nan, index=table.index)))
    seg_y2 = _numeric(table.get("segmentation_bbox_y2", pd.Series(np.nan, index=table.index)))
    intersection = (np.minimum(pose_x2, seg_x2) - np.maximum(pose_x1, seg_x1)).clip(lower=0) * (
        np.minimum(pose_y2, seg_y2) - np.maximum(pose_y1, seg_y1)
    ).clip(lower=0)
    pose_area = (pose_x2 - pose_x1).clip(lower=0) * (pose_y2 - pose_y1).clip(lower=0)
    segmentation_area = (seg_x2 - seg_x1).clip(lower=0) * (seg_y2 - seg_y1).clip(lower=0)
    union = pose_area + segmentation_area - intersection
    table["pose_segmentation_bbox_iou"] = (intersection / union.replace(0, np.nan)).where(
        table["pose_valid"] & table["segmentation_valid"]
    )

    previous_x = table.groupby("animal_id", sort=False)["centroid_x"].shift()
    previous_y = table.groupby("animal_id", sort=False)["centroid_y"].shift()
    table["centroid_jump_px"] = np.hypot(
        table["centroid_x"] - previous_x, table["centroid_y"] - previous_y
    )
    bbox_diagonal = np.hypot(table["bbox_width"], table["bbox_height"])
    image_diagonal = np.hypot(
        _numeric(table.get("image_width", pd.Series(np.nan, index=table.index))),
        _numeric(table.get("image_height", pd.Series(np.nan, index=table.index))),
    )
    table["centroid_jump_threshold_px"] = pd.concat(
        [
            pd.Series(bbox_diagonal * PREDICTION_QC_MAX_JUMP_BOX_DIAGONALS),
            pd.Series(image_diagonal * PREDICTION_QC_MAX_JUMP_IMAGE_FRACTION),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    for layer in ("pose", "segmentation"):
        id_column = f"{layer}_track_id"
        current = table.get(id_column, pd.Series(pd.NA, index=table.index, dtype="string"))
        previous = current.groupby(table["animal_id"], sort=False).shift()
        table[f"{layer}_track_id_changed"] = (
            current.notna()
            & previous.notna()
            & current.astype("string").ne(previous.astype("string"))
        )

    reasons: list[str] = []
    statuses: list[str] = []
    for index in table.index:
        row_reasons: list[str] = []
        pose_valid = bool(table.at[index, "pose_valid"])
        segmentation_valid = bool(table.at[index, "segmentation_valid"])
        if not pose_valid and not segmentation_valid:
            row_reasons.append("missing_primary_detection")
            status = "bad"
        else:
            status = "good"
            if not pose_valid:
                row_reasons.append("missing_pose_detection")
            if not segmentation_valid:
                row_reasons.append("missing_segmentation_detection")
            if table.at[index, "extra_pose_detections"] > 0:
                row_reasons.append("extra_pose_detection")
            if table.at[index, "extra_segmentation_detections"] > 0:
                row_reasons.append("extra_segmentation_detection")
            if table.at[index, "extra_pose_tracks"] > 0:
                row_reasons.append("extra_pose_track")
            if table.at[index, "extra_segmentation_tracks"] > 0:
                row_reasons.append("extra_segmentation_track")
            confidence_value = table.at[index, "primary_detection_confidence"]
            if pd.notna(confidence_value) and confidence_value < PREDICTION_QC_LOW_CONFIDENCE:
                row_reasons.append("low_primary_detection_confidence")
            layer_iou = table.at[index, "pose_segmentation_bbox_iou"]
            if pd.notna(layer_iou) and layer_iou < PREDICTION_QC_MIN_LAYER_IOU:
                row_reasons.append("low_pose_segmentation_agreement")
            jump = table.at[index, "centroid_jump_px"]
            jump_limit = table.at[index, "centroid_jump_threshold_px"]
            if pd.notna(jump) and pd.notna(jump_limit) and jump > jump_limit:
                row_reasons.append("implausible_centroid_jump")
            if bool(table.at[index, "pose_track_id_changed"]):
                row_reasons.append("pose_track_id_changed")
            if bool(table.at[index, "segmentation_track_id_changed"]):
                row_reasons.append("segmentation_track_id_changed")
            if row_reasons:
                status = "warning"
        reasons.append(";".join(row_reasons))
        statuses.append(status)
    table["prediction_qc_status"] = statuses
    table["prediction_qc_reasons"] = reasons
    return table


def build_unified_frame_table(
    pose_raw: pd.DataFrame,
    segmentation_raw: pd.DataFrame,
    config: AnalysisConfig,
    *,
    video_path: str,
    fps: float,
    scale: float,
) -> pd.DataFrame:
    """Construct an identity-aware frame table from pose and mask detections."""
    pose_candidates = _pose_candidates(pose_raw)
    segmentation_candidates = _segmentation_candidates(segmentation_raw, scale)
    count = _frame_count(pose_raw, segmentation_raw, video_path)
    if count <= 0:
        raise AnalysisError("Neither inference CSV contains a usable video frame index.")
    expected_animals = _expected_animal_count(pose_raw, segmentation_raw)
    pose_tracked = pose_candidates["pose_track_id"].notna().any()
    segmentation_tracked = segmentation_candidates["segmentation_track_id"].notna().any()
    tracking_status = "tracked" if pose_tracked or segmentation_tracked else "legacy_untracked"

    if tracking_status == "legacy_untracked":
        identities = [
            {
                "animal_id": "animal_1",
                "segmentation_track_id": None,
                "pose_track_id": None,
                "mean_bbox_iou": None,
                "overlap_frames": 0,
                "association_score": None,
            }
        ]
        diagnostics: dict[str, Any] = {
            "expected_animal_count": 1,
            "tracking_status": tracking_status,
            "pose_tracks": [],
            "segmentation_tracks": [],
            "cross_layer_pairs": [],
            "animal_track_mapping": identities,
            "unmatched_pose_track_ids": [],
            "unmatched_segmentation_track_ids": [],
        }
        pose_layers = [_pose_primary(pose_raw)]
        segmentation_layers = [_segmentation_primary(segmentation_raw, scale)]
    else:
        identities, diagnostics = reconcile_layer_tracks(
            pose_candidates, segmentation_candidates, expected_animals
        )
        diagnostics["tracking_status"] = tracking_status
        pose_layers = []
        segmentation_layers = []
        for identity in identities:
            pose_id = identity["pose_track_id"]
            segmentation_id = identity["segmentation_track_id"]
            pose_layer = pose_candidates.loc[pose_candidates["pose_track_id"].eq(pose_id)].copy()
            segmentation_layer = segmentation_candidates.loc[
                segmentation_candidates["segmentation_track_id"].eq(segmentation_id)
            ].copy()
            # A one-animal recording can safely recover tracker fragments and
            # blank IDs on frames where the selected track is absent. This is
            # intentionally disabled for multi-animal recordings.
            if expected_animals == 1:
                pose_primary = _pose_primary(pose_raw)
                missing = ~pose_primary["frame_index"].isin(pose_layer["frame_index"])
                pose_layer = pd.concat([pose_layer, pose_primary.loc[missing]], ignore_index=True)
                segmentation_primary = _segmentation_primary(segmentation_raw, scale)
                missing = ~segmentation_primary["frame_index"].isin(
                    segmentation_layer["frame_index"]
                )
                segmentation_layer = pd.concat(
                    [segmentation_layer, segmentation_primary.loc[missing]], ignore_index=True
                )
            pose_layers.append(
                pose_layer.sort_values(
                    ["frame_index", "pose_confidence"], ascending=[True, False]
                ).drop_duplicates("frame_index", keep="first")
            )
            segmentation_layers.append(
                segmentation_layer.sort_values(
                    ["frame_index", "segmentation_confidence", "mask_area_px2"],
                    ascending=[True, False, False],
                ).drop_duplicates("frame_index", keep="first")
            )

    tables: list[pd.DataFrame] = []
    for identity, pose, segmentation in zip(identities, pose_layers, segmentation_layers):
        frames = pd.DataFrame({"frame_index": np.arange(count, dtype=int)})
        frames["animal_id"] = identity["animal_id"]
        frames["tracking_status"] = tracking_status
        frames["expected_animal_count"] = len(identities)
        frames["mapped_pose_track_id"] = identity["pose_track_id"]
        frames["mapped_segmentation_track_id"] = identity["segmentation_track_id"]
        frames["cross_layer_mean_bbox_iou"] = identity["mean_bbox_iou"]
        frames["cross_layer_overlap_frames"] = identity["overlap_frames"]
        frames["cross_layer_association_score"] = identity["association_score"]
        table = frames.merge(pose, on="frame_index", how="left", validate="one_to_one")
        table = table.merge(segmentation, on="frame_index", how="left", validate="one_to_one")
        tables.append(table)
    table = (
        pd.concat(tables, ignore_index=True)
        .sort_values(["frame_index", "animal_id"])
        .reset_index(drop=True)
    )

    table["pose_valid"] = table.get("pose_bbox_x1", pd.Series(np.nan, index=table.index)).notna()
    table["segmentation_valid"] = table.get(
        "segmentation_bbox_x1", pd.Series(np.nan, index=table.index)
    ).notna()

    bbox, bbox_source = _first_complete(
        table,
        [
            (
                [
                    "segmentation_bbox_x1",
                    "segmentation_bbox_y1",
                    "segmentation_bbox_x2",
                    "segmentation_bbox_y2",
                ],
                "segmentation_bbox",
            ),
            (
                ["pose_bbox_x1", "pose_bbox_y1", "pose_bbox_x2", "pose_bbox_y2"],
                "pose_bbox",
            ),
        ],
    )
    for column, values in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), bbox):
        table[column] = values
    table["bbox_source"] = bbox_source
    table["bbox_width"] = table["bbox_x2"] - table["bbox_x1"]
    table["bbox_height"] = table["bbox_y2"] - table["bbox_y1"]
    table["bbox_area_px2"] = table["bbox_width"] * table["bbox_height"]

    centroid, centroid_source = _first_complete(
        table,
        [
            (["mask_centroid_x", "mask_centroid_y"], "segmentation_mask"),
            (
                ["segmentation_bbox_center_x", "segmentation_bbox_center_y"],
                "segmentation_bbox",
            ),
            (["pose_bbox_center_x", "pose_bbox_center_y"], "pose_bbox"),
        ],
    )
    table["centroid_x"] = centroid[0]
    table["centroid_y"] = centroid[1]
    table["centroid_source"] = centroid_source
    # Compatibility names let the established notebook-derived plotting and
    # clustering functions consume the same canonical center.
    table["bbox_center_x"] = table["centroid_x"]
    table["bbox_center_y"] = table["centroid_y"]
    smoothed_parts: list[pd.DataFrame] = []
    for _, group in table.groupby("animal_id", sort=False):
        smoothed_parts.append(_smooth_centers(group.sort_values("frame_index"), fps, config))
    smoothed = pd.concat(smoothed_parts).sort_index()
    table["centroid_x_smooth"] = smoothed["bbox_center_x_euro"].reindex(table.index)
    table["centroid_y_smooth"] = smoothed["bbox_center_y_euro"].reindex(table.index)
    table["bbox_center_x_euro"] = table["centroid_x_smooth"]
    table["bbox_center_y_euro"] = table["centroid_y_smooth"]

    table["time_seconds"] = table["frame_index"] / float(fps)
    table["dt_frames"] = 1.0
    table["dt_seconds"] = 1.0 / float(fps)
    grouped = table.groupby("animal_id", sort=False)
    table["dx"] = grouped["centroid_x_smooth"].diff()
    table["dy"] = grouped["centroid_y_smooth"].diff()
    table["distance"] = np.hypot(table["dx"], table["dy"])
    table["distance_px"] = table["distance"]
    table["vx"] = table["dx"] * float(fps)
    table["vy"] = table["dy"] * float(fps)
    table["speed_px_per_frame"] = table["distance"]
    table["speed_px_per_sec"] = table["distance"] * float(fps)
    table["acceleration"] = table.groupby("animal_id", sort=False)[
        "speed_px_per_sec"
    ].diff() * float(fps)
    table["heading"] = np.arctan2(table["vy"], table["vx"])
    table["heading_deg"] = (-np.degrees(table["heading"]) + 360.0) % 360.0
    table["width"] = table["bbox_width"]
    table["height"] = table["bbox_height"]
    table["area"] = table["bbox_area_px2"]
    table["aspect_ratio"] = table["width"] / table["height"].replace(0, np.nan)
    table["area_change"] = table.groupby("animal_id", sort=False)["area"].diff()
    table["aspect_change"] = table.groupby("animal_id", sort=False)["aspect_ratio"].diff()
    table["distance_mm"] = table["distance"] * scale
    table["speed_mm_per_sec"] = table["speed_px_per_sec"] * scale
    table["vx_mm"] = table["vx"] * scale
    table["vy_mm"] = table["vy"] * scale
    table["acceleration_mm_per_sec2"] = table["acceleration"] * scale
    table["width_mm"] = table["width"] * scale
    table["height_mm"] = table["height"] * scale
    table["area_mm2"] = table["area"] * scale**2
    table["cumulative_distance_mm"] = (
        table["distance_mm"].fillna(0).groupby(table["animal_id"], sort=False).cumsum()
    )

    table = _add_prediction_qc(table)

    rois = normalize_rois(config.rois)
    table = assign_roi_labels(
        table,
        rois,
        x_col="centroid_x_smooth",
        y_col="centroid_y_smooth",
    )
    table.loc[
        table["centroid_x_smooth"].isna() | table["centroid_y_smooth"].isna(),
        "roi_label",
    ] = "Missing"
    for name in _keypoint_names(table):
        labels = assign_roi_labels(table, rois, x_col=f"kp_{name}_x", y_col=f"kp_{name}_y")
        table[f"roi_{name}"] = labels["roi_label"]
        table.loc[
            _numeric(table[f"kp_{name}_x"]).isna() | _numeric(table[f"kp_{name}_y"]).isna(),
            f"roi_{name}",
        ] = "Missing"
    # Consolidate the many feature blocks once; downstream CSV and plotting
    # operations then avoid pandas' fragmented-frame performance penalty.
    result = table.copy()
    result.attrs["tracking_diagnostics"] = diagnostics
    return result


def _keypoint_names(df: pd.DataFrame) -> list[str]:
    return sorted(
        column[3:-2]
        for column in df.columns
        if column.startswith("kp_") and column.endswith("_x") and f"{column[:-2]}_y" in df.columns
    )


def _write_roi_tables(df: pd.DataFrame, tables_dir: Path, fps: float) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    animals = df.get("animal_id", pd.Series("animal_1", index=df.index)).astype(str)
    labels = df["roi_label"].fillna("Outside").astype(str)
    duration = _numeric(df.get("dt_seconds", pd.Series(1.0 / fps, index=df.index))).fillna(0)
    work = df[["frame_index", "distance_mm", "speed_mm_per_sec"]].copy()
    work["animal_id"] = animals
    work["roi_label"] = labels
    work["_duration_s"] = duration
    summary = (
        work.groupby(["animal_id", "roi_label"], dropna=False)
        .agg(
            frames=("frame_index", "count"),
            duration_s=("_duration_s", "sum"),
            total_distance_mm=("distance_mm", "sum"),
            average_speed_mm_per_sec=("speed_mm_per_sec", "mean"),
        )
        .reset_index()
        .sort_values(["duration_s", "frames"], ascending=False)
    )
    summary_path = tables_dir / "roi_summary.csv"
    summary.to_csv(summary_path, index=False)

    transitions = pd.DataFrame(columns=["from_roi", "to_roi", "transitions"])
    transition_parts: list[pd.DataFrame] = []
    transition_source = pd.DataFrame(
        {"animal_id": animals, "frame_index": df["frame_index"], "roi_label": labels}
    )
    for animal_id, group in transition_source.groupby("animal_id", sort=False):
        group = group.sort_values("frame_index")
        if len(group) <= 1:
            continue
        pairs = pd.DataFrame(
            {
                "from_roi": group["roi_label"].iloc[:-1].to_numpy(),
                "to_roi": group["roi_label"].iloc[1:].to_numpy(),
            }
        )
        pairs = pairs.loc[
            pairs["from_roi"].ne(pairs["to_roi"])
            & pairs["from_roi"].ne("Missing")
            & pairs["to_roi"].ne("Missing")
        ]
        if pairs.empty:
            continue
        part = pairs.value_counts(["from_roi", "to_roi"]).rename("transitions").reset_index()
        part.insert(0, "animal_id", animal_id)
        transition_parts.append(part)
    if transition_parts:
        transitions = pd.concat(transition_parts, ignore_index=True)
    else:
        transitions = pd.DataFrame(columns=["animal_id", "from_roi", "to_roi", "transitions"])
    transition_path = tables_dir / "roi_transitions.csv"
    transitions.to_csv(transition_path, index=False)

    keypoint_rows: list[dict[str, Any]] = []
    for column in sorted(c for c in df.columns if c.startswith("roi_") and c != "roi_label"):
        for animal_id, group in df.groupby(animals, sort=False):
            for roi, frames in group[column].fillna("Outside").value_counts().items():
                keypoint_rows.append(
                    {
                        "animal_id": str(animal_id),
                        "keypoint": column[4:],
                        "roi_label": str(roi),
                        "frames": int(frames),
                        "duration_s": float(frames) / float(fps),
                    }
                )
    keypoint_path = tables_dir / "keypoint_roi_summary.csv"
    pd.DataFrame(
        keypoint_rows,
        columns=["animal_id", "keypoint", "roi_label", "frames", "duration_s"],
    ).to_csv(keypoint_path, index=False)
    qc_columns = [
        "frame_index",
        "animal_id",
        "prediction_qc_status",
        "prediction_qc_reasons",
        "extra_pose_detections",
        "extra_segmentation_detections",
        "extra_pose_tracks",
        "extra_segmentation_tracks",
        "primary_detection_confidence",
        "pose_segmentation_bbox_iou",
        "centroid_jump_px",
        "centroid_jump_threshold_px",
        "pose_track_id",
        "segmentation_track_id",
        "pose_confidence",
        "segmentation_confidence",
    ]
    qc_path = tables_dir / "prediction_qc_frames.csv"
    df.loc[df["prediction_qc_status"].ne("good"), qc_columns].to_csv(qc_path, index=False)
    return {
        "roi_summary_csv": str(summary_path),
        "roi_transition_csv": str(transition_path),
        "keypoint_roi_summary_csv": str(keypoint_path),
        "prediction_qc_csv": str(qc_path),
        "roi_summary": json.loads(summary.to_json(orient="records")),
    }


def _create_unified_plots(
    df: pd.DataFrame,
    plots_dir: Path,
    *,
    video_path: str,
    rois: list[dict[str, Any]],
) -> list[str]:
    if "animal_id" in df.columns and df["animal_id"].nunique() > 1:
        paths: list[str] = []
        for animal_id, group in df.groupby("animal_id", sort=False):
            paths.extend(
                _create_unified_plots(
                    group.copy(),
                    plots_dir / str(animal_id),
                    video_path=video_path,
                    rois=rois,
                )
            )
        return paths
    paths = create_plots(
        df,
        plots_dir,
        video_path=video_path,
        rois=rois,
        include_motion_diagnostics=False,
    )
    plt, sns = _setup_plotting()
    time_minutes = _numeric(df["time_seconds"]) / 60.0

    confidence_columns = [
        ("pose_confidence", "Pose"),
        ("segmentation_confidence", "Segmentation"),
    ]
    if any(column in df.columns for column, _ in confidence_columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        for column, label in confidence_columns:
            if column in df.columns:
                ax.plot(time_minutes, _numeric(df[column]), linewidth=0.7, label=label)
        ax.set(title="Detection Confidence", xlabel="Time (min)", ylabel="Confidence")
        ax.legend()
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "detection_confidence.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if {"mask_area_mm2", "frame_index"}.issubset(df.columns):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(time_minutes, _numeric(df["mask_area_mm2"]), linewidth=0.8)
        axes[0].set_ylabel("Mask area (mm²)")
        axes[0].set_title("Segmentation Geometry")
        axes[1].plot(time_minutes, _numeric(df["mask_fill_ratio"]), linewidth=0.8)
        axes[1].set(xlabel="Time (min)", ylabel="Mask / box area")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "segmentation_geometry.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if "roi_label" in df.columns:
        order = df["roi_label"].fillna("Outside").value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(10, max(3, len(order) * 0.45 + 1.5)))
        roi_plot = (
            df.assign(roi_label=df["roi_label"].fillna("Outside"))
            .groupby("roi_label", as_index=False)["dt_seconds"]
            .sum()
            .rename(columns={"dt_seconds": "duration_s"})
        )
        sns.barplot(data=roi_plot, x="duration_s", y="roi_label", order=order, ax=ax)
        ax.set(title="Time in ROI", xlabel="Seconds", ylabel="ROI")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "roi_time_seconds.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        roi_speed = (
            df.assign(roi_label=df["roi_label"].fillna("Outside"))
            .groupby("roi_label", as_index=False)["speed_mm_per_sec"]
            .mean()
        )
        fig, ax = plt.subplots(figsize=(9, max(3, len(roi_speed) * 0.45 + 1.5)))
        sns.barplot(data=roi_speed, x="speed_mm_per_sec", y="roi_label", ax=ax)
        ax.set(title="Average Speed by ROI", xlabel="Speed (mm/s)", ylabel="ROI")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "roi_average_speed.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        labels = df["roi_label"].fillna("Outside").astype(str).reset_index(drop=True)
        if len(labels) > 1:
            pairs = pd.DataFrame(
                {"from_roi": labels.iloc[:-1].to_numpy(), "to_roi": labels.iloc[1:].to_numpy()}
            )
            pairs = pairs.loc[
                pairs["from_roi"].ne(pairs["to_roi"])
                & pairs["from_roi"].ne("Missing")
                & pairs["to_roi"].ne("Missing")
            ]
            if not pairs.empty:
                matrix = pd.crosstab(pairs["from_roi"], pairs["to_roi"])
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set(title="ROI Transitions", xlabel="To ROI", ylabel="From ROI")
                fig.tight_layout()
                path = plots_dir / "roi_transitions.png"
                fig.savefig(path, dpi=140)
                plt.close(fig)
                paths.append(str(path))

        categories = list(dict.fromkeys(labels.tolist()))
        category_codes = pd.Categorical(labels, categories=categories).codes
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.step(time_minutes, category_codes, where="post", linewidth=0.8)
        ax.set_yticks(range(len(categories)), categories)
        ax.set(title="ROI Occupancy Timeline", xlabel="Time (min)", ylabel="ROI")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "roi_timeline.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    timing_columns = [
        ("pose_preprocess_ms", "Pose preprocess"),
        ("pose_inference_ms", "Pose inference"),
        ("pose_postprocess_ms", "Pose postprocess"),
        ("segmentation_preprocess_ms", "Segmentation preprocess"),
        ("segmentation_inference_ms", "Segmentation inference"),
        ("segmentation_postprocess_ms", "Segmentation postprocess"),
    ]
    existing_timing = [(column, label) for column, label in timing_columns if column in df.columns]
    if existing_timing:
        column_count = min(3, len(existing_timing))
        row_count = int(math.ceil(len(existing_timing) / column_count))
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(5 * column_count, 3.6 * row_count),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for ax, (column, label) in zip(flat_axes, existing_timing):
            values = _numeric(df[column]).dropna()
            if not values.empty:
                ax.boxplot(values, orientation="horizontal")
                ax.axvline(values.mean(), color="red", linestyle="--", linewidth=1)
            ax.set(title=label, xlabel="Milliseconds")
            ax.set_yticks([])
        for ax in flat_axes[len(existing_timing) :]:
            ax.set_visible(False)
        fig.tight_layout()
        path = plots_dir / "processing_speed_boxplots.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if {"vx_mm", "vy_mm", "frame_index"}.issubset(df.columns):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(time_minutes, _numeric(df["vx_mm"]), linewidth=0.7)
        axes[0].set(ylabel="X velocity (mm/s)", title="Velocity Components")
        axes[1].plot(time_minutes, _numeric(df["vy_mm"]), linewidth=0.7)
        axes[1].set(xlabel="Time (min)", ylabel="Y velocity (mm/s)")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "velocity_components.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if {"speed_mm_per_sec", "frame_index"}.issubset(df.columns):
        fps = 1.0 / max(float(_numeric(df["dt_seconds"]).median()), 1e-9)
        window = max(1, int(round(fps)))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            time_minutes,
            _numeric(df["speed_mm_per_sec"]),
            alpha=0.25,
            linewidth=0.5,
            label="Instantaneous",
        )
        ax.plot(
            time_minutes,
            _numeric(df["speed_mm_per_sec"]).rolling(window, min_periods=1).mean(),
            linewidth=1.0,
            label="1 s mean",
        )
        ax.set(title="Speed", xlabel="Time (min)", ylabel="Speed (mm/s)")
        ax.legend()
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "speed_with_rolling_mean.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if {"acceleration_mm_per_sec2", "dt_seconds"}.issubset(df.columns):
        fps = 1.0 / max(float(_numeric(df["dt_seconds"]).median()), 1e-9)
        window = max(1, int(round(fps)))
        magnitude = _numeric(df["acceleration_mm_per_sec2"]).abs()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            time_minutes,
            magnitude,
            alpha=0.22,
            linewidth=0.5,
            label="Instantaneous magnitude",
        )
        ax.plot(
            time_minutes,
            magnitude.rolling(window, min_periods=1).mean(),
            linewidth=1.0,
            label="1 s mean magnitude",
        )
        ax.set(
            title="Acceleration Magnitude",
            xlabel="Time (min)",
            ylabel="Absolute acceleration (mm/s²)",
        )
        ax.legend()
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "acceleration_magnitude.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if {"cumulative_distance_mm", "time_seconds"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_minutes, _numeric(df["cumulative_distance_mm"]), linewidth=1.0)
        ax.set(title="Cumulative Distance", xlabel="Time (min)", ylabel="Distance (mm)")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "cumulative_distance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    if "prediction_qc_status" in df.columns:
        statuses = ("good", "warning", "bad")
        counts = df["prediction_qc_status"].value_counts()
        colors = ("#3f9b6d", "#e3a327", "#cf4b4b")
        qc_codes = df["prediction_qc_status"].map({"good": 0, "warning": 1, "bad": 2})
        flagged = qc_codes.fillna(2).gt(0)
        if flagged.any():
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=(1, 2))
            count_axis, timeline_axis = axes
            timeline_axis.step(time_minutes, qc_codes, where="post", linewidth=0.8, color="#4c72b0")
            timeline_axis.set(
                xlabel="Time (min)",
                ylabel="QC status",
                yticks=(0, 1, 2),
                yticklabels=("Good", "Warning", "Bad"),
                ylim=(-0.25, 2.25),
            )
        else:
            fig, count_axis = plt.subplots(figsize=(8, 4))
            count_axis.text(
                0.98,
                0.92,
                "No frames flagged",
                transform=count_axis.transAxes,
                ha="right",
                va="top",
                color="#287a50",
                weight="bold",
            )
        count_axis.bar(
            statuses,
            [int(counts.get(status, 0)) for status in statuses],
            color=colors,
        )
        count_axis.set(title="Prediction QC", ylabel="Rows")
        sns.despine(fig)
        fig.tight_layout()
        path = plots_dir / "prediction_qc.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _draw_rois(frame: Any, rois: list[dict[str, Any]], cv2: Any) -> None:
    for roi in reversed(rois):
        if roi["type"] == "polygon":
            points = np.asarray(roi["points"], dtype=np.float64).reshape((-1, 2))
        else:
            points = np.asarray(
                [
                    [roi["x1"], roi["y1"]],
                    [roi["x2"], roi["y1"]],
                    [roi["x2"], roi["y2"]],
                    [roi["x1"], roi["y2"]],
                ],
                dtype=np.float64,
            ).reshape((-1, 2))
        draw_antialiased_polyline(frame, points, (66, 191, 245), cv2, thickness=2)
        x, y = np.mean(points, axis=0).astype(int)
        cv2.putText(
            frame,
            str(roi["name"]),
            (x, max(y, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (66, 191, 245),
            2,
            lineType=cv2.LINE_AA,
        )


def render_unified_annotated_video(
    df: pd.DataFrame,
    video_path: str,
    output_path: Path,
    fps: float,
    rois: list[dict[str, Any]],
) -> str:
    if not video_path or not os.path.isfile(video_path):
        raise AnalysisError("The source video is unavailable for annotated-video export.")
    try:
        import cv2
    except Exception as exc:
        raise AnalysisError(f"OpenCV is required for annotated video export: {exc}") from exc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisError(f"Could not open source video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    rows: dict[int, list[Any]] = {}
    for row in df.itertuples(index=False):
        rows.setdefault(int(row.frame_index), []).append(row)
    names = _keypoint_names(df)
    keypoint_colors = [
        (255, 100, 40),
        (70, 220, 255),
        (200, 80, 255),
        (80, 220, 100),
        (255, 180, 40),
        (180, 180, 255),
    ]
    animal_palette = [
        (255, 210, 70),
        (70, 190, 255),
        (190, 90, 255),
        (80, 220, 120),
        (255, 130, 90),
        (220, 180, 255),
    ]
    animal_ids = list(dict.fromkeys(df.get("animal_id", pd.Series(["animal_1"])).astype(str)))
    animal_colors = {
        animal_id: animal_palette[index % len(animal_palette)]
        for index, animal_id in enumerate(animal_ids)
    }
    frame_index = 0
    try:
        with _open_h264_video_writer(output_path, video_fps, width, height) as writer:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw_rois(frame, rois, cv2)
                for animal_index, row in enumerate(rows.get(frame_index, [])):
                    animal_id = str(getattr(row, "animal_id", "animal_1"))
                    animal_color = animal_colors.get(animal_id, animal_palette[0])
                    polygon = _parse_polygon(getattr(row, "mask_polygon", ""))
                    if len(polygon) >= 3:
                        draw_supersampled_polygon_overlay(
                            frame,
                            polygon,
                            animal_color,
                            cv2,
                            alpha=0.28,
                            supersample=2,
                            outline_thickness=2,
                        )
                    box = [
                        getattr(row, key, math.nan)
                        for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                    ]
                    if all(pd.notna(value) for value in box):
                        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            animal_color,
                            2,
                            lineType=cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            animal_id.replace("_", " ").title(),
                            (x1, max(y1 - 6, 18)),
                            0,
                            0.55,
                            animal_color,
                            2,
                            lineType=cv2.LINE_AA,
                        )

                    keypoints: dict[str, tuple[float, float]] = {}
                    for index, name in enumerate(names):
                        x = getattr(row, f"kp_{name}_x", math.nan)
                        y = getattr(row, f"kp_{name}_y", math.nan)
                        confidence = getattr(row, f"kp_{name}_conf", 1.0)
                        if (
                            pd.notna(x)
                            and pd.notna(y)
                            and (pd.isna(confidence) or float(confidence) > 0)
                        ):
                            point = (float(x), float(y))
                            keypoints[name] = point
                            cv2.circle(
                                frame,
                                (int(round(point[0])), int(round(point[1]))),
                                4,
                                keypoint_colors[index % len(keypoint_colors)],
                                -1,
                                lineType=cv2.LINE_AA,
                            )
                    for first, second in POSE_SKELETON:
                        if first in keypoints and second in keypoints:
                            draw_antialiased_polyline(
                                frame,
                                [keypoints[first], keypoints[second]],
                                animal_color,
                                cv2,
                                thickness=2,
                                closed=False,
                            )

                    cx = getattr(row, "centroid_x_smooth", math.nan)
                    cy = getattr(row, "centroid_y_smooth", math.nan)
                    if pd.notna(cx) and pd.notna(cy):
                        cv2.circle(
                            frame,
                            (int(round(cx)), int(round(cy))),
                            4,
                            animal_color,
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                    text = f"{animal_id}: frame {frame_index}"
                    distance = getattr(row, "cumulative_distance_mm", math.nan)
                    if pd.notna(distance):
                        text += f" | Distance {float(distance):.1f} mm"
                    text_y = 28 + animal_index * 48
                    cv2.putText(
                        frame,
                        text,
                        (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        animal_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    speed = getattr(row, "speed_mm_per_sec", math.nan)
                    roi = getattr(row, "roi_label", "Outside")
                    cv2.putText(
                        frame,
                        f"Speed {float(speed):.1f} mm/s | ROI {roi}"
                        if pd.notna(speed)
                        else f"ROI {roi}",
                        (10, text_y + 22),
                        0,
                        0.55,
                        animal_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                writer.write(frame)
                frame_index += 1
    finally:
        cap.release()
    return str(output_path)


def run_unified_analysis_workflow(
    config: AnalysisConfig,
    *,
    pose_csv: str,
    segmentation_csv: str,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    """Run both inference layers without producing intermediate layer analyses."""
    for label, path in (("pose", pose_csv), ("segmentation", segmentation_csv)):
        if not path or not os.path.isfile(path):
            raise AnalysisError(f"Select a valid {label} inference CSV.")
    output_dir = Path(config.output_dir)
    prepare_analysis_output_dir(
        output_dir,
        generated_files=(
            "analysis.csv",
            "summary.json",
            "analysis_manifest.json",
            "annotated_video.mp4",
        ),
        generated_directories=("tables", "plots", "clustering"),
    )
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    clustering_dir = output_dir / "clustering"
    total = 9

    _progress(progress_callback, 1, total, "Loading pose and segmentation inference files")
    pose_raw = _read_pose_csv(pose_csv)
    segmentation_raw = _read_segmentation_csv(segmentation_csv)
    video_path = config.video_path.strip()
    fps = _infer_fps(pose_raw, video_path, config.fps)
    scale = _mm_per_pixel(config)

    _progress(progress_callback, 2, total, "Selecting the primary animal detection per frame")
    table = build_unified_frame_table(
        pose_raw,
        segmentation_raw,
        config,
        video_path=video_path,
        fps=fps,
        scale=scale,
    )
    tracking_diagnostics = dict(table.attrs.get("tracking_diagnostics") or {})

    cluster_plot_paths: list[str] = []
    cluster_clip_paths: list[str] = []
    if config.run_clustering:
        _progress(progress_callback, 3, total, "Computing UMAP/HDBSCAN behavior clusters")
        clustering_dir.mkdir(parents=True, exist_ok=True)
        clustered_parts: list[pd.DataFrame] = []
        for animal_id, group in table.groupby("animal_id", sort=False):
            animal_dir = (
                clustering_dir / str(animal_id)
                if table["animal_id"].nunique() > 1
                else clustering_dir
            )
            clustered, paths = run_behavior_clustering(
                group.copy(),
                fps,
                animal_dir,
                umap_neighbors=config.umap_neighbors,
                umap_min_dist=config.umap_min_dist,
                hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
            )
            clustered_parts.append(clustered)
            cluster_plot_paths.extend(paths)
        table = pd.concat(clustered_parts, ignore_index=True).sort_values(
            ["frame_index", "animal_id"]
        )
    else:
        _progress(progress_callback, 3, total, "Skipping behavior clustering")

    _progress(progress_callback, 4, total, "Writing authoritative analysis.csv")
    analysis_csv = output_dir / "analysis.csv"
    table.to_csv(analysis_csv, index=False)
    # Derivative products deliberately read the persisted table so there is no
    # hidden in-memory analysis state that can diverge from analysis.csv.
    analysis = pd.read_csv(analysis_csv, low_memory=False)

    _progress(progress_callback, 5, total, "Writing ROI and keypoint summary tables")
    roi_outputs = _write_roi_tables(analysis, tables_dir, fps)

    plot_paths: list[str] = []
    if config.make_plots:
        _progress(progress_callback, 6, total, "Rendering plots from analysis.csv")
        plot_paths = _create_unified_plots(
            analysis,
            plots_dir,
            video_path=video_path,
            rois=normalize_rois(config.rois),
        )
    else:
        _progress(progress_callback, 6, total, "Skipping plots")

    annotated_video = ""
    if config.make_annotated_video:
        _progress(progress_callback, 7, total, "Rendering fused annotated video from analysis.csv")
        annotated_video = render_unified_annotated_video(
            analysis,
            video_path,
            output_dir / "annotated_video.mp4",
            fps,
            normalize_rois(config.rois),
        )
    else:
        _progress(progress_callback, 7, total, "Skipping annotated video")

    if config.run_clustering and config.export_cluster_clips:
        _progress(progress_callback, 8, total, "Exporting behavior cluster clips")
        for animal_id, group in analysis.groupby("animal_id", sort=False):
            animal_dir = (
                clustering_dir / str(animal_id)
                if analysis["animal_id"].nunique() > 1
                else clustering_dir
            )
            cluster_clip_paths.extend(
                export_cluster_clips(
                    group,
                    video_path,
                    animal_dir,
                    fps,
                    config.cluster_clip_length_sec,
                    config.samples_per_cluster,
                )
            )
    else:
        _progress(progress_callback, 8, total, "Finalizing analysis summary")

    total_distance = float(analysis["distance_mm"].sum(skipna=True))
    per_animal = []
    for animal_id, group in analysis.groupby("animal_id", sort=False):
        animal_distance = float(group["distance_mm"].sum(skipna=True))
        per_animal.append(
            {
                "animal_id": str(animal_id),
                "rows": int(len(group)),
                "pose_valid_frames": int(group["pose_valid"].sum()),
                "segmentation_valid_frames": int(group["segmentation_valid"].sum()),
                "total_distance_mm": animal_distance,
                "average_speed_mm_per_sec": float(group["speed_mm_per_sec"].mean(skipna=True)),
            }
        )
    observed = analysis["pose_valid"] | analysis["segmentation_valid"]
    observed_by_frame = observed.groupby(analysis["frame_index"]).sum()
    expected_animals = int(tracking_diagnostics.get("expected_animal_count") or 1)
    count_mismatch_frames = int(observed_by_frame.ne(expected_animals).sum())
    qc_status_counts = {
        str(key): int(value)
        for key, value in analysis["prediction_qc_status"].value_counts().items()
    }
    qc_reason_counts: dict[str, int] = {}
    for value in analysis["prediction_qc_reasons"].fillna("").astype(str):
        for reason in filter(None, value.split(";")):
            qc_reason_counts[reason] = qc_reason_counts.get(reason, 0) + 1
    summary = {
        "analysis_kind": "pose_and_segmentation",
        "frames": int(analysis["frame_index"].nunique()),
        "analysis_rows": int(len(analysis)),
        "expected_animal_count": expected_animals,
        "tracking_status": tracking_diagnostics.get("tracking_status", "legacy_untracked"),
        "animal_count_mismatch_frames": count_mismatch_frames,
        "prediction_qc_status_counts": qc_status_counts,
        "prediction_qc_reason_counts": dict(sorted(qc_reason_counts.items())),
        "prediction_qc_flagged_rows": int(analysis["prediction_qc_status"].ne("good").sum()),
        "fps": float(fps),
        "duration_s": float(analysis["time_seconds"].max()) if len(analysis) else 0.0,
        "mm_per_pixel": float(scale),
        "total_distance_mm": total_distance,
        "total_distance_m": total_distance / 1000.0,
        "average_speed_mm_per_sec": float(analysis["speed_mm_per_sec"].mean(skipna=True)),
        "average_acceleration_mm_per_sec2": float(
            analysis["acceleration_mm_per_sec2"].mean(skipna=True)
        ),
        "pose_valid_frames": int(analysis["pose_valid"].sum()),
        "segmentation_valid_frames": int(analysis["segmentation_valid"].sum()),
        "centroid_source_counts": {
            str(key): int(value)
            for key, value in analysis["centroid_source"].value_counts().items()
        },
        "bbox_source_counts": {
            str(key): int(value) for key, value in analysis["bbox_source"].value_counts().items()
        },
        "segmentation_bbox_source_counts": {
            str(key): int(value)
            for key, value in analysis["segmentation_bbox_source"].value_counts().items()
        },
        "roi_count": len(normalize_rois(config.rois)),
        "roi_summary": roi_outputs["roi_summary"],
        "per_animal": per_animal,
        "tracking_diagnostics": tracking_diagnostics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": 5,
        "analysis_kind": "pose_and_segmentation",
        "authoritative_table": str(analysis_csv),
        "video_path": os.path.abspath(video_path) if video_path else "",
        "pose_inference_csv": os.path.abspath(pose_csv),
        "segmentation_inference_csv": os.path.abspath(segmentation_csv),
        "centroid_precedence": ["segmentation_mask", "segmentation_bbox", "pose_bbox"],
        "bbox_precedence": ["segmentation_bbox", "pose_bbox"],
        "segmentation_bbox_definition": "mask_polygon_bounds_with_inference_bbox_fallback",
        "segmentation_inference_bbox_role": "tracker_or_detector_box_retained_for_diagnostics",
        "prediction_qc": {
            "behavior": "diagnostic_only; rows and measurements are never removed",
            "low_detection_confidence": PREDICTION_QC_LOW_CONFIDENCE,
            "minimum_pose_segmentation_bbox_iou": PREDICTION_QC_MIN_LAYER_IOU,
            "maximum_centroid_jump_box_diagonals": PREDICTION_QC_MAX_JUMP_BOX_DIAGONALS,
            "maximum_centroid_jump_image_fraction": PREDICTION_QC_MAX_JUMP_IMAGE_FRACTION,
            "primary_detection_confidence": "minimum available confidence of the selected pose and segmentation detections",
        },
        "skeleton_edges": [list(edge) for edge in POSE_SKELETON],
        "fps": float(fps),
        "mm_per_pixel": float(scale),
        "rois": normalize_rois(config.rois),
        "columns": list(analysis.columns),
        "tracking": tracking_diagnostics,
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _progress(progress_callback, 9, total, "Unified analysis complete")
    return {
        "layer_id": "combined",
        "analysis_mode": "both",
        "feature_csv": str(analysis_csv),
        "summary_json": str(summary_path),
        "summary": summary,
        "manifest_path": str(manifest_path),
        "plot_paths": plot_paths + cluster_plot_paths,
        "annotated_video": annotated_video,
        "cluster_clip_paths": cluster_clip_paths,
        "output_dir": str(output_dir),
        **roi_outputs,
    }
