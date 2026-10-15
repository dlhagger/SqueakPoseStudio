"""Same-frame distances between raw and smoothed instance centers."""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _identifier(row: dict[str, Any], columns: tuple[str, ...]) -> Any:
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value) or str(value) == "":
            continue
        number = _number(value)
        if number is not None:
            return int(number) if number.is_integer() else number
        return str(value)
    return None


def _center(row: dict[str, Any]) -> tuple[float, float] | None:
    x, y = _number(row.get("mask_centroid_x")), _number(row.get("mask_centroid_y"))
    if x is not None and y is not None:
        return x, y
    for prefix in ("bbox_", "segmentation_bbox_", "pose_bbox_", "inference_bbox_"):
        bounds = [_number(row.get(prefix + suffix)) for suffix in ("x1", "y1", "x2", "y2")]
        if all(value is not None for value in bounds):
            x1, y1, x2, y2 = bounds
            assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
            if x2 > x1 and y2 > y1:
                return (x1 + x2) / 2, (y1 + y2) / 2
    x, y = _number(row.get("bbox_center_x")), _number(row.get("bbox_center_y"))
    return (x, y) if x is not None and y is not None else None


def _smoothed_center(row: dict[str, Any]) -> tuple[float, float] | None:
    # Do not turn a missing observation into a distance via an interpolated center.
    if _center(row) is None:
        return None
    for x_column, y_column in (
        ("centroid_x_smooth", "centroid_y_smooth"),
        ("bbox_center_x_euro", "bbox_center_y_euro"),
    ):
        x, y = _number(row.get(x_column)), _number(row.get(y_column))
        if x is not None and y is not None:
            return x, y
    return None


def add_center_distances(features: pd.DataFrame, mm_per_pixel: float | None) -> pd.DataFrame:
    """Add raw and smoothed JSON distance lists to every observed instance row.

    ``center_distances`` retains its unsmoothed definition. The separate
    ``center_distances_smoothed`` column uses existing filtered centers; it never
    silently substitutes raw centers. Missing centers produce blank cells and
    a valid center with no other observed instances produces ``[]``.
    """
    if mm_per_pixel is not None and (not math.isfinite(mm_per_pixel) or mm_per_pixel <= 0):
        raise ValueError("mm_per_pixel must be finite and positive.")
    output = features.copy()
    if "frame_index" not in features:
        raise ValueError("Center distances require a frame_index column.")
    rows = features.to_dict("records")
    frames = pd.to_numeric(features["frame_index"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    groups: dict[float, list[int]] = {}
    for index, frame in enumerate(frames):
        if np.isfinite(frame):
            groups.setdefault(float(frame), []).append(index)

    def target(index: int, distance: float) -> dict[str, Any]:
        row = rows[index]
        return {
            "class_name": next(
                (
                    str(row[column])
                    for column in (
                        "analysis_class",
                        "class_name",
                        "segmentation_class_name",
                        "pose_class_name",
                    )
                    if column in row and pd.notna(row[column]) and str(row[column])
                ),
                None,
            ),
            "track_id": _identifier(row, ("track_id", "segmentation_track_id", "pose_track_id")),
            "detection_index": _identifier(
                row, ("detection_index", "segmentation_detection_index", "pose_detection_index")
            ),
            "distance_px": distance,
            "distance_mm": distance * mm_per_pixel if mm_per_pixel is not None else None,
        }

    for column, center_function in (
        ("center_distances", _center),
        ("center_distances_smoothed", _smoothed_center),
    ):
        centers = [center_function(row) for row in rows]
        distances: list[list[dict[str, Any]]] = [[] for _ in rows]
        for group in groups.values():
            indices = [index for index in group if centers[index] is not None]
            for left, right in combinations(indices, 2):
                a, b = centers[left], centers[right]
                assert a is not None and b is not None
                distance = math.hypot(a[0] - b[0], a[1] - b[1])
                distances[left].append(target(right, distance))
                distances[right].append(target(left, distance))
        output[column] = [
            json.dumps(items, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if centers[index] is not None and np.isfinite(frames[index])
            else None
            for index, items in enumerate(distances)
        ]
    return output
