"""Segmentation-specific analysis workflow for SqueakPose inference CSVs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis_ops import (
    AnalysisConfig,
    AnalysisError,
    ProgressCallback,
    _draw_roi_overlays,
    _first_video_frame,
    _infer_fps,
    _mm_per_pixel,
    _open_h264_video_writer,
    _progress,
    _setup_plotting,
    _smooth_centers,
    assign_roi_labels,
    create_roi_outputs,
    export_cluster_clips,
    normalize_rois,
    run_behavior_clustering,
)

SEGMENTATION_REQUIRED_COLUMNS = ["frame", "det", "x1", "y1", "x2", "y2", "mask_polygon"]


def is_segmentation_inference_csv(df: pd.DataFrame) -> bool:
    """Return True when a CSV has the SqueakPose segmentation inference schema."""
    return all(col in df.columns for col in SEGMENTATION_REQUIRED_COLUMNS)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_polygon(raw: Any) -> list[tuple[float, float]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw_points = json.loads(text)
        except json.JSONDecodeError:
            return []
    else:
        raw_points = raw

    if not isinstance(raw_points, list):
        return []
    points: list[tuple[float, float]] = []
    for pair in raw_points:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        x = _safe_float(pair[0])
        y = _safe_float(pair[1])
        if not math.isnan(x) and not math.isnan(y):
            points.append((x, y))
    return points


def _polygon_metrics(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 3:
        return {
            "mask_area_px2": math.nan,
            "mask_perimeter_px": math.nan,
            "mask_centroid_x": math.nan,
            "mask_centroid_y": math.nan,
            "mask_orientation_deg": math.nan,
            "mask_major_axis_px": math.nan,
            "mask_minor_axis_px": math.nan,
            "mask_points": float(len(points)),
        }

    xy = np.asarray(points, dtype=float)
    xs = xy[:, 0]
    ys = xy[:, 1]
    xs_next = np.roll(xs, -1)
    ys_next = np.roll(ys, -1)
    cross = xs * ys_next - xs_next * ys
    signed_area = 0.5 * float(cross.sum())
    area = abs(signed_area)

    if abs(signed_area) < 1e-9:
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
    else:
        centroid_x = float(((xs + xs_next) * cross).sum() / (6.0 * signed_area))
        centroid_y = float(((ys + ys_next) * cross).sum() / (6.0 * signed_area))

    diffs = np.diff(np.vstack([xy, xy[0]]), axis=0)
    perimeter = float(np.sqrt((diffs**2).sum(axis=1)).sum())

    centered = xy - np.array([centroid_x, centroid_y])
    orientation = math.nan
    major_axis = math.nan
    minor_axis = math.nan
    if len(centered) >= 2 and np.isfinite(centered).all():
        try:
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            orientation = float(
                (math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])) + 360.0) % 180.0
            )
            major_axis = float(4.0 * math.sqrt(max(eigenvalues[0], 0.0)))
            minor_axis = float(4.0 * math.sqrt(max(eigenvalues[-1], 0.0)))
        except Exception:
            pass

    return {
        "mask_area_px2": float(area),
        "mask_perimeter_px": perimeter,
        "mask_centroid_x": centroid_x,
        "mask_centroid_y": centroid_y,
        "mask_orientation_deg": orientation,
        "mask_major_axis_px": major_axis,
        "mask_minor_axis_px": minor_axis,
        "mask_points": float(len(points)),
    }


def _valid_detection_rows(raw: pd.DataFrame) -> pd.DataFrame:
    detections = raw.copy()
    detections["frame_index"] = pd.to_numeric(detections["frame"], errors="coerce")
    detections["detection_index"] = pd.to_numeric(detections["det"], errors="coerce")
    if "class_id" in detections.columns:
        valid_class = pd.to_numeric(detections["class_id"], errors="coerce").notna()
    else:
        valid_class = pd.Series(True, index=detections.index)
    valid_det = detections["detection_index"].fillna(-1) >= 0
    valid_frame = detections["frame_index"].notna()
    return detections.loc[valid_frame & valid_det & valid_class].copy()


def compute_segmentation_detection_features(raw: pd.DataFrame, mm_per_pixel: float) -> pd.DataFrame:
    """Compute geometry features for every valid segmentation detection row."""
    detections = _valid_detection_rows(raw)
    if detections.empty:
        raise AnalysisError("Segmentation CSV does not contain any valid mask detections.")

    for col in ["frame_index", "detection_index", "class_id", "conf", "x1", "y1", "x2", "y2"]:
        if col in detections.columns:
            detections[col] = pd.to_numeric(detections[col], errors="coerce")

    polygon_points = detections["mask_polygon"].map(_parse_polygon)
    metrics = pd.DataFrame(
        [_polygon_metrics(points) for points in polygon_points], index=detections.index
    )
    detections = pd.concat([detections, metrics], axis=1)

    detections["bbox_x1"] = pd.to_numeric(detections["x1"], errors="coerce")
    detections["bbox_y1"] = pd.to_numeric(detections["y1"], errors="coerce")
    detections["bbox_x2"] = pd.to_numeric(detections["x2"], errors="coerce")
    detections["bbox_y2"] = pd.to_numeric(detections["y2"], errors="coerce")
    detections["bbox_width"] = detections["bbox_x2"] - detections["bbox_x1"]
    detections["bbox_height"] = detections["bbox_y2"] - detections["bbox_y1"]
    detections["bbox_area_px2"] = detections["bbox_width"] * detections["bbox_height"]
    detections["bbox_center_x"] = (detections["bbox_x1"] + detections["bbox_x2"]) / 2.0
    detections["bbox_center_y"] = (detections["bbox_y1"] + detections["bbox_y2"]) / 2.0
    detections["mask_fill_ratio"] = detections["mask_area_px2"] / detections[
        "bbox_area_px2"
    ].replace(0, np.nan)

    detections["bbox_center_x"] = detections["mask_centroid_x"].where(
        detections["mask_centroid_x"].notna(),
        detections["bbox_center_x"],
    )
    detections["bbox_center_y"] = detections["mask_centroid_y"].where(
        detections["mask_centroid_y"].notna(),
        detections["bbox_center_y"],
    )
    detections["confidence"] = pd.to_numeric(detections.get("conf"), errors="coerce")

    detections["bbox_width_mm"] = detections["bbox_width"] * mm_per_pixel
    detections["bbox_height_mm"] = detections["bbox_height"] * mm_per_pixel
    detections["bbox_area_mm2"] = detections["bbox_area_px2"] * (mm_per_pixel**2)
    detections["mask_area_mm2"] = detections["mask_area_px2"] * (mm_per_pixel**2)
    detections["mask_perimeter_mm"] = detections["mask_perimeter_px"] * mm_per_pixel
    detections["mask_major_axis_mm"] = detections["mask_major_axis_px"] * mm_per_pixel
    detections["mask_minor_axis_mm"] = detections["mask_minor_axis_px"] * mm_per_pixel

    counts = detections.groupby("frame_index")["detection_index"].transform("count")
    detections["detections_in_frame"] = counts.astype(int)
    detections["is_primary_detection"] = False

    output_cols = [
        "frame_index",
        "detection_index",
        "detections_in_frame",
        "class_id",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "bbox_area_px2",
        "bbox_center_x",
        "bbox_center_y",
        "mask_centroid_x",
        "mask_centroid_y",
        "mask_area_px2",
        "mask_perimeter_px",
        "mask_fill_ratio",
        "mask_orientation_deg",
        "mask_major_axis_px",
        "mask_minor_axis_px",
        "mask_points",
        "bbox_width_mm",
        "bbox_height_mm",
        "bbox_area_mm2",
        "mask_area_mm2",
        "mask_perimeter_mm",
        "mask_major_axis_mm",
        "mask_minor_axis_mm",
        "mask_polygon",
        "is_primary_detection",
    ]
    return detections[[col for col in output_cols if col in detections.columns]].reset_index(
        drop=True
    )


def _smooth_segmentation_centers(
    df: pd.DataFrame, fps: float, config: AnalysisConfig
) -> pd.DataFrame:
    smoothed = _smooth_centers(df, fps, config)
    return smoothed


def compute_segmentation_track_features(
    detection_features: pd.DataFrame,
    raw: pd.DataFrame,
    fps: float,
    mm_per_pixel: float,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Select one segmentation detection per frame and compute motion features."""
    if detection_features.empty:
        raise AnalysisError("No valid segmentation detections were available for tracking.")

    primary = (
        detection_features.sort_values(
            ["frame_index", "confidence", "mask_area_px2"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset="frame_index", keep="first")
        .sort_values("frame_index")
        .reset_index(drop=True)
    )
    primary["is_primary_detection"] = True
    primary = _smooth_segmentation_centers(primary, fps, config)

    primary["frame_index"] = pd.to_numeric(primary["frame_index"], errors="coerce")
    primary["time_seconds"] = primary["frame_index"] / float(fps or 30.0)
    primary["dx"] = primary["bbox_center_x_euro"].diff()
    primary["dy"] = primary["bbox_center_y_euro"].diff()
    frame_delta = primary["frame_index"].diff()
    primary["dt_frames"] = frame_delta.fillna(1).replace(0, 1)
    primary["dt_seconds"] = primary["dt_frames"] / float(fps or 30.0)
    primary["vx"] = primary["dx"] / primary["dt_seconds"]
    primary["vy"] = primary["dy"] / primary["dt_seconds"]
    primary["distance"] = np.sqrt(primary["dx"] ** 2 + primary["dy"] ** 2)
    primary["speed_px_per_frame"] = primary["distance"] / primary["dt_frames"]
    primary["speed_px_per_sec"] = primary["distance"] / primary["dt_seconds"]
    primary["acceleration"] = primary["speed_px_per_sec"].diff() / primary["dt_seconds"]
    primary["heading"] = np.arctan2(primary["vy"], primary["vx"])

    primary["width"] = primary["bbox_width"]
    primary["height"] = primary["bbox_height"]
    primary["area"] = primary["bbox_area_px2"]
    primary["aspect_ratio"] = primary["width"] / primary["height"].replace(0, np.nan)
    primary["area_change"] = primary["mask_area_px2"].diff().fillna(0)
    primary["aspect_change"] = primary["aspect_ratio"].diff().fillna(0)

    primary["distance_mm"] = primary["distance"] * mm_per_pixel
    primary["speed_mm_per_sec"] = primary["speed_px_per_sec"] * mm_per_pixel
    primary["vx_mm"] = primary["vx"] * mm_per_pixel
    primary["vy_mm"] = primary["vy"] * mm_per_pixel
    primary["acceleration_mm_per_sec2"] = primary["acceleration"] * mm_per_pixel
    primary["width_mm"] = primary["width"] * mm_per_pixel
    primary["height_mm"] = primary["height"] * mm_per_pixel
    primary["area_mm2"] = primary["area"] * (mm_per_pixel**2)
    primary["cumulative_distance_mm"] = primary["distance_mm"].fillna(0).cumsum()
    primary["heading_deg"] = (-np.degrees(primary["heading"]) + 360) % 360

    frame_count = int(pd.to_numeric(raw["frame"], errors="coerce").dropna().nunique())
    valid_frames = set(primary["frame_index"].dropna().astype(int))
    primary["missing_frame_gap_before"] = primary["dt_frames"].fillna(1).clip(lower=1) - 1
    primary["segmentation_valid_frames"] = len(valid_frames)
    primary["segmentation_total_frames"] = frame_count
    return primary


def summarize_segmentation_features(
    primary: pd.DataFrame,
    detections: pd.DataFrame,
    raw: pd.DataFrame,
    fps: float,
    mm_per_pixel: float,
) -> dict[str, Any]:
    raw_frames = pd.to_numeric(raw["frame"], errors="coerce").dropna()
    total_frames = int(raw_frames.nunique()) if not raw_frames.empty else int(len(primary))
    valid_frames = int(primary["frame_index"].nunique())
    no_detection_frames = max(total_frames - valid_frames, 0)
    multi_detection_frames = int((detections.groupby("frame_index").size() > 1).sum())
    total_distance_mm = (
        float(primary["distance_mm"].sum(skipna=True)) if "distance_mm" in primary.columns else 0.0
    )
    return {
        "analysis_kind": "segmentation",
        "frames": int(len(primary)),
        "total_video_frames": total_frames,
        "valid_detection_frames": valid_frames,
        "no_detection_frames": no_detection_frames,
        "multi_detection_frames": multi_detection_frames,
        "detections": int(len(detections)),
        "fps": float(fps),
        "duration_s": float(raw_frames.max() / fps) if not raw_frames.empty and fps else 0.0,
        "mm_per_pixel": float(mm_per_pixel),
        "total_distance_mm": total_distance_mm,
        "total_distance_m": total_distance_mm / 1000.0,
        "average_speed_mm_per_sec": float(primary["speed_mm_per_sec"].mean(skipna=True)),
        "average_acceleration_mm_per_sec2": float(
            primary["acceleration_mm_per_sec2"].mean(skipna=True)
        ),
        "mean_confidence": float(detections["confidence"].mean(skipna=True)),
        "mean_mask_area_px2": float(detections["mask_area_px2"].mean(skipna=True)),
        "mean_mask_area_mm2": float(detections["mask_area_mm2"].mean(skipna=True)),
        "mean_mask_fill_ratio": float(detections["mask_fill_ratio"].mean(skipna=True)),
        "detection_coverage_fraction": float(valid_frames / total_frames) if total_frames else 0.0,
    }


def _line_plot(
    df: pd.DataFrame, x_col: str, y_col: str, path: Path, ylabel: str, title: str
) -> Optional[str]:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    columns = [x_col, y_col]
    if "time_seconds" in df.columns:
        columns.append("time_seconds")
    clean = df[columns].copy()
    clean[x_col] = pd.to_numeric(clean[x_col], errors="coerce")
    clean[y_col] = pd.to_numeric(clean[y_col], errors="coerce")
    clean = clean.dropna(subset=[x_col, y_col])
    if clean.empty:
        return None
    plt, sns = _setup_plotting()
    fig, ax = plt.subplots(figsize=(12, 4))

    if "time_seconds" in clean.columns:
        elapsed = pd.to_numeric(clean["time_seconds"], errors="coerce")
        use_time = elapsed.notna().any()
    else:
        elapsed = pd.Series(dtype=float)
        use_time = False
    x_values = elapsed / 60.0 if use_time else clean[x_col]
    x_label = "Elapsed time (min)" if use_time else "Frame"

    max_plot_points = 4000
    step = max(1, int(math.ceil(len(clean) / max_plot_points)))
    sampled = slice(None, None, step)
    if len(clean) > 2000:
        ax.plot(
            x_values.iloc[sampled],
            clean[y_col].iloc[sampled],
            color="#4c72b0",
            alpha=0.22,
            linewidth=0.65,
            label="Per-frame",
        )
        if use_time:
            time_step = elapsed.diff().dropna()
            time_step = time_step[time_step > 0]
            rolling_window = (
                max(3, int(round(1.0 / float(time_step.median())))) if not time_step.empty else 30
            )
            rolling_label = "1 s rolling mean"
        else:
            rolling_window = max(5, min(60, len(clean) // 300))
            rolling_label = f"{rolling_window}-frame rolling mean"
        rolling = clean[y_col].rolling(rolling_window, center=True, min_periods=1).mean()
        ax.plot(
            x_values.iloc[sampled],
            rolling.iloc[sampled],
            color="#1f5f99",
            linewidth=1.35,
            label=rolling_label,
        )
        ax.legend(frameon=False, loc="upper right")
    else:
        ax.plot(x_values, clean[y_col], color="#4c72b0", linewidth=1.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _confidence_by_detection_index_plot(detections: pd.DataFrame, path: Path) -> Optional[str]:
    required = {"frame_index", "confidence", "detection_index"}
    if not required.issubset(detections.columns):
        return None

    clean = detections[["frame_index", "confidence", "detection_index"]].dropna()
    if clean.empty:
        return None

    clean = clean.copy()
    clean["detection_index"] = pd.to_numeric(clean["detection_index"], errors="coerce")
    clean = clean.dropna(subset=["detection_index"])
    if clean.empty:
        return None
    clean["detection_index"] = clean["detection_index"].astype(int)

    plt, sns = _setup_plotting()
    fig, ax = plt.subplots(figsize=(12, 4))
    unique_detections = sorted(clean["detection_index"].unique())
    palette = sns.color_palette("tab10", n_colors=max(len(unique_detections), 1))
    color_by_detection = dict(zip(unique_detections, palette))

    for detection_index in unique_detections:
        group = clean.loc[clean["detection_index"] == detection_index]
        ax.scatter(
            group["frame_index"],
            group["confidence"],
            s=6,
            alpha=0.72,
            linewidths=0,
            color=color_by_detection[detection_index],
            label=f"det {detection_index}",
        )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Confidence")
    ax.set_title("Mask Confidence by Frame and Detection Index")
    ax.legend(title="Detection index", loc="upper right", frameon=False, markerscale=2)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def create_segmentation_plots(
    primary: pd.DataFrame,
    detections: pd.DataFrame,
    output_dir: Path,
    *,
    video_path: str = "",
    rois: Any = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, sns = _setup_plotting()
    paths: list[str] = []

    confidence_plot = _confidence_by_detection_index_plot(
        detections, output_dir / "segmentation_confidence.png"
    )
    if confidence_plot:
        paths.append(confidence_plot)

    for y_col, ylabel, title, filename in [
        (
            "mask_area_px2",
            "Area (px^2)",
            "Primary Mask Area by Frame",
            "segmentation_mask_area_px2.png",
        ),
        (
            "mask_fill_ratio",
            "Mask / bbox area",
            "Primary Mask Fill Ratio by Frame",
            "segmentation_fill_ratio.png",
        ),
        (
            "speed_mm_per_sec",
            "Speed (mm/s)",
            "Segmentation Centroid Speed",
            "segmentation_speed_mm_per_sec.png",
        ),
        (
            "distance_mm",
            "Distance (mm)",
            "Segmentation Centroid Distance",
            "segmentation_distance_mm.png",
        ),
    ]:
        plot_path = _line_plot(primary, "frame_index", y_col, output_dir / filename, ylabel, title)
        if plot_path:
            paths.append(plot_path)

    if "detections_in_frame" in primary.columns:
        plot_path = _line_plot(
            primary,
            "frame_index",
            "detections_in_frame",
            output_dir / "segmentation_detections_per_frame.png",
            "Detections",
            "Segmentation Detections per Frame",
        )
        if plot_path:
            paths.append(plot_path)

    if "mask_area_px2" in detections.columns:
        clean = detections["mask_area_px2"].dropna()
        if not clean.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(clean, bins=50, ax=ax)
            ax.set_title("Mask Area Distribution")
            ax.set_xlabel("Area (px^2)")
            ax.set_ylabel("Count")
            sns.despine(fig)
            fig.tight_layout()
            path = output_dir / "segmentation_mask_area_distribution.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(str(path))

    if {"bbox_center_x_euro", "bbox_center_y_euro", "frame_index"}.issubset(primary.columns):
        frame = _first_video_frame(video_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        if frame is not None:
            ax.imshow(frame)
        scatter = ax.scatter(
            primary["bbox_center_x_euro"],
            primary["bbox_center_y_euro"],
            c=primary["frame_index"],
            cmap="viridis",
            s=2,
        )
        _draw_roi_overlays(ax, rois or [])
        ax.set_title("Segmentation Centroid Trajectory")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, label="Frame")
        fig.tight_layout()
        path = output_dir / "segmentation_trajectory.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        clean = primary.dropna(subset=["bbox_center_x_euro", "bbox_center_y_euro"])
        if not clean.empty:
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
            else:
                frame_width = int(max(clean["bbox_x2"].max(), clean["bbox_center_x_euro"].max(), 1))
                frame_height = int(
                    max(clean["bbox_y2"].max(), clean["bbox_center_y_euro"].max(), 1)
                )
            heatmap, xedges, yedges = np.histogram2d(
                clean["bbox_center_x_euro"],
                clean["bbox_center_y_euro"],
                bins=50,
                range=[[0, frame_width], [0, frame_height]],
            )
            fig, ax = plt.subplots(figsize=(8, 8))
            if frame is not None:
                ax.imshow(frame)
            ax.imshow(
                heatmap.T,
                cmap="jet",
                alpha=0.65,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin="lower",
                interpolation="bilinear",
            )
            _draw_roi_overlays(ax, rois or [])
            ax.invert_yaxis()
            ax.set_title("Segmentation Centroid Occupancy Heatmap")
            ax.set_xlabel("X position (pixels)")
            ax.set_ylabel("Y position (pixels)")
            fig.tight_layout()
            path = output_dir / "segmentation_occupancy_heatmap.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(str(path))

    return paths


def _mask_area_overlay_text(row: Any) -> str:
    area_val = row.get("mask_area_mm2")
    if pd.isna(area_val):
        return ""
    return f"Mask area: {float(area_val):.1f} mm^2"


def render_segmentation_annotated_video(
    primary: pd.DataFrame,
    video_path: str,
    output_path: Path,
    fps: float,
    rois: Any = None,
) -> Optional[str]:
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        import cv2
    except Exception as exc:
        raise AnalysisError(f"OpenCV is required for annotated video export: {exc}") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    rows_by_frame = {
        int(row["frame_index"]): row
        for _, row in primary.iterrows()
        if not pd.isna(row.get("frame_index"))
    }
    normalized_rois = normalize_rois(rois or [])

    frame_idx = 0
    try:
        with _open_h264_video_writer(output_path, video_fps, width, height) as writer:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                for roi in normalized_rois:
                    x1, y1, x2, y2 = [
                        int(round(float(roi[key]))) for key in ("x1", "y1", "x2", "y2")
                    ]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (66, 191, 245), 2)
                    label = str(roi["name"])
                    (label_width, label_height), _baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2,
                    )
                    label_x = max(2, min(width - label_width - 2, (x1 + x2 - label_width) // 2))
                    label_y = max(
                        label_height + 2,
                        min(height - 2, (y1 + y2 + label_height) // 2),
                    )
                    cv2.putText(
                        frame,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (17, 24, 32),
                        4,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (66, 191, 245),
                        2,
                    )

                row = rows_by_frame.get(frame_idx)
                if row is not None:
                    bbox_vals = [
                        row.get("bbox_x1"),
                        row.get("bbox_y1"),
                        row.get("bbox_x2"),
                        row.get("bbox_y2"),
                    ]
                    if not any(pd.isna(v) for v in bbox_vals):
                        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_vals]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    points = _parse_polygon(row.get("mask_polygon"))
                    if len(points) >= 3:
                        contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            frame, [contour], isClosed=True, color=(0, 255, 255), thickness=2
                        )

                    cx = row.get("bbox_center_x_euro")
                    cy = row.get("bbox_center_y_euro")
                    if not pd.isna(cx) and not pd.isna(cy):
                        cv2.circle(
                            frame,
                            (int(round(float(cx))), int(round(float(cy)))),
                            4,
                            (0, 0, 255),
                            -1,
                        )

                    text = f"Frame: {frame_idx}"
                    speed_val = row.get("speed_mm_per_sec")
                    if not pd.isna(speed_val):
                        text += f" | Speed: {float(speed_val):.1f} mm/s"
                    cv2.putText(
                        frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                    area_text = _mask_area_overlay_text(row)
                    if area_text:
                        cv2.putText(
                            frame,
                            area_text,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                    roi_label = row.get("roi_label")
                    if isinstance(roi_label, str) and roi_label:
                        cv2.putText(
                            frame,
                            f"ROI: {roi_label}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                writer.write(frame)
                frame_idx += 1
    finally:
        cap.release()
    return str(output_path)


def run_segmentation_analysis_workflow(
    config: AnalysisConfig,
    progress_callback: ProgressCallback = None,
    *,
    raw: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    if not config.detections_csv or not os.path.isfile(config.detections_csv):
        raise AnalysisError("Select a valid segmentation CSV.")
    if not config.output_dir:
        raise AnalysisError("Select an output directory.")

    total_steps = 8
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw is None:
        _progress(progress_callback, 1, total_steps, "Loading segmentation CSV")
        raw = pd.read_csv(config.detections_csv).dropna(axis=1, how="all")
    if not is_segmentation_inference_csv(raw):
        missing = [col for col in SEGMENTATION_REQUIRED_COLUMNS if col not in raw.columns]
        raise AnalysisError(f"Segmentation CSV is missing required columns: {', '.join(missing)}")

    video_path = config.video_path.strip()
    fps = _infer_fps(raw.rename(columns={"frame": "frame_index"}), video_path, config.fps)
    scale = _mm_per_pixel(config)
    rois = normalize_rois(config.rois)

    _progress(progress_callback, 2, total_steps, "Parsing segmentation masks")
    detections = compute_segmentation_detection_features(raw, scale)

    _progress(progress_callback, 3, total_steps, "Computing segmentation motion features")
    primary = compute_segmentation_track_features(detections, raw, fps, scale, config)
    if rois:
        _progress(progress_callback, 4, total_steps, "Applying ROI annotations")
        primary = assign_roi_labels(
            primary, rois, x_col="bbox_center_x_euro", y_col="bbox_center_y_euro"
        )
    else:
        _progress(progress_callback, 4, total_steps, "No ROI annotations selected")

    summary = summarize_segmentation_features(primary, detections, raw, fps, scale)
    summary["layer_id"] = "segmentation"
    summary["detections_csv"] = os.path.abspath(config.detections_csv)
    summary["video_path"] = os.path.abspath(video_path) if video_path else ""
    summary["roi_count"] = len(rois)
    summary["rois"] = rois

    roi_outputs: dict[str, Any] = {}
    if rois:
        roi_outputs = create_roi_outputs(primary, output_dir, fps)
        summary["roi_summary"] = roi_outputs.get("roi_summary", [])

    _progress(progress_callback, 5, total_steps, "Writing segmentation feature tables")
    feature_csv = output_dir / "analysis_features.csv"
    detections_csv = output_dir / "segmentation_detections.csv"
    summary_json = output_dir / "analysis_summary.json"
    primary.to_csv(feature_csv, index=False)
    detections.to_csv(detections_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    plot_paths: list[str] = []
    if config.make_plots:
        _progress(progress_callback, 6, total_steps, "Rendering segmentation plots")
        plot_paths = create_segmentation_plots(
            primary, detections, output_dir / "plots", video_path=video_path, rois=rois
        )
    else:
        _progress(progress_callback, 6, total_steps, "Skipping plots")

    annotated_video = None
    if config.make_annotated_video:
        _progress(progress_callback, 7, total_steps, "Rendering segmentation annotated video")
        annotated_video = render_segmentation_annotated_video(
            primary, video_path, output_dir / "annotated_output.mp4", fps, rois=rois
        )
    else:
        _progress(progress_callback, 7, total_steps, "Skipping annotated video")

    cluster_plot_paths: list[str] = []
    cluster_clip_paths: list[str] = []
    if config.run_clustering:
        _progress(progress_callback, 8, total_steps, "Running behavior clustering")
        primary, cluster_plot_paths = run_behavior_clustering(
            primary,
            fps,
            output_dir / "plots",
            umap_neighbors=config.umap_neighbors,
            umap_min_dist=config.umap_min_dist,
            hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
        )
        primary.to_csv(feature_csv, index=False)
        if config.export_cluster_clips:
            cluster_clip_paths = export_cluster_clips(
                primary,
                video_path,
                output_dir,
                fps,
                config.cluster_clip_length_sec,
                config.samples_per_cluster,
            )
    else:
        _progress(progress_callback, 8, total_steps, "Segmentation analysis complete")

    return {
        "feature_csv": str(feature_csv),
        "segmentation_detections_csv": str(detections_csv),
        "summary_json": str(summary_json),
        "summary": summary,
        "plot_paths": plot_paths + roi_outputs.get("roi_plot_paths", []) + cluster_plot_paths,
        "annotated_video": annotated_video or "",
        "cluster_clip_paths": cluster_clip_paths,
        "output_dir": str(output_dir),
        "roi_summary_csv": roi_outputs.get("roi_summary_csv", ""),
        "roi_transition_csv": roi_outputs.get("roi_transition_csv", ""),
        "roi_summary": roi_outputs.get("roi_summary", []),
    }
