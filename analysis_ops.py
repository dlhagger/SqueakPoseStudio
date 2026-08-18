"""Reusable analysis workflow for SqueakPose inference CSV outputs."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from squeakpose.core import commit_staged_paths, remove_path, staging_path_for
from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION, normalize_layer_id
from squeakpose.services.analysis import DEFAULT_ONE_EURO_BETA, DEFAULT_ONE_EURO_MIN_CUTOFF

ProgressCallback = Optional[Callable[[int, int, str], None]]


class AnalysisError(RuntimeError):
    """Raised when an analysis workflow cannot be completed."""


@dataclass
class AnalysisConfig:
    detections_csv: str
    output_dir: str
    layer_id: str = ""
    video_path: str = ""
    fps: float = 0.0
    pixel_distance: float = 1.0
    real_world_distance_mm: float = 1.0
    smooth: bool = True
    min_cutoff: float = DEFAULT_ONE_EURO_MIN_CUTOFF
    beta: float = DEFAULT_ONE_EURO_BETA
    d_cutoff: float = 1.0
    make_plots: bool = True
    make_annotated_video: bool = False
    run_clustering: bool = False
    export_cluster_clips: bool = False
    umap_neighbors: int = 0
    umap_min_dist: float = 0.3
    hdbscan_min_cluster_size: int = 0
    cluster_clip_length_sec: float = 2.0
    samples_per_cluster: int = 1
    rois: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AnalysisConfig":
        return cls(
            detections_csv=str(raw.get("detections_csv") or ""),
            output_dir=str(raw.get("output_dir") or ""),
            layer_id=(normalize_layer_id(raw.get("layer_id")) if raw.get("layer_id") else ""),
            video_path=str(raw.get("video_path") or ""),
            fps=float(raw.get("fps") or 0.0),
            pixel_distance=float(raw.get("pixel_distance") or 1.0),
            real_world_distance_mm=float(raw.get("real_world_distance_mm") or 1.0),
            smooth=bool(raw.get("smooth", True)),
            min_cutoff=float(raw.get("min_cutoff") or DEFAULT_ONE_EURO_MIN_CUTOFF),
            beta=float(raw.get("beta") if raw.get("beta") is not None else DEFAULT_ONE_EURO_BETA),
            d_cutoff=float(raw.get("d_cutoff") or 1.0),
            make_plots=bool(raw.get("make_plots", True)),
            make_annotated_video=bool(raw.get("make_annotated_video", False)),
            run_clustering=bool(raw.get("run_clustering", False)),
            export_cluster_clips=bool(raw.get("export_cluster_clips", False)),
            umap_neighbors=max(0, int(raw.get("umap_neighbors") or 0)),
            umap_min_dist=float(
                raw.get("umap_min_dist") if raw.get("umap_min_dist") is not None else 0.3
            ),
            hdbscan_min_cluster_size=max(0, int(raw.get("hdbscan_min_cluster_size") or 0)),
            cluster_clip_length_sec=float(raw.get("cluster_clip_length_sec") or 2.0),
            samples_per_cluster=max(1, int(raw.get("samples_per_cluster") or 1)),
            rois=normalize_rois(raw.get("rois") or []),
        )


def _progress(callback: ProgressCallback, step: int, total: int, message: str) -> None:
    if callback is not None:
        callback(int(step), int(total), str(message))


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise AnalysisError(f"Detections CSV is missing required columns: {', '.join(missing)}")


def _first_existing_video_path(df: pd.DataFrame) -> str:
    if "video_path" not in df.columns:
        return ""
    for value in df["video_path"].dropna().unique():
        path = str(value).strip()
        if path and os.path.isfile(path):
            return path
    return ""


def _read_video_metadata(video_path: str) -> dict[str, float]:
    if not video_path or not os.path.isfile(video_path):
        return {}
    try:
        import cv2
    except Exception:
        return {}
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return {}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        return {"fps": fps, "width": width, "height": height, "frames": frames}
    finally:
        cap.release()


def _infer_fps(df: pd.DataFrame, video_path: str, requested_fps: float) -> float:
    if requested_fps > 0:
        return float(requested_fps)
    meta = _read_video_metadata(video_path)
    if meta.get("fps", 0.0) > 0:
        return float(meta["fps"])
    if "time_seconds" in df.columns:
        dt = pd.to_numeric(df["time_seconds"], errors="coerce").diff().dropna()
        dt = dt[dt > 0]
        if not dt.empty:
            return float(1.0 / dt.median())
    return 30.0


def _mm_per_pixel(config: AnalysisConfig) -> float:
    if config.pixel_distance > 0 and config.real_world_distance_mm > 0:
        return float(config.real_world_distance_mm / config.pixel_distance)
    return 1.0


def normalize_rois(raw_rois: Any) -> list[dict[str, Any]]:
    """Return validated rectangular ROIs in image-pixel coordinates."""
    if not isinstance(raw_rois, list):
        return []
    rois: list[dict[str, Any]] = []
    for raw in raw_rois:
        if not isinstance(raw, dict):
            continue
        roi_type = str(raw.get("type") or "rect").strip().lower()
        if roi_type != "rect":
            continue
        try:
            x1 = float(raw.get("x1"))
            y1 = float(raw.get("y1"))
            x2 = float(raw.get("x2"))
            y2 = float(raw.get("y2"))
        except (TypeError, ValueError):
            continue
        if any(math.isnan(value) for value in (x1, y1, x2, y2)):
            continue
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left <= 0 or bottom - top <= 0:
            continue
        name = str(raw.get("name") or f"ROI {len(rois) + 1}").strip() or f"ROI {len(rois) + 1}"
        rois.append(
            {
                "name": name,
                "type": "rect",
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom,
            }
        )
    return rois


def assign_roi_labels(
    df: pd.DataFrame,
    rois: list[dict[str, Any]],
    *,
    x_col: str = "bbox_center_x_euro",
    y_col: str = "bbox_center_y_euro",
) -> pd.DataFrame:
    """Annotate each detection row with the rectangular ROI containing its center."""
    out = df.copy()
    out["roi_label"] = "Outside"
    normalized = normalize_rois(rois)
    if not normalized or x_col not in out.columns or y_col not in out.columns:
        return out

    x_values = pd.to_numeric(out[x_col], errors="coerce")
    y_values = pd.to_numeric(out[y_col], errors="coerce")
    for roi in normalized:
        mask = x_values.between(
            float(roi["x1"]), float(roi["x2"]), inclusive="both"
        ) & y_values.between(float(roi["y1"]), float(roi["y2"]), inclusive="both")
        out.loc[mask, "roi_label"] = roi["name"]
    return out


def _smooth_centers(df: pd.DataFrame, fps: float, config: AnalysisConfig) -> pd.DataFrame:
    out = df.copy()
    if not config.smooth:
        out["bbox_center_x_euro"] = out["bbox_center_x"]
        out["bbox_center_y_euro"] = out["bbox_center_y"]
        return out

    try:
        from OneEuroFilter import OneEuroFilter
    except Exception as exc:
        raise AnalysisError(f"Could not import OneEuroFilter: {exc}") from exc

    freq = max(float(fps), 1.0)
    euro_cx = OneEuroFilter(freq, config.min_cutoff, config.beta, config.d_cutoff)
    euro_cy = OneEuroFilter(freq, config.min_cutoff, config.beta, config.d_cutoff)

    filtered_cx: list[float] = []
    filtered_cy: list[float] = []
    for cx, cy in zip(out["bbox_center_x"], out["bbox_center_y"]):
        if pd.isna(cx) or pd.isna(cy):
            filtered_cx.append(math.nan)
            filtered_cy.append(math.nan)
            euro_cx.reset()
            euro_cy.reset()
            continue
        filtered_cx.append(float(euro_cx(float(cx))))
        filtered_cy.append(float(euro_cy(float(cy))))
    out["bbox_center_x_euro"] = filtered_cx
    out["bbox_center_y_euro"] = filtered_cy
    return out


def _prepare_single_detection_per_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["frame_index"]
    ascending = [True]
    if "confidence" in df.columns:
        sort_cols.append("confidence")
        ascending.append(False)
    return (
        df.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(subset="frame_index", keep="first")
        .reset_index(drop=True)
    )


def compute_features(df: pd.DataFrame, fps: float, mm_per_pixel: float) -> pd.DataFrame:
    """Compute the notebook's core per-frame kinematic features."""
    _require_columns(df, ["frame_index", "bbox_center_x", "bbox_center_y"])
    detections = _prepare_single_detection_per_frame(df)

    if "time_seconds" not in detections.columns:
        detections["time_seconds"] = detections["frame_index"] / float(fps)
    detections["frame_index"] = pd.to_numeric(detections["frame_index"], errors="coerce")
    detections["time_seconds"] = pd.to_numeric(detections["time_seconds"], errors="coerce")

    for col in ("bbox_center_x_euro", "bbox_center_y_euro"):
        if col not in detections.columns:
            raw_col = col.replace("_euro", "")
            detections[col] = detections[raw_col]

    detections["dx"] = detections["bbox_center_x_euro"].diff()
    detections["dy"] = detections["bbox_center_y_euro"].diff()

    frame_delta = detections["frame_index"].diff()
    detections["dt_frames"] = frame_delta.fillna(1).replace(0, 1)
    default_dt = 1.0 / float(fps) if fps else np.nan
    time_delta = detections["time_seconds"].diff()
    dt_seconds = time_delta.where(time_delta > 0, detections["dt_frames"] * default_dt)
    detections["dt_seconds"] = dt_seconds.fillna(detections["dt_frames"] * default_dt)

    detections["vx"] = detections["dx"] / detections["dt_seconds"]
    detections["vy"] = detections["dy"] / detections["dt_seconds"]
    detections["distance"] = np.sqrt(detections["dx"] ** 2 + detections["dy"] ** 2)
    detections["speed_px_per_frame"] = detections["distance"] / detections["dt_frames"]
    detections["speed_px_per_sec"] = detections["distance"] / detections["dt_seconds"]
    detections["acceleration"] = detections["speed_px_per_sec"].diff() / detections["dt_seconds"]
    detections["heading"] = np.arctan2(detections["vy"], detections["vx"])

    if "bbox_width" in detections.columns:
        detections["width"] = detections["bbox_width"]
    elif {"bbox_x1", "bbox_x2"}.issubset(detections.columns):
        detections["width"] = detections["bbox_x2"] - detections["bbox_x1"]
    else:
        detections["width"] = np.nan

    if "bbox_height" in detections.columns:
        detections["height"] = detections["bbox_height"]
    elif {"bbox_y1", "bbox_y2"}.issubset(detections.columns):
        detections["height"] = detections["bbox_y2"] - detections["bbox_y1"]
    else:
        detections["height"] = np.nan

    detections["aspect_ratio"] = detections["width"] / detections["height"]
    detections["area"] = detections["width"] * detections["height"]
    detections["area_change"] = detections["area"].diff().fillna(0)
    detections["aspect_change"] = detections["aspect_ratio"].diff().fillna(0)

    detections["distance_mm"] = detections["distance"] * mm_per_pixel
    detections["speed_mm_per_sec"] = detections["speed_px_per_sec"] * mm_per_pixel
    detections["vx_mm"] = detections["vx"] * mm_per_pixel
    detections["vy_mm"] = detections["vy"] * mm_per_pixel
    detections["acceleration_mm_per_sec2"] = detections["acceleration"] * mm_per_pixel
    detections["width_mm"] = detections["width"] * mm_per_pixel
    detections["height_mm"] = detections["height"] * mm_per_pixel
    detections["area_mm2"] = detections["area"] * (mm_per_pixel**2)
    detections["cumulative_distance_mm"] = detections["distance_mm"].fillna(0).cumsum()
    detections["heading_deg"] = (-np.degrees(detections["heading"]) + 360) % 360
    return detections


def summarize_features(df: pd.DataFrame, fps: float, mm_per_pixel: float) -> dict[str, Any]:
    duration_s = (
        float(df["time_seconds"].max()) if "time_seconds" in df.columns and len(df) else 0.0
    )
    total_distance_mm = (
        float(df["distance_mm"].sum(skipna=True)) if "distance_mm" in df.columns else 0.0
    )
    avg_speed = (
        float(df["speed_mm_per_sec"].mean(skipna=True))
        if "speed_mm_per_sec" in df.columns
        else math.nan
    )
    avg_accel = (
        float(df["acceleration_mm_per_sec2"].mean(skipna=True))
        if "acceleration_mm_per_sec2" in df.columns
        else math.nan
    )
    return {
        "frames": int(len(df)),
        "fps": float(fps),
        "duration_s": duration_s,
        "mm_per_pixel": float(mm_per_pixel),
        "total_distance_mm": total_distance_mm,
        "total_distance_m": total_distance_mm / 1000.0,
        "average_speed_mm_per_sec": avg_speed,
        "average_acceleration_mm_per_sec2": avg_accel,
        "mean_confidence": float(df["confidence"].mean(skipna=True))
        if "confidence" in df.columns
        else math.nan,
    }


def _setup_plotting():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _first_video_frame(video_path: str):
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


class _PyAVH264VideoWriter:
    """Stream BGR frames to an isolated, atomic PyAV H.264 export."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        self.output_path = os.path.abspath(os.fspath(output_path))
        self.source_width = int(width)
        self.source_height = int(height)
        self.encoded_width = self.source_width + self.source_width % 2
        self.encoded_height = self.source_height + self.source_height % 2
        self.staged_path = staging_path_for(self.output_path)
        self.process = None
        self.stderr_file = tempfile.TemporaryFile(mode="w+b")
        self.closed = False

        try:
            worker_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "squeakpose",
                "workers",
                "video_encoder.py",
            )
            command = [
                sys.executable,
                worker_path,
                "--output",
                self.staged_path,
                "--fps",
                repr(float(fps)),
                "--width",
                str(self.source_width),
                "--height",
                str(self.source_height),
            ]
            popen_kwargs: dict[str, Any] = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                bufsize=0,
                **popen_kwargs,
            )
            ready = self.process.stdout.readline() if self.process.stdout is not None else b""
            if ready != b"READY\n":
                self.process.wait()
                raise AnalysisError(self._worker_error("PyAV encoder failed to start"))
        except Exception as exc:
            self._discard()
            if isinstance(exc, AnalysisError):
                raise
            raise AnalysisError(
                f"Could not open the PyAV H.264 encoder for {output_path}: {exc}"
            ) from exc

    def __enter__(self) -> "_PyAVH264VideoWriter":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self._discard()
        return False

    def write(self, frame: Any) -> None:
        if self.closed or self.process is None or self.process.stdin is None:
            raise AnalysisError("Cannot write to a closed video export.")
        array = np.asarray(frame)
        expected_shape = (self.source_height, self.source_width, 3)
        if array.shape != expected_shape or array.dtype != np.uint8:
            raise AnalysisError(
                "Video frame does not match the export format: "
                f"expected uint8 BGR {expected_shape}, got {array.dtype} {array.shape}."
            )
        try:
            self.process.stdin.write(array.tobytes(order="C"))
        except (BrokenPipeError, OSError) as exc:
            self.process.wait()
            raise AnalysisError(self._worker_error(f"Could not encode video frame: {exc}")) from exc

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.process is None or self.process.stdin is None:
                raise AnalysisError("The video encoder was not initialized.")
            self.process.stdin.close()
            return_code = self.process.wait()
            if return_code != 0:
                raise AnalysisError(self._worker_error("PyAV could not finalize the video export"))
            self._close_process_handles()
            commit_staged_paths([(self.staged_path, self.output_path)])
            self.closed = True
        except Exception as exc:
            self._discard()
            if isinstance(exc, AnalysisError):
                raise
            raise AnalysisError(
                f"Could not finalize video export {self.output_path}: {exc}"
            ) from exc

    def _discard(self) -> None:
        if self.closed:
            return
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception:
                    pass
        self._close_process_handles()
        try:
            remove_path(self.staged_path)
        except Exception:
            pass
        self.closed = True

    def _worker_error(self, fallback: str) -> str:
        try:
            self.stderr_file.flush()
            self.stderr_file.seek(0)
            detail = self.stderr_file.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        return detail or fallback

    def _close_process_handles(self) -> None:
        if self.process is not None:
            for handle in (self.process.stdin, self.process.stdout):
                if handle is not None and not handle.closed:
                    try:
                        handle.close()
                    except Exception:
                        pass
        if not self.stderr_file.closed:
            self.stderr_file.close()


def _open_h264_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> _PyAVH264VideoWriter:
    """Open the bundled PyAV/libx264 encoder or fail with an actionable error."""
    if width <= 0 or height <= 0:
        raise AnalysisError(f"Cannot export video with invalid frame size {width}x{height}.")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise AnalysisError(f"Cannot export video with invalid frame rate {fps}.")
    return _PyAVH264VideoWriter(output_path, fps, width, height)


def _plot_if_column(
    df: pd.DataFrame, x_col: str, y_col: str, path: Path, ylabel: str, title: str
) -> Optional[str]:
    if y_col not in df.columns or x_col not in df.columns:
        return None
    plt, sns = _setup_plotting()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _draw_roi_overlays(ax, rois: list[dict[str, Any]]) -> None:
    normalized = normalize_rois(rois)
    if not normalized:
        return
    import matplotlib.patches as patches

    for roi in normalized:
        x1 = float(roi["x1"])
        y1 = float(roi["y1"])
        width = float(roi["x2"]) - x1
        height = float(roi["y2"]) - y1
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=1.6,
            edgecolor="#f5b942",
            facecolor="none",
            alpha=0.95,
        )
        ax.add_patch(rect)
        ax.text(
            x1 + width / 2.0,
            y1 + height / 2.0,
            str(roi["name"]),
            color="#111820",
            fontsize=8,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "#f5b942",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def create_roi_outputs(df: pd.DataFrame, output_dir: Path, fps: float) -> dict[str, Any]:
    """Write ROI occupancy/transition summaries and plots."""
    if "roi_label" not in df.columns:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_df = df.copy()
    roi_df["roi_label"] = roi_df["roi_label"].fillna("Outside").astype(str)
    if "dt_seconds" in roi_df.columns:
        duration = pd.to_numeric(roi_df["dt_seconds"], errors="coerce").fillna(0.0)
    else:
        duration = pd.Series(1.0 / float(fps or 30.0), index=roi_df.index)
    roi_df["_roi_duration_s"] = duration

    summary = (
        roi_df.groupby("roi_label", dropna=False)
        .agg(
            frames=("frame_index", "count"),
            duration_s=("_roi_duration_s", "sum"),
            total_distance_mm=("distance_mm", "sum"),
            average_speed_mm_per_sec=("speed_mm_per_sec", "mean"),
        )
        .reset_index()
        .sort_values(["duration_s", "frames"], ascending=False)
    )
    summary_csv = output_dir / "roi_summary.csv"
    summary.to_csv(summary_csv, index=False)

    labels = roi_df["roi_label"].reset_index(drop=True)
    transition_csv = ""
    transition = pd.DataFrame()
    if len(labels) > 1:
        transition = pd.crosstab(
            labels.iloc[:-1].to_numpy(),
            labels.iloc[1:].to_numpy(),
            rownames=["from_roi"],
            colnames=["to_roi"],
        )
        transition_path = output_dir / "roi_transition_matrix.csv"
        transition.to_csv(transition_path)
        transition_csv = str(transition_path)

    plot_paths: list[str] = []
    plt, sns = _setup_plotting()
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, max(3, min(10, len(summary) * 0.45 + 1.5))))
        sns.barplot(data=summary, x="duration_s", y="roi_label", color="#4c72b0", ax=ax)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("ROI")
        ax.set_title("Time in ROI")
        sns.despine(fig)
        fig.tight_layout()
        path = output_dir / "roi_time_seconds.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_paths.append(str(path))

    if not transition.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(transition, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title("ROI Transitions")
        ax.set_xlabel("To ROI")
        ax.set_ylabel("From ROI")
        fig.tight_layout()
        path = output_dir / "roi_transition_matrix.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_paths.append(str(path))

    return {
        "roi_summary_csv": str(summary_csv),
        "roi_transition_csv": transition_csv,
        "roi_plot_paths": plot_paths,
        "roi_summary": _records_for_json(summary),
    }


def create_plots(
    df: pd.DataFrame, output_dir: Path, video_path: str = "", rois: Any = None
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, sns = _setup_plotting()
    paths: list[str] = []

    if "confidence" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.scatterplot(data=df, x="frame_index", y="confidence", edgecolor=None, s=5, ax=ax)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")
        ax.set_title("Detection Confidence by Frame")
        sns.despine(fig)
        fig.tight_layout()
        path = output_dir / "confidence_by_frame.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    speed_cols = [
        ("speed_preprocess_ms", "Preprocess"),
        ("speed_inference_ms", "Inference"),
        ("speed_postprocess_ms", "Postprocess"),
    ]
    existing_speed = [(col, label) for col, label in speed_cols if col in df.columns]
    if existing_speed:
        fig, axes = plt.subplots(
            1, len(existing_speed), figsize=(5 * len(existing_speed), 4), squeeze=False
        )
        for ax, (col, label) in zip(axes[0], existing_speed):
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            ax.boxplot(
                values,
                orientation="horizontal",
                patch_artist=True,
                boxprops={"facecolor": "#4c72b0", "alpha": 0.75},
                medianprops={"color": "white"},
            )
            ax.axvline(values.mean(), color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{label} Time per Frame")
            ax.set_xlabel("Milliseconds")
            ax.set_yticks([])
        fig.tight_layout()
        path = output_dir / "processing_speed_boxplots.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    for y_col, ylabel, title, filename in [
        ("distance_mm", "Distance (mm)", "Distance Traveled by Frame", "distance_mm.png"),
        ("speed_mm_per_sec", "Speed (mm/s)", "Speed by Frame", "speed_mm_per_sec.png"),
        (
            "acceleration_mm_per_sec2",
            "Acceleration (mm/s^2)",
            "Acceleration by Frame",
            "acceleration_mm_per_sec2.png",
        ),
    ]:
        plot_path = _plot_if_column(df, "frame_index", y_col, output_dir / filename, ylabel, title)
        if plot_path:
            paths.append(plot_path)

    if "heading" in df.columns:
        clean_heading = df["heading"].dropna()
        if not clean_heading.empty:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            ax.hist(clean_heading, bins=60, density=True)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_title("Heading Direction Distribution", y=1.1)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            fig.tight_layout()
            path = output_dir / "heading_polar.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(str(path))

    if {"bbox_center_x_euro", "bbox_center_y_euro", "frame_index"}.issubset(df.columns):
        frame = _first_video_frame(video_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        if frame is not None:
            ax.imshow(frame)
        else:
            width = int(df.get("image_width", pd.Series([0])).dropna().max() or 0)
            height = int(df.get("image_height", pd.Series([0])).dropna().max() or 0)
            if width > 0 and height > 0:
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
        scatter = ax.scatter(
            df["bbox_center_x_euro"],
            df["bbox_center_y_euro"],
            c=df["frame_index"],
            cmap="viridis",
            s=2,
        )
        _draw_roi_overlays(ax, rois or [])
        ax.set_title("Trajectory")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, label="Frame")
        fig.tight_layout()
        path = output_dir / "trajectory.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

        clean = df.dropna(subset=["bbox_center_x_euro", "bbox_center_y_euro"])
        if not clean.empty:
            frame_width = int(clean.get("image_width", pd.Series([0])).dropna().max() or 0)
            frame_height = int(clean.get("image_height", pd.Series([0])).dropna().max() or 0)
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
            if frame_width > 0 and frame_height > 0:
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
                ax.set_title("Occupancy Heatmap")
                ax.set_xlabel("X position (pixels)")
                ax.set_ylabel("Y position (pixels)")
                fig.tight_layout()
                path = output_dir / "occupancy_heatmap.png"
                fig.savefig(path, dpi=140)
                plt.close(fig)
                paths.append(str(path))

    return paths


def render_annotated_video(
    df: pd.DataFrame,
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
        for _, row in df.iterrows()
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
                    cv2.putText(
                        frame,
                        str(roi["name"]),
                        (x1 + 4, max(y1 + 18, 18)),
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
                    if not pd.isna(row.get("cumulative_distance_mm")):
                        text += f" | Distance: {float(row['cumulative_distance_mm']):.1f} mm"
                    cv2.putText(
                        frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    speed_val = row.get("speed_mm_per_sec")
                    if not pd.isna(speed_val):
                        cv2.putText(
                            frame,
                            f"Speed: {float(speed_val):.1f} mm/s",
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


def run_behavior_clustering(
    df: pd.DataFrame,
    fps: float,
    output_dir: Path,
    *,
    umap_neighbors: int = 0,
    umap_min_dist: float = 0.3,
    hdbscan_min_cluster_size: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    try:
        import hdbscan
        import umap
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise AnalysisError(f"UMAP/HDBSCAN clustering dependencies are unavailable: {exc}") from exc

    feature_cols = [
        "bbox_center_x",
        "bbox_center_y",
        "speed_mm_per_sec",
        "vx_mm",
        "vy_mm",
        "acceleration_mm_per_sec2",
        "distance_mm",
        "width_mm",
        "height_mm",
        "aspect_ratio",
        "area_mm2",
        "heading_deg",
    ]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise AnalysisError(f"Cannot cluster; missing feature columns: {', '.join(missing)}")

    window_frames = max(int(round(float(fps))), 1)
    clean_features = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    rolled = clean_features.rolling(
        window=window_frames, min_periods=window_frames, center=True
    ).agg(["mean", "std"])
    rolled.columns = [f"{col}_{stat}_1s" for col, stat in rolled.columns]
    behavior_df = rolled.dropna()
    if behavior_df.empty:
        raise AnalysisError(
            "Rolling-window feature table is empty; use a longer video or lower FPS override."
        )

    X_scaled = StandardScaler().fit_transform(behavior_df)
    n_neighbors = (
        int(umap_neighbors) if umap_neighbors > 0 else min(50, max(2, len(behavior_df) - 1))
    )
    n_neighbors = min(max(2, n_neighbors), max(2, len(behavior_df) - 1))
    min_dist = min(max(float(umap_min_dist), 0.0), 1.0)
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=1,
        metric="euclidean",
        random_state=42,
        n_components=2,
    ).fit_transform(X_scaled)

    if hdbscan_min_cluster_size > 0:
        min_cluster_size = min(max(2, int(hdbscan_min_cluster_size)), max(2, len(behavior_df)))
    else:
        min_cluster_size = min(40, max(5, len(behavior_df) // 10))
    cluster_labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min(10, max(1, min_cluster_size // 2)),
        metric="euclidean",
        cluster_selection_epsilon=0.3,
        cluster_selection_method="leaf",
    ).fit_predict(embedding)

    out = df.copy()
    out.loc[behavior_df.index, "behavior_cluster"] = cluster_labels
    out.loc[behavior_df.index, "umap_x"] = embedding[:, 0]
    out.loc[behavior_df.index, "umap_y"] = embedding[:, 1]

    plt, sns = _setup_plotting()
    plot_df = (
        out.loc[behavior_df.index]
        .dropna(subset=["behavior_cluster", "umap_x", "umap_y"])
        .loc[lambda frame: frame["behavior_cluster"] != -1]
        .assign(behavior_cluster=lambda frame: frame["behavior_cluster"].astype(int))
    )
    paths: list[str] = []
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            data=plot_df,
            x="umap_x",
            y="umap_y",
            hue="behavior_cluster",
            palette="tab10",
            s=5,
            linewidth=0,
            ax=ax,
        )
        ax.set_aspect("equal", "datalim")
        ax.set_title("Behavior Clusters (UMAP/HDBSCAN)")
        fig.tight_layout()
        path = output_dir / "behavior_clusters_umap.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return out, paths


def export_cluster_clips(
    df: pd.DataFrame,
    video_path: str,
    output_dir: Path,
    fps: float,
    clip_length_sec: float,
    samples_per_cluster: int,
) -> list[str]:
    if "behavior_cluster" not in df.columns or not video_path or not os.path.isfile(video_path):
        return []
    try:
        import cv2
    except Exception as exc:
        raise AnalysisError(f"OpenCV is required for cluster clip export: {exc}") from exc

    clip_dir = output_dir / "cluster_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    frames_per_clip = max(1, int(round(clip_length_sec * video_fps)))
    frame_count = int(df["frame_index"].max() + 1)
    paths: list[str] = []
    try:
        clusters = sorted(
            c for c in df["behavior_cluster"].dropna().astype(int).unique() if c != -1
        )
        for cluster_id in clusters:
            frames = (
                df.loc[df["behavior_cluster"] == cluster_id, "frame_index"]
                .dropna()
                .astype(int)
                .values
            )
            if len(frames) == 0:
                continue
            sample_indices = np.linspace(
                0, len(frames) - 1, num=min(samples_per_cluster, len(frames)), dtype=int
            )
            for clip_idx, frame_idx in enumerate(frames[sample_indices], start=1):
                start_frame = max(int(frame_idx), 0)
                end_frame = min(start_frame + frames_per_clip, frame_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                clip_path = clip_dir / f"cluster_{cluster_id:02d}_sample_{clip_idx:02d}.mp4"
                with _open_h264_video_writer(clip_path, video_fps, width, height) as writer:
                    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(frame)
                paths.append(str(clip_path))
    finally:
        cap.release()
    return paths


def run_analysis_workflow(
    config: AnalysisConfig, progress_callback: ProgressCallback = None
) -> dict[str, Any]:
    if not config.detections_csv or not os.path.isfile(config.detections_csv):
        raise AnalysisError("Select a valid detections CSV.")
    if not config.output_dir:
        raise AnalysisError("Select an output directory.")

    total_steps = 8
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress_callback, 1, total_steps, "Loading detections CSV")
    raw = pd.read_csv(config.detections_csv).dropna(axis=1, how="all")
    from segmentation_analysis_ops import (
        is_segmentation_inference_csv,
        run_segmentation_analysis_workflow,
    )

    detected_layer = LAYER_SEGMENTATION if is_segmentation_inference_csv(raw) else LAYER_KEYPOINTS
    if config.layer_id and config.layer_id != detected_layer:
        raise AnalysisError(
            f"The selected CSV contains {detected_layer} layer results, "
            f"but this analysis was opened for the {config.layer_id} layer."
        )
    if detected_layer == LAYER_SEGMENTATION:
        result = run_segmentation_analysis_workflow(
            config, progress_callback=progress_callback, raw=raw
        )
        result["layer_id"] = detected_layer
        if isinstance(result.get("summary"), dict):
            result["summary"]["layer_id"] = detected_layer
        return result

    _require_columns(raw, ["frame_index", "bbox_center_x", "bbox_center_y"])

    video_path = config.video_path.strip() or _first_existing_video_path(raw)
    fps = _infer_fps(raw, video_path, config.fps)
    scale = _mm_per_pixel(config)

    _progress(progress_callback, 2, total_steps, "Smoothing detection centers")
    ordered = raw.sort_values("frame_index").reset_index(drop=True)
    smoothed = _smooth_centers(ordered, fps, config)

    _progress(progress_callback, 3, total_steps, "Computing analysis features")
    features = compute_features(smoothed, fps, scale)
    rois = normalize_rois(config.rois)
    if rois:
        _progress(progress_callback, 4, total_steps, "Applying ROI annotations")
        features = assign_roi_labels(features, rois)
    else:
        _progress(progress_callback, 4, total_steps, "No ROI annotations selected")
    summary = summarize_features(features, fps, scale)
    summary["layer_id"] = LAYER_KEYPOINTS
    summary["detections_csv"] = os.path.abspath(config.detections_csv)
    summary["video_path"] = os.path.abspath(video_path) if video_path else ""
    summary["roi_count"] = len(rois)
    summary["rois"] = rois

    roi_outputs: dict[str, Any] = {}
    if rois:
        roi_outputs = create_roi_outputs(features, output_dir, fps)
        summary["roi_summary"] = roi_outputs.get("roi_summary", [])

    _progress(progress_callback, 5, total_steps, "Writing feature table and summary")
    feature_csv = output_dir / "analysis_features.csv"
    summary_json = output_dir / "analysis_summary.json"
    features.to_csv(feature_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    plot_paths: list[str] = []
    if config.make_plots:
        _progress(progress_callback, 6, total_steps, "Rendering plots")
        plot_paths = create_plots(features, output_dir / "plots", video_path=video_path, rois=rois)
    else:
        _progress(progress_callback, 6, total_steps, "Skipping plots")

    annotated_video = None
    if config.make_annotated_video:
        _progress(progress_callback, 7, total_steps, "Rendering annotated video")
        annotated_video = render_annotated_video(
            features, video_path, output_dir / "annotated_output.mp4", fps, rois=rois
        )
    else:
        _progress(progress_callback, 7, total_steps, "Skipping annotated video")

    cluster_plot_paths: list[str] = []
    cluster_clip_paths: list[str] = []
    if config.run_clustering:
        _progress(progress_callback, 8, total_steps, "Running behavior clustering")
        features, cluster_plot_paths = run_behavior_clustering(
            features,
            fps,
            output_dir / "plots",
            umap_neighbors=config.umap_neighbors,
            umap_min_dist=config.umap_min_dist,
            hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
        )
        features.to_csv(feature_csv, index=False)
        if config.export_cluster_clips:
            cluster_clip_paths = export_cluster_clips(
                features,
                video_path,
                output_dir,
                fps,
                config.cluster_clip_length_sec,
                config.samples_per_cluster,
            )
    else:
        _progress(progress_callback, 8, total_steps, "Analysis complete")

    return {
        "layer_id": LAYER_KEYPOINTS,
        "feature_csv": str(feature_csv),
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
