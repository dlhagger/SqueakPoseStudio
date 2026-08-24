"""Persistent per-project-video scale and ROI setup metadata."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from squeakpose.core import atomic_write_text
from squeakpose.json_io import JsonFileError, read_json_file
from squeakpose.project.safety import require_path_within_project
from squeakpose.services.analysis_state import AnalysisAnnotationState

VIDEO_ANALYSIS_SETUP_SCHEMA_VERSION = 1
SUMMARY_RECOVERY_FIELD = "analysis_summary_recovery_checked"
ROIS_CLEARED_FIELD = "rois_cleared"


@dataclass(frozen=True, slots=True)
class VideoAnalysisSetup:
    video_name: str
    frame_width: int
    frame_height: int
    scale_points: tuple[tuple[float, float], ...]
    real_world_distance_mm: float
    rois: tuple[dict[str, Any], ...]
    saved_at: str = ""


def _require_video_name(video_name: str) -> str:
    clean_name = str(video_name).strip()
    if not clean_name or clean_name in {".", ".."} or os.path.basename(clean_name) != clean_name:
        raise ValueError("project video name must be a single filename")
    return clean_name


def video_analysis_setup_path(project_root: str, video_name: str) -> str:
    """Return the project-contained metadata path for one video-library entry."""
    root = os.path.abspath(project_root)
    clean_name = _require_video_name(video_name)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", clean_name).strip("._-") or "video"
    digest = hashlib.sha256(clean_name.casefold().encode("utf-8")).hexdigest()[:12]
    return require_path_within_project(
        root,
        os.path.join(root, "analysis settings", "videos", f"{safe_name}-{digest}.json"),
        purpose="video analysis setup metadata",
        allow_root=False,
    )


def save_video_analysis_setup(
    project_root: str,
    video_name: str,
    *,
    frame_width: int,
    frame_height: int,
    scale_points: Sequence[tuple[float, float]],
    real_world_distance_mm: float,
    rois: Iterable[Mapping[str, Any]],
    rois_cleared: bool = False,
) -> str:
    """Validate and atomically persist one video's analysis setup."""
    clean_name = _require_video_name(video_name)
    real_distance = float(real_world_distance_mm)
    if not math.isfinite(real_distance) or real_distance <= 0:
        raise ValueError("real-world scale distance must be a positive finite number")
    state = AnalysisAnnotationState(
        frame_width=max(0, int(frame_width)),
        frame_height=max(0, int(frame_height)),
        real_world_distance_mm=real_distance,
    )
    state.set_scale_points(scale_points)
    state.replace_rois(rois)
    payload = {
        "schema_version": VIDEO_ANALYSIS_SETUP_SCHEMA_VERSION,
        "video_name": clean_name,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "frame": {"width": state.frame.width, "height": state.frame.height},
        "scale": {
            "points": [[x, y] for x, y in state.scale_points],
            "real_world_distance_mm": state.real_world_distance_mm,
        },
        "rois": state.worker_rois(),
        SUMMARY_RECOVERY_FIELD: True,
        ROIS_CLEARED_FIELD: bool(rois_cleared),
    }
    path = video_analysis_setup_path(project_root, clean_name)
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path


def load_video_analysis_setup(
    project_root: str,
    video_name: str,
) -> VideoAnalysisSetup | None:
    """Load and validate one video's setup, returning ``None`` when absent."""
    clean_name = _require_video_name(video_name)
    path = video_analysis_setup_path(project_root, clean_name)
    payload: Mapping[str, Any] | None = None
    if not os.path.isfile(path):
        return _recover_analysis_summary_setup(project_root, clean_name)
    try:
        payload = read_json_file(path, max_bytes=2 * 1024 * 1024, require_object=True)
    except JsonFileError as exc:
        raise ValueError(f"could not read saved video analysis setup: {exc}") from exc
    if int(payload.get("schema_version") or 0) != VIDEO_ANALYSIS_SETUP_SCHEMA_VERSION:
        raise ValueError("unsupported video analysis setup schema")
    if str(payload.get("video_name") or "") != clean_name:
        raise ValueError("video analysis setup belongs to a different project video")
    frame = payload.get("frame") if isinstance(payload.get("frame"), Mapping) else {}
    scale = payload.get("scale") if isinstance(payload.get("scale"), Mapping) else {}
    width = max(0, int(frame.get("width") or 0))
    height = max(0, int(frame.get("height") or 0))
    real_distance = float(scale.get("real_world_distance_mm") or 1.0)
    if not math.isfinite(real_distance) or real_distance <= 0:
        raise ValueError("saved real-world scale distance is invalid")
    state = AnalysisAnnotationState(
        frame_width=width,
        frame_height=height,
        real_world_distance_mm=real_distance,
    )
    raw_points = scale.get("points")
    if isinstance(raw_points, list):
        state.set_scale_points(raw_points)
    raw_rois = payload.get("rois")
    if isinstance(raw_rois, list):
        state.replace_rois(raw_rois)
    if not state.rois and not payload.get(ROIS_CLEARED_FIELD):
        recovered = _recover_analysis_summary_setup(
            project_root,
            clean_name,
            persist=True,
        )
        if recovered is not None:
            return recovered
    return VideoAnalysisSetup(
        video_name=clean_name,
        frame_width=width,
        frame_height=height,
        scale_points=state.scale_points,
        real_world_distance_mm=state.real_world_distance_mm,
        rois=tuple(state.worker_rois()),
        saved_at=str(payload.get("saved_at") or ""),
    )


def _recover_analysis_summary_setup(
    project_root: str,
    video_name: str,
    *,
    frame_width: int = 0,
    frame_height: int = 0,
    persist: bool = False,
) -> VideoAnalysisSetup | None:
    """Recover legacy ROIs from the newest matching analysis summary once."""
    summaries_root = os.path.join(os.path.abspath(project_root), "analysis outputs")
    matches: list[tuple[float, str, Mapping[str, Any]]] = []
    for layer_name in ("keypoints", "segmentation"):
        layer_root = os.path.join(summaries_root, layer_name)
        try:
            run_names = os.listdir(layer_root)
        except OSError:
            continue
        for run_name in run_names:
            summary_path = os.path.join(layer_root, run_name, "analysis_summary.json")
            try:
                summary = read_json_file(
                    summary_path, max_bytes=4 * 1024 * 1024, require_object=True
                )
                modified = os.path.getmtime(summary_path)
            except (JsonFileError, OSError):
                continue
            recorded_video = str(summary.get("video_path") or "").strip()
            if Path(recorded_video).name.casefold() != video_name.casefold():
                continue
            raw_rois = summary.get("rois")
            if isinstance(raw_rois, list) and raw_rois:
                matches.append((modified, summary_path, summary))
    if not matches:
        return None

    _modified, _summary_path, summary = max(matches, key=lambda item: item[0])
    state = AnalysisAnnotationState(
        # Legacy saved dimensions may have come from a blank CSV canvas in a
        # different layer. Keep recovered ROIs in their original coordinates;
        # the dialog will attach the canonical dimensions of the actual video.
        frame_width=0,
        frame_height=0,
    )
    try:
        state.replace_rois(summary["rois"])
    except (KeyError, TypeError, ValueError):
        return None
    setup = VideoAnalysisSetup(
        video_name=video_name,
        frame_width=state.frame.width,
        frame_height=state.frame.height,
        scale_points=(),
        real_world_distance_mm=1.0,
        rois=tuple(state.worker_rois()),
        saved_at="",
    )
    if persist:
        save_video_analysis_setup(
            project_root,
            video_name,
            frame_width=setup.frame_width,
            frame_height=setup.frame_height,
            scale_points=setup.scale_points,
            real_world_distance_mm=setup.real_world_distance_mm,
            rois=setup.rois,
        )
        return load_video_analysis_setup(project_root, video_name)
    return setup


__all__ = [
    "VIDEO_ANALYSIS_SETUP_SCHEMA_VERSION",
    "VideoAnalysisSetup",
    "load_video_analysis_setup",
    "save_video_analysis_setup",
    "video_analysis_setup_path",
]
