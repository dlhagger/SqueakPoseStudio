"""Qt-free configuration and input validation for analysis runs."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from squeakpose.json_io import JsonFileError, read_json_file
from squeakpose.project.layers import LAYER_DEFINITIONS, normalize_layer_id
from squeakpose.services.video_library import list_project_videos

DEFAULT_ONE_EURO_MIN_CUTOFF = 1.0
DEFAULT_ONE_EURO_BETA = 0.1


class AnalysisConfigError(ValueError):
    """Stable validation failure suitable for presentation by a UI."""

    def __init__(self, code: str, title: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.title = title
        self.message = message


@dataclass(frozen=True)
class AnalysisRunConfig:
    """Validated, detached worker configuration for one analysis run."""

    payload: Mapping[str, Any]
    video_fallback_notice: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.payload)
        result["rois"] = copy.deepcopy(list(self.payload.get("rois", ())))
        return result


@dataclass(frozen=True, slots=True)
class AnalysisCsvContext:
    """Preview metadata discovered from an inference CSV."""

    video_path: str = ""
    width: int = 1280
    height: int = 720


@dataclass(frozen=True, slots=True)
class SegmentationPreview:
    """First CSV frame containing segmentation polygons for the setup canvas."""

    frame_index: int = 0
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectAnalysisInput:
    """One project video and its newest compatible inference CSV, when available."""

    video_name: str
    video_path: str
    csv_path: str = ""
    created_at: str = ""

    @property
    def inference_ready(self) -> bool:
        return bool(self.csv_path)


def _video_identity(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def project_analysis_inputs(project_root: str, layer_id: str) -> tuple[ProjectAnalysisInput, ...]:
    """Pair project-library videos with their newest successful layer inference output."""
    root = os.path.abspath(project_root)
    normalized_layer = normalize_layer_id(layer_id)
    inference_root = os.path.join(root, "inference outputs")
    videos = [
        entry for entry in list_project_videos(os.path.join(root, "videos")) if entry.target_exists
    ]
    runs_dir = os.path.join(root, "inference outputs", "runs")
    newest_by_identity: dict[str, tuple[str, float, str]] = {}
    newest_by_name: dict[str, tuple[str, float, str]] = {}
    try:
        manifest_names = os.listdir(runs_dir)
    except OSError:
        manifest_names = []
    for name in manifest_names:
        if name.startswith(".") or not name.lower().endswith(".json"):
            continue
        manifest_path = os.path.join(runs_dir, name)
        try:
            manifest = read_json_file(manifest_path, max_bytes=1024 * 1024, require_object=True)
            modified = os.path.getmtime(manifest_path)
        except (JsonFileError, OSError):
            continue
        video_path = str(manifest.get("video_path") or "").strip()
        passes = manifest.get("passes")
        if not video_path or not isinstance(passes, list):
            continue
        candidates: list[str] = []
        for item in passes:
            if not isinstance(item, Mapping):
                continue
            if item.get("had_error") or item.get("canceled"):
                continue
            reported_layer = normalize_layer_id(item.get("layer_id"), default="")
            if reported_layer and reported_layer != normalized_layer:
                continue
            csv_path = str(item.get("csv_path") or "").strip()
            csv_path = os.path.abspath(csv_path) if csv_path else ""
            try:
                inside_project_outputs = bool(
                    csv_path and os.path.commonpath((inference_root, csv_path)) == inference_root
                )
            except ValueError:
                inside_project_outputs = False
            if (
                inside_project_outputs
                and os.path.isfile(csv_path)
                and analysis_csv_matches_layer(csv_path, normalized_layer)
            ):
                candidates.append(csv_path)
        if not candidates:
            continue
        created_at = str(manifest.get("created_at") or "")
        manifest_record = (created_at, modified, candidates[-1])
        identity = _video_identity(video_path)
        if manifest_record[:2] >= newest_by_identity.get(identity, ("", -1.0, ""))[:2]:
            newest_by_identity[identity] = manifest_record
        video_name = os.path.basename(video_path).casefold()
        if manifest_record[:2] >= newest_by_name.get(video_name, ("", -1.0, ""))[:2]:
            newest_by_name[video_name] = manifest_record

    options: list[ProjectAnalysisInput] = []
    for entry in videos:
        selected_record = newest_by_identity.get(_video_identity(entry.path))
        if selected_record is None:
            selected_record = newest_by_name.get(entry.name.casefold())
        options.append(
            ProjectAnalysisInput(
                video_name=entry.name,
                video_path=entry.path,
                csv_path=selected_record[2] if selected_record else "",
                created_at=selected_record[0] if selected_record else "",
            )
        )
    return tuple(options)


def _existing_manifest_video_path(csv_path: str) -> str:
    """Resolve an inference CSV's source video through its sibling run manifest."""

    csv_file = Path(csv_path).absolute()
    inference_root = csv_file.parent.parent
    if inference_root.name != "inference outputs":
        return ""

    run_id = ""
    for layer in LAYER_DEFINITIONS.values():
        if csv_file.name.endswith(layer.inference_suffix):
            run_id = csv_file.name[: -len(layer.inference_suffix)]
            break
    if not run_id:
        return ""

    manifest_path = inference_root / "runs" / f"{run_id}.json"
    try:
        manifest = read_json_file(str(manifest_path), max_bytes=1024 * 1024, require_object=True)
    except JsonFileError:
        return ""

    expected_csv = os.path.normcase(os.path.abspath(csv_path))
    matching_pass = False
    passes = manifest.get("passes")
    if isinstance(passes, list):
        for item in passes:
            if not isinstance(item, Mapping):
                continue
            recorded_csv = str(item.get("csv_path") or "").strip()
            if recorded_csv and os.path.normcase(os.path.abspath(recorded_csv)) == expected_csv:
                matching_pass = True
                break
    if not matching_pass:
        return ""

    candidate = str(manifest.get("video_path") or "").strip()
    if candidate and not os.path.isabs(candidate):
        candidate = str(inference_root.parent / candidate)
    if candidate and os.path.isfile(candidate):
        return candidate

    # A project video link may have been retargeted after inference. Prefer the
    # current link with the same filename when the manifest's original path is stale.
    if candidate:
        project_video = inference_root.parent / "videos" / os.path.basename(candidate)
        if project_video.is_file():
            return str(project_video)
    return ""


def safe_analysis_stem(path: str) -> str:
    stem = Path(path).stem if path else "analysis"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "analysis"


def default_analysis_output_dir(
    project_root: str,
    layer_id: str,
    csv_path: str,
    *,
    timestamp: str | None = None,
) -> str:
    stamp = timestamp or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        os.path.abspath(project_root),
        "analysis outputs",
        layer_id,
        f"{safe_analysis_stem(csv_path)}_{stamp}",
    )


def analysis_csv_matches_layer(path: str, layer_id: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            fieldnames = set(next(csv.reader(handle), []))
    except (OSError, csv.Error):
        return False
    is_segmentation = {"frame", "det", "mask_polygon"}.issubset(fieldnames)
    return is_segmentation == (layer_id == "segmentation")


def inspect_analysis_csv(path: str, *, row_limit: int = 1000) -> AnalysisCsvContext:
    """Read preview video and frame dimensions without importing a dataframe stack."""

    video_path = ""
    width = 0
    height = 0
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index >= max(0, int(row_limit)):
                    break
                candidate = str(row.get("video_path") or "").strip()
                if not video_path and candidate and os.path.isfile(candidate):
                    video_path = candidate
                try:
                    width = max(width, int(float(row.get("image_width") or 0)))
                except (TypeError, ValueError):
                    pass
                try:
                    height = max(height, int(float(row.get("image_height") or 0)))
                except (TypeError, ValueError):
                    pass
    except (OSError, csv.Error):
        pass
    if not video_path:
        video_path = _existing_manifest_video_path(path)
    return AnalysisCsvContext(video_path, width or 1280, height or 720)


def load_segmentation_preview(path: str, *, row_limit: int = 10_000) -> SegmentationPreview:
    """Load polygons from the first frame containing a valid segmentation mask."""

    selected_frame: int | None = None
    polygons: list[tuple[tuple[float, float], ...]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index >= max(0, int(row_limit)):
                    break
                try:
                    frame_index = int(float(row.get("frame") or row.get("frame_index") or 0))
                    raw_polygon = json.loads(str(row.get("mask_polygon") or ""))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(raw_polygon, list):
                    continue
                points: list[tuple[float, float]] = []
                for point in raw_polygon:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    try:
                        x, y = float(point[0]), float(point[1])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(x) and math.isfinite(y):
                        points.append((x, y))
                if len(points) < 3:
                    continue
                if selected_frame is None:
                    selected_frame = frame_index
                if frame_index != selected_frame:
                    if frame_index > selected_frame:
                        break
                    continue
                polygons.append(tuple(points))
    except (OSError, csv.Error):
        pass
    return SegmentationPreview(selected_frame or 0, tuple(polygons))


def latest_analysis_csv(directories: Iterable[str], layer_id: str) -> str:
    """Return the newest readable inference CSV matching the active layer."""

    candidates: list[str] = []
    for folder in directories:
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            if name.lower().endswith(".csv") and analysis_csv_matches_layer(path, layer_id):
                candidates.append(path)
    if not candidates:
        return ""
    try:
        return max(candidates, key=os.path.getmtime)
    except OSError:
        return ""


def build_analysis_run_config(
    *,
    layer_id: str,
    detections_csv: str,
    video_path: str,
    output_dir: str,
    pixel_distance: float,
    real_world_distance_mm: float,
    smooth: bool,
    min_cutoff: float,
    beta: float,
    make_plots: bool,
    make_annotated_video: bool,
    run_clustering: bool,
    export_cluster_clips: bool,
    umap_neighbors: int,
    umap_min_dist: float,
    hdbscan_min_cluster_size: int,
    cluster_clip_length_sec: float,
    samples_per_cluster: int,
    rois: Iterable[Mapping[str, Any]],
) -> AnalysisRunConfig:
    """Validate UI-independent invariants and preserve the worker payload schema."""

    csv_path = str(detections_csv).strip()
    if not os.path.isfile(csv_path):
        raise AnalysisConfigError(
            "csv_required",
            "CSV required",
            "Select a valid inference CSV before running analysis.",
        )
    clean_video_path = str(video_path).strip()
    if clean_video_path and not os.path.isfile(clean_video_path):
        raise AnalysisConfigError(
            "invalid_video",
            "Invalid video",
            f"Video file not found:\n{clean_video_path}",
        )
    if float(pixel_distance) <= 0:
        raise AnalysisConfigError(
            "scale_required",
            "Scale required",
            "Draw a two-point scale bar before running analysis.",
        )
    if export_cluster_clips and not run_clustering:
        raise AnalysisConfigError(
            "clustering_required",
            "Clustering required",
            "Enable UMAP/HDBSCAN before exporting cluster clips.",
        )

    payload = {
        "layer_id": layer_id,
        "detections_csv": csv_path,
        "video_path": clean_video_path,
        "output_dir": str(output_dir).strip(),
        "fps": 0.0,
        "pixel_distance": float(pixel_distance),
        "real_world_distance_mm": float(real_world_distance_mm),
        "smooth": bool(smooth),
        "min_cutoff": float(min_cutoff),
        "beta": float(beta),
        "d_cutoff": 1.0,
        "make_plots": bool(make_plots),
        "make_annotated_video": bool(make_annotated_video),
        "run_clustering": bool(run_clustering),
        "export_cluster_clips": bool(export_cluster_clips),
        "umap_neighbors": int(umap_neighbors),
        "umap_min_dist": float(umap_min_dist),
        "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
        "cluster_clip_length_sec": float(cluster_clip_length_sec),
        "samples_per_cluster": int(samples_per_cluster),
        "rois": tuple(copy.deepcopy(dict(roi)) for roi in rois),
    }
    return AnalysisRunConfig(
        payload=payload,
        video_fallback_notice=bool(make_annotated_video and not clean_video_path),
    )
