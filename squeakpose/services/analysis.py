"""Qt-free configuration and input validation for analysis runs."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from squeakpose.json_io import JsonFileError, read_json_file
from squeakpose.project.layers import (
    LAYER_DEFINITIONS,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    normalize_layer_id,
)
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
    primary_bbox: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PosePreviewKeypoint:
    """One named pose keypoint from the primary preview detection."""

    name: str
    x: float
    y: float
    confidence: float = math.nan


@dataclass(frozen=True, slots=True)
class PosePreview:
    """Primary pose detection to overlay on one analysis setup frame."""

    frame_index: int = 0
    bbox: tuple[float, ...] = ()
    keypoints: tuple[PosePreviewKeypoint, ...] = ()
    class_name: str = ""
    confidence: float = math.nan


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


@dataclass(frozen=True, slots=True)
class ProjectAnalysisBundle:
    """All analysis-ready inference inputs discovered for one project video."""

    video_name: str
    video_path: str
    keypoints_csv: str = ""
    segmentation_csv: str = ""
    keypoints_created_at: str = ""
    segmentation_created_at: str = ""

    @property
    def available_layers(self) -> tuple[str, ...]:
        layers: list[str] = []
        if self.keypoints_csv:
            layers.append(LAYER_KEYPOINTS)
        if self.segmentation_csv:
            layers.append(LAYER_SEGMENTATION)
        return tuple(layers)

    @property
    def inference_ready(self) -> bool:
        return bool(self.available_layers)

    @property
    def both_ready(self) -> bool:
        return len(self.available_layers) == 2

    def csv_for_layer(self, layer_id: str) -> str:
        normalized = normalize_layer_id(layer_id)
        return self.keypoints_csv if normalized == LAYER_KEYPOINTS else self.segmentation_csv


def _video_identity(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _relocated_inference_csv(inference_root: str, recorded_path: str, layer_id: str) -> str:
    """Resolve an inference CSV after its project directory has moved."""
    raw = str(recorded_path or "").strip()
    if not raw:
        return ""
    candidate = os.path.abspath(raw)
    try:
        inside_outputs = os.path.commonpath((inference_root, candidate)) == inference_root
    except ValueError:
        inside_outputs = False
    if inside_outputs and os.path.isfile(candidate):
        return candidate

    # Run manifests written before project paths became portable contain the
    # original absolute path. The output filename is run-specific, so it is a
    # safe and unambiguous relocation key within the active layer directory.
    filename = Path(raw).name
    if not filename:
        return ""
    relocated = os.path.join(inference_root, layer_id, filename)
    return relocated if os.path.isfile(relocated) else ""


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
            csv_path = _relocated_inference_csv(
                inference_root, str(item.get("csv_path") or ""), normalized_layer
            )
            if csv_path and analysis_csv_matches_layer(csv_path, normalized_layer):
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


def project_analysis_bundles(project_root: str) -> tuple[ProjectAnalysisBundle, ...]:
    """Discover pose and segmentation inputs together for every project video."""
    keypoints = {
        item.video_name: item for item in project_analysis_inputs(project_root, LAYER_KEYPOINTS)
    }
    segmentation = {
        item.video_name: item for item in project_analysis_inputs(project_root, LAYER_SEGMENTATION)
    }
    names = sorted(set(keypoints) | set(segmentation), key=str.casefold)
    bundles: list[ProjectAnalysisBundle] = []
    for name in names:
        pose = keypoints.get(name)
        segment = segmentation.get(name)
        source = pose or segment
        if source is None:
            continue
        bundles.append(
            ProjectAnalysisBundle(
                video_name=name,
                video_path=source.video_path,
                keypoints_csv=pose.csv_path if pose else "",
                segmentation_csv=segment.csv_path if segment else "",
                keypoints_created_at=pose.created_at if pose else "",
                segmentation_created_at=segment.created_at if segment else "",
            )
        )
    return tuple(bundles)


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
            relocated_csv = _relocated_inference_csv(
                str(inference_root), recorded_csv, csv_file.parent.name
            )
            if relocated_csv and os.path.normcase(relocated_csv) == expected_csv:
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
    video_name: str = "",
) -> str:
    source_video = video_name or _existing_manifest_video_path(csv_path)
    video_stem = safe_analysis_stem(source_video or csv_path)
    return os.path.join(
        os.path.abspath(project_root),
        "analysis outputs",
        video_stem,
        layer_id,
    )


def default_combined_analysis_output_dir(
    project_root: str,
    video_name: str,
) -> str:
    """Return the stable combined-analysis folder for one project video."""
    return os.path.join(
        os.path.abspath(project_root),
        "analysis outputs",
        safe_analysis_stem(video_name),
        "combined",
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
    primary_bbox: tuple[float, ...] = ()
    primary_score = (-math.inf, -math.inf)
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
                confidence = _finite_csv_float(row, "confidence", "conf")
                polygon_area = (
                    abs(
                        sum(
                            x1 * y2 - x2 * y1
                            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
                        )
                    )
                    / 2.0
                )
                score = (
                    confidence if math.isfinite(confidence) else 0.0,
                    polygon_area,
                )
                bbox_values = tuple(
                    _finite_csv_float(row, canonical, legacy)
                    for canonical, legacy in (
                        ("bbox_x1", "x1"),
                        ("bbox_y1", "y1"),
                        ("bbox_x2", "x2"),
                        ("bbox_y2", "y2"),
                    )
                )
                if score > primary_score and all(math.isfinite(value) for value in bbox_values):
                    primary_score = score
                    primary_bbox = bbox_values
    except (OSError, csv.Error):
        pass
    return SegmentationPreview(selected_frame or 0, tuple(polygons), primary_bbox)


def _finite_csv_float(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        try:
            value = float(row.get(name) or "")
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return math.nan


def load_pose_preview(
    path: str,
    *,
    frame_index: int | None = None,
    row_limit: int = 10_000,
) -> PosePreview:
    """Load the highest-confidence animal pose on a requested or first valid frame."""
    requested_frame = int(frame_index) if frame_index is not None else None
    best_frame: int | None = None
    best_score = -math.inf
    best_preview = PosePreview(frame_index=max(0, requested_frame or 0))
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            keypoint_names = tuple(
                match.group(1)
                for field in fieldnames
                if (match := re.fullmatch(r"kp_(.+)_x", field)) is not None
            )
            for index, row in enumerate(reader):
                if index >= max(0, int(row_limit)):
                    break
                raw_frame = _finite_csv_float(row, "frame_index", "frame")
                if not math.isfinite(raw_frame):
                    continue
                row_frame = int(raw_frame)
                if requested_frame is not None and row_frame != requested_frame:
                    continue
                detection_index = _finite_csv_float(row, "detection_index", "det")
                if math.isfinite(detection_index) and detection_index < 0:
                    continue
                if requested_frame is None and best_frame is not None and row_frame > best_frame:
                    break

                bbox_values = tuple(
                    _finite_csv_float(row, canonical, legacy)
                    for canonical, legacy in (
                        ("bbox_x1", "x1"),
                        ("bbox_y1", "y1"),
                        ("bbox_x2", "x2"),
                        ("bbox_y2", "y2"),
                    )
                )
                bbox = bbox_values if all(math.isfinite(value) for value in bbox_values) else ()
                keypoints: list[PosePreviewKeypoint] = []
                for name in keypoint_names:
                    x = _finite_csv_float(row, f"kp_{name}_x")
                    y = _finite_csv_float(row, f"kp_{name}_y")
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    keypoints.append(
                        PosePreviewKeypoint(
                            name=name,
                            x=x,
                            y=y,
                            confidence=_finite_csv_float(row, f"kp_{name}_conf"),
                        )
                    )
                if not bbox and not keypoints:
                    continue

                confidence = _finite_csv_float(row, "confidence", "conf")
                score = confidence if math.isfinite(confidence) else 0.0
                if best_frame is None or row_frame < best_frame:
                    best_frame = row_frame
                    best_score = score
                elif row_frame == best_frame and score <= best_score:
                    continue
                else:
                    best_score = score
                best_preview = PosePreview(
                    frame_index=row_frame,
                    bbox=bbox,
                    keypoints=tuple(keypoints),
                    class_name=str(row.get("class_name") or ""),
                    confidence=confidence,
                )
    except (OSError, csv.Error):
        return PosePreview(frame_index=max(0, requested_frame or 0))
    return best_preview


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


def build_analysis_job_config(
    *,
    analysis_mode: str,
    analysis_inputs: Mapping[str, str],
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
    """Validate a single- or dual-layer analysis job using the existing worker contract."""
    clean_mode = str(analysis_mode or "").strip().lower()
    if clean_mode not in {"both", LAYER_KEYPOINTS, LAYER_SEGMENTATION}:
        raise AnalysisConfigError(
            "invalid_mode", "Analysis mode", "Choose Pose, Segmentation, or Both."
        )
    selected_layers = (
        (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
        if clean_mode == "both"
        else (normalize_layer_id(clean_mode),)
    )

    normalized_inputs = {
        layer: str(analysis_inputs.get(layer) or "").strip()
        for layer in (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
    }
    for layer in selected_layers:
        csv_path = normalized_inputs[layer]
        if not os.path.isfile(csv_path):
            label = "Pose" if layer == LAYER_KEYPOINTS else "Segmentation"
            raise AnalysisConfigError(
                "csv_required",
                f"{label} CSV required",
                f"No readable {label.lower()} inference CSV is available for this video.",
            )
        if not analysis_csv_matches_layer(csv_path, layer):
            raise AnalysisConfigError(
                "wrong_layer",
                "Wrong inference layer",
                f"{os.path.basename(csv_path)} is not a valid {layer} inference CSV.",
            )

    first_layer = selected_layers[0]
    base = build_analysis_run_config(
        layer_id=first_layer,
        detections_csv=normalized_inputs[first_layer],
        video_path=video_path,
        output_dir=output_dir,
        pixel_distance=pixel_distance,
        real_world_distance_mm=real_world_distance_mm,
        smooth=smooth,
        min_cutoff=min_cutoff,
        beta=beta,
        make_plots=make_plots,
        make_annotated_video=make_annotated_video,
        run_clustering=run_clustering,
        export_cluster_clips=export_cluster_clips,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        cluster_clip_length_sec=cluster_clip_length_sec,
        samples_per_cluster=samples_per_cluster,
        rois=rois,
    )
    payload = base.as_dict()
    payload.update(
        {
            "analysis_mode": "both" if len(selected_layers) == 2 else first_layer,
            "analysis_inputs": {layer: normalized_inputs[layer] for layer in selected_layers},
            "selected_layers": list(selected_layers),
            "layer_id": first_layer if len(selected_layers) == 1 else "",
        }
    )
    return AnalysisRunConfig(payload, base.video_fallback_notice)
