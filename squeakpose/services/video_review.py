"""Qt-free planning and result decisions for project video review."""

from __future__ import annotations

import os
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from squeakpose.core import stable_path_id
from squeakpose.project.layers import (
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    normalize_layer_id,
)
from squeakpose.project.paths import ProjectPaths
from squeakpose.project.safety import require_path_within_project
from squeakpose.services.prediction_serialization import rank_prediction_frames

MAX_VIDEO_CACHE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VideoReviewRunPlan:
    meta: dict[str, Any]
    settings: dict[str, Any]
    steps_per_pass: int
    total_steps: int


@dataclass(frozen=True, slots=True)
class VideoReviewCacheDecision:
    predictions_by_layer: dict[str, dict[int, dict]]

    @property
    def has_predictions(self) -> bool:
        return any(self.predictions_by_layer.values())


@dataclass(frozen=True, slots=True)
class VideoReviewPassCompletion:
    predictions: dict[int, dict]
    canceled: bool
    had_error: bool
    error_message: str


@dataclass(frozen=True, slots=True)
class ConfidenceExportPlan:
    """Ranked, not-yet-exported frames selected for a confidence export."""

    candidates: tuple[tuple[int, float, int], ...]
    pending: tuple[tuple[int, float, int], ...]
    selected: tuple[tuple[int, float, int], ...]


def build_video_signature(
    video_path: str | None,
    *,
    total: int,
    fps: float,
) -> dict[str, Any]:
    path = os.path.abspath(video_path) if video_path else ""
    try:
        size = int(os.path.getsize(video_path)) if video_path else 0
        mtime = float(os.path.getmtime(video_path)) if video_path else 0.0
    except Exception:
        size = 0
        mtime = 0.0
    return {
        "path": path,
        "size": size,
        "mtime": mtime,
        "total": int(total),
        "fps": float(fps),
    }


def video_review_cache_path(project_root: str, video_path: str | None) -> str | None:
    if not video_path:
        return None
    cache_dir = ProjectPaths.from_root(project_root).video_prediction_cache
    source_id = stable_path_id(video_path, length=16)
    return require_path_within_project(
        project_root,
        os.path.join(cache_dir, f"{source_id}.json"),
        purpose="video prediction cache",
        allow_root=False,
    )


def plan_video_review_run(
    *,
    video_signature: Mapping[str, Any],
    model_paths: Mapping[str, str],
    review_layers: Sequence[str],
    layer_schemas: Mapping[str, dict],
    start: int,
    end: int,
    stride: int,
    imgsz: int,
    conf: float,
    iou: float,
    kpvis: float | None,
    requested_batch: int,
    effective_batch: int,
    total: int,
    fps: float,
) -> VideoReviewRunPlan:
    normalized_stride = max(1, int(stride))
    steps = max(1, ((int(end) - int(start)) // normalized_stride) + 1)
    layers = [str(layer) for layer in review_layers]
    configured_layers = [str(layer) for layer, path in model_paths.items() if path]
    settings = {
        "start": int(start),
        "end": int(end),
        "stride": normalized_stride,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "batch": int(requested_batch),
        "effective_batch": int(effective_batch),
    }
    meta = {
        "video": dict(video_signature),
        # Cache identity covers every configured model, including a preserved
        # layer that was not selected for this particular prediction pass.
        "model_paths": {layer: model_paths[layer] for layer in configured_layers},
        "layers": layers,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "kpvis": kpvis,
        "start": int(start),
        "end": int(end),
        "stride": normalized_stride,
        "batch": int(requested_batch),
        "initial_effective_batch": int(effective_batch),
        "total": int(total),
        "fps": float(fps),
        "schemas": {layer: layer_schemas.get(layer, {}) for layer in configured_layers},
    }
    return VideoReviewRunPlan(
        meta=meta,
        settings=settings,
        steps_per_pass=steps,
        total_steps=steps * len(layers),
    )


def build_video_review_pass_config(
    *,
    layer_id: str,
    model_path: str,
    video_path: str | None,
    device: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_layer = normalize_layer_id(layer_id)
    return {
        "model_path": model_path or "",
        "video_path": video_path,
        "workflow": layer_definition(normalized_layer).worker_mode,
        "layer_id": normalized_layer,
        "device": device,
        **dict(settings),
    }


def decide_video_review_cache(
    data: Mapping[str, Any],
    *,
    current_video: Mapping[str, Any],
    review_layers: Sequence[str],
    model_paths: Mapping[str, str],
    layer_id: str,
    model_path: str | None,
    workflow: str,
) -> VideoReviewCacheDecision | None:
    meta = data.get("meta", {})
    if not isinstance(meta, Mapping):
        return None
    saved_video = meta.get("video", {})
    if not isinstance(saved_video, Mapping):
        return None
    if saved_video.get("path") != current_video.get("path"):
        return None
    if int(saved_video.get("size", -1)) != int(current_video.get("size", -2)):
        return None
    if abs(float(saved_video.get("mtime", 0.0)) - float(current_video.get("mtime", 0.0))) > 2.0:
        return None

    saved_models = meta.get("model_paths")
    cached_by_layer = data.get("preds_by_layer")
    if isinstance(saved_models, Mapping) and isinstance(cached_by_layer, Mapping):
        for layer in review_layers:
            if str(saved_models.get(layer) or "") != str(model_paths.get(layer) or ""):
                return None
        loaded: dict[str, dict[int, dict[str, Any]]] = {
            LAYER_KEYPOINTS: {},
            LAYER_SEGMENTATION: {},
        }
        try:
            for layer in loaded:
                raw = cached_by_layer.get(layer) or {}
                if isinstance(raw, Mapping):
                    loaded[layer] = {
                        int(frame_index): prediction
                        for frame_index, prediction in raw.items()
                        if isinstance(prediction, dict)
                    }
        except (TypeError, ValueError):
            return None
        return VideoReviewCacheDecision(loaded)

    saved_model = meta.get("model_path")
    if saved_model and model_path and saved_model != model_path:
        return None
    saved_workflow = str(meta.get("workflow", "pose")).strip().lower()
    if saved_workflow != workflow:
        return None
    raw_predictions = data.get("preds", {})
    if not isinstance(raw_predictions, Mapping):
        return None
    try:
        predictions = {
            int(frame_index): prediction
            for frame_index, prediction in raw_predictions.items()
            if isinstance(prediction, dict)
        }
    except (TypeError, ValueError):
        return None
    legacy_layer = normalize_layer_id(layer_id or saved_workflow)
    loaded = {LAYER_KEYPOINTS: {}, LAYER_SEGMENTATION: {}}
    loaded[legacy_layer] = predictions
    return VideoReviewCacheDecision(loaded)


def build_video_review_cache_payload(
    meta: Mapping[str, Any],
    predictions_by_layer: Mapping[str, Mapping[int, dict]],
) -> dict[str, Any]:
    return {
        "meta": dict(meta),
        "preds_by_layer": {
            layer: {str(frame_index): prediction for frame_index, prediction in predictions.items()}
            for layer, predictions in predictions_by_layer.items()
            if predictions
        },
    }


def complete_video_review_pass(
    *,
    partial_predictions: Mapping[int, dict],
    result_event: Mapping[str, Any] | None,
    cancel_requested: bool,
    worker_state: str,
    exit_code: int | None,
    crashed: bool,
    worker_error: str,
    stderr: str,
) -> VideoReviewPassCompletion:
    event = dict(result_event) if result_event is not None else None
    canceled_by_worker = worker_state == "cancelled"
    canceled = bool(cancel_requested or canceled_by_worker)
    if canceled and event is None:
        event = {
            "canceled": True,
            "had_error": False,
            "error_message": "",
            "preds": {},
        }
    if event is None:
        event = {
            "canceled": False,
            "had_error": True,
            "error_message": worker_error or stderr or f"Process exited with code {exit_code}.",
            "preds": {},
        }

    predictions = {int(frame_index): value for frame_index, value in partial_predictions.items()}
    raw_predictions = event.get("preds") or {}
    if isinstance(raw_predictions, Mapping):
        for frame_index, value in raw_predictions.items():
            try:
                predictions[int(frame_index)] = value if isinstance(value, dict) else {"ok": False}
            except (TypeError, ValueError):
                continue

    canceled = bool(event.get("canceled")) or canceled
    had_error = bool(event.get("had_error")) or (
        not canceled
        and (worker_state in {"failed", "start_failed"} or crashed or exit_code not in {None, 0})
    )
    error_message = str(
        event.get("error_message") or stderr or worker_error or "Unknown video prediction error"
    )
    return VideoReviewPassCompletion(
        predictions=predictions,
        canceled=canceled,
        had_error=had_error,
        error_message=error_message,
    )


def exported_frame_indices(
    filenames: Sequence[str],
    *,
    video_base: str,
    source_id: str,
) -> set[int]:
    prefix = f"{_safe_export_component(video_base, fallback='video')}_{_safe_export_component(source_id, fallback='source')}_f"
    pattern = re.compile(
        rf"^{re.escape(prefix)}(\d{{6}})(?:_.*)?\.(?:png|jpg|jpeg|bmp|webp)$",
        re.IGNORECASE,
    )
    indices: set[int] = set()
    for filename in filenames:
        match = pattern.match(str(filename))
        if match:
            indices.add(int(match.group(1)))
    return indices


def select_random_export_frames(
    total_frames: int,
    *,
    already_exported: Sequence[int] = (),
    count: int,
    sampler: Callable[[Sequence[int], int], Sequence[int]] = random.sample,
) -> tuple[int, ...]:
    """Select sorted, unique frame indices that have not already been exported."""

    excluded = {int(index) for index in already_exported}
    available = [index for index in range(max(0, int(total_frames))) if index not in excluded]
    requested = min(max(0, int(count)), len(available))
    if requested == 0:
        return ()
    return tuple(sorted(int(index) for index in sampler(available, requested)))


def available_export_frame_indices(
    total_frames: int,
    *,
    already_exported: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return frame indices not already present in the label queue."""

    excluded = {int(index) for index in already_exported}
    return tuple(index for index in range(max(0, int(total_frames))) if index not in excluded)


def plan_confidence_export(
    predictions: Mapping[int, dict[str, Any]],
    *,
    class_ids: Sequence[int],
    order: str,
    balanced: bool,
    already_exported: Sequence[int] = (),
    count: int | None = None,
) -> ConfidenceExportPlan:
    """Rank predictions and filter frames already present in the label queue."""

    candidates = tuple(
        rank_prediction_frames(
            predictions,
            class_ids=[int(class_id) for class_id in class_ids],
            order="high" if str(order).lower() == "high" else "low",
            balanced=bool(balanced),
        )
    )
    excluded = {int(index) for index in already_exported}
    pending = tuple(candidate for candidate in candidates if int(candidate[0]) not in excluded)
    selected = pending if count is None else pending[: max(0, int(count))]
    return ConfidenceExportPlan(candidates, pending, selected)


def plan_export_frame_path(
    destination_dir: str,
    *,
    video_base: str,
    source_id: str,
    frame_index: int,
    avoid_collisions: bool = True,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> str:
    destination = os.path.abspath(destination_dir)
    base = _safe_export_component(video_base, fallback="video")
    source = _safe_export_component(source_id, fallback="source")
    stem = f"{base}_{source}_f{int(frame_index):06d}"
    candidate = os.path.join(destination, f"{stem}.png")
    suffix = 1
    while avoid_collisions and path_exists(candidate):
        candidate = os.path.join(destination, f"{stem}_{suffix}.png")
        suffix += 1
    if os.path.commonpath([destination, os.path.abspath(candidate)]) != destination:
        raise ValueError("export path escapes destination directory")
    return candidate


def _safe_export_component(value: str, *, fallback: str) -> str:
    component = os.path.basename(str(value).strip())
    component = re.sub(r"[\x00-\x1f\x7f]+", "_", component)
    return fallback if component in {"", ".", ".."} else component
