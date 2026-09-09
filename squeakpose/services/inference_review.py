"""Project-wide, model-specific ranking of frames that need inference review."""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

from squeakpose.project.layers import (
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    normalize_layer_id,
)
from squeakpose.services.analysis import project_analysis_inputs
from squeakpose.services.video_library import list_project_videos

RANK_OVERALL = "overall"
RANK_POSE_CONFIDENCE = "pose_confidence"
RANK_SEGMENTATION_CONFIDENCE = "segmentation_confidence"
RANK_KEYPOINT_CONFIDENCE = "keypoint_confidence"
RANKING_MODES = (
    RANK_OVERALL,
    RANK_POSE_CONFIDENCE,
    RANK_SEGMENTATION_CONFIDENCE,
    RANK_KEYPOINT_CONFIDENCE,
)

LOW_DETECTION_CONFIDENCE = 0.35
LOW_KEYPOINT_CONFIDENCE = 0.50
_STATUS_SEVERITY = {"good": 0, "warning": 1, "bad": 2}
_REASON_WEIGHTS = {
    "missing_detection": 55.0,
    "missing_some_detections": 35.0,
    "extra_detection": 28.0,
    "track_count_mismatch": 25.0,
    "track_id_changed": 22.0,
    "low_detection_confidence": 18.0,
    "low_keypoint_confidence": 15.0,
}


@dataclass(frozen=True, slots=True)
class InferenceReviewOverlay:
    """One model detection reconstructed from an inference CSV row."""

    animal_id: str
    bbox: tuple[float, ...] = ()
    mask_polygon: tuple[tuple[float, float], ...] = ()
    keypoints: tuple[tuple[str, float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class InferenceReviewCandidate:
    """Frame-level quality record for one inference model and layer."""

    layer_id: str
    model_path: str
    video_name: str
    video_path: str
    inference_csv: str
    frame_index: int
    time_seconds: float | None
    status: str
    reasons: tuple[str, ...]
    detection_confidence: float | None
    keypoint_confidence: float | None
    detections_in_frame: int
    tracks_in_frame: int | None
    expected_animal_count: int
    priority_score: float
    track_ids: tuple[str, ...] = ()
    overlays: tuple[InferenceReviewOverlay, ...] = ()
    run_id: str = ""

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.layer_id,
            os.path.normcase(os.path.abspath(self.video_path)),
            os.path.normcase(os.path.abspath(self.model_path)) if self.model_path else "",
            self.run_id or os.path.normcase(os.path.abspath(self.inference_csv)),
            self.frame_index,
        )


@dataclass(frozen=True, slots=True)
class InferenceReviewScanResult:
    """Bounded candidate index and whole-project scan statistics for one layer."""

    layer_id: str
    candidates: tuple[InferenceReviewCandidate, ...]
    videos_total: int
    videos_with_inference: int
    frames_scanned: int
    warning_frames: int
    bad_frames: int
    model_paths: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelInferenceSource:
    layer_id: str
    model_path: str
    video_name: str
    video_path: str
    inference_csv: str
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class _SourceScanResult:
    candidates: tuple[InferenceReviewCandidate, ...]
    frames_scanned: int
    warning_frames: int
    bad_frames: int


class _ReverseRankingKey:
    """Invert tuple ordering so a min-heap exposes its worst retained item."""

    __slots__ = ("key",)

    def __init__(self, key: tuple) -> None:
        self.key = key

    def __lt__(self, other: _ReverseRankingKey) -> bool:
        return self.key > other.key


class _BoundedCandidatePools:
    """Keep only the best candidates for each ranking mode."""

    def __init__(self, layer_id: str, pool_per_mode: int) -> None:
        self.limit = max(1, int(pool_per_mode))
        self.modes = [
            RANK_OVERALL,
            RANK_POSE_CONFIDENCE if layer_id == LAYER_KEYPOINTS else RANK_SEGMENTATION_CONFIDENCE,
        ]
        if layer_id == LAYER_KEYPOINTS:
            self.modes.append(RANK_KEYPOINT_CONFIDENCE)
        self.heaps: dict[str, list[tuple[_ReverseRankingKey, int, InferenceReviewCandidate]]] = {
            mode: [] for mode in self.modes
        }
        self._sequence = 0

    def add(self, candidate: InferenceReviewCandidate) -> None:
        for mode, heap in self.heaps.items():
            if not _candidate_supports_mode(candidate, mode):
                continue
            ranking_key = _ranking_key(candidate, mode)
            entry = (_ReverseRankingKey(ranking_key), self._sequence, candidate)
            self._sequence += 1
            if len(heap) < self.limit:
                heapq.heappush(heap, entry)
            elif ranking_key < heap[0][0].key:
                heapq.heapreplace(heap, entry)

    def candidates(self) -> tuple[InferenceReviewCandidate, ...]:
        retained: dict[tuple[str, str, str, str, int], InferenceReviewCandidate] = {}
        for heap in self.heaps.values():
            for _, _, candidate in heap:
                retained[candidate.key] = candidate
        return tuple(retained.values())


@dataclass(slots=True)
class _FrameAccumulator:
    source: ModelInferenceSource
    frame_index: int
    time_seconds: float | None = None
    confidences: list[float] = field(default_factory=list)
    keypoint_confidences: list[float] = field(default_factory=list)
    reported_detection_counts: list[int] = field(default_factory=list)
    reported_track_counts: list[int] = field(default_factory=list)
    expected_counts: list[int] = field(default_factory=list)
    track_ids: set[str] = field(default_factory=set)
    valid_detection_rows: int = 0
    overlays: list[InferenceReviewOverlay] = field(default_factory=list)

    def add(self, row: Mapping[str, str]) -> None:
        time_value = _finite(row.get("time_seconds"))
        if self.time_seconds is None and time_value is not None:
            self.time_seconds = time_value
        _append_int(self.reported_detection_counts, row.get("detections_in_frame"))
        _append_int(self.reported_track_counts, row.get("tracks_in_frame"))
        _append_int(self.expected_counts, row.get("expected_animal_count"))
        detection_index = _finite(row.get("detection_index"))
        if detection_index is None:
            detection_index = _finite(row.get("det"))
        confidence = _layer_confidence(row, self.source.layer_id)
        bbox = _layer_bbox(row, self.source.layer_id)
        valid = (detection_index is None or detection_index >= 0) and (
            confidence is not None or len(bbox) == 4
        )
        if not valid:
            return
        self.valid_detection_rows += 1
        if confidence is not None:
            self.confidences.append(confidence)
        track_id = str(row.get("track_id") or "").strip()
        if track_id and track_id.lower() not in {"nan", "none", "-1"}:
            self.track_ids.add(track_id)
        if self.source.layer_id == LAYER_KEYPOINTS:
            for column, raw_value in row.items():
                if column.startswith("kp_") and column.endswith("_conf"):
                    _append_finite(self.keypoint_confidences, raw_value)
        self.overlays.append(_overlay_from_inference_row(row, self.source.layer_id))

    def finish(self) -> InferenceReviewCandidate:
        detection_count = max(
            self.reported_detection_counts or [self.valid_detection_rows],
            default=self.valid_detection_rows,
        )
        expected = max(self.expected_counts or [1])
        track_count = max(self.reported_track_counts) if self.reported_track_counts else None
        confidence = _minimum(self.confidences)
        keypoint_confidence = _minimum(self.keypoint_confidences)
        reasons: list[str] = []
        if detection_count <= 0:
            reasons.append("missing_detection")
            status = "bad"
        else:
            status = "good"
            if detection_count < expected:
                reasons.append("missing_some_detections")
            if detection_count > expected:
                reasons.append("extra_detection")
            if track_count is not None and track_count != expected:
                reasons.append("track_count_mismatch")
            if confidence is not None and confidence < LOW_DETECTION_CONFIDENCE:
                reasons.append("low_detection_confidence")
            if (
                self.source.layer_id == LAYER_KEYPOINTS
                and keypoint_confidence is not None
                and keypoint_confidence < LOW_KEYPOINT_CONFIDENCE
            ):
                reasons.append("low_keypoint_confidence")
            if reasons:
                status = "warning"
        reason_tuple = tuple(sorted(set(reasons)))
        return InferenceReviewCandidate(
            layer_id=self.source.layer_id,
            model_path=self.source.model_path,
            video_name=self.source.video_name,
            video_path=self.source.video_path,
            inference_csv=self.source.inference_csv,
            frame_index=self.frame_index,
            time_seconds=self.time_seconds,
            status=status,
            reasons=reason_tuple,
            detection_confidence=confidence,
            keypoint_confidence=keypoint_confidence,
            detections_in_frame=detection_count,
            tracks_in_frame=track_count,
            expected_animal_count=expected,
            priority_score=_priority_score(
                status,
                reason_tuple,
                detection_confidence=confidence,
                keypoint_confidence=keypoint_confidence,
            ),
            track_ids=tuple(sorted(self.track_ids)),
            overlays=tuple(self.overlays),
            run_id=self.source.run_id,
        )


def discover_model_inference_sources(
    project_root: str,
    layer_id: str,
) -> tuple[ModelInferenceSource, ...]:
    """Resolve each model's newest successful output for every project video."""

    normalized_layer = normalize_layer_id(layer_id)
    return tuple(
        ModelInferenceSource(
            layer_id=normalized_layer,
            model_path=item.model_path or _model_path_from_csv(item.csv_path),
            video_name=item.video_name,
            video_path=item.video_path,
            inference_csv=item.csv_path,
            run_id=item.run_id,
        )
        for item in project_analysis_inputs(project_root, normalized_layer, newest_per_model=True)
        if item.csv_path
    )


def scan_project_inference_quality(
    project_root: str,
    *,
    layer_id: str = LAYER_KEYPOINTS,
    pool_per_mode_per_video: int = 150,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> InferenceReviewScanResult:
    """Stream one model layer's outputs and retain bounded review candidate pools."""

    root = os.path.abspath(project_root)
    normalized_layer = normalize_layer_id(layer_id)
    project_videos = [
        entry for entry in list_project_videos(os.path.join(root, "videos")) if entry.target_exists
    ]
    sources = discover_model_inference_sources(root, normalized_layer)
    issues: list[str] = []
    retained: dict[tuple[str, str, str, str, int], InferenceReviewCandidate] = {}
    frames_scanned = warning_frames = bad_frames = 0
    successful_video_keys: set[str] = set()
    total_sources = len(sources)
    for source_index, source in enumerate(sources, start=1):
        if cancel_requested is not None and cancel_requested():
            break
        if progress_callback is not None:
            progress_callback(source_index - 1, total_sources, source.video_name)
        try:
            scan = _scan_inference_source(
                source,
                pool_per_mode=pool_per_mode_per_video,
                cancel_requested=cancel_requested,
            )
        except (OSError, csv.Error, ValueError) as exc:
            issues.append(f"{source.video_name}: {exc}")
            continue
        successful_video_keys.add(os.path.normcase(os.path.abspath(source.video_path)))
        frames_scanned += scan.frames_scanned
        warning_frames += scan.warning_frames
        bad_frames += scan.bad_frames
        for candidate in scan.candidates:
            retained[candidate.key] = candidate
    if progress_callback is not None:
        progress_callback(total_sources, total_sources, "Complete")
    inferred_names = {source.video_name.casefold() for source in sources}
    for entry in project_videos:
        if entry.name.casefold() not in inferred_names:
            issues.append(f"{entry.name}: {normalized_layer} inference output not found")
    return InferenceReviewScanResult(
        layer_id=normalized_layer,
        candidates=tuple(retained.values()),
        videos_total=len(project_videos),
        videos_with_inference=len(successful_video_keys),
        frames_scanned=frames_scanned,
        warning_frames=warning_frames,
        bad_frames=bad_frames,
        model_paths=tuple(sorted({source.model_path for source in sources if source.model_path})),
        issues=tuple(issues),
    )


def rank_inference_review_candidates(
    candidates: Iterable[InferenceReviewCandidate],
    *,
    mode: str = RANK_OVERALL,
    limit: int = 25,
    per_video_limit: int = 5,
    min_frame_spacing: int = 15,
    video_name: str = "",
    model_path: str = "",
    status: str = "",
    reason: str = "",
) -> tuple[InferenceReviewCandidate, ...]:
    """Rank and temporally de-duplicate candidates for one selected model."""

    normalized_mode = mode if mode in RANKING_MODES else RANK_OVERALL
    requested_video = str(video_name).casefold()
    requested_model = os.path.normcase(os.path.abspath(model_path)) if model_path else ""
    requested_status = str(status).strip().lower()
    requested_reason = str(reason).strip()
    filtered = [
        candidate
        for candidate in candidates
        if (not requested_video or candidate.video_name.casefold() == requested_video)
        and (
            not requested_model
            or os.path.normcase(os.path.abspath(candidate.model_path)) == requested_model
        )
        and (not requested_status or candidate.status == requested_status)
        and (not requested_reason or requested_reason in candidate.reasons)
        and _candidate_supports_mode(candidate, normalized_mode)
    ]
    filtered.sort(key=lambda candidate: _ranking_key(candidate, normalized_mode))
    selected: list[InferenceReviewCandidate] = []
    frames_by_video: dict[str, list[int]] = {}
    cap = max(1, int(per_video_limit))
    spacing = max(0, int(min_frame_spacing))
    for candidate in filtered:
        video_key = os.path.normcase(os.path.abspath(candidate.video_path))
        prior_frames = frames_by_video.setdefault(video_key, [])
        if len(prior_frames) >= cap:
            continue
        if spacing and any(abs(candidate.frame_index - prior) <= spacing for prior in prior_frames):
            continue
        selected.append(candidate)
        prior_frames.append(candidate.frame_index)
        if len(selected) >= max(1, int(limit)):
            break
    return tuple(selected)


def candidate_reasons(candidates: Iterable[InferenceReviewCandidate]) -> tuple[str, ...]:
    return tuple(sorted({reason for candidate in candidates for reason in candidate.reasons}))


def _scan_inference_source(
    source: ModelInferenceSource,
    *,
    pool_per_mode: int,
    cancel_requested: Callable[[], bool] | None,
) -> _SourceScanResult:
    pools = _BoundedCandidatePools(source.layer_id, pool_per_mode)
    current: _FrameAccumulator | None = None
    previous_track_ids: tuple[str, ...] = ()
    frames_scanned = warning_frames = bad_frames = 0

    def finish_frame(accumulator: _FrameAccumulator) -> None:
        nonlocal previous_track_ids, frames_scanned, warning_frames, bad_frames
        candidate = accumulator.finish()
        candidate = _flag_track_id_change(candidate, previous_track_ids)
        previous_track_ids = candidate.track_ids or previous_track_ids
        frames_scanned += 1
        warning_frames += candidate.status == "warning"
        bad_frames += candidate.status == "bad"
        pools.add(candidate)

    with open(source.inference_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not ({"frame_index", "frame"} & set(reader.fieldnames)):
            raise ValueError("inference CSV does not contain a frame index")
        for row_index, row in enumerate(reader):
            if cancel_requested is not None and row_index % 1000 == 0 and cancel_requested():
                break
            frame_value = _finite(row.get("frame_index"))
            if frame_value is None:
                frame_value = _finite(row.get("frame"))
            if frame_value is None:
                continue
            frame_index = int(frame_value)
            if current is None or current.frame_index != frame_index:
                if current is not None:
                    finish_frame(current)
                current = _FrameAccumulator(source=source, frame_index=frame_index)
            current.add(row)
    if current is not None:
        finish_frame(current)
    return _SourceScanResult(
        candidates=pools.candidates(),
        frames_scanned=frames_scanned,
        warning_frames=warning_frames,
        bad_frames=bad_frames,
    )


def _flag_track_id_change(
    candidate: InferenceReviewCandidate,
    previous_ids: tuple[str, ...],
) -> InferenceReviewCandidate:
    if not previous_ids or not candidate.track_ids or candidate.track_ids == previous_ids:
        return candidate
    reasons = tuple(sorted(set(candidate.reasons) | {"track_id_changed"}))
    status = "bad" if candidate.status == "bad" else "warning"
    return replace(
        candidate,
        status=status,
        reasons=reasons,
        priority_score=_priority_score(
            status,
            reasons,
            detection_confidence=candidate.detection_confidence,
            keypoint_confidence=candidate.keypoint_confidence,
        ),
    )


def _candidate_supports_mode(candidate: InferenceReviewCandidate, mode: str) -> bool:
    if mode == RANK_KEYPOINT_CONFIDENCE:
        return candidate.layer_id == LAYER_KEYPOINTS and candidate.keypoint_confidence is not None
    if mode == RANK_POSE_CONFIDENCE:
        return candidate.layer_id == LAYER_KEYPOINTS and candidate.detection_confidence is not None
    if mode == RANK_SEGMENTATION_CONFIDENCE:
        return (
            candidate.layer_id == LAYER_SEGMENTATION and candidate.detection_confidence is not None
        )
    return True


def _ranking_key(candidate: InferenceReviewCandidate, mode: str) -> tuple:
    if mode in {RANK_POSE_CONFIDENCE, RANK_SEGMENTATION_CONFIDENCE}:
        metric = candidate.detection_confidence
    elif mode == RANK_KEYPOINT_CONFIDENCE:
        metric = candidate.keypoint_confidence
    else:
        return (
            -candidate.priority_score,
            candidate.detection_confidence if candidate.detection_confidence is not None else -1.0,
            candidate.video_name.casefold(),
            candidate.frame_index,
        )
    return (
        metric if metric is not None else math.inf,
        -_STATUS_SEVERITY.get(candidate.status, 0),
        -candidate.priority_score,
        candidate.video_name.casefold(),
        candidate.frame_index,
    )


def _priority_score(
    status: str,
    reasons: Sequence[str],
    *,
    detection_confidence: float | None,
    keypoint_confidence: float | None,
) -> float:
    score = {"good": 0.0, "warning": 50.0, "bad": 100.0}.get(status, 0.0)
    score += sum(_REASON_WEIGHTS.get(reason, 8.0) for reason in reasons)
    if detection_confidence is not None:
        score += max(0.0, 1.0 - detection_confidence) * 20.0
    if keypoint_confidence is not None:
        score += max(0.0, 1.0 - keypoint_confidence) * 10.0
    return score


def _model_path_from_csv(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle), {})
    except (OSError, csv.Error):
        return ""
    return str(row.get("model_path") or "").strip()


def _layer_confidence(row: Mapping[str, str], layer_id: str) -> float | None:
    return _finite(row.get("confidence" if layer_id == LAYER_KEYPOINTS else "conf"))


def _layer_bbox(row: Mapping[str, str], layer_id: str) -> tuple[float, ...]:
    names = (
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
        if layer_id == LAYER_KEYPOINTS
        else ("x1", "y1", "x2", "y2")
    )
    values = tuple(_finite(row.get(name)) for name in names)
    if not all(value is not None for value in values):
        return ()
    return tuple(value for value in values if value is not None)


def _overlay_from_inference_row(
    row: Mapping[str, str],
    layer_id: str,
) -> InferenceReviewOverlay:
    bbox = _layer_bbox(row, layer_id)
    polygon: tuple[tuple[float, float], ...] = ()
    if layer_id == LAYER_SEGMENTATION:
        try:
            raw_polygon = json.loads(str(row.get("mask_polygon") or ""))
            parsed = []
            if isinstance(raw_polygon, list):
                for point in raw_polygon:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        x = _finite(point[0])
                        y = _finite(point[1])
                        if x is not None and y is not None:
                            parsed.append((x, y))
            if len(parsed) >= 3:
                polygon = tuple(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    keypoints: list[tuple[str, float, float, float]] = []
    if layer_id == LAYER_KEYPOINTS:
        names = sorted(
            column[3:-2] for column in row if column.startswith("kp_") and column.endswith("_x")
        )
        for name in names:
            x = _finite(row.get(f"kp_{name}_x"))
            y = _finite(row.get(f"kp_{name}_y"))
            confidence = _finite(row.get(f"kp_{name}_conf"))
            if x is not None and y is not None:
                keypoints.append((name, x, y, confidence if confidence is not None else math.nan))
    track_id = str(row.get("track_id") or "").strip()
    animal_id = f"animal_{track_id}" if track_id and track_id not in {"-1", "nan"} else "animal"
    return InferenceReviewOverlay(
        animal_id=animal_id,
        bbox=bbox,
        mask_polygon=polygon,
        keypoints=tuple(keypoints),
    )


def _append_finite(destination: list[float], raw: object) -> None:
    value = _finite(raw)
    if value is not None:
        destination.append(value)


def _append_int(destination: list[int], raw: object) -> None:
    value = _finite(raw)
    if value is not None:
        destination.append(max(0, int(value)))


def _finite(raw: object) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _minimum(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return min(materialized) if materialized else None


__all__ = [
    "InferenceReviewCandidate",
    "InferenceReviewOverlay",
    "InferenceReviewScanResult",
    "ModelInferenceSource",
    "RANK_KEYPOINT_CONFIDENCE",
    "RANK_OVERALL",
    "RANK_POSE_CONFIDENCE",
    "RANK_SEGMENTATION_CONFIDENCE",
    "candidate_reasons",
    "discover_model_inference_sources",
    "rank_inference_review_candidates",
    "scan_project_inference_quality",
]
