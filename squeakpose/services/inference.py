"""Qt-free planning and result aggregation for project video inference."""

from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from squeakpose.core import atomic_write_text
from squeakpose.json_io import read_json_file
from squeakpose.project.layers import (
    LAYER_DEFINITIONS,
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    normalize_layer_id,
)
from squeakpose.project.safety import ProjectPathError, require_path_within_project

ManifestWriter = Callable[[str, str], None]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class InferenceJobPlan:
    run_id: str
    job_index: int
    job_total: int
    layer_id: str
    workflow: str
    model_path: str
    video_path: str
    csv_path: str
    preview_path: str
    classes: tuple[str, ...]
    keypoint_names: tuple[str, ...]
    device: str
    batch_size: int
    total_frames: int
    fps: float

    @property
    def display_name(self) -> str:
        return layer_definition(self.layer_id).display_name

    def worker_config(self) -> dict[str, Any]:
        """Return the existing inference worker JSON payload."""
        return {
            "layer_id": self.layer_id,
            "mode": self.workflow,
            "model_path": self.model_path,
            "video_path": self.video_path,
            "csv_path": self.csv_path,
            "preview_path": self.preview_path,
            "classes": list(self.classes),
            "kp_names": list(self.keypoint_names),
            "device": self.device,
            "batch_size": self.batch_size,
            "total_frames": self.total_frames,
            "fps": self.fps,
        }


@dataclass(frozen=True, slots=True)
class InferenceRunPlan:
    project_root: str
    run_id: str
    created_at: str
    video_path: str
    manifest_path: str
    jobs: tuple[InferenceJobPlan, ...]

    @property
    def output_directories(self) -> tuple[str, ...]:
        paths = {os.path.dirname(self.manifest_path)}
        paths.update(os.path.dirname(job.csv_path) for job in self.jobs)
        return tuple(sorted(paths, key=str.casefold))


@dataclass(frozen=True, slots=True)
class InferencePassResult:
    layer_id: str
    workflow: str
    model_path: str
    csv_path: str
    preview_path: str = ""
    rows_written: int = 0
    processed_frames: int = 0
    canceled: bool = False
    had_error: bool = False
    error_message: str = ""

    @property
    def discard_paths(self) -> tuple[str, ...]:
        if self.rows_written > 0 or not (self.had_error or self.canceled):
            return ()
        return tuple(path for path in (self.csv_path, self.preview_path) if path)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceRunSummary:
    run_id: str
    video_path: str
    canceled: bool
    results: tuple[InferencePassResult, ...]
    manifest_path: str
    manifest_error: str = ""

    @property
    def failed_count(self) -> int:
        return sum(result.had_error for result in self.results)

    @property
    def canceled_count(self) -> int:
        return sum(result.canceled for result in self.results)

    @property
    def successful_count(self) -> int:
        return sum(not result.had_error and not result.canceled for result in self.results)

    @property
    def details(self) -> tuple[str, ...]:
        lines: list[str] = []
        for result in self.results:
            name = layer_definition(result.layer_id).display_name
            if result.had_error:
                lines.append(f"{name}: failed — {result.error_message or 'failed'}")
            elif result.canceled:
                lines.append(f"{name}: canceled ({result.rows_written} rows retained)")
            else:
                detail = f"{name}: {result.rows_written} rows → {result.csv_path}"
                if result.preview_path:
                    detail += f"\nPreview → {result.preview_path}"
                lines.append(detail)
        if self.manifest_path:
            lines.append(f"Run manifest: {self.manifest_path}")
        elif self.manifest_error:
            lines.append(f"Run manifest failed: {self.manifest_error}")
        return tuple(lines)


@dataclass(frozen=True, slots=True)
class VideoInferenceStatus:
    """Project inference history summarized for one source video."""

    video_path: str
    successful_layers: tuple[str, ...]
    latest_created_at: str
    run_count: int


def video_identity(video_path: str) -> str:
    """Return a stable comparison key for local files and project symlinks."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(video_path))))


def project_video_inference_statuses(project_root: str) -> dict[str, VideoInferenceStatus]:
    """Index successful inference layers from the project's persisted run manifests.

    Invalid, incomplete, or unrelated files are ignored so a damaged manifest cannot
    prevent the inference picker from opening.
    """
    runs_dir = os.path.join(os.path.abspath(project_root), "inference outputs", "runs")
    if not os.path.isdir(runs_dir):
        return {}

    history: dict[str, dict[str, Any]] = {}
    try:
        names = sorted(os.listdir(runs_dir), key=str.casefold)
    except OSError:
        return {}
    for name in names:
        if name.startswith(".") or not name.lower().endswith(".json"):
            continue
        manifest_path = os.path.join(runs_dir, name)
        try:
            payload = read_json_file(manifest_path, require_object=True)
        except (OSError, UnicodeError, ValueError):
            continue
        if not isinstance(payload, dict) or not str(payload.get("video_path") or ""):
            continue
        video_path = str(payload["video_path"])
        key = video_identity(video_path)
        record = history.setdefault(
            key,
            {"video_path": video_path, "layers": set(), "latest": "", "runs": 0},
        )
        record["runs"] += 1
        created_at = str(payload.get("created_at") or "")
        if created_at >= record["latest"]:
            record["latest"] = created_at
            record["video_path"] = video_path
        passes = payload.get("passes")
        if not isinstance(passes, list):
            continue
        for inference_pass in passes:
            if not isinstance(inference_pass, dict):
                continue
            if inference_pass.get("had_error") or inference_pass.get("canceled"):
                continue
            layer_id = normalize_layer_id(inference_pass.get("layer_id"), default="")
            if layer_id in LAYER_DEFINITIONS:
                record["layers"].add(layer_id)

    return {
        key: VideoInferenceStatus(
            video_path=str(record["video_path"]),
            successful_layers=tuple(
                layer_id for layer_id in LAYER_DEFINITIONS if layer_id in record["layers"]
            ),
            latest_created_at=str(record["latest"]),
            run_count=int(record["runs"]),
        )
        for key, record in history.items()
    }


def configured_inference_layers(
    active_layer: str,
    model_paths: Mapping[str, str],
) -> tuple[str, ...]:
    """Return configured layers in active-first project order."""
    normalized_models: dict[str, str] = {}
    for raw_layer_id, model_path in model_paths.items():
        layer_id = normalize_layer_id(raw_layer_id, default="")
        if layer_id in LAYER_DEFINITIONS and str(model_path or ""):
            normalized_models[layer_id] = str(model_path)
    ordered = (normalize_layer_id(active_layer), *LAYER_DEFINITIONS)
    return tuple(dict.fromkeys(layer_id for layer_id in ordered if layer_id in normalized_models))


def create_inference_run_id(
    video_path: str,
    *,
    created_at: datetime.datetime | None = None,
    token: str | None = None,
) -> str:
    """Create a filesystem-safe, collision-resistant ID for one project run."""
    timestamp = (created_at or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    safe_token = re.sub(r"[^A-Za-z0-9]+", "", str(token or uuid.uuid4().hex))[:12]
    if not safe_token:
        raise ValueError("inference run token must contain a letter or number")
    return f"{safe_stem}_{timestamp}_{safe_token}"


def plan_inference_run(
    *,
    project_root: str,
    video_path: str,
    active_layer: str,
    model_paths: Mapping[str, str],
    pose_classes: Sequence[str] = (),
    segmentation_classes: Sequence[str] = (),
    keypoint_names: Sequence[str] = (),
    device: str = "cpu",
    batch_size: int = 1,
    total_frames: int = 0,
    fps: float = 0.0,
    created_at: datetime.datetime | None = None,
    run_id: str | None = None,
) -> InferenceRunPlan:
    """Plan ordered per-layer worker jobs with project-contained output paths."""
    root = os.path.abspath(project_root)
    if not str(video_path or ""):
        raise ValueError("video_path must not be empty")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")

    timestamp = created_at or datetime.datetime.now()
    resolved_run_id = run_id or create_inference_run_id(video_path, created_at=timestamp)
    if not _RUN_ID_RE.fullmatch(resolved_run_id) or resolved_run_id in {".", ".."}:
        raise ValueError(f"invalid inference run id: {resolved_run_id!r}")

    normalized_models = {
        normalize_layer_id(layer_id, default=""): str(path or "")
        for layer_id, path in model_paths.items()
        if normalize_layer_id(layer_id, default="") in LAYER_DEFINITIONS
    }
    layer_ids = configured_inference_layers(active_layer, normalized_models)
    if not layer_ids:
        raise ValueError("at least one inference model must be configured")

    manifest_path = require_path_within_project(
        root,
        os.path.join(root, "inference outputs", "runs", f"{resolved_run_id}.json"),
        purpose="inference run manifest",
        allow_root=False,
    )
    jobs: list[InferenceJobPlan] = []
    for index, layer_id in enumerate(layer_ids, start=1):
        layer = layer_definition(layer_id)
        output_root = require_path_within_project(
            root,
            os.path.join(root, "inference outputs", layer.id),
            purpose=f"{layer.id} inference output directory",
            allow_root=False,
        )
        csv_path = require_path_within_project(
            root,
            os.path.join(output_root, f"{resolved_run_id}{layer.inference_suffix}"),
            purpose=f"{layer.id} inference CSV",
            allow_root=False,
        )
        preview_path = ""
        if layer_id == LAYER_DEPTH:
            preview_path = require_path_within_project(
                root,
                f"{os.path.splitext(csv_path)[0]}_preview.mp4",
                purpose="depth inference preview",
                allow_root=False,
            )
        if layer_id == LAYER_KEYPOINTS:
            classes = tuple(str(value) for value in pose_classes)
            keypoints = tuple(str(value) for value in keypoint_names)
        elif layer_id == LAYER_SEGMENTATION:
            classes = tuple(str(value) for value in segmentation_classes)
            keypoints = ()
        else:
            classes = ()
            keypoints = ()
        jobs.append(
            InferenceJobPlan(
                run_id=resolved_run_id,
                job_index=index,
                job_total=len(layer_ids),
                layer_id=layer_id,
                workflow=layer.worker_mode,
                model_path=normalized_models[layer_id],
                video_path=str(video_path),
                csv_path=csv_path,
                preview_path=preview_path,
                classes=classes,
                keypoint_names=keypoints,
                device=str(device or "cpu"),
                batch_size=int(batch_size),
                total_frames=max(0, int(total_frames)),
                fps=max(0.0, float(fps)),
            )
        )
    return InferenceRunPlan(
        project_root=root,
        run_id=resolved_run_id,
        created_at=timestamp.isoformat(),
        video_path=str(video_path),
        manifest_path=manifest_path,
        jobs=tuple(jobs),
    )


def prepare_inference_run(plan: InferenceRunPlan) -> None:
    """Create only the validated output directories required by a run plan."""
    for directory in plan.output_directories:
        contained = require_path_within_project(
            plan.project_root,
            directory,
            purpose="inference output directory",
            allow_root=False,
        )
        os.makedirs(contained, exist_ok=True)


def aggregate_inference_result(
    job: InferenceJobPlan,
    event: Mapping[str, Any] | None,
    *,
    project_root: str,
    exit_code: int = 0,
    crashed: bool = False,
    cancel_requested: bool = False,
    stderr: str = "",
) -> InferencePassResult:
    """Combine a worker event and process outcome without discarding partial rows."""
    payload = dict(event or {})
    if payload.get("event") == "error":
        payload = {
            "rows_written": 0,
            "processed_frames": 0,
            "canceled": False,
            "had_error": True,
            "error_message": str(payload.get("error_message") or "Inference worker error"),
        }
    if not event:
        payload = {
            "rows_written": 0,
            "processed_frames": 0,
            "canceled": bool(cancel_requested),
            "had_error": not bool(cancel_requested),
            "error_message": str(stderr or f"Process exited with code {exit_code}."),
        }

    rows_written = max(0, int(payload.get("rows_written") or 0))
    canceled = bool(payload.get("canceled")) or bool(cancel_requested)
    had_error = bool(payload.get("had_error")) or (
        not canceled and (bool(crashed) or int(exit_code) != 0)
    )
    error_message = str(
        payload.get("error_message") or stderr or ("Unknown inference error" if had_error else "")
    )

    csv_path, path_error = _result_path(
        project_root,
        payload.get("csv_path"),
        job.csv_path,
        purpose="inference result CSV",
    )
    preview_path, preview_error = _result_path(
        project_root,
        payload.get("preview_path"),
        job.preview_path,
        purpose="inference result preview",
        optional=True,
    )
    unsafe_error = path_error or preview_error
    if unsafe_error:
        had_error = True
        error_message = f"{error_message}; {unsafe_error}".strip("; ")

    return InferencePassResult(
        layer_id=job.layer_id,
        workflow=job.workflow,
        model_path=job.model_path,
        csv_path=csv_path,
        preview_path=preview_path,
        rows_written=rows_written,
        processed_frames=max(0, int(payload.get("processed_frames") or 0)),
        canceled=canceled,
        had_error=had_error,
        error_message=error_message,
    )


def build_inference_manifest(
    plan: InferenceRunPlan,
    results: Sequence[InferencePassResult],
    *,
    canceled: bool,
) -> dict[str, Any]:
    """Build the schema-compatible project inference manifest payload."""
    return {
        "schema_version": 1,
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "video_path": plan.video_path,
        "canceled": bool(canceled),
        "passes": [result.as_dict() for result in results],
    }


def finalize_inference_run(
    plan: InferenceRunPlan,
    results: Sequence[InferencePassResult],
    *,
    canceled: bool = False,
    writer: ManifestWriter = atomic_write_text,
) -> InferenceRunSummary:
    """Persist the final manifest and return a UI-neutral structured summary."""
    result_items = tuple(results)
    run_canceled = bool(canceled or any(result.canceled for result in result_items))
    manifest = build_inference_manifest(plan, result_items, canceled=run_canceled)
    manifest_path = ""
    manifest_error = ""
    try:
        prepare_inference_run(plan)
        writer(plan.manifest_path, json.dumps(manifest, indent=2))
        manifest_path = plan.manifest_path
    except Exception as exc:
        manifest_error = str(exc) or type(exc).__name__
    return InferenceRunSummary(
        run_id=plan.run_id,
        video_path=plan.video_path,
        canceled=run_canceled,
        results=result_items,
        manifest_path=manifest_path,
        manifest_error=manifest_error,
    )


class InferenceRunAccumulator:
    """Collect at most one result per planned layer before finalization."""

    def __init__(self, plan: InferenceRunPlan):
        self.plan = plan
        self._jobs = {job.layer_id: job for job in plan.jobs}
        self._results: list[InferencePassResult] = []
        self._recorded_layers: set[str] = set()

    @property
    def results(self) -> tuple[InferencePassResult, ...]:
        return tuple(self._results)

    @property
    def pending_jobs(self) -> tuple[InferenceJobPlan, ...]:
        return tuple(job for job in self.plan.jobs if job.layer_id not in self._recorded_layers)

    def record(
        self,
        job: InferenceJobPlan,
        event: Mapping[str, Any] | None,
        *,
        exit_code: int = 0,
        crashed: bool = False,
        cancel_requested: bool = False,
        stderr: str = "",
    ) -> InferencePassResult:
        planned = self._jobs.get(job.layer_id)
        if planned != job:
            raise ValueError("inference result does not belong to this run plan")
        if job.layer_id in self._recorded_layers:
            raise ValueError(f"inference result already recorded for layer: {job.layer_id}")
        result = aggregate_inference_result(
            job,
            event,
            project_root=self.plan.project_root,
            exit_code=exit_code,
            crashed=crashed,
            cancel_requested=cancel_requested,
            stderr=stderr,
        )
        self._results.append(result)
        self._recorded_layers.add(job.layer_id)
        return result

    def finalize(
        self,
        *,
        canceled: bool = False,
        writer: ManifestWriter = atomic_write_text,
    ) -> InferenceRunSummary:
        return finalize_inference_run(
            self.plan,
            self._results,
            canceled=canceled,
            writer=writer,
        )


def _result_path(
    project_root: str,
    reported_path: Any,
    planned_path: str,
    *,
    purpose: str,
    optional: bool = False,
) -> tuple[str, str]:
    raw_path = str(reported_path or planned_path or "")
    if optional and not raw_path:
        return "", ""
    try:
        return (
            require_path_within_project(
                project_root,
                raw_path,
                purpose=purpose,
                allow_root=False,
            ),
            "",
        )
    except ProjectPathError as exc:
        return planned_path, str(exc)


__all__ = [
    "InferenceJobPlan",
    "InferencePassResult",
    "InferenceRunAccumulator",
    "InferenceRunPlan",
    "InferenceRunSummary",
    "VideoInferenceStatus",
    "aggregate_inference_result",
    "build_inference_manifest",
    "configured_inference_layers",
    "create_inference_run_id",
    "finalize_inference_run",
    "plan_inference_run",
    "prepare_inference_run",
    "project_video_inference_statuses",
    "video_identity",
]
