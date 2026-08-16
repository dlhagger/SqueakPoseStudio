"""Qt-free state and path planning for the depth assistant."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from squeakpose.core import remove_path, staging_path_for
from squeakpose.project.safety import require_path_within_project

DepthViewMode = Literal["original", "depth", "overlay"]
VALID_DEPTH_VIEW_MODES = frozenset({"original", "depth", "overlay"})
DEFAULT_RANGE_TEXT = "No saved depth range · Near = bright"
INVALID_RANGE_TEXT = "Depth range unavailable · Near = bright"
DEFAULT_PROBE_TEXT = "Right-click the image to sample raw depth."


@dataclass(frozen=True, slots=True)
class DepthRangeSummary:
    p02_depth: float
    p98_depth: float
    median_depth: float
    min_depth: float | None = None
    max_depth: float | None = None
    valid_pixels: int | None = None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> DepthRangeSummary | None:
        try:
            low = float(metadata["p02_depth"])
            high = float(metadata["p98_depth"])
            median = float(metadata["median_depth"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (low, high, median)):
            return None
        try:
            minimum = _optional_finite_float(metadata.get("min_depth"))
            maximum = _optional_finite_float(metadata.get("max_depth"))
            valid_pixels = _optional_nonnegative_int(metadata.get("valid_pixels"))
        except (TypeError, ValueError):
            return None
        return cls(
            p02_depth=low,
            p98_depth=high,
            median_depth=median,
            min_depth=minimum,
            max_depth=maximum,
            valid_pixels=valid_pixels,
        )


@dataclass(frozen=True, slots=True)
class DepthProbe:
    x: int
    y: int
    depth: float | None
    valid: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DepthProbe:
        x = int(value.get("x", 0))
        y = int(value.get("y", 0))
        raw_depth = value.get("depth")
        try:
            depth = float(raw_depth) if raw_depth is not None else None
        except (TypeError, ValueError):
            depth = None
        valid = bool(value.get("valid", depth is not None))
        if depth is None or not math.isfinite(depth) or depth <= 0 or not valid:
            depth = None
            valid = False
        return cls(x=x, y=y, depth=depth, valid=valid)

    def as_mapping(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "depth": self.depth, "valid": self.valid}


@dataclass
class DepthAssistantSnapshot:
    view_mode: DepthViewMode
    image_name: str
    metadata: dict[str, Any] | None
    probes: list[DepthProbe]
    probe_error: str


@dataclass
class DepthAssistantState:
    """Per-image depth display, metadata, and probe state."""

    view_mode: DepthViewMode = "depth"
    image_name: str = ""
    metadata: dict[str, Any] | None = None
    probes: list[DepthProbe] = field(default_factory=list)
    probe_error: str = ""
    max_probes: int = 6
    _undo_snapshots: list[DepthAssistantSnapshot] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def depth_range(self) -> DepthRangeSummary | None:
        if self.metadata is None:
            return None
        return DepthRangeSummary.from_metadata(self.metadata)

    @property
    def has_valid_metadata(self) -> bool:
        return self.depth_range is not None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_snapshots)

    def set_view_mode(self, mode: str) -> DepthViewMode:
        normalized = normalize_depth_view_mode(mode)
        self.view_mode = normalized
        return normalized

    def load_image(
        self,
        image_name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        probe_error: str = "",
    ) -> None:
        normalized_name = str(image_name or "")
        changed = normalized_name != self.image_name
        self.image_name = normalized_name
        self.set_metadata(metadata)
        self.probe_error = str(probe_error or "")
        if changed:
            self.probes = []
            self._undo_snapshots.clear()

    def set_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        self.metadata = deepcopy(dict(metadata)) if metadata is not None else None

    def add_probe(self, probe: Mapping[str, Any] | DepthProbe) -> DepthProbe:
        normalized = probe if isinstance(probe, DepthProbe) else DepthProbe.from_mapping(probe)
        self.probes.append(normalized)
        limit = max(1, int(self.max_probes))
        self.probes = self.probes[-limit:]
        return normalized

    def clear_probes(self) -> None:
        self.probes = []

    def clear(self) -> None:
        self.image_name = ""
        self.metadata = None
        self.probes = []
        self.probe_error = ""
        self._undo_snapshots.clear()

    def range_text(self) -> str:
        if self.metadata is None:
            return DEFAULT_RANGE_TEXT
        summary = self.depth_range
        if summary is None:
            return INVALID_RANGE_TEXT
        unit = depth_unit(self.metadata)
        if unit == "m":
            return (
                f"Range (2–98%): {summary.p02_depth:.3f}–{summary.p98_depth:.3f} m · "
                f"median {summary.median_depth:.3f} m · Near = bright"
            )
        return (
            f"Range (2–98%): {summary.p02_depth:.3f}–{summary.p98_depth:.3f} relative · "
            f"median {summary.median_depth:.3f} relative · Near = bright"
        )

    def probe_text(self) -> str:
        if not self.probes:
            return self.probe_error or DEFAULT_PROBE_TEXT
        lines = ["Pixel probes:"]
        for index, probe in enumerate(self.probes, start=1):
            lines.append(
                f"{index}. ({probe.x}, {probe.y}): {format_depth_value(probe.depth, self.metadata)}"
            )
        valid = [probe.depth for probe in self.probes[-2:] if probe.depth is not None]
        if len(valid) == 2:
            delta = abs(float(valid[1]) - float(valid[0]))
            lines.append(f"Δ last two: {format_depth_value(delta, self.metadata)}")
        return "\n".join(lines)

    def snapshot(self) -> DepthAssistantSnapshot:
        return DepthAssistantSnapshot(
            view_mode=self.view_mode,
            image_name=self.image_name,
            metadata=deepcopy(self.metadata),
            probes=deepcopy(self.probes),
            probe_error=self.probe_error,
        )

    def restore(self, snapshot: DepthAssistantSnapshot) -> None:
        self.view_mode = normalize_depth_view_mode(snapshot.view_mode)
        self.image_name = str(snapshot.image_name)
        self.metadata = deepcopy(snapshot.metadata)
        self.probes = deepcopy(snapshot.probes)
        self.probe_error = str(snapshot.probe_error)

    def push_undo_snapshot(self) -> DepthAssistantSnapshot:
        snapshot = self.snapshot()
        self._undo_snapshots.append(deepcopy(snapshot))
        return snapshot

    def undo(self) -> bool:
        if not self._undo_snapshots:
            return False
        self.restore(self._undo_snapshots.pop())
        return True


@dataclass(frozen=True, slots=True)
class DepthPredictionTargetPlan:
    final_map: str
    final_preview: str
    final_metadata: str
    staged_map: str
    staged_preview: str
    staged_metadata: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DepthPredictionTargetPlan:
        """Restore the legacy prediction-target dictionary representation."""
        return cls(
            final_map=str(value.get("final_map") or ""),
            final_preview=str(value.get("final_preview") or ""),
            final_metadata=str(value.get("final_metadata") or ""),
            staged_map=str(value.get("staged_map") or ""),
            staged_preview=str(value.get("staged_preview") or ""),
            staged_metadata=str(value.get("staged_metadata") or ""),
        )

    def as_mapping(self) -> dict[str, str]:
        return {
            "final_map": self.final_map,
            "final_preview": self.final_preview,
            "final_metadata": self.final_metadata,
            "staged_map": self.staged_map,
            "staged_preview": self.staged_preview,
            "staged_metadata": self.staged_metadata,
        }

    def worker_paths(self) -> dict[str, str]:
        return {
            "depth_map_path": self.staged_map,
            "depth_preview_path": self.staged_preview,
            "depth_metadata_path": self.staged_metadata,
        }

    def replacements(self) -> tuple[tuple[str, str], ...]:
        replacements = (
            (self.staged_map, self.final_map),
            (self.staged_preview, self.final_preview),
            (self.staged_metadata, self.final_metadata),
        )
        if not all(staged and final for staged, final in replacements):
            raise ValueError("Depth prediction output transaction is incomplete.")
        return replacements

    def staged_paths(self) -> tuple[str, str, str]:
        return self.staged_map, self.staged_preview, self.staged_metadata


@dataclass(frozen=True, slots=True)
class DepthArtifactPlan:
    """Resolved artifacts and alignment expected for one displayed image."""

    image_name: str
    map_path: str
    metadata_path: str
    preview_path: str
    expected_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class DepthArtifactLoadResult:
    plan: DepthArtifactPlan
    depth_map: Any = None
    metadata: dict[str, Any] | None = None
    map_error: str = ""
    metadata_error: str = ""
    preview_available: bool = False


def plan_depth_artifacts(
    *,
    depth_image_dir: str,
    depth_preview_dir: str,
    image_name: str,
    image_width: int,
    image_height: int,
    project_root: str | None = None,
) -> DepthArtifactPlan:
    """Resolve depth sidecars without reading arrays, JSON, or preview pixels."""
    normalized_name = os.path.basename(str(image_name or ""))
    stem = os.path.splitext(normalized_name)[0]
    if not stem:
        raise ValueError("A source image name is required for depth artifacts.")
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("Depth artifact alignment requires positive image dimensions.")
    paths = {
        "map": os.path.abspath(os.path.join(depth_image_dir, f"{stem}.npy")),
        "metadata": os.path.abspath(os.path.join(depth_image_dir, f"{stem}_depth.json")),
        "preview": os.path.abspath(os.path.join(depth_preview_dir, f"{stem}_depth.png")),
    }
    if project_root:
        for kind, path in paths.items():
            paths[kind] = require_path_within_project(
                project_root,
                path,
                purpose=f"depth {kind} artifact",
                allow_root=False,
            )
    return DepthArtifactPlan(
        image_name=normalized_name,
        map_path=paths["map"],
        metadata_path=paths["metadata"],
        preview_path=paths["preview"],
        expected_shape=(height, width),
    )


def load_depth_artifacts(
    plan: DepthArtifactPlan,
    *,
    array_reader: Callable[[str], Any] | None,
    metadata_reader: Callable[[str], Mapping[str, Any]] | None,
    is_file: Callable[[str], bool] = os.path.isfile,
) -> DepthArtifactLoadResult:
    """Load and validate depth data through injected, UI-independent readers."""
    depth_map: Any = None
    map_error = ""
    if array_reader is None:
        map_error = "NumPy is unavailable; pixel sampling is disabled."
    elif not is_file(plan.map_path):
        map_error = "No raw depth map is available for pixel sampling."
    else:
        try:
            candidate = array_reader(plan.map_path)
            dimensions = int(getattr(candidate, "ndim"))
            shape = tuple(int(value) for value in getattr(candidate, "shape"))
            if dimensions != 2:
                raise ValueError(f"expected 2 dimensions, received {dimensions}")
            if shape != plan.expected_shape:
                raise ValueError(f"map {shape} does not match image {plan.expected_shape}")
            depth_map = candidate
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            map_error = f"Pixel sampling unavailable: {exc}"

    metadata: dict[str, Any] | None = None
    metadata_error = ""
    if metadata_reader is not None and is_file(plan.metadata_path):
        try:
            loaded_metadata = metadata_reader(plan.metadata_path)
            if not isinstance(loaded_metadata, Mapping):
                raise ValueError("depth metadata must be a JSON object")
            metadata = deepcopy(dict(loaded_metadata))
        except (OSError, TypeError, ValueError) as exc:
            metadata_error = f"Depth metadata unavailable: {exc}"

    return DepthArtifactLoadResult(
        plan=plan,
        depth_map=depth_map,
        metadata=metadata,
        map_error=map_error,
        metadata_error=metadata_error,
        preview_available=is_file(plan.preview_path),
    )


def plan_depth_prediction_targets(
    *,
    depth_image_dir: str,
    depth_preview_dir: str,
    image_path: str,
    project_root: str | None = None,
    staging_factory: Callable[[str], str] = staging_path_for,
) -> DepthPredictionTargetPlan:
    """Reserve sibling staging files for one atomic depth prediction commit."""
    image_stem = os.path.splitext(os.path.basename(str(image_path)))[0] or "image"
    final = {
        "map": os.path.abspath(os.path.join(depth_image_dir, f"{image_stem}.npy")),
        "preview": os.path.abspath(os.path.join(depth_preview_dir, f"{image_stem}_depth.png")),
        "metadata": os.path.abspath(os.path.join(depth_image_dir, f"{image_stem}_depth.json")),
    }
    if project_root:
        for kind, path in final.items():
            final[kind] = require_path_within_project(
                project_root,
                path,
                purpose=f"depth prediction {kind}",
                allow_root=False,
            )

    staged: dict[str, str] = {}
    try:
        for kind, path in final.items():
            candidate = os.path.abspath(staging_factory(path))
            if candidate == path or os.path.dirname(candidate) != os.path.dirname(path):
                raise ValueError("Depth staging paths must be siblings of their final targets.")
            staged[kind] = candidate
    except Exception:
        for path in staged.values():
            try:
                remove_path(path)
            except OSError:
                pass
        raise
    return DepthPredictionTargetPlan(
        final_map=final["map"],
        final_preview=final["preview"],
        final_metadata=final["metadata"],
        staged_map=staged["map"],
        staged_preview=staged["preview"],
        staged_metadata=staged["metadata"],
    )


def normalize_depth_view_mode(value: str) -> DepthViewMode:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_DEPTH_VIEW_MODES:
        return cast(DepthViewMode, normalized)
    return "depth"


def depth_unit(metadata: Mapping[str, Any] | None) -> str:
    units = str((metadata or {}).get("units") or "estimated_meters").strip().lower()
    if units in {"m", "meter", "meters", "metre", "metres", "estimated_meters"}:
        return "m"
    return "relative"


def format_depth_value(value: float | None, metadata: Mapping[str, Any] | None = None) -> str:
    if value is None:
        return "invalid"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "invalid"
    unit = depth_unit(metadata)
    return f"{numeric:.3f} m" if unit == "m" else f"{numeric:.3f} relative"


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("depth metadata values must be finite")
    return numeric


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 0:
        raise ValueError("valid pixel count must be nonnegative")
    return numeric
