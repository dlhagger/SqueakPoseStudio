"""Controller boundary for depth display and pixel probing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from squeakpose.annotation.depth import (
    DepthArtifactLoadResult,
    DepthArtifactPlan,
    DepthAssistantSnapshot,
    DepthAssistantState,
    DepthProbe,
    DepthViewMode,
    load_depth_artifacts,
)


class DepthSampler(Protocol):
    def __call__(
        self,
        depth_map: Any,
        *,
        x: float,
        y: float,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DepthProbeAttempt:
    probe: DepthProbe | None = None
    error: str = ""

    @property
    def accepted(self) -> bool:
        return self.probe is not None


def _ignore_state(_snapshot: DepthAssistantSnapshot) -> None:
    pass


@dataclass(frozen=True, slots=True)
class DepthControllerCallbacks:
    state_changed: Callable[[DepthAssistantSnapshot], None] = _ignore_state


class DepthAssistantController:
    """Own per-image depth assistant state behind an explicit sampling dependency."""

    def __init__(
        self,
        *,
        sampler: DepthSampler,
        state: DepthAssistantState | None = None,
        callbacks: DepthControllerCallbacks | None = None,
    ) -> None:
        self.state = state or DepthAssistantState()
        self._sampler = sampler
        self._callbacks = callbacks or DepthControllerCallbacks()
        self._depth_map: Any = None

    @property
    def depth_map(self) -> Any:
        return self._depth_map

    def set_view_mode(self, mode: str) -> DepthViewMode:
        normalized = self.state.set_view_mode(mode)
        self._emit_state()
        return normalized

    def load_image(
        self,
        image_name: str,
        *,
        depth_map: Any = None,
        metadata: Mapping[str, Any] | None = None,
        probe_error: str = "",
    ) -> None:
        self.state.load_image(image_name, metadata=metadata, probe_error=probe_error)
        self._depth_map = depth_map
        self._emit_state()

    def load_artifacts(
        self,
        plan: DepthArtifactPlan,
        *,
        array_reader: Callable[[str], Any] | None,
        metadata_reader: Callable[[str], Mapping[str, Any]] | None,
        is_file: Callable[[str], bool],
    ) -> DepthArtifactLoadResult:
        """Load a planned artifact set and bind its validated sampling state."""
        result = load_depth_artifacts(
            plan,
            array_reader=array_reader,
            metadata_reader=metadata_reader,
            is_file=is_file,
        )
        self.load_image(
            plan.image_name,
            depth_map=result.depth_map,
            metadata=result.metadata,
            probe_error=result.map_error,
        )
        return result

    def clear_image(self) -> None:
        self._depth_map = None
        self.state.clear()
        self._emit_state()

    def probe(self, x: float, y: float) -> DepthProbeAttempt:
        if self._depth_map is None:
            error = self.state.probe_error or "No aligned raw depth map is available."
            return DepthProbeAttempt(error=error)
        try:
            sampled = self._sampler(self._depth_map, x=float(x), y=float(y))
            probe = self.state.add_probe(sampled)
        except (IndexError, TypeError, ValueError) as exc:
            return DepthProbeAttempt(error=str(exc))
        self._emit_state()
        return DepthProbeAttempt(probe=probe)

    def clear_probes(self) -> bool:
        if not self.state.probes:
            return False
        self.state.clear_probes()
        self._emit_state()
        return True

    def _emit_state(self) -> DepthAssistantSnapshot:
        snapshot = self.state.snapshot()
        self._callbacks.state_changed(snapshot)
        return snapshot


__all__ = [
    "DepthAssistantController",
    "DepthControllerCallbacks",
    "DepthProbeAttempt",
    "DepthSampler",
]
