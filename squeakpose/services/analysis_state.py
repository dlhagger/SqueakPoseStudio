"""Qt-free scale and region-of-interest state for analysis runs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FrameDimensions:
    width: int = 0
    height: int = 0


@dataclass(frozen=True, slots=True)
class AnalysisROI:
    """A named rectangular region in frame-pixel coordinates."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    type: str = "rect"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_name: str = "ROI",
        frame: FrameDimensions | None = None,
    ) -> AnalysisROI:
        left, right = sorted((float(value["x1"]), float(value["x2"])))
        top, bottom = sorted((float(value["y1"]), float(value["y2"])))
        if frame is not None:
            if frame.width > 0:
                left = min(max(left, 0.0), float(frame.width))
                right = min(max(right, 0.0), float(frame.width))
            if frame.height > 0:
                top = min(max(top, 0.0), float(frame.height))
                bottom = min(max(bottom, 0.0), float(frame.height))
        name = str(value.get("name") or "").strip() or default_name
        return cls(name=name, x1=left, y1=top, x2=right, y2=bottom)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_worker_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class AnalysisAnnotationSnapshot:
    frame: FrameDimensions
    scale_points: tuple[tuple[float, float], ...]
    pixel_distance: float
    real_world_distance_mm: float
    rois: tuple[AnalysisROI, ...]

    @property
    def mm_per_pixel(self) -> float | None:
        if self.pixel_distance <= 0:
            return None
        return self.real_world_distance_mm / self.pixel_distance


class AnalysisAnnotationState:
    """Own analysis scale and ROI transitions independently of the Qt canvas."""

    def __init__(
        self,
        *,
        frame_width: int = 0,
        frame_height: int = 0,
        real_world_distance_mm: float = 1.0,
    ) -> None:
        self._frame = FrameDimensions()
        self._scale_points: tuple[tuple[float, float], ...] = ()
        self._pixel_distance = 0.0
        self._real_world_distance_mm = float(real_world_distance_mm)
        self._rois: list[AnalysisROI] = []
        self.set_frame_dimensions(frame_width, frame_height)

    @property
    def frame(self) -> FrameDimensions:
        return self._frame

    @property
    def scale_points(self) -> tuple[tuple[float, float], ...]:
        return self._scale_points

    @property
    def pixel_distance(self) -> float:
        return self._pixel_distance

    @property
    def real_world_distance_mm(self) -> float:
        return self._real_world_distance_mm

    @property
    def mm_per_pixel(self) -> float | None:
        if self._pixel_distance <= 0:
            return None
        return self._real_world_distance_mm / self._pixel_distance

    @property
    def rois(self) -> tuple[AnalysisROI, ...]:
        return tuple(self._rois)

    def set_frame_dimensions(self, width: int, height: int) -> FrameDimensions:
        self._frame = FrameDimensions(max(0, int(width)), max(0, int(height)))
        return self._frame

    def set_scale_points(
        self,
        points: Sequence[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        self._scale_points = tuple((float(x), float(y)) for x, y in points[:2])
        if len(self._scale_points) == 2:
            (x1, y1), (x2, y2) = self._scale_points
            self._pixel_distance = math.hypot(x2 - x1, y2 - y1)
        else:
            self._pixel_distance = 0.0
        return self._scale_points

    def set_pixel_distance(self, distance: float) -> float:
        """Accept a canvas-computed distance while retaining the selected points."""
        self._pixel_distance = max(0.0, float(distance))
        return self._pixel_distance

    def set_real_world_distance(self, distance_mm: float) -> float:
        self._real_world_distance_mm = float(distance_mm)
        return self._real_world_distance_mm

    def clear_scale(self) -> None:
        self._scale_points = ()
        self._pixel_distance = 0.0

    def add_roi(
        self,
        value: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> AnalysisROI:
        default_name = f"ROI {len(self._rois) + 1}"
        payload = dict(value)
        if name is not None:
            payload["name"] = name
        roi = AnalysisROI.from_mapping(
            payload,
            default_name=default_name,
            frame=self._frame,
        )
        self._rois.append(roi)
        return roi

    def replace_rois(self, rois: Iterable[Mapping[str, Any] | AnalysisROI]) -> None:
        self._rois = []
        for value in rois:
            if isinstance(value, AnalysisROI):
                value = value.as_worker_dict()
            self.add_roi(value)

    def delete_roi(self, index: int) -> bool:
        if not 0 <= int(index) < len(self._rois):
            return False
        del self._rois[int(index)]
        return True

    def clear_rois(self) -> None:
        self._rois = []

    def clear(self) -> None:
        self.clear_scale()
        self.clear_rois()

    def worker_rois(self) -> list[dict[str, Any]]:
        """Return detached dictionaries in the existing worker JSON shape."""
        return [roi.as_worker_dict() for roi in self._rois]

    def snapshot(self) -> AnalysisAnnotationSnapshot:
        return AnalysisAnnotationSnapshot(
            frame=self._frame,
            scale_points=self._scale_points,
            pixel_distance=self._pixel_distance,
            real_world_distance_mm=self._real_world_distance_mm,
            rois=tuple(self._rois),
        )

    def restore(self, snapshot: AnalysisAnnotationSnapshot) -> None:
        self._frame = snapshot.frame
        self._scale_points = tuple(snapshot.scale_points)
        self._pixel_distance = float(snapshot.pixel_distance)
        self._real_world_distance_mm = float(snapshot.real_world_distance_mm)
        self._rois = list(snapshot.rois)


__all__ = [
    "AnalysisAnnotationSnapshot",
    "AnalysisAnnotationState",
    "AnalysisROI",
    "FrameDimensions",
]
