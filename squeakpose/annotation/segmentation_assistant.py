"""Qt-free planning helpers for prompt-based segmentation assistants."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from squeakpose.annotation.segmentation import Point, PromptPoint


@dataclass(frozen=True, slots=True)
class SamPromptRequest:
    """A complete, model-agnostic prompt inference request."""

    source: str
    class_id: int
    prompts: tuple[PromptPoint, ...]

    @property
    def points(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y, _label in self.prompts]

    @property
    def labels(self) -> list[int]:
        return [int(label) for _x, _y, label in self.prompts]

    def predict_kwargs(self) -> dict[str, Any]:
        """Return the narrow kwargs accepted by Ultralytics-style predictors."""
        return {
            "source": self.source,
            "points": self.points,
            "labels": self.labels,
            "verbose": False,
        }


@dataclass(frozen=True, slots=True)
class SamContourResult:
    points: tuple[Point, ...]
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class SamContourSelection:
    result: SamContourResult | None = None
    failure: Literal["", "no_masks", "no_polygon"] = ""


def discover_sam_weight_candidates(
    project_root: str,
    *,
    default_filename: str,
    list_directory: Callable[[str], Iterable[str]] = os.listdir,
    is_file: Callable[[str], bool] = os.path.isfile,
) -> tuple[str, ...]:
    """Discover project-local SAM3 weights in deterministic preference order."""
    raw_root = str(project_root or "")
    if not raw_root:
        return ()
    root = os.path.abspath(raw_root)
    try:
        names = sorted(str(name) for name in list_directory(root))
    except (OSError, TypeError, ValueError):
        return ()

    exact: list[str] = []
    prefix: list[str] = []
    other: list[str] = []
    for name in names:
        path = os.path.join(root, name)
        if not is_file(path):
            continue
        lower = name.lower()
        if not lower.endswith((".pt", ".pth")) or "sam3" not in lower:
            continue
        if lower == default_filename.lower():
            exact.append(path)
        elif lower.startswith("sam3"):
            prefix.append(path)
        else:
            other.append(path)
    return tuple(exact + prefix + other)


def select_existing_sam_weight(
    candidates: Iterable[str],
    *,
    is_file: Callable[[str], bool] = os.path.isfile,
) -> str | None:
    """Select the first existing candidate after absolute-path de-duplication."""
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.abspath(str(candidate or ""))
        if not candidate or normalized in seen:
            continue
        seen.add(normalized)
        if is_file(normalized):
            return normalized
    return None


def select_sam_contour(results: Sequence[Any] | Iterable[Any]) -> SamContourResult | None:
    """Choose the highest-confidence usable contour from predictor results."""
    return inspect_sam_contour(results).result


def inspect_sam_contour(
    results: Sequence[Any] | Iterable[Any],
) -> SamContourSelection:
    """Select a contour while preserving user-facing empty-result semantics."""
    try:
        result = next(iter(results))
    except (StopIteration, TypeError):
        return SamContourSelection(failure="no_masks")

    masks = getattr(result, "masks", None)
    if masks is None:
        return SamContourSelection(failure="no_masks")
    try:
        contours = getattr(masks, "xy")
        if len(contours) == 0:
            return SamContourSelection(failure="no_masks")
    except (AttributeError, TypeError):
        return SamContourSelection(failure="no_masks")

    scores = _confidence_values(getattr(result, "boxes", None))
    best_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
    score = float(scores[best_index]) if scores else 0.0
    if best_index >= len(contours):
        best_index = 0
        score = float(scores[0]) if scores else 0.0

    try:
        points = tuple(
            (float(point[0]), float(point[1])) for point in contours[best_index] if len(point) >= 2
        )
    except (IndexError, TypeError, ValueError):
        return SamContourSelection(failure="no_polygon")
    if len(points) < 3:
        return SamContourSelection(failure="no_polygon")
    return SamContourSelection(result=SamContourResult(points=points, score=score))


def _confidence_values(boxes: Any) -> list[float]:
    confidence = getattr(boxes, "conf", None)
    if confidence is None:
        return []
    try:
        cpu_value = confidence.cpu() if callable(getattr(confidence, "cpu", None)) else confidence
        raw_values = (
            cpu_value.tolist() if callable(getattr(cpu_value, "tolist", None)) else cpu_value
        )
        return [float(value) for value in raw_values]
    except (TypeError, ValueError):
        return []


__all__ = [
    "SamContourResult",
    "SamContourSelection",
    "SamPromptRequest",
    "discover_sam_weight_candidates",
    "inspect_sam_contour",
    "select_existing_sam_weight",
    "select_sam_contour",
]
