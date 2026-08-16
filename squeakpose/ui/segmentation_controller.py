"""Controller boundary for segmentation editing and prompt prediction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from squeakpose.annotation.documents import (
    SegmentationAnnotationDocument,
    SegmentationDocumentSnapshot,
)
from squeakpose.annotation.segmentation import (
    Point,
    PromptPoint,
    SegmentationEditSnapshot,
    SegmentationEntry,
)
from squeakpose.annotation.segmentation_assistant import (
    SamContourSelection,
    SamPromptRequest,
    inspect_sam_contour,
)


@dataclass(frozen=True, slots=True)
class SegmentationPromptRequest:
    """Qt-free input for a SAM-like prompt predictor."""

    class_id: int
    prompts: tuple[PromptPoint, ...]


@dataclass(frozen=True, slots=True)
class SegmentationPromptResult:
    points: tuple[Point, ...]
    score: float = 0.0


def _ignore_state(_snapshot: SegmentationEditSnapshot) -> None:
    pass


def _ignore_document(_snapshot: SegmentationDocumentSnapshot) -> None:
    pass


@dataclass(frozen=True, slots=True)
class SegmentationControllerCallbacks:
    state_changed: Callable[[SegmentationEditSnapshot], None] = _ignore_state
    document_changed: Callable[[SegmentationDocumentSnapshot], None] = _ignore_document


class SegmentationAnnotationController:
    """Coordinate segmentation state, its document, and an optional predictor."""

    def __init__(
        self,
        document: SegmentationAnnotationDocument,
        *,
        predict: Callable[[SegmentationPromptRequest], SegmentationPromptResult] | None = None,
        callbacks: SegmentationControllerCallbacks | None = None,
    ) -> None:
        self.document = document
        self.state = document.to_edit_state()
        self._predict = predict
        self._callbacks = callbacks or SegmentationControllerCallbacks()

    def bind_document(self, document: SegmentationAnnotationDocument) -> None:
        """Rebind storage when the composition root loads another frame."""
        self.document = document

    def replace_document(
        self,
        entries: Mapping[int, Mapping[str, Any]],
        *,
        selected_target: int | None = None,
    ) -> None:
        self.document.load_annotations(entries)
        self.state.reset(
            accepted_masks=self.document.snapshot(),
            selected_target=selected_target,
        )
        self._emit_all()

    def select_target(self, class_id: int | None, *, clear_prompts: bool = True) -> None:
        self.state.select_target(class_id)
        if clear_prompts:
            self.state.clear_prompt_state()
        self._emit_state()

    def add_prompt(self, x: float, y: float, *, positive: bool = True) -> PromptPoint:
        prompt = self.state.add_prompt(x, y, positive=positive)
        self._emit_state()
        return prompt

    def remove_last_prompt(self) -> PromptPoint | None:
        removed = self.state.remove_last_prompt()
        if removed is not None:
            self.state.clear_preview()
            self._emit_state()
        return removed

    def clear_prompts(self) -> None:
        self.state.clear_prompt_state()
        self._emit_state()

    def set_preview(self, points: Sequence[Point], *, score: float = 0.0) -> None:
        self.state.set_preview(points, score)
        self._emit_state()

    def request_preview(self) -> SegmentationPromptResult:
        """Run the injected predictor without granting it access to UI state."""
        if self._predict is None:
            raise RuntimeError("No segmentation prompt predictor is configured.")
        if self.state.selected_target is None:
            raise ValueError("Select a segmentation class before requesting a preview.")
        if not self.state.prompt_points:
            raise ValueError("At least one segmentation prompt is required.")
        request = SegmentationPromptRequest(
            class_id=self.state.selected_target,
            prompts=tuple(self.state.prompt_points),
        )
        result = self._predict(request)
        self.state.set_preview(result.points, result.score)
        self._emit_state()
        return result

    def build_prompt_request(self, source: str) -> SamPromptRequest:
        """Build predictor inputs without exposing controller or UI state."""
        if self.state.selected_target is None:
            raise ValueError("Select a segmentation class before requesting a preview.")
        if not self.state.prompt_points:
            raise ValueError("At least one segmentation prompt is required.")
        normalized_source = str(source or "")
        if not normalized_source:
            raise ValueError("A segmentation image source is required.")
        return SamPromptRequest(
            source=normalized_source,
            class_id=self.state.selected_target,
            prompts=tuple(self.state.prompt_points),
        )

    def apply_prediction_results(self, results: Sequence[Any]) -> SegmentationPromptResult | None:
        """Select a model contour and publish it as the editable preview."""
        contour = self.coordinate_prediction_results(results).result
        if contour is None:
            return None
        return SegmentationPromptResult(points=contour.points, score=contour.score)

    def coordinate_prediction_results(
        self,
        results: Sequence[Any],
    ) -> SamContourSelection:
        """Publish a usable contour while retaining empty-result diagnostics."""
        selection = inspect_sam_contour(results)
        contour = selection.result
        if contour is None:
            return selection
        self.state.set_preview(contour.points, contour.score)
        self._emit_state()
        return SamContourSelection(
            result=contour,
            failure=selection.failure,
        )

    def accept_preview(self) -> SegmentationEntry | None:
        if self.state.selected_target is None or not self.state.has_preview:
            return None
        self.state.push_undo_snapshot()
        accepted = self.state.accept_preview()
        self.document.apply_edit_state(self.state)
        self._emit_all()
        return accepted

    def upsert_polygon(
        self,
        class_id: int,
        points: Sequence[Point],
        *,
        score: float = 0.0,
        record_undo: bool = True,
    ) -> SegmentationEntry:
        normalized = [(float(x), float(y)) for x, y in points]
        if len(normalized) < 3:
            raise ValueError("A segmentation polygon requires at least three points.")
        if record_undo:
            self.state.push_undo_snapshot()
        entry: SegmentationEntry = {
            "class_id": int(class_id),
            "segments": normalized,
            "score": float(score),
        }
        stored = self.state.set_accepted_entry(class_id, entry)
        self.document.apply_edit_state(self.state)
        self._emit_all()
        return stored

    def remove_mask(
        self,
        class_id: int | None = None,
        *,
        record_undo: bool = True,
    ) -> bool:
        target = self.state.selected_target if class_id is None else int(class_id)
        if target is None or target not in self.state.accepted_masks:
            return False
        if record_undo:
            self.state.push_undo_snapshot()
        removed = self.state.clear_accepted_mask(target)
        self.document.apply_edit_state(self.state)
        self._emit_all()
        return removed

    def undo(self) -> bool:
        if not self.state.undo():
            return False
        self.document.apply_edit_state(self.state)
        self._emit_all()
        return True

    def _emit_state(self) -> SegmentationEditSnapshot:
        snapshot = self.state.snapshot()
        self._callbacks.state_changed(snapshot)
        return snapshot

    def _emit_document(self) -> SegmentationDocumentSnapshot:
        snapshot = self.document.typed_snapshot()
        self._callbacks.document_changed(snapshot)
        return snapshot

    def _emit_all(self) -> None:
        self._emit_state()
        self._emit_document()


__all__ = [
    "SegmentationAnnotationController",
    "SegmentationControllerCallbacks",
    "SegmentationPromptRequest",
    "SegmentationPromptResult",
]
