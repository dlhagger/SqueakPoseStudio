"""Controller boundary for pose annotation editing.

The controller owns no widgets.  A view renders the snapshots supplied through the
callbacks and forwards user intent through the methods below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    PoseDocumentSnapshot,
)
from squeakpose.annotation.models import BoundingBox, KeypointEntry
from squeakpose.annotation.pose import PoseEditSnapshot, PoseEditState, PoseEntry


def _ignore_state(_snapshot: PoseEditSnapshot) -> None:
    pass


def _ignore_document(_snapshot: PoseDocumentSnapshot) -> None:
    pass


@dataclass(frozen=True, slots=True)
class PoseControllerCallbacks:
    """Explicit presentation hooks used by a pose view."""

    state_changed: Callable[[PoseEditSnapshot], None] = _ignore_state
    document_changed: Callable[[PoseDocumentSnapshot], None] = _ignore_document


class PoseAnnotationController:
    """Coordinate a pose document and the currently selected pose edit."""

    def __init__(
        self,
        document: PoseAnnotationDocument,
        *,
        keypoint_order_for: Callable[[int], Sequence[str]],
        canonical_names: Sequence[str] = (),
        callbacks: PoseControllerCallbacks | None = None,
    ) -> None:
        self.document = document
        self.state = PoseEditState()
        self._keypoint_order_for = keypoint_order_for
        self._canonical_names = tuple(str(name) for name in canonical_names)
        self._callbacks = callbacks or PoseControllerCallbacks()

    def bind_document(self, document: PoseAnnotationDocument) -> None:
        """Rebind frame storage while preserving the active schema and selection."""
        self.document = document

    def configure_schema(
        self,
        *,
        keypoint_order_for: Callable[[int], Sequence[str]] | None = None,
        canonical_names: Sequence[str] | None = None,
    ) -> None:
        """Refresh schema dependencies after project class configuration changes."""
        if keypoint_order_for is not None:
            self._keypoint_order_for = keypoint_order_for
        if canonical_names is not None:
            self._canonical_names = tuple(str(name) for name in canonical_names)

    def select_class(self, class_id: int | None) -> PoseEditSnapshot:
        """Select a class and restore its cached annotation, if present."""
        if class_id is None:
            self.state.select_class(None)
        else:
            normalized_id = int(class_id)
            self.state.select_class(
                normalized_id,
                self._keypoint_order_for(normalized_id),
                canonical_names=self._canonical_names,
                entry=self.document.get(normalized_id),
            )
        return self._emit_state()

    def replace_document(self, entries: Mapping[int, Mapping[str, Any]]) -> None:
        """Load a frame document and refresh the selected edit without state leakage."""
        selected = self.state.active_class_id
        self.document.load_annotations(entries)
        self._emit_document()
        self.select_class(selected)

    def set_box(self, box: BoundingBox) -> BoundingBox:
        self.state.push_undo_snapshot()
        stored = self.state.set_box(box)
        self._discard_active_annotation()
        self._emit_all()
        return stored

    def replace_box_preserving_keypoints(self, box: BoundingBox) -> BoundingBox:
        """Replace the active box without discarding keypoints, with undo support."""
        self.state.push_undo_snapshot()
        stored = self.state.replace_box(box)
        self._sync_document(require_complete=True)
        self._emit_all()
        return stored

    def add_next_keypoint(
        self,
        x: float,
        y: float,
        *,
        visibility: int = 2,
        display_name: str | None = None,
    ) -> KeypointEntry | None:
        if self.state.box is None or self.state.next_keypoint_name is None:
            return None
        self.state.push_undo_snapshot()
        added = self.state.add_next_keypoint(
            x,
            y,
            visibility=visibility,
            display_name=display_name,
        )
        self._sync_document(require_complete=True)
        self._emit_all()
        return added

    def mark_next_invisible(self) -> KeypointEntry | None:
        return self.add_next_keypoint(0.0, 0.0, visibility=0)

    def set_visibility(
        self,
        name: str,
        visibility: int,
        *,
        record_undo: bool = True,
    ) -> bool:
        if name not in self.state.keypoints:
            return False
        if record_undo:
            self.state.push_undo_snapshot()
        changed = self.state.set_visibility(name, visibility)
        self._sync_document(require_complete=True)
        self._emit_all()
        return changed

    def delete_keypoint(self, name: str, *, record_undo: bool = True) -> bool:
        if name not in self.state.keypoints:
            return False
        if record_undo:
            self.state.push_undo_snapshot()
        changed = self.state.delete_keypoint(name)
        self._sync_document(require_complete=True)
        self._emit_all()
        return changed

    def delete_box(self, *, record_undo: bool = True) -> bool:
        if self.state.box is None and not self.state.keypoints:
            return False
        if record_undo:
            self.state.push_undo_snapshot()
        changed = self.state.delete_box()
        self._discard_active_annotation()
        self._emit_all()
        return changed

    def apply_template(
        self,
        template: Mapping[str, Any],
        *,
        image_width: float,
        image_height: float,
    ) -> bool:
        if self.state.active_class_id is None:
            return False
        self.state.push_undo_snapshot()
        applied = self.state.apply_template(
            template,
            image_width=image_width,
            image_height=image_height,
        )
        self._sync_document(require_complete=True)
        self._emit_all()
        return applied

    def template_for_active_class(
        self,
        class_name: str,
        *,
        image_width: float,
        image_height: float,
    ) -> PoseEntry | None:
        return self.state.to_template(
            class_name,
            image_width=image_width,
            image_height=image_height,
        )

    def commit(self, *, require_complete: bool = True) -> bool:
        stored = self._sync_document(require_complete=require_complete)
        self._emit_document()
        return stored

    def undo(self) -> bool:
        if not self.state.undo():
            return False
        self._sync_document(require_complete=True)
        self._emit_all()
        return True

    def _sync_document(self, *, require_complete: bool) -> bool:
        class_id = self.state.active_class_id
        if class_id is None:
            return False
        annotation = self.document.apply_edit_state(
            self.state,
            require_complete=require_complete,
        )
        if annotation is None:
            self.document.delete_annotation(class_id)
            return False
        return True

    def _discard_active_annotation(self) -> None:
        if self.state.active_class_id is not None:
            self.document.delete_annotation(self.state.active_class_id)

    def _emit_state(self) -> PoseEditSnapshot:
        snapshot = self.state.snapshot()
        self._callbacks.state_changed(snapshot)
        return snapshot

    def _emit_document(self) -> PoseDocumentSnapshot:
        snapshot = self.document.typed_snapshot()
        self._callbacks.document_changed(snapshot)
        return snapshot

    def _emit_all(self) -> None:
        self._emit_state()
        self._emit_document()


__all__ = ["PoseAnnotationController", "PoseControllerCallbacks"]
