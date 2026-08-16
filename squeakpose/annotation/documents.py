"""Layer-specific annotation state independent of Qt rendering."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from squeakpose.annotation.pose import PoseEditState
from squeakpose.annotation.segmentation import SegmentationEditState


@dataclass(frozen=True, slots=True)
class PoseKeypointValue:
    name: str
    x: float
    y: float
    visibility: int = 2
    index: int = -1
    canonical_index: int = -1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PoseKeypointValue":
        return cls(
            name=str(value.get("name") or ""),
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            visibility=int(value.get("vis", 2)),
            index=int(value.get("idx", -1)),
            canonical_index=int(value.get("canon_idx", -1)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "vis": self.visibility,
            "idx": self.index,
            "canon_idx": self.canonical_index,
        }


@dataclass(frozen=True, slots=True)
class PoseAnnotationValue:
    class_id: int
    box: tuple[float, float, float, float]
    keypoints: tuple[PoseKeypointValue, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        class_id: int,
        value: Mapping[str, Any],
    ) -> "PoseAnnotationValue":
        bbox = value.get("bbox")
        if not isinstance(bbox, Mapping):
            raise ValueError("pose annotation requires a bbox mapping")
        points = value.get("keypoints") or ()
        return cls(
            class_id=int(class_id),
            box=(
                float(bbox.get("x", 0.0)),
                float(bbox.get("y", 0.0)),
                float(bbox.get("w", 0.0)),
                float(bbox.get("h", 0.0)),
            ),
            keypoints=tuple(
                PoseKeypointValue.from_mapping(point)
                for point in points
                if isinstance(point, Mapping)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = self.box
        return {
            "class_id": self.class_id,
            "bbox": {"x": x, "y": y, "w": width, "h": height},
            "keypoints": [point.as_dict() for point in self.keypoints],
        }


@dataclass(frozen=True, slots=True)
class SegmentationAnnotationValue:
    class_id: int
    segments: tuple[tuple[float, float], ...]
    score: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        class_id: int,
        value: Mapping[str, Any],
    ) -> "SegmentationAnnotationValue":
        segments = value.get("segments") or ()
        return cls(
            class_id=int(class_id),
            segments=tuple((float(point[0]), float(point[1])) for point in segments),
            score=float(value.get("score", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "segments": list(self.segments),
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class PoseDocumentSnapshot:
    annotations: tuple[PoseAnnotationValue, ...]


@dataclass(frozen=True, slots=True)
class SegmentationDocumentSnapshot:
    annotations: tuple[SegmentationAnnotationValue, ...]


class AnnotationDocument(MutableMapping[int, dict[str, Any]]):
    """Mutable per-class annotation state with explicit replacement semantics."""

    layer_id = "unknown"
    workflow = "unknown"  # Compatibility identifier used by existing workers.

    def __init__(self, entries: Mapping[int, dict[str, Any]] | None = None):
        self._entries: dict[int, dict[str, Any]] = {}
        self.replace(entries or {})

    def __getitem__(self, class_id: int) -> dict[str, Any]:
        return self._entries[int(class_id)]

    def __setitem__(self, class_id: int, entry: dict[str, Any]) -> None:
        normalized_id = int(class_id)
        if not isinstance(entry, dict):
            raise TypeError("annotation entry must be a dictionary")
        copied = deepcopy(entry)
        copied["class_id"] = normalized_id
        self._entries[normalized_id] = copied

    def __delitem__(self, class_id: int) -> None:
        del self._entries[int(class_id)]

    def __iter__(self) -> Iterator[int]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def replace(self, entries: Mapping[int, dict[str, Any]]) -> None:
        replacement: dict[int, dict[str, Any]] = {}
        for class_id, entry in entries.items():
            normalized_id = int(class_id)
            copied = deepcopy(entry)
            copied["class_id"] = normalized_id
            replacement[normalized_id] = copied
        self._entries = replacement

    def replace_entries(self, entries: Mapping[int, dict[str, Any]]) -> None:
        """Compatibility-safe named replacement without direct mapping mutation."""
        self.replace(entries)

    def upsert_entry(self, class_id: int, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Insert or replace one serialized entry and return a detached copy."""
        self[int(class_id)] = dict(entry)
        return deepcopy(self._entries[int(class_id)])

    def delete_entry(self, class_id: int) -> bool:
        """Delete one entry, returning whether it existed."""
        normalized_id = int(class_id)
        if normalized_id not in self._entries:
            return False
        del self._entries[normalized_id]
        return True

    def load_entries(self, entries: Mapping[int, dict[str, Any]]) -> None:
        """Load the existing serialized cache representation."""
        self.replace(entries)

    def export_entries(self) -> dict[int, dict[str, Any]]:
        """Export the existing serialized cache representation detached from state."""
        return self.snapshot()

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return deepcopy(self._entries)

    def is_complete(self, class_id: int, **_requirements: Any) -> bool:
        return int(class_id) in self._entries


class KeypointAnnotationDocument(AnnotationDocument):
    layer_id = "keypoints"
    workflow = "pose"

    def annotation(self, class_id: int) -> PoseAnnotationValue | None:
        entry = self._entries.get(int(class_id))
        if entry is None:
            return None
        return PoseAnnotationValue.from_mapping(class_id, entry)

    def replace_annotations(self, annotations: Iterable[PoseAnnotationValue]) -> None:
        self.replace({annotation.class_id: annotation.as_dict() for annotation in annotations})

    def load_annotations(
        self,
        entries: Mapping[int, Mapping[str, Any]],
    ) -> tuple[PoseAnnotationValue, ...]:
        """Load serialized pose entries and return their typed values."""
        self.replace({int(class_id): dict(entry) for class_id, entry in entries.items()})
        return self.export_annotations()

    def upsert_annotation(self, annotation: PoseAnnotationValue) -> PoseAnnotationValue:
        self[annotation.class_id] = annotation.as_dict()
        stored = self.annotation(annotation.class_id)
        assert stored is not None
        return stored

    def delete_annotation(self, class_id: int) -> bool:
        return self.delete_entry(class_id)

    def export_annotations(self) -> tuple[PoseAnnotationValue, ...]:
        return tuple(
            PoseAnnotationValue.from_mapping(class_id, entry)
            for class_id, entry in sorted(self._entries.items())
        )

    def typed_snapshot(self) -> PoseDocumentSnapshot:
        return PoseDocumentSnapshot(self.export_annotations())

    def restore_typed_snapshot(self, snapshot: PoseDocumentSnapshot) -> None:
        self.replace_annotations(snapshot.annotations)

    def apply_edit_state(
        self,
        state: PoseEditState,
        *,
        require_complete: bool = True,
    ) -> PoseAnnotationValue | None:
        entry = state.to_annotation_entry(require_complete=require_complete)
        if entry is None or state.active_class_id is None:
            return None
        annotation = PoseAnnotationValue.from_mapping(state.active_class_id, entry)
        return self.upsert_annotation(annotation)

    def to_edit_state(
        self,
        class_id: int,
        keypoint_order: Sequence[str] = (),
        *,
        canonical_names: Sequence[str] = (),
    ) -> PoseEditState:
        state = PoseEditState()
        state.select_class(
            class_id,
            keypoint_order,
            canonical_names=canonical_names,
            entry=deepcopy(self._entries.get(int(class_id))),
        )
        return state

    def is_complete(
        self,
        class_id: int,
        *,
        required_keypoints: list[str] | None = None,
        **_requirements: Any,
    ) -> bool:
        entry = self.get(int(class_id))
        if not entry:
            return False
        bbox = entry.get("bbox")
        if not isinstance(bbox, dict):
            return False
        if float(bbox.get("w", 0.0) or 0.0) <= 0:
            return False
        if float(bbox.get("h", 0.0) or 0.0) <= 0:
            return False
        required = list(required_keypoints or [])
        available = {
            str(point.get("name") or "")
            for point in entry.get("keypoints", [])
            if isinstance(point, dict)
        }
        return all(name in available for name in required)


class SegmentationAnnotationDocument(AnnotationDocument):
    layer_id = "segmentation"
    workflow = "segmentation"

    def annotation(self, class_id: int) -> SegmentationAnnotationValue | None:
        entry = self._entries.get(int(class_id))
        if entry is None:
            return None
        return SegmentationAnnotationValue.from_mapping(class_id, entry)

    def replace_annotations(
        self,
        annotations: Iterable[SegmentationAnnotationValue],
    ) -> None:
        self.replace({annotation.class_id: annotation.as_dict() for annotation in annotations})

    def load_annotations(
        self,
        entries: Mapping[int, Mapping[str, Any]],
    ) -> tuple[SegmentationAnnotationValue, ...]:
        """Load serialized segmentation entries and return their typed values."""
        self.replace({int(class_id): dict(entry) for class_id, entry in entries.items()})
        return self.export_annotations()

    def upsert_annotation(
        self,
        annotation: SegmentationAnnotationValue,
    ) -> SegmentationAnnotationValue:
        self[annotation.class_id] = annotation.as_dict()
        stored = self.annotation(annotation.class_id)
        assert stored is not None
        return stored

    def delete_annotation(self, class_id: int) -> bool:
        return self.delete_entry(class_id)

    def export_annotations(self) -> tuple[SegmentationAnnotationValue, ...]:
        return tuple(
            SegmentationAnnotationValue.from_mapping(class_id, entry)
            for class_id, entry in sorted(self._entries.items())
        )

    def typed_snapshot(self) -> SegmentationDocumentSnapshot:
        return SegmentationDocumentSnapshot(self.export_annotations())

    def restore_typed_snapshot(self, snapshot: SegmentationDocumentSnapshot) -> None:
        self.replace_annotations(snapshot.annotations)

    def apply_edit_state(self, state: SegmentationEditState) -> None:
        self.replace_entries(state.accepted_masks)

    def to_edit_state(self, *, selected_target: int | None = None) -> SegmentationEditState:
        state = SegmentationEditState()
        state.reset(
            accepted_masks=self.snapshot(),
            selected_target=selected_target,
        )
        return state

    def is_complete(self, class_id: int, **_requirements: Any) -> bool:
        entry = self.get(int(class_id))
        if not entry:
            return False
        points = entry.get("segments")
        return isinstance(points, list) and len(points) >= 3


# Public compatibility name for integrations using the pre-layer terminology.
PoseAnnotationDocument = KeypointAnnotationDocument


__all__ = [
    "AnnotationDocument",
    "KeypointAnnotationDocument",
    "PoseAnnotationDocument",
    "PoseAnnotationValue",
    "PoseDocumentSnapshot",
    "PoseKeypointValue",
    "SegmentationAnnotationDocument",
    "SegmentationAnnotationValue",
    "SegmentationDocumentSnapshot",
]
