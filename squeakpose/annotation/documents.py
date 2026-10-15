"""Layer-specific annotation state independent of Qt rendering."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

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
    instance_id: str = ""

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
            instance_id=str(value.get("instance_id", "")),
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
            **({"instance_id": self.instance_id} if self.instance_id else {}),
            "class_id": self.class_id,
            "bbox": {"x": x, "y": y, "w": width, "h": height},
            "keypoints": [point.as_dict() for point in self.keypoints],
        }


@dataclass(frozen=True, slots=True)
class SegmentationAnnotationValue:
    class_id: int
    segments: tuple[tuple[float, float], ...]
    score: float = 0.0
    instance_id: str = ""

    @classmethod
    def from_mapping(
        cls,
        class_id: int,
        value: Mapping[str, Any],
    ) -> "SegmentationAnnotationValue":
        segments = value.get("segments") or ()
        return cls(
            class_id=int(class_id),
            instance_id=str(value.get("instance_id", "")),
            segments=tuple((float(point[0]), float(point[1])) for point in segments),
            score=float(value.get("score", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **({"instance_id": self.instance_id} if self.instance_id else {}),
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


class AnnotationEntries(dict[int, dict[str, Any]]):
    """Legacy class view carrying the complete frame through compatibility callers."""

    instances: list[dict[str, Any]]
    selected_instances: dict[int, str]


class AnnotationDocument(MutableMapping[int, dict[str, Any]]):
    """Instance storage with a backward-compatible selected-entry view per class.

    Mapping methods address the selected animal of a class. Use ``all_entries``
    or typed snapshots for whole-frame operations; converting this mapping to a
    plain dict intentionally yields only its legacy class view. Instance IDs are
    editor-local and are not animal identities or columns in YOLO label files.
    """

    layer_id = "unknown"
    workflow = "unknown"  # Compatibility identifier used by existing workers.

    def __init__(self, entries: Mapping[int, dict[str, Any]] | None = None):
        self._entries: dict[int, dict[str, Any]] = {}
        self._parked: dict[str, dict[str, Any]] = {}
        self._selected: dict[int, str] = {}
        self._order: list[str] = []
        self.replace({} if entries is None else entries)

    def __getitem__(self, class_id: int) -> dict[str, Any]:
        return self._entries[int(class_id)]

    def __setitem__(self, class_id: int, entry: dict[str, Any]) -> None:
        normalized_id = int(class_id)
        if not isinstance(entry, dict):
            raise TypeError("annotation entry must be a dictionary")
        copied = deepcopy(entry)
        copied["class_id"] = normalized_id
        selected = self._selected.get(
            normalized_id, str(copied.get("instance_id") or f"class:{normalized_id}")
        )
        self._selected[normalized_id] = selected
        if selected != f"class:{normalized_id}":
            copied["instance_id"] = selected
        self._entries[normalized_id] = copied
        identifier = self.entry_id(copied)
        if identifier not in self._order:
            self._order.append(identifier)

    def __delitem__(self, class_id: int) -> None:
        del self._entries[int(class_id)]

    def __iter__(self) -> Iterator[int]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def entry_id(entry: Mapping[str, Any]) -> str:
        return str(entry.get("instance_id") or f"class:{int(entry['class_id'])}")

    def all_entries(self) -> list[dict[str, Any]]:
        """Detached annotations, including non-selected instances of each class."""
        entries = list(self._parked.values()) + list(self._entries.values())
        return deepcopy(sorted(entries, key=lambda entry: self._order.index(self.entry_id(entry))))

    def instance_ids(self, class_id: int) -> list[str]:
        ids = [self.entry_id(e) for e in self.all_entries() if e["class_id"] == class_id]
        selected = self.selected_instance_id(class_id)
        if selected not in ids:
            ids.append(selected)
        return ids

    def selected_instance_id(self, class_id: int) -> str:
        return self._selected.get(class_id, f"class:{class_id}")

    def start_instance(self, class_id: int) -> str:
        """Create a new editing slot without replacing another animal."""
        class_id = int(class_id)
        previous = self._entries.pop(class_id, None)
        if previous is not None:
            self._parked[self.entry_id(previous)] = previous
        identifier = uuid4().hex
        self._selected[class_id] = identifier
        return identifier

    def select_instance(self, class_id: int, instance_id: str) -> None:
        class_id = int(class_id)
        if instance_id == self.selected_instance_id(class_id):
            return
        target = self._parked.get(instance_id)
        if target is None or target["class_id"] != class_id:
            raise KeyError(instance_id)
        previous = self._entries.pop(class_id, None)
        if previous is not None:
            self._parked[self.entry_id(previous)] = previous
        self._entries[class_id] = self._parked.pop(instance_id)
        self._selected[class_id] = instance_id

    def clear(self) -> None:
        self._entries.clear()
        self._parked.clear()
        self._selected.clear()
        self._order.clear()

    def replace_instance_entries(self, entries: Iterable[Mapping[str, Any]]) -> None:
        self.clear()
        for raw in entries:
            entry = deepcopy(dict(raw))
            cid = int(entry["class_id"])
            if cid in self._entries:
                previous = self._entries.pop(cid)
                self._parked[self.entry_id(previous)] = previous
                if not entry.get("instance_id"):
                    entry["instance_id"] = uuid4().hex
            identifier = self.entry_id(entry)
            if identifier in self._parked:
                raise ValueError("duplicate annotation instance ID")
            self._selected[cid] = identifier
            self[cid] = entry

    def replace(self, entries: Mapping[int, Mapping[str, Any]]) -> None:
        instances = getattr(entries, "instances", None)
        if instances is not None:
            self.clear()
            self._entries = {int(cid): deepcopy(dict(entry)) for cid, entry in entries.items()}
            active_ids = {self.entry_id(entry) for entry in self._entries.values()}
            self._parked = {
                self.entry_id(entry): deepcopy(entry)
                for entry in instances
                if self.entry_id(entry) not in active_ids
            }
            self._selected = deepcopy(getattr(entries, "selected_instances", {}))
            self._order = [self.entry_id(entry) for entry in instances]
            return
        self.clear()
        for cid, entry in entries.items():
            self[int(cid)] = dict(entry)

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
        result = AnnotationEntries(deepcopy(self._entries))
        result.instances = self.all_entries()
        result.selected_instances = dict(self._selected)
        return result

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
        self.replace_instance_entries(annotation.as_dict() for annotation in annotations)

    def load_annotations(
        self,
        entries: Mapping[int, Mapping[str, Any]],
    ) -> tuple[PoseAnnotationValue, ...]:
        """Load serialized pose entries and return their typed values."""
        self.replace(entries)
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
            for entry in sorted(self.all_entries(), key=lambda entry: int(entry["class_id"]))
            for class_id in [int(entry["class_id"])]
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
        self.replace_instance_entries(annotation.as_dict() for annotation in annotations)

    def load_annotations(
        self,
        entries: Mapping[int, Mapping[str, Any]],
    ) -> tuple[SegmentationAnnotationValue, ...]:
        """Load serialized segmentation entries and return their typed values."""
        self.replace(entries)
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
            for entry in sorted(self.all_entries(), key=lambda entry: int(entry["class_id"]))
            for class_id in [int(entry["class_id"])]
        )

    def typed_snapshot(self) -> SegmentationDocumentSnapshot:
        return SegmentationDocumentSnapshot(self.export_annotations())

    def restore_typed_snapshot(self, snapshot: SegmentationDocumentSnapshot) -> None:
        self.replace_annotations(snapshot.annotations)

    def apply_edit_state(self, state: SegmentationEditState) -> None:
        for cid in list(self._entries):
            if cid not in state.accepted_masks:
                del self[cid]
        for cid, entry in state.accepted_masks.items():
            self[cid] = entry

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
