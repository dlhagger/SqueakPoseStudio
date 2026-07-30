"""Workflow-specific annotation state independent of Qt rendering."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from typing import Any


class AnnotationDocument(MutableMapping[int, dict[str, Any]]):
    """Mutable per-class annotation state with explicit replacement semantics."""

    workflow = "unknown"

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

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return deepcopy(self._entries)

    def is_complete(self, class_id: int, **_requirements: Any) -> bool:
        return int(class_id) in self._entries


class PoseAnnotationDocument(AnnotationDocument):
    workflow = "pose"

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
    workflow = "segmentation"

    def is_complete(self, class_id: int, **_requirements: Any) -> bool:
        entry = self.get(int(class_id))
        if not entry:
            return False
        points = entry.get("segments")
        return isinstance(points, list) and len(points) >= 3
