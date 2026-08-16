"""Qt-free edit state for keypoint pose annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from squeakpose.annotation.models import BoundingBox, Keypoint, KeypointEntry

PoseEntry = dict[str, Any]


@dataclass
class PoseEditSnapshot:
    """Detached copy of an active pose edit suitable for undo history."""

    active_class_id: int | None
    keypoint_order: list[str]
    canonical_names: list[str]
    box: BoundingBox | None
    keypoints: dict[str, KeypointEntry]


@dataclass
class PoseEditState:
    """Active box and ordered keypoint state independent of Qt rendering."""

    active_class_id: int | None = None
    keypoint_order: list[str] = field(default_factory=list)
    canonical_names: list[str] = field(default_factory=list)
    box: BoundingBox | None = None
    keypoints: dict[str, KeypointEntry] = field(default_factory=dict)
    _undo_snapshots: list[PoseEditSnapshot] = field(default_factory=list, init=False, repr=False)

    @property
    def next_keypoint_name(self) -> str | None:
        for name in self.keypoint_order:
            if name not in self.keypoints:
                return name
        return None

    @property
    def current_keypoint_index(self) -> int:
        next_name = self.next_keypoint_name
        return (
            len(self.keypoint_order) if next_name is None else self.keypoint_order.index(next_name)
        )

    @property
    def is_complete(self) -> bool:
        return self.box is not None and self.next_keypoint_name is None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_snapshots)

    def select_class(
        self,
        class_id: int | None,
        keypoint_order: Sequence[str] = (),
        *,
        canonical_names: Sequence[str] = (),
        entry: Mapping[str, Any] | None = None,
    ) -> None:
        """Replace the active class and optionally restore its cached annotation."""
        self.active_class_id = None if class_id is None else int(class_id)
        self.keypoint_order = [str(name) for name in keypoint_order]
        self.canonical_names = [str(name) for name in canonical_names]
        self.box = None
        self.keypoints = {}
        self._undo_snapshots.clear()
        if entry is not None and self.active_class_id is not None:
            self.load_annotation(entry)

    def set_box(self, box: BoundingBox) -> BoundingBox:
        """Set a box for the active class and start its keypoints again."""
        stored = self.replace_box(box)
        self.keypoints.clear()
        return stored

    def replace_box(self, box: BoundingBox) -> BoundingBox:
        """Replace the active box while preserving already placed keypoints."""
        class_id = self._required_class_id()
        self.box = BoundingBox(
            x=float(box.x),
            y=float(box.y),
            w=float(box.w),
            h=float(box.h),
            class_id=class_id,
        )
        return deepcopy(self.box)

    def add_next_keypoint(
        self,
        x: float,
        y: float,
        *,
        visibility: int = 2,
        display_name: str | None = None,
    ) -> KeypointEntry | None:
        """Add the next missing required keypoint in declared order."""
        name = self.next_keypoint_name
        if self.box is None or name is None:
            return None
        class_id = self._required_class_id()
        vis = int(visibility)
        point = Keypoint(
            x=0.0 if vis == 0 else float(x),
            y=0.0 if vis == 0 else float(y),
            class_id=class_id,
            name=name,
        )
        entry = KeypointEntry(
            name=name,
            display_name=str(display_name) if display_name is not None else name,
            kp=point,
            visibility=vis,
        )
        self.keypoints[name] = entry
        return deepcopy(entry)

    def mark_next_invisible(self) -> KeypointEntry | None:
        return self.add_next_keypoint(0.0, 0.0, visibility=0)

    def set_visibility(self, name: str, visibility: int) -> bool:
        entry = self.keypoints.get(str(name))
        if entry is None:
            return False
        entry.visibility = int(visibility)
        return True

    def delete_keypoint(self, name: str) -> bool:
        return self.keypoints.pop(str(name), None) is not None

    def delete_box(self) -> bool:
        changed = self.box is not None or bool(self.keypoints)
        self.box = None
        self.keypoints.clear()
        return changed

    def clear(self) -> None:
        self.box = None
        self.keypoints.clear()

    def load_annotation(self, entry: Mapping[str, Any]) -> bool:
        """Load the current annotation-cache dictionary representation."""
        if self.active_class_id is None:
            return False
        bbox = entry.get("bbox")
        if not isinstance(bbox, Mapping):
            self.clear()
            return False
        class_id = self._required_class_id()
        self.box = BoundingBox(
            x=float(bbox.get("x", 0.0)),
            y=float(bbox.get("y", 0.0)),
            w=float(bbox.get("w", 0.0)),
            h=float(bbox.get("h", 0.0)),
            class_id=class_id,
        )
        indexed: dict[int, Mapping[str, Any]] = {}
        named: dict[str, Mapping[str, Any]] = {}
        for raw in entry.get("keypoints", []):
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name") or "")
            if name:
                named[name] = raw
            try:
                index = int(raw.get("idx", -1))
            except (TypeError, ValueError):
                index = -1
            if index >= 0:
                indexed[index] = raw

        self.keypoints = {}
        for index, name in enumerate(self.keypoint_order):
            raw = named.get(name) or indexed.get(index)
            if raw is None:
                continue
            visibility = int(raw.get("vis", 2))
            point = Keypoint(
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                class_id=class_id,
                name=name,
            )
            self.keypoints[name] = KeypointEntry(name, name, point, visibility)
        return True

    def apply_template(
        self,
        template: Mapping[str, Any],
        *,
        image_width: float,
        image_height: float,
    ) -> bool:
        """Apply the existing normalized template representation to the active class."""
        if self.active_class_id is None:
            return False
        bbox = template.get("bbox", {})
        if not isinstance(bbox, Mapping):
            bbox = {}
        width = float(image_width)
        height = float(image_height)
        box_width = float(bbox.get("w", 1.0))
        box_height = float(bbox.get("h", 1.0))
        center_x = float(bbox.get("xc", 0.5))
        center_y = float(bbox.get("yc", 0.5))
        self.box = BoundingBox(
            x=(center_x - box_width / 2.0) * width,
            y=(center_y - box_height / 2.0) * height,
            w=box_width * width,
            h=box_height * height,
            class_id=self.active_class_id,
        )

        indexed: dict[int, Mapping[str, Any]] = {}
        for raw in template.get("keypoints", []):
            if not isinstance(raw, Mapping):
                continue
            try:
                index = int(raw.get("idx", -1))
            except (TypeError, ValueError):
                continue
            if index >= 0:
                indexed[index] = raw

        self.keypoints = {}
        for index, name in enumerate(self.keypoint_order):
            raw = indexed.get(index)
            if raw is None:
                visibility = 0
                x = y = 0.0
            else:
                visibility = int(raw.get("vis", 2))
                x = 0.0 if visibility == 0 else float(raw.get("x", 0.0)) * width
                y = 0.0 if visibility == 0 else float(raw.get("y", 0.0)) * height
            point = Keypoint(x=x, y=y, class_id=self.active_class_id, name=name)
            self.keypoints[name] = KeypointEntry(name, name, point, visibility)
        return True

    def to_annotation_entry(self, *, require_complete: bool = True) -> PoseEntry | None:
        """Build the current annotation-cache dictionary representation."""
        if self.active_class_id is None or self.box is None:
            return None
        if require_complete and not self.is_complete:
            return None
        keypoints = []
        for index, name in enumerate(self.keypoint_order):
            entry = self.keypoints.get(name)
            if entry is None:
                continue
            keypoints.append(
                {
                    "name": name,
                    "x": float(entry.kp.x),
                    "y": float(entry.kp.y),
                    "vis": int(entry.visibility),
                    "idx": index,
                    "canon_idx": self._canonical_index(name),
                }
            )
        return {
            "class_id": self.active_class_id,
            "bbox": {
                "x": float(self.box.x),
                "y": float(self.box.y),
                "w": float(self.box.w),
                "h": float(self.box.h),
            },
            "keypoints": keypoints,
        }

    def to_template(
        self, class_name: str, *, image_width: float, image_height: float
    ) -> PoseEntry | None:
        """Build the normalized template dictionary used by the current UI."""
        entry = self.to_annotation_entry()
        if entry is None or self.box is None:
            return None
        width = max(1.0, float(image_width))
        height = max(1.0, float(image_height))
        keypoints = []
        for point in entry["keypoints"]:
            visibility = int(point["vis"])
            keypoints.append(
                {
                    "name": point["name"],
                    "idx": point["idx"],
                    "canon_idx": point["canon_idx"],
                    "x": 0.0 if visibility == 0 else float(point["x"]) / width,
                    "y": 0.0 if visibility == 0 else float(point["y"]) / height,
                    "vis": visibility,
                }
            )
        return {
            "class": str(class_name),
            "bbox": {
                "xc": (self.box.x + self.box.w / 2.0) / width,
                "yc": (self.box.y + self.box.h / 2.0) / height,
                "w": self.box.w / width,
                "h": self.box.h / height,
            },
            "keypoints": keypoints,
        }

    def snapshot(self) -> PoseEditSnapshot:
        return PoseEditSnapshot(
            active_class_id=self.active_class_id,
            keypoint_order=deepcopy(self.keypoint_order),
            canonical_names=deepcopy(self.canonical_names),
            box=deepcopy(self.box),
            keypoints=deepcopy(self.keypoints),
        )

    def restore(self, snapshot: PoseEditSnapshot) -> None:
        self.active_class_id = snapshot.active_class_id
        self.keypoint_order = deepcopy(snapshot.keypoint_order)
        self.canonical_names = deepcopy(snapshot.canonical_names)
        self.box = deepcopy(snapshot.box)
        self.keypoints = deepcopy(snapshot.keypoints)

    def push_undo_snapshot(self) -> PoseEditSnapshot:
        snapshot = self.snapshot()
        self._undo_snapshots.append(deepcopy(snapshot))
        return snapshot

    def undo(self) -> bool:
        if not self._undo_snapshots:
            return False
        self.restore(self._undo_snapshots.pop())
        return True

    def _canonical_index(self, name: str) -> int:
        try:
            return self.canonical_names.index(name)
        except ValueError:
            return -1

    def _required_class_id(self) -> int:
        if self.active_class_id is None:
            raise ValueError("an active class is required")
        return self.active_class_id
