"""Qt-free annotation domain models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float
    class_id: int

    def to_yolo(
        self,
        img_w: float,
        img_h: float,
    ) -> tuple[int, float, float, float, float]:
        if img_w <= 0 or img_h <= 0:
            raise ValueError("image dimensions must be positive")
        xc = (self.x + self.w / 2) / img_w
        yc = (self.y + self.h / 2) / img_h
        return self.class_id, xc, yc, self.w / img_w, self.h / img_h


@dataclass(slots=True)
class Keypoint:
    x: float
    y: float
    class_id: int
    name: str

    def to_yolo(self, img_w: float, img_h: float) -> tuple[int, float, float, str]:
        if img_w <= 0 or img_h <= 0:
            raise ValueError("image dimensions must be positive")
        return self.class_id, self.x / img_w, self.y / img_h, self.name


@dataclass(slots=True)
class KeypointEntry:
    name: str
    display_name: str
    kp: Keypoint
    visibility: int


@dataclass(slots=True)
class Annotation:
    ann_id: int
    bbox: BoundingBox
    keypoints: dict[str, KeypointEntry]
    order: list[str]
