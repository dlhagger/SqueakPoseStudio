"""Narrow QGraphicsScene presentation for annotations and saved-layer context."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from squeakpose.annotation.graphics import BoxItem, KeypointItem
from squeakpose.annotation.models import BoundingBox, Keypoint
from squeakpose.project.layers import normalize_layer_id


def _ignore_item(_item: QGraphicsItem) -> None:
    pass


@dataclass(frozen=True, slots=True)
class PoseReferenceKeypoint:
    keypoint: Keypoint
    visibility: int = 2
    label_text: str = ""


class CanvasScenePresenter:
    """Construct visual scene items without owning files, state, or input routing."""

    DEPTH_PROBE_COLORS = (
        "#73d7ff",
        "#ffd166",
        "#82e0aa",
        "#ff8fab",
        "#c7a0ff",
        "#f6bd60",
    )

    def __init__(
        self,
        scene: QGraphicsScene,
        *,
        track_item: Callable[[QGraphicsItem], None] = _ignore_item,
        untrack_item: Callable[[QGraphicsItem], None] = _ignore_item,
    ) -> None:
        self.scene = scene
        self._track_item = track_item
        self._untrack_item = untrack_item
        self.reference_items: list[QGraphicsItem] = []
        self.prompt_items: list[QGraphicsItem] = []
        self.depth_probe_items: list[QGraphicsItem] = []

    @staticmethod
    def segmentation_color(class_id: int, alpha: int = 255) -> QColor:
        return QColor.fromHsv(int((int(class_id) * 47) % 360), 210, 245, int(alpha))

    @staticmethod
    def polygon_path(points: Sequence[tuple[float, float]]) -> QPainterPath | None:
        if len(points) < 3:
            return None
        path = QPainterPath()
        path.moveTo(float(points[0][0]), float(points[0][1]))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        path.closeSubpath()
        return path

    def add_background(self, pixmap: QPixmap) -> QGraphicsPixmapItem:
        item = QGraphicsPixmapItem(pixmap)
        item.setZValue(0)
        self.scene.addItem(item)
        return item

    def add_depth_display(
        self,
        pixmap: QPixmap,
        *,
        image_width: int,
        image_height: int,
        overlay: bool,
    ) -> QGraphicsPixmapItem:
        prepared = self._scaled_pixmap(pixmap, image_width, image_height)
        item = QGraphicsPixmapItem(prepared)
        item.setZValue(1.0)
        item.setOpacity(0.55 if overlay else 1.0)
        self.scene.addItem(item)
        self._track_item(item)
        return item

    def add_segmentation_mask(
        self,
        class_id: int,
        points: Sequence[tuple[float, float]],
        *,
        label_text: str,
        preview: bool = False,
        color: QColor | None = None,
    ) -> QGraphicsPathItem | None:
        normalized = [(float(x), float(y)) for x, y in points]
        path = self.polygon_path(normalized)
        if path is None:
            return None
        display_color = QColor(color) if color is not None else self.segmentation_color(class_id)
        item = QGraphicsPathItem(path)
        pen = QPen(display_color)
        pen.setCosmetic(True)
        pen.setWidth(2 if preview else 3)
        if preview:
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        fill_color = QColor(display_color)
        fill_color.setAlpha(52 if preview else 76)
        item.setBrush(QBrush(fill_color))
        item.setZValue(4.5 if preview else 4.0)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, not preview)
        item.seg_class_id = int(class_id)
        item.seg_points = normalized
        item.seg_preview = bool(preview)
        self._add_segmentation_frame(
            item,
            path,
            f"{label_text} (preview)" if preview else str(label_text),
            preview=preview,
        )
        self.scene.addItem(item)
        self._track_item(item)
        return item

    def update_segmentation_geometry(
        self,
        item: QGraphicsPathItem | None,
        points: Sequence[tuple[float, float]],
    ) -> bool:
        if item is None:
            return False
        normalized = [(float(x), float(y)) for x, y in points]
        path = self.polygon_path(normalized)
        if path is None:
            return False
        item.seg_points = normalized
        item.setPath(path)
        self._position_segmentation_frame(item, path)
        return True

    def add_prompt_marker(
        self,
        x: float,
        y: float,
        *,
        positive: bool,
    ) -> tuple[QGraphicsItem, ...]:
        radius = 5.0
        color = Qt.GlobalColor.green if positive else Qt.GlobalColor.red
        marker = QGraphicsEllipseItem(x - radius, y - radius, radius * 2.0, radius * 2.0)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(2)
        marker.setPen(pen)
        marker.setBrush(QBrush(Qt.GlobalColor.transparent))
        marker.setZValue(8.0)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        items: list[QGraphicsItem] = [marker]
        if not positive:
            items.extend(
                (
                    QGraphicsLineItem(
                        x - radius + 1.0,
                        y - radius + 1.0,
                        x + radius - 1.0,
                        y + radius - 1.0,
                    ),
                    QGraphicsLineItem(
                        x - radius + 1.0,
                        y + radius - 1.0,
                        x + radius - 1.0,
                        y - radius + 1.0,
                    ),
                )
            )
            for line in items[1:]:
                line_pen = QPen(color)
                line_pen.setCosmetic(True)
                line_pen.setWidth(2)
                line.setPen(line_pen)
                line.setZValue(8.1)
                line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        for item in items:
            self.scene.addItem(item)
            self._track_item(item)
            self.prompt_items.append(item)
        return tuple(items)

    def clear_prompts(self) -> None:
        self._remove_owned(self.prompt_items, untrack=True)

    def render_depth_probes(
        self,
        probes: Sequence[Mapping[str, Any]],
        *,
        format_value: Callable[[Any], str] | None = None,
    ) -> tuple[QGraphicsItem, ...]:
        self.clear_depth_probes()
        formatter = format_value or self._default_depth_value
        for index, probe in enumerate(probes, start=1):
            color = QColor(self.DEPTH_PROBE_COLORS[(index - 1) % len(self.DEPTH_PROBE_COLORS)])
            x = float(probe["x"])
            y = float(probe["y"])
            marker = QGraphicsEllipseItem(-5.0, -5.0, 10.0, 10.0)
            marker.setPos(x + 0.5, y + 0.5)
            marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            marker.setAcceptHoverEvents(False)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(2)
            marker.setPen(pen)
            marker.setBrush(QBrush(QColor(10, 15, 18, 190)))
            marker.setZValue(20.0)
            text_item = QGraphicsSimpleTextItem(f"{index} · {formatter(probe.get('depth'))}")
            text_item.setBrush(QBrush(color))
            text_item.setPos(x + 8.5, y - 10.5)
            text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            text_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            text_item.setAcceptHoverEvents(False)
            text_item.setZValue(20.0)
            self.scene.addItem(marker)
            self.scene.addItem(text_item)
            self.depth_probe_items.extend((marker, text_item))
        return tuple(self.depth_probe_items)

    def clear_depth_probes(self) -> None:
        self._remove_owned(self.depth_probe_items)

    def add_reference_item(
        self,
        item: QGraphicsItem,
        *,
        layer_id: str,
        opacity: float,
        z_value: float = 1.0,
    ) -> QGraphicsItem:
        item.reference_layer_id = normalize_layer_id(layer_id)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setAcceptHoverEvents(False)
        item.setOpacity(float(opacity))
        item.setZValue(float(z_value))
        self.scene.addItem(item)
        self.reference_items.append(item)
        return item

    def add_depth_reference(
        self,
        pixmap: QPixmap,
        *,
        layer_id: str,
        image_width: int,
        image_height: int,
    ) -> QGraphicsPixmapItem:
        item = QGraphicsPixmapItem(self._scaled_pixmap(pixmap, image_width, image_height))
        self.add_reference_item(item, layer_id=layer_id, opacity=0.42, z_value=0.5)
        return item

    def add_segmentation_reference(
        self,
        class_id: int,
        points: Sequence[tuple[float, float]],
        *,
        label_text: str,
        layer_id: str,
        color: QColor = QColor(104, 164, 207),
    ) -> QGraphicsPathItem | None:
        normalized = [(float(x), float(y)) for x, y in points]
        path = self.polygon_path(normalized)
        if path is None:
            return None
        item = QGraphicsPathItem(path)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(48)
        item.setBrush(QBrush(fill))
        item.seg_class_id = int(class_id)
        item.seg_points = normalized
        item.seg_preview = False
        label_item = QGraphicsSimpleTextItem(f"{label_text} · Segmentation", item)
        label_item.setBrush(QBrush(color))
        label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        label_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        bounds = path.boundingRect()
        label_item.setPos(bounds.left() + 4.0, bounds.top() + 4.0)
        label_item.setVisible(False)
        item.seg_label_item = label_item
        self.add_reference_item(item, layer_id=layer_id, opacity=0.50)
        return item

    def add_pose_reference(
        self,
        bbox: BoundingBox,
        keypoints: Sequence[PoseReferenceKeypoint],
        *,
        class_name: str,
        layer_id: str,
        keypoint_radius: int,
        keypoint_font_px: int,
        show_keypoint_labels: bool = False,
        color: QColor = QColor(104, 164, 207),
    ) -> tuple[BoxItem, tuple[KeypointItem, ...]]:
        box_item = BoxItem(bbox, f"{class_name} · Keypoints")
        box_item.set_reference_style(color, show_label=False)
        self.add_reference_item(box_item, layer_id=layer_id, opacity=0.52)
        rendered: list[KeypointItem] = []
        for reference in keypoints:
            item = KeypointItem(reference.keypoint, keypoint_radius, keypoint_font_px)
            item.visibility = int(reference.visibility)
            item.update_appearance()
            if reference.label_text and item.visibility > 0:
                item.text_item.setText(reference.label_text)
            item.set_reference_style(color, show_label=show_keypoint_labels)
            self.add_reference_item(
                item,
                layer_id=layer_id,
                opacity=0.90 if show_keypoint_labels else 0.52,
            )
            rendered.append(item)
        return box_item, tuple(rendered)

    def clear_references(self) -> None:
        self._remove_owned(self.reference_items)

    def forget_scene_items(self) -> None:
        """Drop ownership lists after an external whole-scene clear."""
        self.reference_items.clear()
        self.prompt_items.clear()
        self.depth_probe_items.clear()

    def _add_segmentation_frame(
        self,
        item: QGraphicsPathItem,
        path: QPainterPath,
        label_text: str,
        *,
        preview: bool,
    ) -> None:
        frame_color = QColor(32, 78, 255)
        frame_item = QGraphicsRectItem(item)
        frame_pen = QPen(frame_color)
        frame_pen.setWidth(2)
        frame_pen.setCosmetic(True)
        if preview:
            frame_pen.setStyle(Qt.PenStyle.DashLine)
        frame_item.setPen(frame_pen)
        frame_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        frame_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        frame_item.setAcceptHoverEvents(False)
        frame_item.setZValue(0.3)
        label_bg = QGraphicsRectItem(item)
        label_bg.setBrush(QBrush(frame_color))
        label_bg.setPen(QPen(Qt.PenStyle.NoPen))
        label_bg.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        label_bg.setAcceptHoverEvents(False)
        label_bg.setZValue(0.4)
        label_item = QGraphicsSimpleTextItem(label_text, item)
        label_font = QFont()
        label_font.setPixelSize(12)
        label_item.setFont(label_font)
        label_item.setBrush(QBrush(Qt.GlobalColor.white))
        label_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        label_item.setZValue(0.5)
        item.seg_frame_item = frame_item
        item.seg_label_bg = label_bg
        item.seg_label_item = label_item
        self._position_segmentation_frame(item, path)

    def _position_segmentation_frame(
        self,
        item: QGraphicsPathItem,
        path: QPainterPath,
    ) -> None:
        frame_item = getattr(item, "seg_frame_item", None)
        label_bg = getattr(item, "seg_label_bg", None)
        label_item = getattr(item, "seg_label_item", None)
        if frame_item is None or label_bg is None or label_item is None:
            return
        bounds = path.boundingRect()
        frame_item.setRect(bounds)
        text_rect = label_item.boundingRect()
        badge_width = text_rect.width() + 8.0
        badge_height = text_rect.height() + 2.0
        badge_x = bounds.left() + 2.0
        badge_y = bounds.top() - badge_height - 2.0
        if badge_y < self.scene.sceneRect().top():
            badge_y = bounds.bottom() + 2.0
        label_bg.setRect(badge_x, badge_y, badge_width, badge_height)
        label_item.setPos(badge_x + 4.0, badge_y + 1.0)

    def _remove_owned(self, items: list[QGraphicsItem], *, untrack: bool = False) -> None:
        for item in list(items):
            if item.scene() is self.scene:
                self.scene.removeItem(item)
            if untrack:
                self._untrack_item(item)
        items.clear()

    @staticmethod
    def _scaled_pixmap(pixmap: QPixmap, width: int, height: int) -> QPixmap:
        if pixmap.width() == int(width) and pixmap.height() == int(height):
            return pixmap
        return pixmap.scaled(
            int(width),
            int(height),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    @staticmethod
    def _default_depth_value(value: Any) -> str:
        return f"{float(value):.3f} m" if value is not None else "invalid"


__all__ = ["CanvasScenePresenter", "PoseReferenceKeypoint"]
