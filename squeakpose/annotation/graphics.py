"""Reusable annotation graphics items and labeling view."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontDatabase,
    QFontInfo,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from squeakpose.annotation.models import BoundingBox, Keypoint


def _ui_font(px: int) -> QFont:
    font = QFont()
    available = set(QFontDatabase.families())
    system_family = QFontDatabase.systemFont(
        QFontDatabase.SystemFont.GeneralFont
    ).family()
    for family in ("Fira Sans", system_family, "Segoe UI", "Arial", "Helvetica"):
        if family and family in available:
            font.setFamily(family)
            if QFontInfo(font).family() == family:
                break
    font.setPixelSize(px)
    return font


class BoxItem(QGraphicsRectItem):
    HANDLE = 8
    MIN_W = 6
    MIN_H = 6
    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 4, 8

    def __init__(self, bbox: BoundingBox, class_name: str):
        super().__init__(0, 0, max(self.MIN_W, bbox.w), max(self.MIN_H, bbox.h))
        self.setPos(bbox.x, bbox.y)
        self.bbox = bbox
        self.class_name = class_name

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(2)

        pen = QPen(Qt.GlobalColor.blue)
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.setPen(pen)

        self._label_bg = QGraphicsRectItem(self)
        self._label_bg.setBrush(QBrush(Qt.GlobalColor.blue))
        self._label_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self._label_bg.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_bg.setZValue(0.0)

        self._label = QGraphicsSimpleTextItem(class_name, self)
        self._label.setFont(_ui_font(12))
        self._label.setBrush(QBrush(Qt.GlobalColor.white))
        self._label.setZValue(0.1)
        self._label_pad_x = 4.0
        self._label_pad_y = 1.0
        self._reposition_label()

        self._drag_edges = 0
        self._press_rect = QRectF()
        self._press_pos_local = QPointF()
        self._press_item_pos = QPointF()

        self.update_model()  # ✅ now exists

    def _reposition_label(self):
        """Keep class label outside the bbox to avoid keypoint overlap.

        Preferred placement: above top-left corner. If the box is near the top
        edge of the scene, place the label just below the box instead.
        """
        margin = 2.0
        text_rect = self._label.boundingRect()
        bg_w = text_rect.width() + (self._label_pad_x * 2.0)
        bg_h = text_rect.height() + (self._label_pad_y * 2.0)
        x = margin
        y = -(bg_h + margin)

        if self.scene():
            sr = self.scene().sceneRect()
            # If above would clip off-screen, move label below the box.
            if self.pos().y() + y < sr.top():
                y = self.rect().height() + margin

        self._label_bg.setRect(x, y, bg_w, bg_h)
        self._label.setPos(x + self._label_pad_x, y + self._label_pad_y)

    # --- only outline is clickable/selectable ---
    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        stroker = QPainterPathStroker()
        stroker.setWidth(max(6.0, float(self.HANDLE) * 2.0))  # clickable band
        return stroker.createStroke(path)

    def contains(self, point: QPointF) -> bool:
        return self.shape().contains(point)

    # --- edge hit/cursor ---
    def _hit_edges(self, p_local: QPointF) -> int:
        r = self.rect()
        tol = max(6.0, float(self.HANDLE))
        edges = 0
        if abs(p_local.x() - r.left())   <= tol: edges |= self.LEFT
        if abs(p_local.x() - r.right())  <= tol: edges |= self.RIGHT
        if abs(p_local.y() - r.top())    <= tol: edges |= self.TOP
        if abs(p_local.y() - r.bottom()) <= tol: edges |= self.BOTTOM
        return edges

    def _cursor_for_edges(self, edges: int):
        # diagonals first, then single-axis
        if edges in (self.TOP | self.LEFT, self.BOTTOM | self.RIGHT):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (self.TOP | self.RIGHT, self.BOTTOM | self.LEFT):
            return Qt.CursorShape.SizeBDiagCursor
        if edges & (self.LEFT | self.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edges & (self.TOP | self.BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeAllCursor

    # --- events ---
    def hoverMoveEvent(self, event):
        edges = self._hit_edges(event.pos())
        self.setCursor(self._cursor_for_edges(edges) if edges else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._hit_edges(event.pos())
            if not edges:
                # clicked inside -> let keypoints receive it
                event.ignore()
                return
            self._drag_edges = edges
            self._press_rect = QRectF(self.rect())
            self._press_pos_local = QPointF(event.pos())
            self._press_item_pos = QPointF(self.pos())
            self.setZValue(2.5)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_edges:
            delta = event.pos() - self._press_pos_local
            self._apply_resize(self._drag_edges, delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_edges:
            self._drag_edges = 0
            self.setZValue(2)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # --- resize & clamp ---
    def _apply_resize(self, edges: int, delta_local: QPointF):
        new_rect = QRectF(self._press_rect)
        new_pos = QPointF(self._press_item_pos)

        if edges & self.LEFT:
            new_pos.setX(self._press_item_pos.x() + delta_local.x())
            new_rect.setWidth(max(self.MIN_W, self._press_rect.width() - delta_local.x()))
        if edges & self.RIGHT:
            new_rect.setWidth(max(self.MIN_W, self._press_rect.width() + delta_local.x()))
        if edges & self.TOP:
            new_pos.setY(self._press_item_pos.y() + delta_local.y())
            new_rect.setHeight(max(self.MIN_H, self._press_rect.height() - delta_local.y()))
        if edges & self.BOTTOM:
            new_rect.setHeight(max(self.MIN_H, self._press_rect.height() + delta_local.y()))

        if self.scene():
            sr = self.scene().sceneRect()
            new_pos.setX(max(sr.left(), new_pos.x()))
            new_pos.setY(max(sr.top(),  new_pos.y()))
            new_pos.setX(min(new_pos.x(), sr.right() - new_rect.width()))
            new_pos.setY(min(new_pos.y(), sr.bottom() - new_rect.height()))

        self.setPos(new_pos)
        self.setRect(0, 0, new_rect.width(), new_rect.height())
        self.update_model()

    # --- sync bbox dataclass ---
    def update_model(self):
        self.bbox.x = self.pos().x()
        self.bbox.y = self.pos().y()
        self.bbox.w = self.rect().width()
        self.bbox.h = self.rect().height()
        self._reposition_label()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            sr = self.scene().sceneRect()
            r = self.rect()
            new_pos = value
            nx = min(max(new_pos.x(), sr.left()), sr.right() - r.width())
            ny = min(max(new_pos.y(), sr.top()),  sr.bottom() - r.height())
            return QPointF(nx, ny)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            self._reposition_label()
        elif change in (QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
                        QGraphicsItem.GraphicsItemChange.ItemTransformChange):
            self.update_model()
        return super().itemChange(change, value)


class KeypointItem(QGraphicsEllipseItem):
    """
    Precise keypoint:
      - position in scene coords (setPos)
      - local rect centered at origin
      - ignores transformations (constant on-screen size)
      - cosmetic pen; clamped to image bounds
    """
    def __init__(self, kp: Keypoint, pixel_radius: int = 4, font_px: int = 10):
        super().__init__(-pixel_radius, -pixel_radius, pixel_radius*2, pixel_radius*2)
        self.kp = kp
        self.visibility = 2
        self._pixel_radius = max(1, pixel_radius)
        self._font_px = max(6, font_px)

        self.setPos(kp.x, kp.y)  # scene-space anchor
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(3)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

        color = Qt.GlobalColor.red
        pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
        self.setPen(pen); self.setBrush(QBrush(color))

        self.text_item = QGraphicsSimpleTextItem(kp.name, self)
        self.text_item.setFont(_ui_font(self._font_px))
        self.text_item.setBrush(QBrush(color))

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(0); shadow.setOffset(1, 1); shadow.setColor(Qt.GlobalColor.black)
        self.text_item.setGraphicsEffect(shadow)

        self._reposition_label()

    def _reposition_label(self):
        self.text_item.setPos(self._pixel_radius + 2, -self._pixel_radius - 2)

    def refresh_display_sizes(self, pixel_radius: int, font_px: int):
        self._pixel_radius = max(1, pixel_radius)
        self._font_px = max(6, font_px)
        self.prepareGeometryChange()
        self.setRect(-self._pixel_radius, -self._pixel_radius, self._pixel_radius*2, self._pixel_radius*2)
        self.text_item.setFont(_ui_font(self._font_px))
        self._reposition_label()

    def update_appearance(self):
        # 2 = visible (red), 1 = occluded (yellow), 0 = invisible/not present (gray dashed)
        if self.visibility == 2:
            color = Qt.GlobalColor.red
            pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
            self.setPen(pen); self.setBrush(QBrush(color))
            self.text_item.setBrush(QBrush(color))
            self.text_item.setVisible(True)
        elif self.visibility == 1:
            color = Qt.GlobalColor.yellow
            pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
            self.setPen(pen); self.setBrush(QBrush(color))
            self.text_item.setBrush(QBrush(color))
            self.text_item.setVisible(True)
        else:  # self.visibility == 0
            color = Qt.GlobalColor.lightGray
            pen = QPen(color); pen.setStyle(Qt.PenStyle.DashLine); pen.setWidth(1); pen.setCosmetic(True)
            self.setPen(pen)
            self.setBrush(QBrush(Qt.GlobalColor.transparent))
            self.text_item.setBrush(QBrush(color))
            # optional: keep labels hidden for invisible to reduce clutter
            self.text_item.setVisible(False)

    def toggle_visibility(self):
        # Cycle: 2 (visible) -> 1 (occluded) -> 0 (invisible) -> 2 ...
        if self.visibility == 2:
            self.visibility = 1
        elif self.visibility == 1:
            self.visibility = 0
        else:
            self.visibility = 2
        self.update_appearance()

    def update_model(self):
        p = self.pos()
        self.kp.x = p.x()
        self.kp.y = p.y()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            sr = self.scene().sceneRect()
            p = value
            x = min(max(p.x(), sr.left()), sr.right())
            y = min(max(p.y(), sr.top()),  sr.bottom())
            return QPointF(x, y)
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.update_model()
        return super().itemChange(change, value)


class LabelView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, app_ref):
        super().__init__(scene)
        self.app = app_ref
        self._start_pos: Optional[QPointF] = None
        self._crosshair_v: Optional[QGraphicsLineItem] = None
        self._crosshair_h: Optional[QGraphicsLineItem] = None
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._drawing_cancelled = False
        self._seg_brush_active = False
        self._seg_brush_add = True
        self._seg_brush_last_pos: Optional[QPointF] = None
        self._seg_brush_cursor_ring: Optional[QGraphicsEllipseItem] = None

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        if self.app.mode == 'panzoom':
            old_pos = self.mapToScene(event.position().toPoint())
            zoom_in_factor = 1.05
            zoom_out_factor = 1 / zoom_in_factor
            zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
            new_scale = self.transform().m11() * zoom_factor
            if new_scale < 1.0:
                zoom_factor = 1.0 / self.transform().m11()
            elif new_scale > 8.0:
                zoom_factor = 8.0 / self.transform().m11()
            self.scale(zoom_factor, zoom_factor)
            self.app.update_zoom_label()
            new_pos = self.mapToScene(event.position().toPoint())
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        else:
            super().wheelEvent(event)

    def _remove_crosshairs(self):
        if self._crosshair_v:
            owner_scene = self._crosshair_v.scene()
            if owner_scene is not None:
                owner_scene.removeItem(self._crosshair_v)
            self._crosshair_v = None
        if self._crosshair_h:
            owner_scene = self._crosshair_h.scene()
            if owner_scene is not None:
                owner_scene.removeItem(self._crosshair_h)
            self._crosshair_h = None

    def _ensure_crosshairs(self):
        """Create crosshair items with consistent styling if they do not exist."""
        if self._crosshair_v is None:
            self._crosshair_v = QGraphicsLineItem()
            self._crosshair_v.setZValue(10)
            pen = QPen(Qt.GlobalColor.cyan)
            pen.setCosmetic(True)
            self._crosshair_v.setPen(pen)
            self.scene().addItem(self._crosshair_v)
        if self._crosshair_h is None:
            self._crosshair_h = QGraphicsLineItem()
            self._crosshair_h.setZValue(10)
            pen = QPen(Qt.GlobalColor.cyan)
            pen.setCosmetic(True)
            self._crosshair_h.setPen(pen)
            self.scene().addItem(self._crosshair_h)

    def _update_crosshairs(self, scene_pos: QPointF):
        """Ensure crosshairs exist and update them to intersect at scene_pos."""
        if not self.scene():
            return
        self._ensure_crosshairs()
        img_bounds = self.scene().sceneRect()
        self._crosshair_v.setLine(scene_pos.x(), img_bounds.top(), scene_pos.x(), img_bounds.bottom())
        self._crosshair_h.setLine(img_bounds.left(), scene_pos.y(), img_bounds.right(), scene_pos.y())

    def draw_crosshairs_at(self, global_pos: QPoint):
        scene_pos = self.mapToScene(self.mapFromGlobal(global_pos))
        self._update_crosshairs(scene_pos)

    def _should_show_seg_brush_cursor(self) -> bool:
        try:
            return (
                self.app._is_seg_workflow()
                and self.app.mode == "segedit"
                and self.app._is_seg_edit_tool_brush()
                and self.scene() is not None
            )
        except Exception:
            return False

    def _ensure_seg_brush_cursor_ring(self):
        if self._seg_brush_cursor_ring is not None and self._seg_brush_cursor_ring.scene() is self.scene():
            return
        self._seg_brush_cursor_ring = QGraphicsEllipseItem()
        pen = QPen(QColor(118, 188, 255, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self._seg_brush_cursor_ring.setPen(pen)
        self._seg_brush_cursor_ring.setBrush(QBrush(Qt.GlobalColor.transparent))
        self._seg_brush_cursor_ring.setZValue(9.2)
        self._seg_brush_cursor_ring.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._seg_brush_cursor_ring.setAcceptHoverEvents(False)
        self._seg_brush_cursor_ring.setVisible(False)
        self.scene().addItem(self._seg_brush_cursor_ring)

    def _hide_seg_brush_cursor(self):
        ring = self._seg_brush_cursor_ring
        if ring is not None:
            ring.setVisible(False)

    def _reset_seg_brush_cursor(self):
        ring = self._seg_brush_cursor_ring
        if ring is None:
            return
        try:
            if ring.scene() is not None:
                ring.scene().removeItem(ring)
        except Exception:
            pass
        self._seg_brush_cursor_ring = None

    def _update_seg_brush_cursor(self, scene_pos: QPointF):
        if not self._should_show_seg_brush_cursor():
            self._hide_seg_brush_cursor()
            return
        self._ensure_seg_brush_cursor_ring()
        ring = self._seg_brush_cursor_ring
        if ring is None:
            return
        radius = max(2.0, float(getattr(self.app, "seg_brush_radius", 8)))
        x = float(scene_pos.x())
        y = float(scene_pos.y())
        ring.setRect(x - radius, y - radius, radius * 2.0, radius * 2.0)
        ring.setVisible(True)

    def refresh_seg_brush_cursor(self):
        if not self._should_show_seg_brush_cursor():
            self._hide_seg_brush_cursor()
            return
        vp = self.viewport().mapFromGlobal(QCursor.pos())
        if not self.viewport().rect().contains(vp):
            self._hide_seg_brush_cursor()
            return
        self._update_seg_brush_cursor(self.mapToScene(vp))

    def _cancel_draw(self):
        self._drawing_cancelled = True
        if self._temp_rect:
            owner_scene = self._temp_rect.scene()
            if owner_scene is not None:
                owner_scene.removeItem(self._temp_rect)
            self._temp_rect = None
        self._start_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self.app.mode == "segment":
            self.app._clear_seg_prompt_state()
            self.app.update_status_bar("Segmentation prompts cleared.")
        else:
            self.app.update_status_bar("Box drawing cancelled.")

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if self.app.mode == "segedit":
            if event.button() == Qt.MouseButton.LeftButton and self.app._start_seg_brush(scene_pos, add=True):
                self._seg_brush_active = True
                self._seg_brush_add = True
                self._seg_brush_last_pos = scene_pos
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton and self.app._start_seg_brush(scene_pos, add=False):
                self._seg_brush_active = True
                self._seg_brush_add = False
                self._seg_brush_last_pos = scene_pos
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.app.mode == 'panzoom':
                super().mousePressEvent(event)
            elif self.app.mode == 'bbox':
                # Do NOT clear here — clear only when committing a valid rect
                self._start_pos = scene_pos
                self._drawing_cancelled = False
                self._remove_crosshairs()
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif self.app.mode == 'keypoint':
                self.app.add_keypoint(scene_pos)
            elif self.app.mode == 'segment':
                self.app._add_seg_prompt(scene_pos, positive=True)
                event.accept()
                return
        elif event.button() == Qt.MouseButton.RightButton and self.app.mode == "segment":
            self.app._add_seg_prompt(scene_pos, positive=False)
            event.accept()
            return
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self._update_seg_brush_cursor(scene_pos)
        if self.app.mode == "segedit" and self._seg_brush_active:
            if self.app._apply_seg_brush(scene_pos, add=self._seg_brush_add, prev_scene_pos=self._seg_brush_last_pos):
                self._seg_brush_last_pos = scene_pos
            event.accept()
            return
        if (self.app.mode == 'bbox' and self._start_pos is None) or (self.app.mode in {'keypoint', 'segment'}):
            self._update_crosshairs(scene_pos)
        elif self._start_pos and self.app.mode == 'bbox':
            self._remove_crosshairs()
            end_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._start_pos, end_pos).normalized()
            if not self._temp_rect:
                self._temp_rect = QGraphicsRectItem(rect)
                pen = QPen(Qt.GlobalColor.yellow); pen.setWidth(2); pen.setCosmetic(True)
                self._temp_rect.setPen(pen); self._temp_rect.setZValue(1.5)
                self.scene().addItem(self._temp_rect)
            else:
                self._temp_rect.setRect(rect)
        elif self.app.mode == 'panzoom':
            self._remove_crosshairs()
            super().mouseMoveEvent(event)

    def enterEvent(self, event):
        self.refresh_seg_brush_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hide_seg_brush_cursor()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.app.mode == "segedit" and self._seg_brush_active:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.app._apply_seg_brush(scene_pos, add=self._seg_brush_add, prev_scene_pos=self._seg_brush_last_pos)
            self._seg_brush_active = False
            self._seg_brush_last_pos = None
            self.app._finish_seg_brush()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos and self.app.mode == 'bbox':
            if not self._drawing_cancelled:
                end_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._start_pos, end_pos).normalized()
                if rect.width() >= 2 and rect.height() >= 2:
                    self.app.add_bbox(rect)
            if self._temp_rect:
                owner_scene = self._temp_rect.scene()
                if owner_scene is not None:
                    owner_scene.removeItem(self._temp_rect)
                self._temp_rect = None
            self._start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.app.mode == 'panzoom':
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetTransform()
            self.app.update_zoom_label()
