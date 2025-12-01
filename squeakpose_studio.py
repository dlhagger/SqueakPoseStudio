#!/usr/bin/env python3
import sys, os, shutil, json, random, yaml, re, shlex, platform, csv, datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem,
    QGraphicsSimpleTextItem, QGraphicsLineItem, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QSplashScreen, QMessageBox,
    QDialog, QFrame, QStatusBar, QGraphicsDropShadowEffect, QSizePolicy,
    QProgressDialog, QDialogButtonBox, QTabWidget, QSlider, QSpinBox, QDoubleSpinBox, QProgressBar,
    QInputDialog, QFileDialog, QFormLayout, QLineEdit, QTextEdit
)
from PyQt6.QtGui import (
    QPixmap, QPen, QBrush, QKeySequence, QFont, QPainter, QShortcut,
    QFontDatabase, QIcon, QCursor, QPainterPath, QPainterPathStroker,
    QFontInfo
)
from PyQt6.QtCore import (
    Qt, QRectF, QPointF, QTimer, QPoint, QProcess
)

# --- cross-platform UI font helper ---
def _ui_font(px: int) -> QFont:
    f = QFont()
    available = set(QFontDatabase.families())
    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    ordered = ["Fira Sans", system_family, "Segoe UI", "Arial", "Helvetica"]
    seen = set()
    for family in ordered:
        if not family or family in seen:
            continue
        seen.add(family)
        if family in available:
            f.setFamily(family)
            if QFontInfo(f).family() == family:
                break
    f.setPixelSize(px)
    return f

# CV2

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None

# Ultralytics YOLO

from ultralytics import YOLO

# Preferred device order: CUDA → MPS → CPU
try:
    import torch as _torch
except Exception:
    _torch = None

def _auto_device() -> str:
    try:
        if _torch is not None:
            if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                return 'cuda'
            # On macOS, MPS can be present but not fully usable; check both built and available
            if hasattr(_torch, 'backends') and hasattr(_torch.backends, 'mps'):
                mps = _torch.backends.mps
                if getattr(mps, 'is_built', lambda: False)() and getattr(mps, 'is_available', lambda: False)():
                    return 'mps'
        return 'cpu'
    except Exception:
        return 'cpu'

# =========================
# Data Classes
# =========================

@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float
    class_id: int

    def to_yolo(self, img_w: float, img_h: float) -> Tuple[int, float, float, float, float]:
        xc = (self.x + self.w / 2) / img_w
        yc = (self.y + self.h / 2) / img_h
        return (self.class_id, xc, yc, self.w / img_w, self.h / img_h)


@dataclass(slots=True)
class Keypoint:
    x: float
    y: float
    class_id: int
    name: str

    def to_yolo(self, img_w: float, img_h: float) -> Tuple[int, float, float, str]:
        return (self.class_id, self.x / img_w, self.y / img_h, self.name)


# =========================
# Graphics Items
# =========================

class CongratsPopup(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎉 SqueakPose Studio")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        emoji = QLabel("🐭🧀🎉")
        emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emoji.setStyleSheet("font-size: 48px;")

        message = QLabel("All images have been labeled!\nAmazing work, Squeaker!")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet("font-size: 18px; padding: 10px;")

        ok_btn = QPushButton("Let's Go!")
        ok_btn.setStyleSheet("padding: 8px 16px; font-size: 14px;")
        ok_btn.clicked.connect(self.accept)

        layout.addWidget(emoji)
        layout.addWidget(message)
        layout.addWidget(ok_btn)
        self.setLayout(layout)
        self.setFixedSize(350, 300)

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

        self._label = QGraphicsSimpleTextItem(class_name, self)
        self._label.setFont(_ui_font(12))
        self._label.setBrush(QBrush(Qt.GlobalColor.blue))
        self._label.setPos(2, 2)

        self._drag_edges = 0
        self._press_rect = QRectF()
        self._press_pos_local = QPointF()
        self._press_item_pos = QPointF()

        self.update_model()  # ✅ now exists

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
        self._label.setPos(2, 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            sr = self.scene().sceneRect()
            r = self.rect()
            new_pos = value
            nx = min(max(new_pos.x(), sr.left()), sr.right() - r.width())
            ny = min(max(new_pos.y(), sr.top()),  sr.bottom() - r.height())
            return QPointF(nx, ny)
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




# =========================
# View
# =========================

class LabelView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, app_ref):
        super().__init__(scene)
        self.app = app_ref
        self._start_pos: Optional[QPointF] = None
        self._crosshair_v: Optional[QGraphicsLineItem] = None
        self._crosshair_h: Optional[QGraphicsLineItem] = None
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._drawing_cancelled = False

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)

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
            self.scene().removeItem(self._crosshair_v)
            self._crosshair_v = None
        if self._crosshair_h:
            self.scene().removeItem(self._crosshair_h)
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

    def _cancel_draw(self):
        self._drawing_cancelled = True
        if self._temp_rect:
            self.scene().removeItem(self._temp_rect)
            self._temp_rect = None
        self._start_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.app.update_status_bar("Box drawing cancelled.")

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
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
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if (self.app.mode == 'bbox' and self._start_pos is None) or (self.app.mode == 'keypoint'):
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

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos and self.app.mode == 'bbox':
            if not self._drawing_cancelled:
                end_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._start_pos, end_pos).normalized()
                if rect.width() >= 2 and rect.height() >= 2:
                    # Clear only when committing a valid rectangle
                    self.app._remove_all_boxes_and_keypoints()
                    self.app.add_bbox(rect)
            if self._temp_rect:
                self.scene().removeItem(self._temp_rect)
                self._temp_rect = None
            self._start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.app.mode == 'panzoom':
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetTransform()
            self.app.update_zoom_label()


# =========================
# Video Review Pan/Zoom View
# =========================

class VideoView(QGraphicsView):
    """Lightweight pan/zoom view for the video reviewer.
    - Mouse wheel: zoom in/out centered on cursor
    - Left-drag: pan (ScrollHandDrag)
    - Double-click: reset zoom
    - Shortcuts (+/-) handled by the dialog via QShortcut
    """
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_in = 1.05
        zoom_out = 1.0 / zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        # clamp zoom between 10% and 800%
        cur = self.transform().m11()
        new_scale = cur * factor
        if new_scale < 0.10:
            factor = 0.10 / cur
        elif new_scale > 8.0:
            factor = 8.0 / cur
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetTransform()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def reset_view(self):
        self.resetTransform()

# =========================
# Main Application
# =========================

class LabelingApp(QMainWindow):

    def _ensure_label_files(self, class_file: str, keypoint_file: str) -> tuple[list[str], list[str]]:
        """
        Ensure class and keypoint name files exist WITHOUT any UI prompts.
        - If either path is empty, place files next to the labels directory.
        - If a file is missing or empty, create it with sensible defaults.
        - Returns (classes, kp_names).
        """
        # Defaults
        default_classes = ["mouse"]
        default_kps = ["nose", "head", "left_ear", "right_ear", "back", "tail_base"]

        # Resolve fallback locations if the provided paths are empty
        project_root = os.path.dirname(self.label_dir) if self.label_dir else os.getcwd()
        if not class_file:
            class_file = os.path.join(project_root, "classes.txt")
        if not keypoint_file:
            keypoint_file = os.path.join(project_root, "keypoints.txt")

        # Ensure parent dirs exist
        try:
            cf_dir = os.path.dirname(class_file)
            kf_dir = os.path.dirname(keypoint_file)
            if cf_dir:
                os.makedirs(cf_dir, exist_ok=True)
            if kf_dir and kf_dir != cf_dir:
                os.makedirs(kf_dir, exist_ok=True)
        except Exception:
            pass

        # Helper to write a list to file
        def _write_lines(path: str, items: list[str]):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for s in items:
                        f.write(s + "\n")
            except Exception:
                pass

        # Create files if missing
        if not os.path.exists(class_file):
            _write_lines(class_file, default_classes)
        if not os.path.exists(keypoint_file):
            _write_lines(keypoint_file, default_kps)

        # Read them (if empty, backfill defaults)
        def _read_nonempty_lines(path: str) -> list[str]:
            try:
                lines = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
                if not lines:
                    return []
                return lines
            except Exception:
                return []

        classes = _read_nonempty_lines(class_file)
        kp_names = _read_nonempty_lines(keypoint_file)

        if not classes:
            classes = default_classes
            _write_lines(class_file, classes)
        if not kp_names:
            kp_names = default_kps
            _write_lines(keypoint_file, kp_names)

        return classes, kp_names

    def refresh_image_list(self):
        """Reload the images_to_label directory file list (used after exporting a frame from video)."""
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        try:
            self.images = sorted(f for f in os.listdir(self.image_dir) if f.lower().endswith(exts))
        except Exception:
            pass
    def __init__(self, image_dir: str, label_dir: str, class_file: str, keypoint_file: str):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        os.makedirs(self.label_dir, exist_ok=True)

        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        self.images = sorted(f for f in os.listdir(self.image_dir) if f.lower().endswith(exts))
        self.current_idx = 0

        # Ensure classes.txt and keypoints.txt exist (never prompt, always silent)
        self.classes, self.kp_names = self._ensure_label_files(class_file, keypoint_file)

        self.mode = 'panzoom'
        self.bboxes: List[BoundingBox] = []
        self.kps: List[Keypoint] = []
        self.current_kp_idx = 0
        self.predict_model_path: Optional[str] = None
        self.predict_model: Optional[YOLO] = None
        self.nav_filter = 'all'  # 'all' | 'labeled' | 'unlabeled'

        # keypoint display (screen-space)
        self.kp_pixel_radius = 4
        self.kp_font_px = 10
        self._precision_active = False

        self._log_path = os.path.join(os.path.dirname(__file__), 'squeakpose_debug.log')
        self._predict_busy = False
        # Auto-select device once at startup
        self._device = _auto_device()
        print(f"🧠 Inference device: {self._device}")
        # Build UI and load first image
        self._setup_ui()
        self.load_image()
    def closeEvent(self, event):
        super().closeEvent(event)

    # ---------- UI Setup ----------

    def _setup_ui(self):
        self.setWindowTitle('SqueakPose Studio')
        central = QWidget()
        self.setCentralWidget(central)

        self.scene = QGraphicsScene()
        self.view = LabelView(self.scene, self)

        # Floating mode panel
        self.mode_buttons_frame = QFrame(self.view)
        self.mode_buttons_frame.setStyleSheet("""
            background-color: rgba(60, 63, 65, 200);
            border: 1px solid #555;
            border-radius: 8px;
        """)
        self.mode_buttons_frame.setFixedWidth(140)

        mode_layout = QVBoxLayout(self.mode_buttons_frame)
        mode_layout.setContentsMargins(8, 8, 8, 8)
        mode_layout.setSpacing(5)

        self.mode_title = QLabel("Labeling Mode")
        self.mode_title.setStyleSheet("font-weight: bold; font-size: 10pt; color: #e0e0e0;")
        mode_layout.addWidget(self.mode_title)

        self.panzoom_btn = QPushButton('Pan/Zoom (1)')
        self.bbox_btn = QPushButton('BBox (2)')
        self.keypoint_btn = QPushButton('Keypoint (3)')
        self.predict_btn = QPushButton('Predict (4)')

        for btn, mode_name in [(self.panzoom_btn, 'panzoom'),
                               (self.bbox_btn, 'bbox'),
                               (self.keypoint_btn, 'keypoint')]:
            btn.clicked.connect(lambda checked, m=mode_name: self.set_mode(m))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3f41;
                    color: #e0e0e0;
                    border: none;
                    padding: 8px;
                    text-align: left;
                }
                QPushButton:hover { background-color: #505357; }
            """)
            mode_layout.addWidget(btn)

        self.predict_btn.clicked.connect(lambda checked: self.set_mode('predict'))
        mode_layout.addWidget(self.predict_btn)
        mode_layout.addStretch()
        self.mode_buttons_frame.show()
        self.center_mode_panel()

        # Top bar
        self.class_selector = QComboBox()
        self.class_selector.addItems(self.classes)

        top_layout = QHBoxLayout()

        # Browse filter selector
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Labeled", "Unlabeled"])
        self.filter_combo.setToolTip("Which images to browse with Prev/Next")
        self.filter_combo.currentTextChanged.connect(lambda t: self._set_nav_filter(t.lower()))
        top_layout.addWidget(QLabel("Browse:"))
        top_layout.addWidget(self.filter_combo)

        # Browse (no save)
        btn_prev = QPushButton('◀ Prev')
        btn_prev.clicked.connect(self.prev_index)
        top_layout.addWidget(btn_prev)

        btn_next = QPushButton('Next ▶')
        btn_next.clicked.connect(self.next_index)
        top_layout.addWidget(btn_next)

        # Workflow
        btn_complete = QPushButton('Complete → Next Unlabeled')
        btn_complete.clicked.connect(self.complete_and_next_unlabeled)
        top_layout.addWidget(btn_complete)

        btn_skip = QPushButton('Skip → Next Unlabeled')
        btn_skip.clicked.connect(self.skip_to_next_unlabeled)
        top_layout.addWidget(btn_skip)

        # Save
        btn_save = QPushButton('Save')
        btn_save.clicked.connect(self.save_labels)
        top_layout.addWidget(btn_save)

        # Spacer so loader/class sit on the right
        top_layout.addStretch()

        # --- Video tools (opens a separate review dialog; doesn't touch Images tab) ---
        btn_video = QPushButton("Video")
        btn_video.setToolTip("Predict an entire video, then review frames with overlays")
        btn_video.clicked.connect(self.open_video_reviewer)
        top_layout.addWidget(btn_video)

        # Model + class
        top_layout.addWidget(QLabel("Class:"))
        top_layout.addWidget(self.class_selector)
        
        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.view)
        bottom_controls = QHBoxLayout()

        btn_normalize = QPushButton("Validate Labels")
        btn_normalize.setToolTip("Rewrite labels_all files and ensure matching images exist in images_all")
        btn_normalize.clicked.connect(self.normalize_labels_all)
        bottom_controls.addWidget(btn_normalize)

        btn_export_dataset = QPushButton("Export Dataset")
        btn_export_dataset.setToolTip("Split images_all/labels_all into train/val and regenerate dataset.yaml")
        btn_export_dataset.clicked.connect(self.export_dataset)
        bottom_controls.addWidget(btn_export_dataset)

        btn_train = QPushButton("Train Model")
        btn_train.setToolTip("Launch a training run for a selected dataset")
        btn_train.clicked.connect(self.open_train_dialog)
        bottom_controls.addWidget(btn_train)

        bottom_controls.addStretch()

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        bottom_controls.addWidget(load_model_btn)

        self.inference_btn = QPushButton("Inference")
        self.inference_btn.setToolTip("Select a video, run YOLO, and export per-frame metrics to CSV")
        self.inference_btn.clicked.connect(self.run_video_inference)
        bottom_controls.addWidget(self.inference_btn)

        layout.addLayout(bottom_controls)
        central.setLayout(layout)

        # reflect initial nav filter in the dropdown
        try:
            mapping = {"all": 0, "labeled": 1, "unlabeled": 2}
            self.filter_combo.setCurrentIndex(mapping.get(self.nav_filter, 0))
        except Exception:
            pass

        # --- legend (bottom-left) ---
        self.legend_frame = QFrame(self.view)
        self.legend_frame.setStyleSheet("""
            background-color: rgba(60, 63, 65, 200);
            border: 1px solid #555;
            border-radius: 8px;
        """)
        # don't lock width; let it resize
        self.legend_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        legend_layout = QVBoxLayout(self.legend_frame)
        legend_layout.setContentsMargins(8, 8, 8, 8)
        legend_layout.setSpacing(5)

        self.legend_title = QLabel("Keypoint Visibility")
        self.legend_title.setStyleSheet("font-weight: bold; font-size: 10pt; color: #e0e0e0;")
        legend_layout.addWidget(self.legend_title)

        # multiline, can wrap, can expand
        self.legend_label = QLabel(
            "Keys:  🔴 Visible   🟡 Occluded   ⚪ Invisible (v=0)\n"
            "L: toggle labels    -/= point size    [ / ] text size    0: mark next invisible    Shift+0: selected → invisible"
        )
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet("font-size: 10pt; color: #e0e0e0;")
        self.legend_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        legend_layout.addWidget(self.legend_label)

        self.legend_frame.hide()


        # Floating zoom HUD
        self.zoom_frame = QFrame(self.view)
        self.zoom_frame.setStyleSheet("""
            background-color: rgba(60, 63, 65, 200);
            border: 1px solid #555;
            border-radius: 8px;
        """)
        self.zoom_frame.setFixedWidth(120)

        zoom_layout = QVBoxLayout(self.zoom_frame)
        zoom_layout.setContentsMargins(8, 8, 8, 8)
        zoom_layout.setSpacing(5)

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setStyleSheet("font-size: 10pt; color: #e0e0e0;")
        zoom_layout.addWidget(self.zoom_label)

        self.zoom_frame.move(10, 150)
        self.zoom_frame.hide()

        # Status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # Shortcuts
        self._bind_shortcuts()

    # ---------- Navigation helpers ----------

    def _find_next_unlabeled(self, start_from: int) -> int:
        """Return index of next frame without a label file. If none, returns current index."""
        total = len(self.images)
        idx = start_from
        for _ in range(total):
            idx = (idx + 1) % total
            base = os.path.splitext(self.images[idx])[0]
            label_file = os.path.join(self.label_dir, f"{base}.txt")
            if not os.path.exists(label_file):
                return idx
        return start_from  # all labeled

    # ---------- Navigation filtering ----------
    def _is_labeled_index(self, idx: int) -> bool:
        base = os.path.splitext(self.images[idx])[0]
        label_file = os.path.join(self.label_dir, f"{base}.txt")
        return os.path.exists(label_file)

    def _filtered_indices(self) -> list[int]:
        if not self.images:
            return []
        if self.nav_filter == 'all':
            return list(range(len(self.images)))
        elif self.nav_filter == 'labeled':
            return [i for i in range(len(self.images)) if self._is_labeled_index(i)]
        else:  # 'unlabeled'
            return [i for i in range(len(self.images)) if not self._is_labeled_index(i)]

    def _set_nav_filter(self, mode: str):
        if mode not in ('all', 'labeled', 'unlabeled'):
            return
        self.nav_filter = mode
        fi = self._filtered_indices()
        if not fi:
            self.update_status_bar(f"No images match filter: {mode}.")
            return
        if self.current_idx not in fi:
            self.current_idx = fi[0]
        self.update_status_bar(f"Browsing: {mode} ({fi.index(self.current_idx)+1}/{len(fi)})")
        self.load_image()

    def prev_index(self):
        fi = self._filtered_indices()
        if not fi:
            self.update_status_bar("No images found for current filter.")
            return
        if self.current_idx not in fi:
            self.current_idx = fi[0]
        else:
            pos = fi.index(self.current_idx)
            self.current_idx = fi[(pos - 1) % len(fi)]
        self.mode = 'bbox'
        self.load_image()

    def next_index(self):
        fi = self._filtered_indices()
        if not fi:
            self.update_status_bar("No images found for current filter.")
            return
        if self.current_idx not in fi:
            self.current_idx = fi[0]
        else:
            pos = fi.index(self.current_idx)
            self.current_idx = fi[(pos + 1) % len(fi)]
        self.mode = 'bbox'
        self.load_image()

    def complete_and_next_unlabeled(self):
        if not self._is_fully_labeled():
            QMessageBox.information(self, "Incomplete",
                                    "Place one bounding box and all keypoints to complete this frame.")
            return
        self.save_labels()
        next_idx = self._find_next_unlabeled(self.current_idx)
        if next_idx == self.current_idx:
            popup = CongratsPopup(); popup.exec()
            return
        self.current_idx = next_idx
        self.mode = 'bbox'
        self.load_image()

    def skip_to_next_unlabeled(self):
        next_idx = self._find_next_unlabeled(self.current_idx)
        if next_idx == self.current_idx:
            popup = CongratsPopup(); popup.exec()
            return
        self.current_idx = next_idx
        self.mode = 'bbox'
        self.load_image()

    # ---------- Prediction ----------

    def load_model(self):
        model_file, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model file", "", "Model Files (*.pt *.yaml *.onnx)"
        )
        if not model_file:
            return
        try:
            self.predict_model = YOLO(model_file)
            self.predict_model_path = model_file
            # Re-detect device in case hardware/availability changed
            self._device = _auto_device()
            print(f"🧠 Inference device: {self._device}")
            QMessageBox.information(self, "Model Loaded", f"Loaded model:\n{os.path.basename(model_file)}")
        except Exception as e:
            QMessageBox.warning(self, "Model Load Error", f"Could not load model:\n{e}")

    def run_video_inference(self):
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV:\n\n  pip install opencv-python")
            return
        if not self.predict_model:
            QMessageBox.information(self, "No Model", "Load a model before running inference.")
            return

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video for inference",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.mpg *.mpeg);;All Files (*)",
        )
        if not video_path:
            return

        # Gather basic video metadata for progress/time stamps
        total_frames = 0
        fps = 0.0
        try:
            cap_meta = _cv2.VideoCapture(video_path)
            if cap_meta is not None and cap_meta.isOpened():
                total_frames = int(cap_meta.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap_meta.get(_cv2.CAP_PROP_FPS) or 0.0)
            if cap_meta is not None:
                cap_meta.release()
        except Exception:
            total_frames = 0
            fps = 0.0

        device_name = str(getattr(self, "_device", "cpu")).lower()
        default_batch = 16 if device_name in {"cuda", "mps"} else 4
        batch_size, ok = QInputDialog.getInt(
            self,
            "Batch Size",
            "Frames per batch (larger uses more VRAM/RAM but speeds up inference):",
            value=max(1, default_batch),
            min=1,
            max=256,
        )
        if not ok:
            return

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_root = os.path.join(parent_dir, "inference outputs")
        try:
            os.makedirs(output_root, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Output Error", f"Could not create output directory:\n{output_root}\n\n{e}")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_name = f"{base_name}_{timestamp}.csv"
        csv_path = os.path.join(output_root, csv_name)

        # Pre-compute CSV schema (frame/detection metrics + keypoints)
        def _kp_key(name: str, idx: int) -> str:
            safe = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or f"kp{idx}").strip().lower())
            safe = safe.strip("_") or f"kp{idx}"
            return safe

        kp_columns = []
        for idx, kp_name in enumerate(self.kp_names):
            key = _kp_key(kp_name, idx)
            kp_columns.extend([
                f"kp_{key}_x",
                f"kp_{key}_y",
                f"kp_{key}_conf",
                f"kp_{key}_x_norm",
                f"kp_{key}_y_norm",
            ])

        fieldnames = [
            "video_path",
            "model_path",
            "frame_index",
            "time_seconds",
            "detections_in_frame",
            "detection_index",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_width",
            "bbox_height",
            "bbox_area",
            "bbox_center_x",
            "bbox_center_y",
            "bbox_center_x_norm",
            "bbox_center_y_norm",
            "bbox_width_norm",
            "bbox_height_norm",
            "image_width",
            "image_height",
            "speed_preprocess_ms",
            "speed_inference_ms",
            "speed_postprocess_ms",
        ] + kp_columns

        rows: list[dict] = []
        canceled = False
        had_error = False
        error_message = ""
        model_path = getattr(self, "predict_model_path", "")

        prog = QProgressDialog("Running inference…", "Cancel", 0, 0 if total_frames <= 0 else total_frames, self)
        prog.setWindowTitle("Video Inference")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        if total_frames <= 0:
            prog.setRange(0, 0)  # busy indicator for unknown length

        # Avoid concurrent predictions
        was_busy = getattr(self, "_predict_busy", False)
        self._predict_busy = True
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(False)
        if hasattr(self, "inference_btn"):
            self.inference_btn.setEnabled(False)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cap = _cv2.VideoCapture(video_path)
            if cap is None or not cap.isOpened():
                QMessageBox.warning(self, "Video Error", f"Unable to open video:\n{video_path}")
                return

            frames: list = []
            frame_indices: list[int] = []
            processed_frames = 0

            def process_batch() -> bool:
                nonlocal frames, frame_indices, rows, canceled, processed_frames, had_error, error_message
                if not frames:
                    return True

                batch_frames = frames[:]
                batch_indices = frame_indices[:]
                frames.clear()
                frame_indices.clear()

                try:
                    predict_args = dict(
                        source=batch_frames,
                        imgsz=640,
                        conf=0.25,
                        iou=0.5,
                        device=self._device,
                        verbose=False,
                    )
                    if batch_size > 0:
                        predict_args["batch"] = batch_size
                    results_list = list(self.predict_model.predict(**predict_args))
                except Exception as e:
                    had_error = True
                    error_message = str(e)
                    return False

                for fi, result in zip(batch_indices, results_list):
                    if prog.wasCanceled():
                        canceled = True
                        break

                    img_h, img_w = 0, 0
                    if hasattr(result, "orig_shape") and result.orig_shape:
                        img_h, img_w = int(result.orig_shape[0]), int(result.orig_shape[1])
                    elif hasattr(result, "orig_img") and getattr(result, "orig_img") is not None:
                        img_h, img_w = result.orig_img.shape[:2]

                    detections = int(len(result.boxes) if result.boxes is not None else 0)
                    speed = getattr(result, "speed", {}) or {}
                    time_seconds = (fi / fps) if fps > 0 else ""

                    if detections == 0:
                        row = {
                            "video_path": video_path,
                            "model_path": model_path,
                            "frame_index": fi,
                            "time_seconds": time_seconds,
                            "detections_in_frame": 0,
                            "detection_index": -1,
                            "track_id": "",
                            "class_id": "",
                            "class_name": "",
                            "confidence": "",
                            "bbox_x1": "",
                            "bbox_y1": "",
                            "bbox_x2": "",
                            "bbox_y2": "",
                            "bbox_width": "",
                            "bbox_height": "",
                            "bbox_area": "",
                            "bbox_center_x": "",
                            "bbox_center_y": "",
                            "bbox_center_x_norm": "",
                            "bbox_center_y_norm": "",
                            "bbox_width_norm": "",
                            "bbox_height_norm": "",
                            "image_width": img_w,
                            "image_height": img_h,
                            "speed_preprocess_ms": speed.get("preprocess"),
                            "speed_inference_ms": speed.get("inference"),
                            "speed_postprocess_ms": speed.get("postprocess"),
                        }
                        for col in kp_columns:
                            row[col] = ""
                        rows.append(row)
                    else:
                        xyxy = result.boxes.xyxy.cpu().tolist()
                        xywh = result.boxes.xywh.cpu().tolist()
                        confs = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [None] * detections
                        cls_list = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [0] * detections
                        ids_raw = getattr(result.boxes, "id", None)
                        if ids_raw is not None:
                            try:
                                ids_list = ids_raw.cpu().tolist()
                            except Exception:
                                ids_list = [None] * detections
                        else:
                            ids_list = [None] * detections

                        kp_abs = []
                        kp_norm = []
                        if hasattr(result, "keypoints") and result.keypoints is not None:
                            try:
                                kp_abs = result.keypoints.data.cpu().tolist()
                                kp_norm = result.keypoints.xyn.cpu().tolist() if hasattr(result.keypoints, "xyn") else []
                            except Exception:
                                kp_abs = []
                                kp_norm = []

                        for det_idx in range(detections):
                            x1, y1, x2, y2 = xyxy[det_idx]
                            cx, cy, w, h = xywh[det_idx]
                            area = w * h
                            cls_id = int(cls_list[det_idx]) if det_idx < len(cls_list) else 0
                            class_name = ""
                            if hasattr(result, "names") and result.names:
                                class_name = result.names.get(cls_id, "")
                            if not class_name and 0 <= cls_id < len(self.classes):
                                class_name = self.classes[cls_id]

                            kp_values = kp_abs[det_idx] if det_idx < len(kp_abs) else []
                            kp_norm_values = kp_norm[det_idx] if det_idx < len(kp_norm) else []

                            row = {
                                "video_path": video_path,
                                "model_path": model_path,
                                "frame_index": fi,
                                "time_seconds": time_seconds,
                                "detections_in_frame": detections,
                                "detection_index": det_idx,
                                "track_id": ids_list[det_idx] if det_idx < len(ids_list) else "",
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": confs[det_idx] if det_idx < len(confs) else "",
                                "bbox_x1": x1,
                                "bbox_y1": y1,
                                "bbox_x2": x2,
                                "bbox_y2": y2,
                                "bbox_width": w,
                                "bbox_height": h,
                                "bbox_area": area,
                                "bbox_center_x": cx,
                                "bbox_center_y": cy,
                                "bbox_center_x_norm": (cx / img_w) if img_w else "",
                                "bbox_center_y_norm": (cy / img_h) if img_h else "",
                                "bbox_width_norm": (w / img_w) if img_w else "",
                                "bbox_height_norm": (h / img_h) if img_h else "",
                                "image_width": img_w,
                                "image_height": img_h,
                                "speed_preprocess_ms": speed.get("preprocess"),
                                "speed_inference_ms": speed.get("inference"),
                                "speed_postprocess_ms": speed.get("postprocess"),
                            }

                            for idx_kp, kp_name in enumerate(self.kp_names):
                                key = _kp_key(kp_name, idx_kp)
                                abs_val = kp_values[idx_kp] if idx_kp < len(kp_values) else [None, None, None]
                                norm_val = kp_norm_values[idx_kp] if idx_kp < len(kp_norm_values) else [None, None]
                                row[f"kp_{key}_x"] = abs_val[0] if abs_val and abs_val[0] is not None else ""
                                row[f"kp_{key}_y"] = abs_val[1] if abs_val and abs_val[1] is not None else ""
                                row[f"kp_{key}_conf"] = abs_val[2] if abs_val and len(abs_val) > 2 else ""
                                row[f"kp_{key}_x_norm"] = norm_val[0] if norm_val and norm_val[0] is not None else ""
                                row[f"kp_{key}_y_norm"] = norm_val[1] if norm_val and norm_val[1] is not None else ""

                            rows.append(row)

                    processed_frames += 1
                    if total_frames > 0:
                        prog.setValue(min(processed_frames, total_frames))
                        prog.setLabelText(f"Inferencing frame {processed_frames}/{total_frames}")
                    else:
                        prog.setLabelText(f"Inferencing frame {processed_frames}")
                    QApplication.processEvents()

                    if canceled:
                        break

                return not canceled and not had_error

            frame_idx = 0
            try:
                while not canceled and not had_error:
                    if prog.wasCanceled():
                        canceled = True
                        break

                    ok, frame = cap.read()
                    if not ok:
                        break

                    frames.append(frame)
                    frame_indices.append(frame_idx)
                    frame_idx += 1

                    if len(frames) >= batch_size:
                        if not process_batch():
                            break

                if not canceled and not had_error and frames:
                    process_batch()
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            had_error = True
            error_message = str(e)
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()
            self._predict_busy = was_busy
            if hasattr(self, "predict_btn"):
                self.predict_btn.setEnabled(True)
            if hasattr(self, "inference_btn"):
                self.inference_btn.setEnabled(True)

        if had_error:
            QMessageBox.critical(self, "Inference Error", f"An error occurred during inference:\n{error_message}")
            return

        if not rows:
            if canceled:
                QMessageBox.information(self, "Inference Canceled", "Inference canceled before any results were generated.")
            return

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            QMessageBox.warning(self, "Write Error", f"Failed to write CSV:\n{csv_path}\n\n{e}")
            return

        message = f"Saved {len(rows)} row(s) to:\n{csv_path}"
        if canceled:
            message = "Inference canceled early.\n" + message
        QMessageBox.information(self, "Inference Complete", message)

    def set_mode(self, mode: str):
        if mode == 'predict':
            if not self.predict_model:
                QMessageBox.information(self, "No Model", "Please click 'Load Model' first.")
                return
            if not self.images:
                self.update_status_bar("No images to predict.")
                return
            if self._predict_busy:
                self.update_status_bar("Prediction already running...")
                return
            self.run_prediction_on_current_image()
            return

        self.mode = mode
        self._update_status()

        if hasattr(self.view, '_remove_crosshairs'):
            self.view._remove_crosshairs()

        if self.mode == 'panzoom':
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.mode == 'bbox':
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, 'draw_crosshairs_at'):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == 'keypoint':
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, 'draw_crosshairs_at'):
                self.view.draw_crosshairs_at(QCursor.pos())

    def run_prediction_on_current_image(self):
        if not self.predict_model or not self.images:
            return
        img_path = os.path.join(self.image_dir, self.images[self.current_idx])

        if self._predict_busy:
            self.update_status_bar("Prediction already running...")
            return

        self._predict_busy = True
        if hasattr(self, 'predict_btn'):
            self.predict_btn.setEnabled(False)
        self.update_status_bar("Running prediction...")

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            results_list = self.predict_model.predict(
                source=img_path,
                imgsz=640,
                conf=0.25,
                iou=0.5,
                device=self._device,
                verbose=False
            )
            results = results_list[0]
            self._apply_prediction(results)
        except Exception as e:
            import traceback, datetime
            tb = traceback.format_exc()
            try:
                with open(self._log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f"\n[{datetime.datetime.now().isoformat()}] Sync prediction error on {img_path}\n{tb}\n")
            except Exception:
                pass
            self._on_predict_error(str(e))
        finally:
            QApplication.restoreOverrideCursor()
            # ensure reset on ALL paths
            self._predict_busy = False
            if hasattr(self, 'predict_btn'):
                self.predict_btn.setEnabled(True)


    def _apply_prediction(self, results):
        try:
            self._remove_all_boxes_and_keypoints()

            # Re-enable predict button / clear busy state
            self._predict_busy = False
            if hasattr(self, 'predict_btn'):
                self.predict_btn.setEnabled(True)

            # Select highest-confidence detection if multiple; warn if >1
            idx = 0
            if results.boxes is not None and len(results.boxes) > 0:
                if hasattr(results.boxes, 'conf') and results.boxes.conf is not None:
                    # pick the highest-confidence index
                    import torch
                    try:
                        idx = int(results.boxes.conf.argmax().item())
                    except Exception:
                        idx = 0
                else:
                    idx = 0
                if len(results.boxes) > 1:
                    self.update_status_bar(f"Prediction returned {len(results.boxes)} detections; using the top-confidence one.")

                xyxy = results.boxes.xyxy.cpu().tolist()[idx]
                cls = int(results.boxes.cls.cpu().tolist()[idx]) if results.boxes.cls is not None else 0
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                bb = BoundingBox(x1, y1, w, h, cls)
                item = BoxItem(bb, self.classes[bb.class_id] if bb.class_id < len(self.classes) else str(bb.class_id))
                self.scene.addItem(item)
                self.bboxes.append(bb)

            # Apply keypoints for the chosen instance (index matches `idx` above)
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                kps_list = results.keypoints.data.cpu().numpy().tolist()
                if kps_list:
                    use_idx = min(idx, len(kps_list) - 1)
                    inst = kps_list[use_idx]
                    for idx_pt, (x, y, vis) in enumerate(inst):
                        name = self.kp_names[idx_pt] if idx_pt < len(self.kp_names) else f"kp{idx_pt}"
                        kp_obj = Keypoint(x, y, self.class_selector.currentIndex(), name)
                        kp_item = KeypointItem(kp_obj, self.kp_pixel_radius, self.kp_font_px)
                        kp_item.visibility = int(vis) if vis in (0, 1, 2) else (2 if vis > 0 else 1)
                        kp_item.update_appearance()
                        self.scene.addItem(kp_item)
                        self.kps.append(kp_obj)
                    self.current_kp_idx = min(len(self.kp_names), len(self.kps))

            self._update_status()
            self.update_status_bar("Prediction applied.")
        except Exception as e:
            import traceback, datetime
            tb = traceback.format_exc()
            try:
                with open(self._log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f"\n[{datetime.datetime.now().isoformat()}] Apply-prediction error on {self.images[self.current_idx] if self.images else 'N/A'}\n{tb}\n")
            except Exception:
                pass
            self._on_predict_error(tb)
            return

    def _on_predict_error(self, error_text: str):
        # Reset busy state and re-enable button
        self._predict_busy = False
        if hasattr(self, 'predict_btn'):
            self.predict_btn.setEnabled(True)
        # Surface the error to the user and point to the log
        try:
            QMessageBox.critical(self, "Prediction Error",
                                 f"An error occurred during prediction.\n\nDetails:\n{error_text[:1000]}\n\nA full traceback was written to:\n{self._log_path}")
        except Exception:
            pass
        self.update_status_bar("Prediction failed. See log for details.")

    def _reset_zoom(self):
        self.view.resetTransform()
        self.update_zoom_label()

    def mark_current_kp_invisible(self):
        """Mark the next required keypoint as invisible (v=0) and advance."""
        if self.mode != 'keypoint':
            self.update_status_bar("Switch to Keypoint mode to mark invisible (press 3).")
            return
        if not self.bboxes:
            self.update_status_bar("Place a bounding box first.")
            return
        if self.current_kp_idx >= len(self.kp_names):
            self.update_status_bar("All keypoints already placed.")
            return

        name = self.kp_names[self.current_kp_idx]
        cid = self.class_selector.currentIndex()

        # Use (0,0) for invisibles; YOLO ignores coords when v=0
        kp = Keypoint(0.0, 0.0, cid, name)
        item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
        item.visibility = 0
        item.update_appearance()

        # Keep it in the scene so saving picks it up (subtle visual)
        self.scene.addItem(item)
        self.kps.append(kp)

        # Advance to next missing name
        if hasattr(self, "_sync_current_kp_idx"):
            self._sync_current_kp_idx()
        else:
            self.current_kp_idx = min(self.current_kp_idx + 1, len(self.kp_names))

        self._update_status()
        self.update_status_bar(f"Marked '{name}' invisible (v=0).")

    def set_selected_invisible(self):
        """Convert selected keypoints to invisible (v=0) without moving them."""
        changed = False
        for it in self.scene.selectedItems():
            if isinstance(it, KeypointItem):
                it.visibility = 0
                it.update_appearance()
                changed = True
        if changed:
            self.update_status_bar("Selected keypoints set to invisible (v=0).")


    # ---------- Shortcuts & input ----------

    def _bind_shortcuts(self):
        # modes & core actions
        mapping = {
            '1': lambda: self.set_mode('panzoom'),
            '2': lambda: self.set_mode('bbox'),
            '3': lambda: self.set_mode('keypoint'),
            '4': lambda: self.set_mode('predict'),
            'S': self.save_labels,
            'Z': self.undo,
            'V': self.toggle_selected_visibility,
            'R': self._reset_zoom,  # <-- refresh zoom label too
            Qt.Key.Key_Delete: self.delete_selected,
            Qt.Key.Key_Backspace: self.delete_selected,  # optional: Mac-friendly
            Qt.Key.Key_P: self.prev_index,
            Qt.Key.Key_N: self.next_index,
        }
        for key, func in mapping.items():
            QShortcut(QKeySequence(key), self).activated.connect(func)

        # cancel drawing
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(self.view._cancel_draw)

        # Size controls + label toggle
        QShortcut(QKeySequence('='), self).activated.connect(lambda: self._bump_kp_size(+1))
        QShortcut(QKeySequence('-'), self).activated.connect(lambda: self._bump_kp_size(-1))
        QShortcut(QKeySequence(']'), self).activated.connect(lambda: self._bump_kp_font(+1))
        QShortcut(QKeySequence('['), self).activated.connect(lambda: self._bump_kp_font(-1))
        QShortcut(QKeySequence('L'), self).activated.connect(self._toggle_kp_labels)

        # Invisible keypoints
        QShortcut(QKeySequence('0'), self).activated.connect(self.mark_current_kp_invisible)
        QShortcut(QKeySequence('Shift+0'), self).activated.connect(self.set_selected_invisible)

        # Workflow jumps
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.complete_and_next_unlabeled)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self.complete_and_next_unlabeled)
        QShortcut(QKeySequence('K'), self).activated.connect(self.skip_to_next_unlabeled)
        QShortcut(QKeySequence('Meta+Return'), self).activated.connect(self.complete_and_next_unlabeled)  # optional: macOS

    def keyPressEvent(self, event):
        # Space = temporary pan
        if event.key() == Qt.Key.Key_Space:
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.setCursor(Qt.CursorShape.OpenHandCursor)
            return

        # Arrow keys:
        # - If we're in keypoint mode AND at least one KeypointItem is selected -> nudge
        # - Else -> browse frames prev/next
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            selected_kp = any(isinstance(it, KeypointItem) for it in self.scene.selectedItems())
            if self.mode == 'keypoint' and selected_kp:
                step = 0.5
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    step = 3.0
                dx = dy = 0
                if event.key() == Qt.Key.Key_Left:  dx = -step
                elif event.key() == Qt.Key.Key_Right: dx = step
                elif event.key() == Qt.Key.Key_Up:   dy = -step
                elif event.key() == Qt.Key.Key_Down: dy = step
                for it in self.scene.selectedItems():
                    if isinstance(it, KeypointItem):
                        it.moveBy(dx, dy)
                        it.update_model()
                return
            else:
                # browse
                if event.key() == Qt.Key.Key_Left:
                    self.prev_index()
                elif event.key() == Qt.Key.Key_Right:
                    self.next_index()
                return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.set_mode(self.mode)  # restore cursor/drag per current tool
            return
        super().keyReleaseEvent(event)

    def _bump_kp_size(self, d):
        self.kp_pixel_radius = max(1, self.kp_pixel_radius + d)
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                it.refresh_display_sizes(self.kp_pixel_radius, self.kp_font_px)

    def _bump_kp_font(self, d):
        self.kp_font_px = max(6, self.kp_font_px + d)
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                it.refresh_display_sizes(self.kp_pixel_radius, self.kp_font_px)

    def _toggle_kp_labels(self):
        any_visible = any(isinstance(it, KeypointItem) and it.text_item.isVisible() for it in self.scene.items())
        new_vis = not any_visible
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                it.text_item.setVisible(new_vis)
        self.update_status_bar("Keypoint labels " + ("shown" if new_vis else "hidden"))

    def update_status_bar(self, msg: str):
        self.status.showMessage(msg, 2500)

    def _kp_text(self) -> str:
        if self.mode == 'keypoint':
            return f"Next: {self.kp_names[self.current_kp_idx]}  ({self.current_kp_idx}/{len(self.kp_names)})" \
                   if self.current_kp_idx < len(self.kp_names) else "All keypoints placed"
        return ""

    # ---------- Image load / navigation ----------

    def load_image(self):
        if hasattr(self.view, '_remove_crosshairs'):
            self.view._remove_crosshairs()

        self.scene.clear()
        if not self.images:
            return

        img_path = os.path.join(self.image_dir, self.images[self.current_idx])
        pix = QPixmap(img_path)
        if pix.isNull():
            self.update_status_bar(f"Failed to load image: {self.images[self.current_idx]}")
            return
        self.img_w, self.img_h = pix.width(), pix.height()
        self.scene.setSceneRect(0, 0, self.img_w, self.img_h)
        bg_item = QGraphicsPixmapItem(pix)
        bg_item.setZValue(0)
        self.scene.addItem(bg_item)

        self.bboxes.clear()
        self.kps.clear()
        self.current_kp_idx = 0

        base = os.path.splitext(self.images[self.current_idx])[0]
        label_file = os.path.join(self.label_dir, f"{base}.txt")

        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                parts = lines[0].split()
                if len(parts) >= 5:
                    cid = int(parts[0]); xc = float(parts[1]); yc = float(parts[2])
                    w = float(parts[3]); h = float(parts[4])
                    x = (xc - w / 2) * self.img_w
                    y = (yc - h / 2) * self.img_h
                    w_pix = w * self.img_w
                    h_pix = h * self.img_h
                    bbox = BoundingBox(x, y, w_pix, h_pix, cid)
                    item = BoxItem(bbox, self.classes[cid] if cid < len(self.classes) else str(cid))
                    self.scene.addItem(item)
                    self.bboxes.append(bbox)

                    kp_data = parts[5:]
                    for idx in range(0, len(kp_data), 3):
                        if idx + 2 >= len(kp_data):
                            break
                        xn = float(kp_data[idx])
                        yn = float(kp_data[idx + 1])
                        vis = int(kp_data[idx + 2])
                        if idx // 3 < len(self.kp_names):
                            kp_name = self.kp_names[idx // 3]
                            x_pix = xn * self.img_w
                            y_pix = yn * self.img_h
                            kp = Keypoint(x_pix, y_pix, cid, kp_name)
                            kp_item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
                            kp_item.visibility = vis
                            kp_item.update_appearance()
                            self.scene.addItem(kp_item)
                            self.kps.append(kp)

        self.current_kp_idx = min(len(self.kp_names), len(self.kps))
        self._update_status()
        scene_center = self.scene.sceneRect().center()
        self.view.centerOn(scene_center)

    def add_bbox(self, rect: QRectF):
        self._remove_all_boxes_and_keypoints()
        cid = self.class_selector.currentIndex()
        class_name = self.classes[cid]
        bbox = BoundingBox(rect.x(), rect.y(), rect.width(), rect.height(), cid)
        item = BoxItem(bbox, class_name)
        self.scene.addItem(item)
        self.bboxes.append(bbox)
        self.update_status_bar("Box added. Switch to Keypoint mode (3).")

    def add_keypoint(self, pos: QPointF):
        if not self.bboxes:
            self.update_status_bar("Place a bounding box first.")
            return
        if self.current_kp_idx >= len(self.kp_names):
            self.update_status_bar("All keypoints placed for this frame.")
            return
        cid = self.class_selector.currentIndex()
        name = self.kp_names[self.current_kp_idx]
        kp = Keypoint(pos.x(), pos.y(), cid, name)
        item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
        self.scene.addItem(item)
        self.kps.append(kp)
        self.current_kp_idx += 1
        self._update_status()

    def delete_selected(self):
        for item in list(self.scene.selectedItems()):
            if isinstance(item, BoxItem):
                if item.bbox in self.bboxes:
                    self.bboxes.remove(item.bbox)
            if isinstance(item, KeypointItem):
                if item.kp in self.kps:
                    self.kps.remove(item.kp)
            self.scene.removeItem(item)
        self.current_kp_idx = min(self.current_kp_idx, len(self.kp_names), len(self.kps))
        self._update_status()

    def undo(self):
        if self.mode == 'keypoint' and self.kps:
            kp = self.kps.pop()
            for it in list(self.scene.items()):
                if isinstance(it, KeypointItem) and it.kp is kp:
                    self.scene.removeItem(it)
                    break
            self.current_kp_idx = max(0, self.current_kp_idx - 1)
            self._update_status()
        elif self.mode == 'bbox' and self.bboxes:
            bb = self.bboxes.pop()
            for it in list(self.scene.items()):
                if isinstance(it, BoxItem) and it.bbox is bb:
                    self.scene.removeItem(it)
                    break
            for it in list(self.scene.items()):
                if isinstance(it, KeypointItem):
                    self.scene.removeItem(it)
            self.kps.clear()
            self.current_kp_idx = 0
            self._update_status()

    def _is_fully_labeled(self) -> bool:
        return len(self.bboxes) == 1 and self.current_kp_idx == len(self.kp_names)

    def _update_status(self):
        buttons = {'panzoom': self.panzoom_btn, 'bbox': self.bbox_btn,
                   'keypoint': self.keypoint_btn, 'predict': self.predict_btn}
        for mode_name, button in buttons.items():
            if self.mode == mode_name:
                button.setStyleSheet("background-color: #505357; font-weight: bold;")
            else:
                button.setStyleSheet("")

        # Show filtered index / total in status bar
        fi = self._filtered_indices()
        if fi and self.current_idx in fi:
            idx_in_view = fi.index(self.current_idx) + 1
            self.status.showMessage(f"Viewing {self.nav_filter}: {idx_in_view}/{len(fi)}", 2000)

        if self.mode == 'keypoint':
            self.legend_frame.show()
            self._layout_overlays()
            self.zoom_frame.hide()
            frame_height = self.legend_frame.sizeHint().height()
            view_height = self.view.viewport().height()
            self.legend_frame.move(10, view_height - frame_height - 10)
            self.update_status_bar(self._kp_text())
        elif self.mode == 'panzoom':
            self.legend_frame.hide()
            self.zoom_frame.show()
            frame_height = self.zoom_frame.sizeHint().height()
            view_height = self.view.viewport().height()
            self.zoom_frame.move(10, view_height - frame_height - 10)
            self.update_zoom_label()
        else:
            self.legend_frame.hide()
            self.zoom_frame.hide()

    def toggle_selected_visibility(self):
        for item in self.scene.selectedItems():
            if isinstance(item, KeypointItem):
                item.toggle_visibility()

    def update_zoom_label(self):
        zoom = int(self.view.transform().m11() * 100)
        self.zoom_label.setText(f"Zoom: {zoom}%")

    def center_mode_panel(self):
        frame_height = self.mode_buttons_frame.sizeHint().height()
        view_height = self.view.viewport().height()
        y_centered = (view_height - frame_height) // 2
        self.mode_buttons_frame.move(10, y_centered)

    def _layout_overlays(self):
        """Dynamically position and size legend / zoom overlays."""
        vw = self.view.viewport().width()
        vh = self.view.viewport().height()

        # --- compute a content-based preferred width ---
        fm = self.legend_label.fontMetrics()
        ch = fm.horizontalAdvance('M')  # approx width of one character
        # Aim for ~26 characters per line (+ padding), wrap the rest
        preferred = int(ch * 26 + 20)

        # Clamp between sensible bounds and a fraction of viewport
        w = max(220, min(preferred, int(vw * 0.32), 360))

        # Apply width and position bottom-left
        self.legend_frame.setFixedWidth(w)
        h = self.legend_frame.sizeHint().height()
        self.legend_frame.move(10, vh - h - 10)

        # If you also show the zoom HUD, keep it in the same corner
        if hasattr(self, "zoom_frame") and self.zoom_frame.isVisible():
            zh = self.zoom_frame.sizeHint().height()
            self.zoom_frame.move(10, vh - zh - 10)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'mode_buttons_frame'):
            self.center_mode_panel()
            self._layout_overlays()

    def _remove_all_boxes_and_keypoints(self):
        for it in list(self.scene.items()):
            if isinstance(it, (BoxItem, KeypointItem)):
                self.scene.removeItem(it)
        self.bboxes.clear()
        self.kps.clear()
        self.current_kp_idx = 0
        self._update_status()

    # ---------- Save ----------
    def _collect_keypoints_by_name(self) -> dict[str, tuple[Keypoint, int]]:
        """Return {kp_name: (Keypoint, visibility)} for all KeypointItems in the scene.
        If there are duplicates by name, the last one found wins."""
        out: dict[str, tuple[Keypoint, int]] = {}
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                out[it.kp.name] = (it.kp, getattr(it, "visibility", 2))
        return out

    def _sync_current_kp_idx(self):
        """Advance index to the first *missing* required name, counting from the start of kp_names."""
        name_to_entry = self._collect_keypoints_by_name()
        count = 0
        for name in self.kp_names:
            if name in name_to_entry:
                count += 1
            else:
                break
        self.current_kp_idx = min(count, len(self.kp_names))

    def save_labels(self):
        if not self.images:
            return

        if len(self.bboxes) != 1:
            QMessageBox.warning(self, "Save Error", "Exactly one bounding box is required per image.")
            return
        if self.current_kp_idx != len(self.kp_names):
            QMessageBox.warning(self, "Save Error", "All keypoints must be placed before saving.")
            return

        base = os.path.splitext(self.images[self.current_idx])[0]

        project_root = os.path.dirname(self.label_dir)
        images_to_label_dir = self.image_dir
        images_all_dir = os.path.join(project_root, "images_all")
        labels_all_dir = os.path.join(project_root, "labels_all")
        annotations_dir = os.path.join(project_root, "annotations")
        os.makedirs(images_all_dir, exist_ok=True)
        os.makedirs(labels_all_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        label_out_path = os.path.join(labels_all_dir, f"{base}.txt")
        annotated_out_path = os.path.join(annotations_dir, f"{base}_annotated.png")
        image_out_path = os.path.join(images_all_dir, self.images[self.current_idx])

        bbox_item = next((it for it in self.scene.items() if isinstance(it, BoxItem)), None)
        if bbox_item is None:
            QMessageBox.warning(self, "Save Error", "No bounding box found.")
            return
        bbox_item.update_model()
        cid, xc, yc, w, h = bbox_item.bbox.to_yolo(self.img_w, self.img_h)
        bbox_cid = cid

        # Collect by keypoint name from the scene
        name_to_entry: dict[str, tuple[Keypoint, int]] = {}
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                # (Keypoint, visibility). Make sure visibility is an int.
                it.kp.class_id = bbox_cid
                name_to_entry[it.kp.name] = (it.kp, int(getattr(it, "visibility", 2)))

        # Verify every required name exists; show which are missing
        missing = [n for n in self.kp_names if n not in name_to_entry]
        if missing:
            # Fill any missing with (0,0, v=0)
            for n in missing:
                kp = Keypoint(0.0, 0.0, bbox_cid, n)
                name_to_entry[n] = (kp, 0)


        # Build YOLO line in the **fixed order** defined by self.kp_names
        line = f"{bbox_cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for name in self.kp_names:
            kp, vis = name_to_entry[name]
            if int(vis) == 0:
                xn = 0.0
                yn = 0.0
            else:
                xn = kp.x / self.img_w
                yn = kp.y / self.img_h
            line += f" {xn:.6f} {yn:.6f} {int(vis)}"

        with open(label_out_path, 'w', encoding='utf-8') as f:
            f.write(line + "\n")
        print(f"✅ Saved label to {label_out_path}")

        rect = QRectF(0, 0, self.img_w, self.img_h)
        pm = QPixmap(int(rect.width()), int(rect.height()))
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        self.scene.render(painter, target=QRectF(pm.rect()), source=rect)
        painter.end()
        pm.save(annotated_out_path)
        print(f"✅ Saved annotated image to {annotated_out_path}")

        src = os.path.join(images_to_label_dir, self.images[self.current_idx])
        if os.path.exists(src):
            shutil.copy2(src, image_out_path)
            print(f"✅ Copied original image to {image_out_path}")
        else:
            print(f"⚠️ Warning: Original image {src} not found!")

    # ---------- Video ----------
    def export_dataset(self):
        """Split images_all/labels_all into train/val sets and regenerate dataset.yaml."""
        base = os.path.dirname(__file__)
        images_all_dir = os.path.join(base, "images_all")
        labels_all_dir = os.path.join(base, "labels_all")

        if not os.path.isdir(images_all_dir):
            QMessageBox.information(self, "No images_all directory",
                                    f"Expected {images_all_dir} to exist.")
            return
        if not os.path.isdir(labels_all_dir):
            QMessageBox.information(self, "No labels_all directory",
                                    f"Expected {labels_all_dir} to exist.")
            return

        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        images = [f for f in os.listdir(images_all_dir) if f.lower().endswith(exts)]
        if not images:
            QMessageBox.information(self, "Nothing to export",
                                    "images_all does not contain any images.")
            return

        ratio, ok = QInputDialog.getDouble(
            self,
            "Train/Val Split",
            "Train split ratio (0.1 – 0.95):",
            0.8,
            0.1,
            0.95,
            2
        )
        if not ok:
            return

        dataset_choice, ok_choice = QInputDialog.getItem(
            self,
            "Dataset Type",
            "Choose dataset format:",
            ["Pose (keypoints)", "Detection (bbox only)"],
            0,
            False
        )
        if not ok_choice:
            return
        pose_mode = dataset_choice.startswith("Pose")


        base_datasets_dir = os.path.join(base, "datasets", "pose" if pose_mode else "detect")
        os.makedirs(base_datasets_dir, exist_ok=True)

        images_train_dir = os.path.join(base_datasets_dir, "images", "train")
        images_val_dir = os.path.join(base_datasets_dir, "images", "val")
        labels_train_dir = os.path.join(base_datasets_dir, "labels", "train")
        labels_val_dir = os.path.join(base_datasets_dir, "labels", "val")

        dataset_dirs = [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]
        existing = any(os.path.isdir(d) and os.listdir(d) for d in dataset_dirs)
        if existing:
            confirm = QMessageBox.question(
                self,
                "Overwrite dataset?",
                "Existing train/val folders contain files. Overwrite them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            for d in dataset_dirs:
                if os.path.isdir(d):
                    try:
                        shutil.rmtree(d, ignore_errors=True)
                    except Exception:
                        pass

        random.shuffle(images)
        train_count = int(len(images) * ratio)
        if train_count <= 0 and len(images) > 0:
            train_count = 1
        if train_count >= len(images) and len(images) > 1:
            train_count = len(images) - 1
        train_images = images[:train_count]
        val_images = images[train_count:]

        targets = [
            (train_images, images_train_dir, labels_train_dir),
            (val_images, images_val_dir, labels_val_dir),
        ]
        for _, img_dir, lbl_dir in targets:
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

        total = len(train_images) + len(val_images)
        prog = QProgressDialog("Copying dataset…", "Cancel", 0, total, self)
        prog.setWindowTitle("Export Dataset")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        missing_labels: list[str] = []
        copied = 0
        canceled = False

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for group, img_dir, lbl_dir in targets:
                for img_file in group:
                    if prog.wasCanceled():
                        canceled = True
                        break
                    src_img = os.path.join(images_all_dir, img_file)
                    dst_img = os.path.join(img_dir, img_file)
                    try:
                        shutil.copy2(src_img, dst_img)
                    except Exception as e:
                        missing_labels.append(f"{img_file}: copy image failed ({e})")
                        continue

                    base_name = os.path.splitext(img_file)[0]
                    label_src = os.path.join(labels_all_dir, f"{base_name}.txt")
                    label_dst = os.path.join(lbl_dir, f"{base_name}.txt")
                    if os.path.exists(label_src):
                        if pose_mode:
                            try:
                                shutil.copy2(label_src, label_dst)
                            except Exception as e:
                                missing_labels.append(f"{base_name}.txt: copy failed ({e})")
                        else:
                            try:
                                det_lines: list[str] = []
                                with open(label_src, "r", encoding="utf-8") as lf:
                                    for raw in lf:
                                        parts = raw.strip().split()
                                        if not parts:
                                            continue
                                        if len(parts) < 5:
                                            missing_labels.append(f"{base_name}.txt: insufficient columns for detection")
                                            continue
                                        det_lines.append(" ".join(parts[:5]))
                                if det_lines:
                                    with open(label_dst, "w", encoding="utf-8") as out:
                                        out.write("\n".join(det_lines) + "\n")
                                else:
                                    missing_labels.append(f"{base_name}.txt: no usable bbox rows")
                            except Exception as e:
                                missing_labels.append(f"{base_name}.txt: convert failed ({e})")
                    else:
                        missing_labels.append(f"{base_name}.txt: missing")

                    copied += 1
                    prog.setValue(copied)
                    prog.setLabelText(f"Copying {img_file}")
                    QApplication.processEvents()
                if canceled:
                    break
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()

        if canceled:
            QMessageBox.information(self, "Export canceled",
                                    "Dataset export was canceled. Partially copied files may remain.")
            return

        dataset_yaml_path = os.path.join(base_datasets_dir, "dataset.yaml")
        if pose_mode:
            try:
                from dataset_builder import create_dataset_yaml
            except Exception as e:
                QMessageBox.warning(self, "dataset_builder import error",
                                    f"Could not import dataset_builder.create_dataset_yaml:\n{e}")
                return
            try:
                create_dataset_yaml(base_datasets_dir, self.classes, self.kp_names)
            except Exception as e:
                QMessageBox.warning(self, "dataset.yaml error",
                                    f"Failed to create dataset.yaml:\n{e}")
                return
        else:
            detect_yaml = {
                "path": base_datasets_dir,
                "train": "images/train",
                "val": "images/val",
                "nc": len(self.classes),
                "names": self.classes,
            }
            try:
                with open(dataset_yaml_path, "w", encoding="utf-8") as yf:
                    yaml.safe_dump(detect_yaml, yf, sort_keys=False)
            except Exception as e:
                QMessageBox.warning(self, "dataset.yaml error",
                                    f"Failed to write detection dataset.yaml:\n{e}")
                return

        summary = (f"Train images: {len(train_images)}\n"
                   f"Val images: {len(val_images)}\n"
                   f"Format: {'Pose (keypoints)' if pose_mode else 'Detection (bbox)'}\n"
                   f"dataset.yaml written to: {dataset_yaml_path}")
        if missing_labels:
            summary += ("\n\nWarnings:\n" +
                        "\n".join(missing_labels[:10]))
            if len(missing_labels) > 10:
                summary += f"\n…{len(missing_labels) - 10} more"

        QMessageBox.information(self, "Dataset exported", summary)
        self.update_status_bar("Dataset export complete.")

    def normalize_labels_all(self):
        base = os.path.dirname(__file__)
        labels_dir = self.label_dir
        images_all_dir = os.path.join(base, "images_all")
        images_to_label_dir = self.image_dir

        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
        if not label_files:
            QMessageBox.information(self, "No labels", "labels_all does not contain any .txt files.")
            return

        kp_count = len(self.kp_names)
        expected_kp_values = kp_count * 3
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

        prog = QProgressDialog("Validating labels…", "Cancel", 0, len(label_files), self)
        prog.setWindowTitle("Validate Labels")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        normalized = 0
        untouched = 0
        copied_images = 0
        warnings: list[str] = []

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        canceled = False
        try:
            for idx, fname in enumerate(sorted(label_files), start=1):
                if prog.wasCanceled():
                    canceled = True
                    break

                stem = os.path.splitext(fname)[0]
                label_path = os.path.join(labels_dir, fname)
                try:
                    with open(label_path, "r", encoding="utf-8") as lf:
                        lines = [ln.strip() for ln in lf if ln.strip()]
                except Exception as e:
                    warnings.append(f"{fname}: read error ({e})")
                    prog.setValue(idx)
                    QApplication.processEvents()
                    continue

                if not lines:
                    warnings.append(f"{fname}: empty file")
                    prog.setValue(idx)
                    QApplication.processEvents()
                    continue

                first = lines[0].split()
                if len(first) < 5:
                    warnings.append(f"{fname}: expected at least 5 values, found {len(first)}")
                    prog.setValue(idx)
                    QApplication.processEvents()
                    continue

                try:
                    cid = int(round(float(first[0])))
                    bbox_vals = [float(first[i]) for i in range(1, 5)]
                except Exception as e:
                    warnings.append(f"{fname}: parse error ({e})")
                    prog.setValue(idx)
                    QApplication.processEvents()
                    continue

                kp_vals_raw = first[5:]
                normalized_kp: list[tuple[float, float, int]] = []

                for i in range(expected_kp_values // 3):
                    base_idx = i * 3
                    if base_idx + 2 < len(kp_vals_raw):
                        try:
                            xn = float(kp_vals_raw[base_idx])
                            yn = float(kp_vals_raw[base_idx + 1])
                            vis = int(round(float(kp_vals_raw[base_idx + 2])))
                        except Exception:
                            xn = 0.0
                            yn = 0.0
                            vis = 0
                    else:
                        xn = 0.0
                        yn = 0.0
                        vis = 0
                    vis = max(0, min(2, vis))
                    normalized_kp.append((xn, yn, vis))

                line = f"{cid} {bbox_vals[0]:.6f} {bbox_vals[1]:.6f} {bbox_vals[2]:.6f} {bbox_vals[3]:.6f}"
                for xn, yn, vis in normalized_kp:
                    line += f" {xn:.6f} {yn:.6f} {vis}"

                original_raw = lines[0]
                already_single_line = len(lines) == 1
                line_changed = (not already_single_line) or (line != original_raw)

                if line_changed:
                    try:
                        with open(label_path, "w", encoding="utf-8") as lf:
                            lf.write(line + "\n")
                    except Exception as e:
                        warnings.append(f"{fname}: write error ({e})")
                        prog.setValue(idx)
                        QApplication.processEvents()
                        continue
                    normalized += 1
                else:
                    untouched += 1

                image_found = False
                for ext in exts:
                    candidate = os.path.join(images_all_dir, stem + ext)
                    if os.path.exists(candidate):
                        image_found = True
                        break

                if not image_found:
                    for ext in exts:
                        src = os.path.join(images_to_label_dir, stem + ext)
                        if os.path.exists(src):
                            dst = os.path.join(images_all_dir, os.path.basename(src))
                            try:
                                os.makedirs(images_all_dir, exist_ok=True)
                                shutil.copy2(src, dst)
                                copied_images += 1
                                image_found = True
                            except Exception as e:
                                warnings.append(f"{stem}{ext}: copy failed ({e})")
                            break

                if not image_found:
                    warnings.append(f"{fname}: no matching image found in images_all or images_to_label")

                prog.setValue(idx)
                prog.setLabelText(f"Normalizing {fname}")
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()

        if canceled:
            QMessageBox.information(self, "Normalization canceled",
                                    "Operation canceled. Some files may have been processed already.")
            return

        if normalized == 0 and copied_images == 0 and not warnings:
            summary = "All label files already normalized. No changes made."
        else:
            parts: list[str] = []
            if normalized:
                parts.append(f"Normalized {normalized} label file(s).")
            if untouched and normalized:
                parts.append(f"{untouched} file(s) were already normalized.")
            elif untouched and not normalized:
                parts.append(f"{untouched} file(s) already normalized.")
            parts.append(f"Copied {copied_images} missing image(s) into images_all.")
            summary = "\n".join(parts)
        if warnings:
            summary += "\n\nWarnings:\n" + "\n".join(warnings[:10])
            if len(warnings) > 10:
                summary += f"\n…{len(warnings) - 10} more"

        QMessageBox.information(self, "Normalization complete", summary)
        self.update_status_bar("Label normalization complete.")

    def open_train_dialog(self):
        dlg = TrainDialog(self, default_dataset=os.path.join(os.path.dirname(__file__), "datasets"))
        dlg.exec()

    def open_video_reviewer(self):
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV:\n\n  pip install opencv-python")
            return
        # It’s okay if no model is loaded yet; dialog will warn before predicting
        dlg = VideoReviewDialog(self, self.predict_model, self._device, self.kp_names, self.classes)
        dlg.exec()

class VideoReviewDialog(QDialog):
    """
    Modal tool that:
      1) Loads a video
      2) Runs YOLO predict synchronously over a chosen frame range (with a modal QProgressDialog)
      3) Lets you scrub a timeline and see bbox+keypoint overlays and confidence
    """
    def __init__(self, parent, model, device: str, kp_names: list[str], classes: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Video Review (beta)")
        self.resize(980, 700)

        self.model = model            # may be None until user loads
        self.device = device
        self.kp_names = kp_names
        self.classes = classes
        self.model_path = getattr(parent, 'predict_model_path', None)

        # runtime state
        self.cap = None
        self.path: Optional[str] = None
        self.base: str = ""
        self.total: int = 0
        self.fps: float = 0.0
        self.cur: int = 0
        self.preds: dict[int, dict] = {}
        self._last_frame_bgr = None  # holds the current raw frame for export

        # build all widgets/layouts
        self._build_ui()

    def _build_ui(self):
        # --- UI ---
        top = QVBoxLayout(self)

        # Bar 1: file + params
        row = QHBoxLayout()
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self._choose_video)
        row.addWidget(self.btn_load)

        row.addSpacing(8)
        self.info = QLabel("No video loaded")
        row.addWidget(self.info)
        row.addStretch()

        row.addWidget(QLabel("Start"))
        self.spin_start = QSpinBox(); self.spin_start.setRange(0, 0); self.spin_start.setValue(0); row.addWidget(self.spin_start)

        row.addWidget(QLabel("End"))
        self.spin_end = QSpinBox(); self.spin_end.setRange(0, 0); self.spin_end.setValue(0); row.addWidget(self.spin_end)

        row.addWidget(QLabel("Stride"))
        self.spin_stride = QSpinBox(); self.spin_stride.setRange(1, 1000); self.spin_stride.setValue(5); row.addWidget(self.spin_stride)

        row.addWidget(QLabel("Conf≥"))
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.0, 1.0); self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.25); row.addWidget(self.spin_conf)

        row.addWidget(QLabel("IoU"))
        self.spin_iou = QDoubleSpinBox(); self.spin_iou.setRange(0.0, 1.0); self.spin_iou.setSingleStep(0.05); self.spin_iou.setValue(0.50); row.addWidget(self.spin_iou)

        # keypoint visibility threshold (map kp conf → visible/occluded)
        row.addWidget(QLabel("kp≥"))
        self.spin_kpvis = QDoubleSpinBox()
        self.spin_kpvis.setRange(0.0, 1.0)
        self.spin_kpvis.setSingleStep(0.05)
        self.spin_kpvis.setValue(0.50)  # >= → visible (red); < → occluded (yellow)
        row.addWidget(self.spin_kpvis)

        # Batch size spinner for prediction
        row.addWidget(QLabel("Batch"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(-1, 256)
        self.spin_batch.setSpecialValueText("Auto")
        self.spin_batch.setValue(8)  # default batch size
        row.addWidget(self.spin_batch)

        self.btn_predict = QPushButton("Predict Range (sync)")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self._predict_sync)
        row.addWidget(self.btn_predict)

        top.addLayout(row)

        # Graphics view (pan/zoom enabled)
        self.scene = QGraphicsScene()
        self.view = VideoView(self.scene)
        top.addWidget(self.view, 1)

        # Timeline
        bar2 = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        bar2.addWidget(self.slider)
        self.lbl_idx = QLabel("0/0")
        bar2.addWidget(self.lbl_idx)
        top.addLayout(bar2)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        # NEW: bottom-right export button
        self.btn_send = QPushButton("Send Frame → Images")
        self.btn_send.setToolTip("Save current frame to the labeler's images_to_label folder")
        self.btn_send.setEnabled(False)
        self.btn_send.clicked.connect(self._export_current_frame_to_images)
        buttons.addButton(self.btn_send, QDialogButtonBox.ButtonRole.ActionRole)

        # NEW: export N lowest-confidence frames
        self.btn_send_low = QPushButton("Send Lowest…")
        self.btn_send_low.setToolTip("Export N lowest-confidence predicted frames to the labeler")
        self.btn_send_low.setEnabled(False)
        self.btn_send_low.clicked.connect(self._export_low_confidence_frames)
        buttons.addButton(self.btn_send_low, QDialogButtonBox.ButtonRole.ActionRole)

        # NEW: export N highest-confidence frames
        self.btn_send_high = QPushButton("Send Highest…")
        self.btn_send_high.setToolTip("Export N highest-confidence predicted frames to the labeler")
        self.btn_send_high.setEnabled(False)
        self.btn_send_high.clicked.connect(self._export_high_confidence_frames)
        buttons.addButton(self.btn_send_high, QDialogButtonBox.ButtonRole.ActionRole)

        # NEW: export N random frames (no predictions required)
        self.btn_send_random = QPushButton("Send Random…")
        self.btn_send_random.setToolTip("Export N random frames to the labeler for fresh labeling")
        self.btn_send_random.setEnabled(False)
        self.btn_send_random.clicked.connect(self._export_random_frames)
        buttons.addButton(self.btn_send_random, QDialogButtonBox.ButtonRole.ActionRole)

        # Shortcut: Shift+E exports N lowest (asks for N)
        self._exportN_shortcut = QShortcut(QKeySequence("Shift+E"), self)
        self._exportN_shortcut.activated.connect(self._export_low_confidence_frames)

        # Shortcut: Shift+H exports N highest (asks for N)
        self._exportNhigh_shortcut = QShortcut(QKeySequence("Shift+H"), self)
        self._exportNhigh_shortcut.activated.connect(self._export_high_confidence_frames)

        # Shortcut: Shift+R exports N random frames
        self._exportNrandom_shortcut = QShortcut(QKeySequence("Shift+R"), self)
        self._exportNrandom_shortcut.activated.connect(self._export_random_frames)

        top.addWidget(buttons)

        # overlay items
        self._overlay_items: list[QGraphicsItem] = []

        # Arrow-key timeline stepping (Left/Right = ±1 frame)
        self._left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._left_shortcut.setAutoRepeat(True)
        self._left_shortcut.activated.connect(lambda: self._step(-1))

        self._right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._right_shortcut.setAutoRepeat(True)
        self._right_shortcut.activated.connect(lambda: self._step(+1))

        self._export_shortcut = QShortcut(QKeySequence("E"), self)
        self._export_shortcut.activated.connect(self._export_current_frame_to_images)

        # Zoom shortcuts for the view
        self._zoom_in_sc = QShortcut(QKeySequence("+"), self)
        self._zoom_in_sc.activated.connect(lambda: self.view.scale(1.05, 1.05))
        self._zoom_out_sc = QShortcut(QKeySequence("-"), self)
        self._zoom_out_sc.activated.connect(lambda: self.view.scale(1/1.05, 1/1.05))
        self._zoom_reset_sc = QShortcut(QKeySequence("R"), self)
        self._zoom_reset_sc.activated.connect(self.view.reset_view)
    def _export_high_confidence_frames(self):
        """Export the top-N highest-confidence predicted frames to the labeler's images_to_label folder.
        Skips any frames that are already present on disk (dedupe across restarts)."""
        # Preconditions
        if not self.preds:
            QMessageBox.information(self, "No predictions", "Run prediction first, then try again.")
            return
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        # Ask for N
        N, ok = QInputDialog.getInt(self, "Send Highest Confidence", "How many frames?", 25, 1, 100000, 1)
        if not ok:
            return

        # Destination
        parent = self.parent()
        dest_dir = getattr(parent, "image_dir", None)
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label folder.")
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not create destination folder:\n{e}")
            return

        # Sort by confidence DESC
        items = sorted(
            [(idx, float(p.get("conf", 0.0))) for idx, p in self.preds.items() if p.get("ok")],
            key=lambda t: t[1],
            reverse=True
        )

        exported = 0
        skipped = 0
        errors = 0

        for frame_idx, _ in items:
            if exported >= N:
                break

            out_name = f"{self.base}_f{frame_idx:06d}.png"
            out_path = os.path.join(dest_dir, out_name)

            # Dedupe across restarts by checking file existence
            if os.path.exists(out_path):
                skipped += 1
                continue

            # Seek and write this frame
            try:
                self.cap.set(_cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    errors += 1
                    continue
                # Write BGR frame as PNG
                if not _cv2.imwrite(out_path, frame):
                    errors += 1
                    continue
                exported += 1
            except Exception:
                errors += 1
                continue

        # Let the labeler refresh its list if available
        try:
            if hasattr(parent, "refresh_image_list"):
                parent.refresh_image_list()
        except Exception:
            pass

        QMessageBox.information(
            self,
            "Export Complete",
            f"Exported {exported} frame(s) to images_to_label.\n"
            f"Skipped {skipped} (already existed).\n"
            f"Errors: {errors}."
        )
    # ---------- caching ----------
    def _cache_path(self) -> Optional[str]:
        if not self.path:
            return None
        return os.path.abspath(self.path) + ".sqp_preds.json"

    def _video_signature(self) -> dict:
        try:
            return {
                "path": os.path.abspath(self.path) if self.path else "",
                "size": int(os.path.getsize(self.path)) if self.path else 0,
                "mtime": float(os.path.getmtime(self.path)) if self.path else 0.0,
                "total": int(self.total),
                "fps": float(self.fps),
            }
        except Exception:
            return {
                "path": os.path.abspath(self.path) if self.path else "",
                "size": 0,
                "mtime": 0.0,
                "total": int(self.total),
                "fps": float(self.fps),
            }

    def _load_cache_if_valid(self) -> bool:
        fp = self._cache_path()
        if not fp or not os.path.exists(fp):
            return False
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            vid = meta.get("video", {})
            cur = self._video_signature()

            # Same file check (path/size) and mtime within a couple seconds
            if (vid.get("path") != cur.get("path")) or (int(vid.get("size", -1)) != int(cur.get("size", -2))):
                return False
            if abs(float(vid.get("mtime", 0.0)) - float(cur.get("mtime", 0.0))) > 2.0:
                return False

            # Optional: require the same model if both are known
            mp_saved = meta.get("model_path")
            if mp_saved and self.model_path and (mp_saved != self.model_path):
                return False

            preds = data.get("preds", {})
            self.preds = {int(k): v for k, v in preds.items()}
            return bool(self.preds)
        except Exception:
            return False

    def _save_cache(self, meta: dict):
        fp = self._cache_path()
        if not fp:
            return
        data = {
            "meta": meta,
            "preds": {str(k): v for k, v in self.preds.items()},
        }
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    # ---------- video load ----------
    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
        if not path:
            return
        self._open_video(path)

    def _open_video(self, path: str):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None

        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
            return

        cap = _cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            QMessageBox.warning(self, "Video Error", "Failed to open video.")
            return

        self.cap = cap
        self.path = path
        self.base = os.path.splitext(os.path.basename(path))[0]
        self.total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)

        w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.info.setText(f"{self.base} — {w}x{h} @ {self.fps:.2f} fps — {self.total} frames")
        self.spin_start.setRange(0, max(0, self.total - 1)); self.spin_start.setValue(0)
        self.spin_end.setRange(0, max(0, self.total - 1)); self.spin_end.setValue(max(0, self.total - 1))
        self.slider.setRange(0, max(0, self.total - 1))
        self.btn_predict.setEnabled(True)
        if hasattr(self, "btn_send"):
            self.btn_send.setEnabled(True)
        if hasattr(self, "btn_send_low"):
            self.btn_send_low.setEnabled(False)
        if hasattr(self, "btn_send_high"):
            self.btn_send_high.setEnabled(False)
        if hasattr(self, "btn_send_random"):
            self.btn_send_random.setEnabled(True)
        # Try to load cached predictions; if present, enable timeline immediately
        if self._load_cache_if_valid():
            self.slider.setEnabled(True)
            if hasattr(self, "btn_send_low"):
                self.btn_send_low.setEnabled(bool(self.preds))
            if hasattr(self, "btn_send_high"):
                self.btn_send_high.setEnabled(bool(self.preds))
            cached_keys = sorted(self.preds.keys())
            self._seek(cached_keys[0] if cached_keys else 0, show_only=False)
        else:
            self.slider.setEnabled(False)   # enable after predictions
            self._seek(0, show_only=True)

        # Reset pan/zoom whenever a new video is opened
        if hasattr(self, "view") and hasattr(self.view, "reset_view"):
            self.view.reset_view()

    # ---------- prediction (sync, modal progress) ----------
    def _predict_sync(self):
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        if self.model is None:
            QMessageBox.information(self, "No model", "Click 'Load Model' in the main window first.")
            return

        start = int(self.spin_start.value())
        end = int(self.spin_end.value())
        stride = max(1, int(self.spin_stride.value()))
        conf = float(self.spin_conf.value())
        iou = float(self.spin_iou.value())
        bs = int(self.spin_batch.value()) if hasattr(self, "spin_batch") else 1
        imgsz = 640
        kpvis = float(self.spin_kpvis.value()) if hasattr(self, "spin_kpvis") else 0.5
        batch_kwargs = {} if bs <= 0 else {"batch": bs}

        if end < start:
            QMessageBox.warning(self, "Range Error", "End must be ≥ Start.")
            return

        # modal progress dialog
        steps = max(1, ((end - start) // stride) + 1)
        prog = QProgressDialog("Running prediction…", "Cancel", 0, steps, self)
        prog.setWindowTitle("Predicting")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        # Set initial position once
        self.cap.set(_cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        done = 0
        self.preds.clear()

        frames: list = []
        frame_indices: list[int] = []

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            while idx <= end:
                if prog.wasCanceled():
                    break
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break

                frames.append(frame)
                frame_indices.append(idx)

                # If we filled a batch or reached the end, run prediction on the batch
                if len(frames) >= bs or (idx + stride) > end:
                    try:
                        results_list = self.model.predict(
                            source=frames,
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            device=self.device,
                            verbose=False,
                            **batch_kwargs,
                        )
                        results_list = list(results_list)
                        for fi, res in zip(frame_indices, results_list):
                            pred = self._extract_top(res)
                            self.preds[fi] = pred
                    except Exception as e:
                        # if batch fails, mark each frame in the batch as failed
                        for fi in frame_indices:
                            self.preds[fi] = {"ok": False, "error": str(e)}

                    done += len(frames)
                    prog.setValue(min(done, steps))
                    prog.setLabelText(f"Predicting frames {frame_indices[0]}–{frame_indices[-1]}")
                    QApplication.processEvents()

                    # reset batch accumulators
                    frames = []
                    frame_indices = []

                idx += stride
                if idx <= end:
                    self.cap.set(_cv2.CAP_PROP_POS_FRAMES, idx)
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()

        # persist cache and enable review
        if self.preds:
            try:
                meta = {
                    "video": self._video_signature(),
                    "model_path": self.model_path,
                    "imgsz": imgsz,
                    "conf": conf,
                    "iou": iou,
                    "kpvis": kpvis,
                    "start": start,
                    "end": end,
                    "stride": stride,
                    "total": self.total,
                    "fps": self.fps,
                    "classes": self.classes,
                    "kp_names": self.kp_names,
                }
                self._save_cache(meta)
            except Exception:
                pass
            self.slider.setEnabled(True)
            if hasattr(self, "btn_send_low"):
                self.btn_send_low.setEnabled(True)
            if hasattr(self, "btn_send_high"):
                self.btn_send_high.setEnabled(True)
            self._seek(start, show_only=False)

    @staticmethod
    def _extract_top(results) -> dict:
        """Pick highest-conf detection; keep kp 'v' as confidence float (0..1)."""
        out = {"ok": False, "conf": 0.0, "cls": 0, "xyxy": None, "kps": []}
        try:
            if results.boxes is None or len(results.boxes) == 0:
                return out
            import numpy as _np
            confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else _np.zeros((len(results.boxes),), dtype=_np.float32)
            i = int(confs.argmax())
            out["conf"] = float(confs[i])
            out["cls"] = int(results.boxes.cls.cpu().numpy()[i]) if results.boxes.cls is not None else 0
            out["xyxy"] = [float(v) for v in results.boxes.xyxy.cpu().numpy()[i].tolist()]
            # keep keypoint confidence as float
            if hasattr(results, "keypoints") and results.keypoints is not None:
                kps = results.keypoints.data.cpu().numpy()  # (N, K, 3)
                if kps.shape[0] > i:
                    out["kps"] = [[float(x), float(y), float(v)] for (x, y, v) in kps[i]]
            out["ok"] = True
        except Exception:
            pass
        return out

    # ---------- timeline / overlay ----------
    def _step(self, delta: int):
        """Jump the timeline by `delta` frames (negative for left, positive for right)."""
        if self.cap is None or self.total <= 0:
            return
        new_idx = max(0, min(self.total - 1, self.cur + int(delta)))
        if new_idx == self.cur:
            return
        # Update the slider (for UI) and seek the frame (even if slider is disabled)
        try:
            self.slider.setValue(new_idx)
        except Exception:
            pass
        self._seek(new_idx, show_only=False)

    def _on_slider(self, idx: int):
        self._seek(int(idx), show_only=False)

    def _seek(self, frame_idx: int, show_only: bool):
        if self.cap is None:
            return
        self.cur = frame_idx
        self.lbl_idx.setText(f"{self.cur+1}/{self.total}")

        self.cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        # remember the raw BGR frame so we can export it to the labeler
        self._last_frame_bgr = frame.copy()
        pix = self._cv_to_qpix(frame)
        self.scene.clear()
        self.scene.setSceneRect(0, 0, pix.width(), pix.height())
        self.scene.addPixmap(pix)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        if not show_only:
            self._draw_overlay_for(frame_idx)

    def _export_current_frame_to_images(self):
        """Export the *currently displayed* frame into the labeler's images_to_label folder.
        Skips if this frame index has already been exported (dedupe across restarts)."""
        parent = self.parent()
        dest_dir = getattr(parent, "image_dir", None)
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label directory.")
            return
        if self._last_frame_bgr is None:
            QMessageBox.information(self, "No frame", "Load a video and seek to a frame first.")
            return

        # Filesystem-based dedupe: do not export if this frame has already been exported
        existing = self._existing_export_indices()
        if self.cur in existing:
            QMessageBox.information(
                self,
                "Already exported",
                f"Frame {self.cur} is already in images_to_label.\nSkipping duplicate export."
            )
            return

        try:
            os.makedirs(dest_dir, exist_ok=True)
            base_name = f"{self.base}_f{self.cur:06d}.png"
            out_path = os.path.join(dest_dir, base_name)

            if _cv2 is None:
                QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
                return

            ok = _cv2.imwrite(out_path, self._last_frame_bgr)
            if not ok:
                QMessageBox.warning(self, "Export Error", "cv2.imwrite failed to save the image.")
                return

            # Refresh the labeler file list if available
            if hasattr(parent, "refresh_image_list"):
                try:
                    parent.refresh_image_list()
                except Exception:
                    pass

            QMessageBox.information(self, "Exported", f"Saved: {out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export frame:\n{e}")
            
    def _export_random_frames(self):
        """Export N random frames from the loaded video for fresh labeling."""
        if self.cap is None or self.total <= 0:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        parent = self.parent()
        dest_dir = getattr(parent, "image_dir", None)
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label directory.")
            return
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not create destination folder:\n{e}")
            return

        available = list(range(self.total))
        existing = self._existing_export_indices()
        if existing:
            available = [idx for idx in available if idx not in existing]
        if not available:
            QMessageBox.information(self, "Nothing to export", "Every frame from this video is already in images_to_label.")
            return

        max_n = len(available)
        default_n = min(25, max_n)
        n, ok = QInputDialog.getInt(
            self,
            "Export Random Frames",
            "How many random frames should I send to the labeler?",
            default_n,
            1,
            max_n,
            1,
        )
        if not ok or n <= 0:
            return

        count = min(n, len(available))
        selected = random.sample(available, count)
        selected.sort()

        prog = QProgressDialog("Saving frames…", "Cancel", 0, len(selected), self)
        prog.setWindowTitle("Exporting")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        saved = 0
        failed: list[tuple[int, str]] = []
        cur_pos = int(self.cur)

        for i, fi in enumerate(selected, start=1):
            if prog.wasCanceled():
                break

            self.cap.set(_cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failed.append((fi, "read-failed"))
            else:
                base_name = f"{self.base}_f{fi:06d}.png"
                dest_path = os.path.join(dest_dir, base_name)
                suffix = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{self.base}_f{fi:06d}_{suffix}.png")
                    suffix += 1
                if _cv2.imwrite(dest_path, frame):
                    saved += 1
                else:
                    failed.append((fi, "write-failed"))

            prog.setValue(i)
            prog.setLabelText(f"Exporting frame {fi}")
            QApplication.processEvents()

        canceled = prog.wasCanceled()
        prog.close()

        try:
            self._seek(cur_pos, show_only=False)
        except Exception:
            pass

        if hasattr(parent, "refresh_image_list"):
            try:
                parent.refresh_image_list()
                parent.update_status_bar(f"Exported {saved} random frame(s) to images_to_label")
            except Exception:
                pass

        if saved > 0:
            msg = f"Saved {saved} random frame(s) to:\n{dest_dir}"
            if canceled and saved < len(selected):
                msg += "\n\nExport canceled before completing all requested frames."
            QMessageBox.information(self, "Export complete", msg)
        else:
            title = "Export canceled" if canceled else "No frames saved"
            detail = "Export was canceled." if canceled else "Nothing was written."
            if failed:
                detail += "\n\nIssues:\n" + "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
                if len(failed) > 10:
                    detail += f"\n…{len(failed) - 10} more"
            QMessageBox.information(self, title, detail)

        if failed:
            msg = "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
            more = "" if len(failed) <= 10 else f"\n…{len(failed) - 10} more"
            QMessageBox.warning(self, "Some exports failed", f"{saved} succeeded, {len(failed)} failed.\n\n{msg}{more}")
            
    def _existing_export_indices(self) -> set[int]:
        """Scan the labeler's images_to_label folder for frames already exported for this video."""
        out: set[int] = set()
        parent = self.parent()
        dest_dir = getattr(parent, "image_dir", None)
        if not dest_dir or not os.path.isdir(dest_dir):
            return out
        try:
            import re
            # Matches: {base}_f000123.png (optionally with suffixes, any common image ext)
            pat = re.compile(rf"^{re.escape(self.base)}_f(\d{{6}})(?:_.*)?\.(?:png|jpg|jpeg|bmp|webp)$", re.IGNORECASE)
            for fn in os.listdir(dest_dir):
                m = pat.match(fn)
                if m:
                    try:
                        out.add(int(m.group(1)))
                    except Exception:
                        pass
        except Exception:
            pass
        return out
    
    def _export_low_confidence_frames(self):
        self._export_predictions_by_confidence(order="low")

    def _export_high_confidence_frames(self):
        self._export_predictions_by_confidence(order="high")

    def _export_predictions_by_confidence(self, order: str):
        order_key = (order or "low").lower()
        if order_key not in {"low", "high"}:
            order_key = "low"

        if not self.preds:
            QMessageBox.information(self, "No predictions", "Run Predict Range first to generate predictions.")
            return
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        candidates = [(fi, float(p.get("conf", 0.0))) for fi, p in self.preds.items() if p.get("ok")]
        if not candidates:
            QMessageBox.information(self, "No predictions", "No successful predictions available to export.")
            return

        if order_key == "low":
            candidates.sort(key=lambda t: t[1])
            order_label = "lowest"
            dialog_title = "Export Lowest Confidence"
        else:
            candidates.sort(key=lambda t: (-t[1], t[0]))
            order_label = "highest"
            dialog_title = "Export Highest Confidence"
        conf_map = {fi: conf for fi, conf in candidates}

        already = self._existing_export_indices()
        pending = [fi for fi, _ in candidates if fi not in already]
        if not pending:
            QMessageBox.information(self, "Nothing to export", f"All {order_label}-confidence frames are already exported.")
            return

        max_n = len(pending)
        default_n = min(25, max_n)
        n, ok = QInputDialog.getInt(
            self,
            dialog_title,
            "How many frames should I send to the labeler?",
            default_n,
            1,
            max_n,
            1,
        )
        if not ok or n <= 0:
            return

        selected = pending[:min(n, len(pending))]

        parent = self.parent()
        dest_dir = getattr(parent, "image_dir", None)
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label directory.")
            return
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not create destination folder:\n{e}")
            return

        prog = QProgressDialog("Saving frames…", "Cancel", 0, len(selected), self)
        prog.setWindowTitle("Exporting")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        saved = 0
        saved_confs: list[float] = []
        failed: list[tuple[int, str]] = []
        cur_pos = int(self.cur)

        for i, fi in enumerate(selected, start=1):
            if prog.wasCanceled():
                break

            self.cap.set(_cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failed.append((fi, "read-failed"))
            else:
                base_name = f"{self.base}_f{fi:06d}.png"
                dest_path = os.path.join(dest_dir, base_name)
                suffix = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{self.base}_f{fi:06d}_{suffix}.png")
                    suffix += 1
                if _cv2.imwrite(dest_path, frame):
                    saved += 1
                    saved_confs.append(conf_map.get(fi, 0.0))
                else:
                    failed.append((fi, "write-failed"))

            prog.setValue(i)
            prog.setLabelText(f"Exporting frame {fi}")
            QApplication.processEvents()

        canceled = prog.wasCanceled()
        prog.close()

        try:
            self._seek(cur_pos, show_only=False)
        except Exception:
            pass

        if hasattr(parent, "refresh_image_list"):
            try:
                parent.refresh_image_list()
                parent.update_status_bar(f"Exported {saved} frame(s) to images_to_label")
            except Exception:
                pass

        if saved > 0:
            msg = f"Saved {saved} frame(s) to:\n{dest_dir}"
            if saved_confs:
                lo = min(saved_confs)
                hi = max(saved_confs)
                msg += f"\nConfidence range of exported set: {lo:.2f}–{hi:.2f}"
            if canceled and saved < len(selected):
                msg += "\n\nExport canceled before completing all requested frames."
            QMessageBox.information(self, "Export complete", msg)
        else:
            title = "Export canceled" if canceled else "No frames saved"
            detail = "Export was canceled." if canceled else "Nothing was written."
            if failed:
                detail += "\n\nIssues:\n" + "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
                if len(failed) > 10:
                    detail += f"\n…{len(failed) - 10} more"
            QMessageBox.information(self, title, detail)

        if failed:
            msg = "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
            more = "" if len(failed) <= 10 else f"\n…{len(failed) - 10} more"
            QMessageBox.warning(self, "Some exports failed", f"{saved} succeeded, {len(failed)} failed.\n\n{msg}{more}")

    def _draw_overlay_for(self, frame_idx: int):
        # clear old
        for it in getattr(self, "_overlay_items", []):
            try: self.scene.removeItem(it)
            except Exception: pass
        self._overlay_items = []

        p = self.preds.get(frame_idx)
        if not p or not p.get("ok"):
            return

        # ---- Bounding box (blue, thicker) ----
        if p.get("xyxy"):
            x1, y1, x2, y2 = p["xyxy"]
            r = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(Qt.GlobalColor.blue); pen.setWidth(3); pen.setCosmetic(True)
            r.setPen(pen); r.setZValue(5)
            self.scene.addItem(r); self._overlay_items.append(r)

            # class + confidence (bigger, blue)
            cls_id = p.get("cls", 0)
            cls_name = self.classes[cls_id] if 0 <= cls_id < len(self.classes) else str(cls_id)
            t = QGraphicsSimpleTextItem(f"{cls_name} {p.get('conf', 0.0):.2f}")
            t.setFont(_ui_font(24))
            t.setBrush(QBrush(Qt.GlobalColor.blue))
            t.setPos(x1 + 2, y1 + 2); t.setZValue(6)
            self.scene.addItem(t); self._overlay_items.append(t)

        # ---- Keypoints (map kp conf → visibility) ----
        thr = float(self.spin_kpvis.value()) if hasattr(self, "spin_kpvis") else 0.5
        for i, kp in enumerate(p.get("kps", [])):
            x, y, conf = kp
            vis = 2 if conf >= thr else 1  # 2=visible(red), 1=occluded(yellow)

            if vis == 2:
                color = Qt.GlobalColor.red; fill = QBrush(color); style = Qt.PenStyle.SolidLine
            elif vis == 1:
                color = Qt.GlobalColor.yellow; fill = QBrush(color); style = Qt.PenStyle.SolidLine
            else:
                color = Qt.GlobalColor.lightGray; fill = QBrush(Qt.GlobalColor.transparent); style = Qt.PenStyle.DashLine

            dot = QGraphicsEllipseItem(-4, -4, 8, 8)  # slightly larger dot
            dot.setPos(x, y)
            pen = QPen(color); pen.setCosmetic(True); pen.setWidth(2); pen.setStyle(style)
            dot.setPen(pen); dot.setBrush(fill); dot.setZValue(7)
            self.scene.addItem(dot); self._overlay_items.append(dot)

            # label next to kp
            name = self.kp_names[i] if i < len(self.kp_names) else f"kp{i}"
            lbl = QGraphicsSimpleTextItem(name)
            lbl.setFont(_ui_font(18))
            lbl.setBrush(QBrush(color))
            lbl.setPos(x + 8, y - 16); lbl.setZValue(8)
            lbl.setVisible(vis != 0)  # hide if invisible
            self.scene.addItem(lbl); self._overlay_items.append(lbl)

    @staticmethod
    def _cv_to_qpix(frame_bgr) -> QPixmap:
        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        from PyQt6.QtGui import QImage
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def reject(self):
        # cleanup
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().reject()

class TrainDialog(QDialog):
    """Dialog scaffold for future YOLO training integration."""

    MODEL_OPTIONS = {
        "YOLOv11n (nano)": "yolo11n.yaml",
        "YOLOv11s (small)": "yolo11s.yaml",
        "YOLOv11m (medium)": "yolo11m.yaml",
        "YOLOv11l (large)": "yolo11l.yaml",
        "YOLOv11x (xlarge)": "yolo11x.yaml",
    }

    def __init__(self, parent, default_dataset: str):
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.resize(520, 360)

        self.default_dataset = default_dataset
        self.device = _auto_device()
        self.training_running = False

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Dataset selector
        ds_row = QHBoxLayout()
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("Select dataset folder (contains images/ and labels/)")
        if os.path.isdir(default_dataset):
            self.dataset_edit.setText(default_dataset)
        ds_row.addWidget(self.dataset_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dataset)
        ds_row.addWidget(browse_btn)
        form.addRow("Dataset path:", ds_row)

        # Model choice
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.MODEL_OPTIONS.keys())
        form.addRow("Model size:", self.model_combo)

        # Device info
        self.device_label = QLabel(self.device.upper())
        form.addRow("Device:", self.device_label)

        # Task selection
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "Auto (from dataset)",
            "Detection",
            "Pose",
        ])
        form.addRow("Training task:", self.task_combo)

        # Hyperparameters
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(50)
        form.addRow("Epochs:", self.epoch_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(0, 512)
        self.batch_spin.setSpecialValueText("Auto")
        self.batch_spin.setValue(0)
        form.addRow("Batch size:", self.batch_spin)

        self.batch_hint = QLabel("")
        self.batch_hint.setStyleSheet("color: #bbbbbb; font-size: 9pt;")
        form.addRow("", self.batch_hint)

        layout.addLayout(form)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Training output will appear here once the integration is complete.")
        layout.addWidget(self.log_view, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)

        self.run_btn = QPushButton("Start Training")
        self.run_btn.clicked.connect(self._start_training)
        button_box.addButton(self.run_btn, QDialogButtonBox.ButtonRole.ActionRole)

        layout.addWidget(button_box)

        self._configure_batch_controls()

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select dataset directory",
            self.dataset_edit.text() or self.default_dataset,
        )
        if path:
            self.dataset_edit.setText(path)

    def _configure_batch_controls(self):
        if self.device == 'cuda':
            self.batch_spin.setValue(0)
            self.batch_spin.setEnabled(False)
            self.batch_hint.setText("CUDA detected → using automatic batch sizing.")
        elif self.device == 'mps':
            default = max(1, self.batch_spin.value() or 16)
            self.batch_spin.setValue(default)
            self.batch_spin.setEnabled(True)
            self.batch_hint.setText("MPS detected → choose a manual batch size that fits memory.")
        else:
            default = self.batch_spin.value() or 16
            self.batch_spin.setValue(default)
            self.batch_spin.setEnabled(True)
            self.batch_hint.setText("CPU detected → adjust batch size as needed (lower values use less memory).")

    def _log(self, message: str):
        self.log_view.append(message)
        self.log_view.ensureCursorVisible()
        QApplication.processEvents()

    def _resolve_model_config(self, base_cfg: str, task_value: Optional[str]) -> tuple[str, Optional[str]]:
        cfg = base_cfg
        notice = None
        if task_value == "pose" and "-pose" not in cfg.lower():
            if cfg.lower().endswith(".yaml"):
                cfg = cfg[:-5] + "-pose.yaml"
            else:
                cfg = cfg + "-pose.yaml"
            notice = "Pose task detected → switched to pose variant of the model config."
        elif task_value == "detect" and "-pose" in cfg.lower():
            cfg = cfg.replace("-pose", "")
            notice = "Detection task selected → using detection variant of the model config."
        return cfg, notice

    def _infer_task_from_yaml(self, yaml_path: str) -> Optional[str]:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return None
        if isinstance(data, dict):
            if "kpt_shape" in data or "kp_names" in data:
                return "pose"
        return "detect"

    def _start_training(self):
        if self.training_running:
            QMessageBox.information(self, "Training running", "A training session is already in progress.")
            return

        dataset_path = self.dataset_edit.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, "Dataset required", "Select a dataset folder before starting training.")
            return
        if not os.path.isdir(dataset_path):
            QMessageBox.warning(self, "Invalid dataset", f"Folder not found:\n{dataset_path}")
            return

        data_yaml = os.path.join(dataset_path, "dataset.yaml")
        if os.path.isfile(data_yaml):
            resolved = data_yaml
        elif dataset_path.lower().endswith((".yaml", ".yml")) and os.path.isfile(dataset_path):
            resolved = dataset_path
        else:
            QMessageBox.warning(
                self,
                "dataset.yaml missing",
                "Could not find dataset.yaml in the selected folder.\n"
                "Select the dataset root (contains dataset.yaml) or the YAML file directly."
            )
            return

        model_label = self.model_combo.currentText()
        base_model_cfg = self.MODEL_OPTIONS[model_label]
        epochs = self.epoch_spin.value()
        batch = self.batch_spin.value()
        batch_display = "auto" if batch <= 0 else str(batch)

        if self.device == 'mps' and batch <= 0:
            QMessageBox.warning(
                self,
                "Batch size required",
                "Automatic batch sizing is unavailable on Apple MPS.\n"
                "Set a positive batch size before starting training."
            )
            return

        task_selection = self.task_combo.currentText()
        if task_selection.startswith("Auto"):
            inferred_task = self._infer_task_from_yaml(resolved)
            task_value = inferred_task if inferred_task in {"pose", "detect"} else None
        elif task_selection.startswith("Detection"):
            task_value = "detect"
        else:
            task_value = "pose"

        model_cfg, cfg_notice = self._resolve_model_config(base_model_cfg, task_value)
        if cfg_notice:
            self._log(cfg_notice)

        try:
            from ultralytics import YOLO
        except Exception as e:
            QMessageBox.warning(self, "ultralytics missing",
                                f"Could not import ultralytics.YOLO:\n{e}\n\nInstall with:\n  pip install ultralytics")
            return

        self._log(f"Starting training for {model_label} ({model_cfg})")
        self._log(f"- dataset: {resolved}")
        self._log(f"- device: {self.device}")
        self._log(f"- epochs: {epochs}")
        self._log(f"- batch size: {batch_display}")
        if task_value:
            self._log(f"- task: {task_value}")
        self._log("Running ultralyticsYOLO.train() — progress will stream to the active terminal.")
        self._log("")

        batch_param = -1 if batch <= 0 else int(batch)

        params = {
            "data": resolved,
            "epochs": epochs,
            "device": self.device,
            "exist_ok": False,
            "batch": batch_param,
        }
        if task_value:
            params["task"] = task_value

        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.run_btn.setEnabled(False)
        self.training_running = True

        try:
            model = YOLO(model_cfg)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)
            self.training_running = False
            self._log(f"Failed to load model config '{model_cfg}': {e}")
            QMessageBox.critical(self, "Model load error",
                                 f"Could not create YOLO model from {model_cfg}.\n\nDetails:\n{e}")
            return

        try:
            results = model.train(**params)
            save_dir = getattr(results, "save_dir", None)
            if save_dir:
                self._log(f"Training complete. Artifacts saved to: {save_dir}")
            else:
                self._log("Training complete.")
            QMessageBox.information(self, "Training complete",
                                    "YOLO training finished. Review the terminal/logs for metrics.")
        except Exception as e:
            self._log(f"Training failed: {e}")
            QMessageBox.critical(self, "Training error", f"Training failed:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)
            self.training_running = False

# =========================
# Entrypoint
# =========================

if __name__ == '__main__':
    app = QApplication(sys.argv)

    base = os.path.dirname(__file__)

    app.setApplicationName("SqueakPose Studio")
    app.setApplicationDisplayName("SqueakPose Studio")

    icon_path = os.path.join(base, "squeakpose_studio_logo.png")
    app.setWindowIcon(QIcon(icon_path))

    splash_pix = QPixmap(os.path.join(base, "squeakpose_studio_logo.png"))
    splash = QSplashScreen(splash_pix, Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
    splash.show(); app.processEvents(); splash.raise_(); splash.activateWindow()
    screen = app.primaryScreen(); screen_geometry = screen.availableGeometry()
    x = (screen_geometry.width() - splash_pix.width()) // 2
    y = (screen_geometry.height() - splash_pix.height()) // 2
    splash.move(x, y)

    # folders
    for folder in [
        os.path.join(base, 'images_to_label'),
        os.path.join(base, 'images_all'),
        os.path.join(base, 'labels_all'),
        os.path.join(base, 'annotations'),
        os.path.join(base, 'fonts')
    ]:
        os.makedirs(folder, exist_ok=True)

    img_dir = os.path.join(base, 'images_to_label')
    lbl_dir = os.path.join(base, 'labels_all')
    cls_file = os.path.join(base, 'classes.txt')
    kp_file = os.path.join(base, 'keypoints.txt')
    font_path = os.path.join(base, 'fonts', 'FiraSans-Regular.ttf')

    missing_files = []
    if not os.path.isfile(cls_file):
        missing_files.append(("classes.txt", cls_file))
    if not os.path.isfile(kp_file):
        missing_files.append(("keypoints.txt", kp_file))
    if missing_files:
        splash.close()
        details = "\n".join(f"- {name}: {path}" for name, path in missing_files)
        QMessageBox.critical(
            None,
            "Project Setup Incomplete",
            "The label tool needs both classes.txt and keypoints.txt.\n\n"
            f"Missing file(s):\n{details}\n\nAdd them to the project and restart SqueakPose Studio."
        )
        sys.exit(1)

    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        print("✅ Fira Sans font loaded successfully." if font_id != -1 else "⚠️ Failed to load Fira Sans font.")

    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()

    dark_stylesheet = f"""
    QWidget {{
        background-color: #2b2b2b;
        color: #e0e0e0;
        font-family: 'Fira Sans', '{system_family}', 'Arial', 'Helvetica';
        font-size: 11pt;
    }}
    QPushButton {{
        background-color: #3c3f41;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 6px;
    }}
    QPushButton:hover {{ background-color: #505357; }}
    QComboBox, QLabel {{
        background-color: #3c3f41;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 4px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #2b2b2b;
        selection-background-color: #606366;
    }}
    QGraphicsView {{ background-color: #1e1e1e; }}
    """
    app.setStyleSheet(dark_stylesheet)

    def start_main_window():
        window = LabelingApp(img_dir, lbl_dir, cls_file, kp_file)
        splash.finish(window)
        window.show()
        screen = app.primaryScreen(); screen_geometry = screen.availableGeometry()
        window_width = window.frameGeometry().width(); window_height = window.frameGeometry().height()
        x = (screen_geometry.width() - window_width) // 2
        y = (screen_geometry.height() - window_height) // 2
        window.move(x, y); window.raise_(); window.activateWindow(); window._update_status()

    QTimer.singleShot(1000, start_main_window)
    sys.exit(app.exec())
