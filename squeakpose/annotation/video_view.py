"""Graphics view used by the video-review dialog."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView


class VideoView(QGraphicsView):
    """Pannable video frame view with bounded cursor-centered zoom."""

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event) -> None:
        zoom_in = 1.05
        factor = zoom_in if event.angleDelta().y() > 0 else 1.0 / zoom_in
        current = self.transform().m11()
        new_scale = current * factor
        if new_scale < 0.10:
            factor = 0.10 / current
        elif new_scale > 8.0:
            factor = 8.0 / current
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetTransform()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def reset_view(self) -> None:
        self.resetTransform()
