"""QGraphicsScene presentation for validated depth artifacts and probes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from squeakpose.annotation.depth import (
    DepthArtifactLoadResult,
    DepthAssistantState,
    DepthProbe,
    DepthViewMode,
    normalize_depth_view_mode,
)

_PROBE_COLORS = (
    "#73d7ff",
    "#ffd166",
    "#82e0aa",
    "#ff8fab",
    "#c7a0ff",
    "#f6bd60",
)


class DepthRangeView(Protocol):
    def set_range_text(self, text: str) -> None: ...

    def set_probe_text(self, text: str, *, can_clear: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class DepthPreviewDecision:
    """Detached choice for composing the original image and saved preview."""

    mode: DepthViewMode
    show_original: bool
    preview_path: str
    preview_opacity: float
    status_message: str

    @property
    def show_preview(self) -> bool:
        return bool(self.preview_path)


@dataclass(frozen=True, slots=True)
class DepthPreviewPresentation:
    decision: DepthPreviewDecision
    preview_item: QGraphicsPixmapItem | None = None
    status_message: str = ""


def decide_depth_preview(
    artifacts: DepthArtifactLoadResult,
    mode: str,
) -> DepthPreviewDecision:
    """Preserve the main-window depth/original/overlay display choices."""
    normalized = normalize_depth_view_mode(mode)
    if normalized == "original":
        return DepthPreviewDecision(
            mode=normalized,
            show_original=True,
            preview_path="",
            preview_opacity=0.0,
            status_message="Original image displayed; a saved depth map remains available.",
        )
    if not artifacts.preview_available:
        return DepthPreviewDecision(
            mode=normalized,
            show_original=True,
            preview_path="",
            preview_opacity=0.0,
            status_message="No saved depth map for this image. Select Predict to create one.",
        )
    overlay = normalized == "overlay"
    return DepthPreviewDecision(
        mode=normalized,
        show_original=True,
        preview_path=artifacts.plan.preview_path,
        preview_opacity=0.55 if overlay else 1.0,
        status_message=(
            "Saved depth overlay displayed."
            if overlay
            else "Saved depth map displayed (near = bright)."
        ),
    )


class DepthPreviewPresenter:
    """Render depth preview and probe items without owning artifact or scene state."""

    def __init__(
        self,
        scene: QGraphicsScene,
        *,
        range_view: DepthRangeView | None = None,
        track_item: Callable[[QGraphicsItem], None] | None = None,
        pixmap_loader: Callable[[str], QPixmap] = QPixmap,
    ) -> None:
        self.scene = scene
        self.range_view = range_view
        self._track_item = track_item
        self._pixmap_loader = pixmap_loader
        self._preview_item: QGraphicsPixmapItem | None = None
        self._probe_items: list[QGraphicsItem] = []

    @property
    def preview_item(self) -> QGraphicsPixmapItem | None:
        return self._preview_item

    @property
    def probe_items(self) -> tuple[QGraphicsItem, ...]:
        return tuple(self._probe_items)

    def present_preview(
        self,
        artifacts: DepthArtifactLoadResult,
        *,
        mode: str,
        image_width: int,
        image_height: int,
    ) -> DepthPreviewPresentation:
        """Add only the optional preview layer; the source image remains scene-owned."""
        self.clear_preview()
        decision = decide_depth_preview(artifacts, mode)
        if not decision.show_preview:
            return DepthPreviewPresentation(
                decision=decision,
                status_message=decision.status_message,
            )

        pixmap = self._pixmap_loader(decision.preview_path)
        if pixmap.isNull():
            # The legacy path silently retained the original item when an existing
            # preview file could not be decoded.
            return DepthPreviewPresentation(decision=decision)
        width = int(image_width)
        height = int(image_height)
        if width > 0 and height > 0 and (pixmap.width() != width or pixmap.height() != height):
            pixmap = pixmap.scaled(
                width,
                height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        item = QGraphicsPixmapItem(pixmap)
        item.setZValue(1.0)
        item.setOpacity(decision.preview_opacity)
        self.scene.addItem(item)
        self._preview_item = item
        if self._track_item is not None:
            self._track_item(item)
        return DepthPreviewPresentation(
            decision=decision,
            preview_item=item,
            status_message=decision.status_message,
        )

    def present_state(self, state: DepthAssistantState) -> None:
        """Project domain-owned range/probe text onto the existing panel."""
        if self.range_view is None:
            return
        self.range_view.set_range_text(state.range_text())
        self.range_view.set_probe_text(state.probe_text(), can_clear=bool(state.probes))

    def present_probe_markers(
        self,
        probes: Iterable[DepthProbe | Mapping[str, Any]],
        *,
        active_depth_layer: bool,
    ) -> tuple[QGraphicsItem, ...]:
        """Replace numbered, transform-independent scene markers for current probes."""
        self.clear_probe_markers()
        if not active_depth_layer:
            return ()
        normalized = [
            probe if isinstance(probe, DepthProbe) else DepthProbe.from_mapping(probe)
            for probe in probes
        ]
        for index, probe in enumerate(normalized, start=1):
            color = QColor(_PROBE_COLORS[(index - 1) % len(_PROBE_COLORS)])
            marker = QGraphicsEllipseItem(-5.0, -5.0, 10.0, 10.0)
            marker.setPos(float(probe.x) + 0.5, float(probe.y) + 0.5)
            marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            marker.setAcceptHoverEvents(False)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(2)
            marker.setPen(pen)
            marker.setBrush(QBrush(QColor(10, 15, 18, 190)))
            marker.setZValue(20.0)

            value_text = f"{probe.depth:.3f} m" if probe.depth is not None else "invalid"
            text_item = QGraphicsSimpleTextItem(f"{index} · {value_text}")
            text_item.setBrush(QBrush(color))
            text_item.setPos(float(probe.x) + 8.5, float(probe.y) - 10.5)
            text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            text_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            text_item.setAcceptHoverEvents(False)
            text_item.setZValue(20.0)
            self.scene.addItem(marker)
            self.scene.addItem(text_item)
            self._probe_items.extend((marker, text_item))
        return self.probe_items

    def clear_preview(self) -> None:
        item = self._preview_item
        self._preview_item = None
        if item is not None:
            self._remove_item(item)

    def clear_probe_markers(self) -> None:
        items = self._probe_items
        self._probe_items = []
        for item in items:
            self._remove_item(item)

    def clear(self) -> None:
        self.clear_preview()
        self.clear_probe_markers()

    def _remove_item(self, item: QGraphicsItem) -> None:
        try:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        except RuntimeError:
            # QGraphicsScene.clear() may already have deleted the wrapped C++ item.
            pass


__all__ = [
    "DepthPreviewDecision",
    "DepthPreviewPresentation",
    "DepthPreviewPresenter",
    "DepthRangeView",
    "decide_depth_preview",
]
