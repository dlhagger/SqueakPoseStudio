"""Canvas overlay presentation without graphics-scene or application ownership."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from squeakpose.ui.canvas_hud import LayerContextHud
from squeakpose.ui.style import apply_panel_shadow, hud_stylesheet

KEYPOINT_LEGEND_TEXT = (
    "Keys:  🔴 Visible   🟡 Occluded   ⚪ Invisible (v=0)\n"
    "L: toggle labels   -/= point size   [/] text size\n"
    "0: mark next invisible   Shift+0: selected → invisible"
)


@dataclass(frozen=True)
class CanvasPresentationState:
    """The small, render-only state needed by canvas HUD widgets."""

    editing: str
    references: str = ""
    mode: str = ""
    zoom_scale: float = 1.0


class KeypointLegendHud(QFrame):
    """Present the existing keypoint visibility and shortcut legend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(hud_stylesheet())
        apply_panel_shadow(self)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(6)

        self.title_label = QLabel("Keypoint Visibility")
        self.title_label.setObjectName("hudTitle")
        layout.addWidget(self.title_label)

        self.legend_label = QLabel(KEYPOINT_LEGEND_TEXT)
        self.legend_label.setWordWrap(True)
        self.legend_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self.legend_label)

    def fit_to_viewport(self, viewport_width: int) -> None:
        """Apply the legacy bounded, font-aware legend width."""
        character_width = self.legend_label.fontMetrics().horizontalAdvance("M")
        preferred = int(character_width * 30 + 24)
        width = max(250, min(preferred, int(viewport_width * 0.42), 420))
        self.setFixedWidth(width)
        self.adjustSize()


class ZoomHud(QFrame):
    """Present the current canvas transform scale as a percentage."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(hud_stylesheet())
        apply_panel_shadow(self)
        self.setFixedWidth(132)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.value_label = QLabel("Zoom: 100%")
        self.value_label.setObjectName("zoomValue")
        layout.addWidget(self.value_label)

    def set_scale(self, scale: float) -> None:
        self.value_label.setText(f"Zoom: {int(float(scale) * 100)}%")


class CanvasHudPresenter(QObject):
    """Own and position canvas HUD widgets using explicit presentation state.

    The presenter is deliberately unaware of ``QGraphicsScene`` and ``LabelingApp``.
    Its caller supplies already-derived layer text, mode, zoom scale, and viewport
    dimensions.
    """

    def __init__(self, canvas_parent: QWidget) -> None:
        super().__init__(canvas_parent)
        self.layer_context = LayerContextHud(canvas_parent)
        self.legend = KeypointLegendHud(canvas_parent)
        self.zoom = ZoomHud(canvas_parent)
        self.legend.hide()
        self.zoom.hide()

    def apply(self, state: CanvasPresentationState) -> None:
        self.set_context(editing=state.editing, references=state.references)
        self.set_zoom_scale(state.zoom_scale)
        self.set_mode(state.mode)

    def set_context(self, *, editing: str, references: str = "") -> None:
        self.layer_context.set_context(editing=editing, references=references)

    def set_zoom_scale(self, scale: float) -> None:
        self.zoom.set_scale(scale)

    def set_mode(self, mode: str) -> None:
        self.legend.setVisible(mode == "keypoint")
        self.zoom.setVisible(mode == "panzoom")

    def layout_overlays(self, *, viewport_width: int, viewport_height: int) -> None:
        """Place the top-left context and bottom-left mode overlays."""
        self.layout_context()

        x = 10
        cursor_y = max(0, int(viewport_height)) - 10
        if not self.legend.isHidden():
            self.legend.fit_to_viewport(max(0, int(viewport_width)))
            legend_height = self.legend.sizeHint().height()
            top = max(10, cursor_y - legend_height)
            self.legend.move(x, top)
            cursor_y = top - 8
        if not self.zoom.isHidden():
            zoom_height = self.zoom.sizeHint().height()
            self.zoom.move(x, max(10, cursor_y - zoom_height))

    def layout_context(self) -> None:
        """Place and raise the persistent top-left layer context."""
        self.layer_context.adjustSize()
        self.layer_context.move(10, 10)
        self.layer_context.raise_()

    def show_context(self) -> None:
        self.layer_context.show()


__all__ = [
    "CanvasHudPresenter",
    "CanvasPresentationState",
    "KEYPOINT_LEGEND_TEXT",
    "KeypointLegendHud",
    "ZoomHud",
]
