"""Small canvas HUD presentation components without scene ownership."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from squeakpose.ui.style import apply_panel_shadow, hud_stylesheet


class LayerContextHud(QFrame):
    """Present the active edit layer and visible reference-layer summary."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(hud_stylesheet())
        apply_panel_shadow(self, blur=14, y_offset=2, alpha=75)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(1)
        self.editing_label = QLabel("")
        self.editing_label.setObjectName("layerEditing")
        self.reference_label = QLabel("")
        self.reference_label.setObjectName("layerReference")
        layout.addWidget(self.editing_label)
        layout.addWidget(self.reference_label)

    def set_context(self, *, editing: str, references: str = "") -> None:
        self.editing_label.setText(str(editing))
        self.reference_label.setText(str(references))
        self.reference_label.setVisible(bool(str(references)))
        self.adjustSize()


__all__ = ["LayerContextHud"]
