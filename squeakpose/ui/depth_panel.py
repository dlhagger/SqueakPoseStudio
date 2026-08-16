"""Depth display, range, and model controls independent of window ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from squeakpose.annotation.depth import DEFAULT_PROBE_TEXT, DEFAULT_RANGE_TEXT
from squeakpose.ui.style import ThemedComboBox, apply_panel_shadow, sidebar_stylesheet


def _ignore_mode(_mode: str) -> None:
    pass


def _ignore_model(_path: str) -> None:
    pass


def _ignore_action() -> None:
    pass


@dataclass(frozen=True, slots=True)
class DepthDisplayCallbacks:
    mode_changed: Callable[[str], None] = _ignore_mode
    clear_probes: Callable[[], None] = _ignore_action


@dataclass(frozen=True, slots=True)
class DepthModelCallbacks:
    select_model: Callable[[str], None] = _ignore_model
    choose_model: Callable[[], None] = _ignore_action


class _DepthFrame(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolPanel")
        self.setStyleSheet(sidebar_stylesheet())
        apply_panel_shadow(self)
        self.panel_layout = QVBoxLayout(self)
        self.panel_layout.setContentsMargins(10, 9, 10, 9)
        self.panel_layout.setSpacing(6)
        heading = QLabel(title)
        heading.setObjectName("panelTitle")
        self.panel_layout.addWidget(heading)


def _button(text: str, callback: Callable[[], None], tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setMinimumHeight(30)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(lambda _checked=False: callback())
    return button


class DepthDisplayPanel(_DepthFrame):
    def __init__(
        self,
        *,
        mode: str = "depth",
        callbacks: DepthDisplayCallbacks | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self.callbacks = callbacks or DepthDisplayCallbacks()
        super().__init__("Depth Display", parent)
        self.mode_combo = ThemedComboBox()
        self.mode_combo.addItem("Original", "original")
        self.mode_combo.addItem("Depth", "depth")
        self.mode_combo.addItem("Overlay", "overlay")
        self.mode_combo.setToolTip(
            "Compare the source image, standalone depth map, or a blended overlay"
        )
        self.mode_combo.setMinimumHeight(32)
        self.panel_layout.addWidget(self.mode_combo)
        self.set_mode(mode, emit=False)
        self.mode_combo.currentIndexChanged.connect(self._emit_mode)

    def set_mode(self, mode: str, *, emit: bool = False) -> None:
        index = self.mode_combo.findData(str(mode).strip().lower())
        if index < 0:
            index = self.mode_combo.findData("depth")
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)
        if emit:
            self.callbacks.mode_changed(str(self.mode_combo.currentData()))

    def _emit_mode(self, _index: int) -> None:
        self.callbacks.mode_changed(str(self.mode_combo.currentData()))


class DepthRangePanel(_DepthFrame):
    def __init__(
        self,
        callbacks: DepthDisplayCallbacks | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self.callbacks = callbacks or DepthDisplayCallbacks()
        super().__init__("Depth Range", parent)
        self.range_label = QLabel(DEFAULT_RANGE_TEXT)
        self.range_label.setWordWrap(True)
        self.range_label.setStyleSheet("color: #9fb0bd; font-size: 9pt;")
        self.range_label.setToolTip(
            "Depth values are estimated meters. The preview uses inverse depth, "
            "so brighter colors indicate surfaces closer to the camera."
        )
        self.panel_layout.addWidget(self.range_label)
        self.probe_label = QLabel(DEFAULT_PROBE_TEXT)
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet("color: #c8d4dc; font-size: 9pt;")
        self.panel_layout.addWidget(self.probe_label)
        self.clear_btn = _button(
            "Clear Probes",
            self.callbacks.clear_probes,
            "Remove depth sample markers from the current image",
        )
        self.clear_btn.setMinimumHeight(28)
        self.clear_btn.setEnabled(False)
        self.panel_layout.addWidget(self.clear_btn)

    def set_range_text(self, text: str) -> None:
        self.range_label.setText(str(text))

    def set_probe_text(self, text: str, *, can_clear: bool) -> None:
        self.probe_label.setText(str(text))
        self.clear_btn.setEnabled(bool(can_clear))


class DepthModelPanel(_DepthFrame):
    def __init__(
        self,
        callbacks: DepthModelCallbacks | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self.callbacks = callbacks or DepthModelCallbacks()
        super().__init__("Depth Assistant", parent)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("fieldLabel")
        self.panel_layout.addWidget(self.status_label)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)
        self.official_model_btn = QPushButton("YOLO26 Depth ▾")
        self.official_model_btn.setToolTip(
            "Choose an official depth model; Ultralytics downloads it on first use"
        )
        menu = QMenu(self.official_model_btn)
        for size, description in (
            ("n", "Nano — fastest"),
            ("s", "Small"),
            ("m", "Medium"),
            ("l", "Large"),
            ("x", "Extra large — most accurate"),
        ):
            action = menu.addAction(description)
            action.triggered.connect(
                lambda _checked=False, model_size=size: self.callbacks.select_model(
                    f"yolo26{model_size}-depth.pt"
                )
            )
        self.official_model_btn.setMenu(menu)
        self.choose_model_btn = _button(
            "Choose…",
            self.callbacks.choose_model,
            "Choose a custom Ultralytics depth checkpoint",
        )
        self.clear_model_btn = _button(
            "Clear Model",
            lambda: self.callbacks.select_model(""),
        )
        for button in (self.official_model_btn, self.choose_model_btn, self.clear_model_btn):
            button.setMinimumHeight(30)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.grid.addWidget(self.official_model_btn, 0, 0)
        self.grid.addWidget(self.choose_model_btn, 0, 1)
        self.grid.addWidget(self.clear_model_btn, 1, 0, 1, 2)
        self.panel_layout.addLayout(self.grid)

    def set_model_status(self, text: str, *, tooltip: str = "", can_clear: bool = False) -> None:
        self.status_label.setText(str(text))
        self.status_label.setToolTip(str(tooltip))
        self.clear_model_btn.setEnabled(bool(can_clear))


__all__ = [
    "DepthDisplayCallbacks",
    "DepthDisplayPanel",
    "DepthModelCallbacks",
    "DepthModelPanel",
    "DepthRangePanel",
]
