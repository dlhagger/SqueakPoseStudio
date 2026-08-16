"""Navigation and layer-selection controls with explicit callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from squeakpose.project.layers import LAYER_DEPTH, LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.ui.style import (
    ThemedComboBox,
    apply_panel_shadow,
    sidebar_stylesheet,
    style_combo_popup,
)

NAV_FILTERS = ("all", "labeled", "unlabeled")


def _ignore_text(_value: str) -> None:
    pass


def _ignore_visibility(_layer_id: str, _visible: bool) -> None:
    pass


def _ignore_action() -> None:
    pass


@dataclass(frozen=True, slots=True)
class NavigationPanelCallbacks:
    filter_changed: Callable[[str], None] = _ignore_text
    layer_changed: Callable[[str], None] = _ignore_text
    visibility_changed: Callable[[str, bool], None] = _ignore_visibility
    previous: Callable[[], None] = _ignore_action
    next: Callable[[], None] = _ignore_action
    complete: Callable[[], None] = _ignore_action
    skip: Callable[[], None] = _ignore_action
    save: Callable[[], None] = _ignore_action
    delete_image: Callable[[], None] = _ignore_action


def _popup(combo: ThemedComboBox, *, max_items: int) -> None:
    combo.setMaxVisibleItems(max_items)
    popup = QListView(combo)
    popup.setUniformItemSizes(True)
    popup.setSpacing(2)
    popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
    popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    style_combo_popup(popup)
    combo.setView(popup)


def _button(text: str, callback: Callable[[], None], tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setMinimumHeight(30)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(lambda _checked=False: callback())
    return button


class NavigationPanel(QFrame):
    """Browse, layer visibility, and frame-operation controls."""

    def __init__(
        self,
        *,
        active_filter: str = "all",
        active_layer: str = LAYER_KEYPOINTS,
        layer_visibility: Mapping[str, bool] | None = None,
        callbacks: NavigationPanelCallbacks | None = None,
        embedded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.callbacks = callbacks or NavigationPanelCallbacks()
        if not embedded:
            self.setObjectName("ToolPanel")
            self.setStyleSheet(sidebar_stylesheet())
            apply_panel_shadow(self)
        layout = QVBoxLayout(self)
        if embedded:
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(6)
        if not embedded:
            title = QLabel("Navigation & Labeling")
            title.setObjectName("panelTitle")
            layout.addWidget(title)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        browse_label = QLabel("Browse")
        browse_label.setObjectName("fieldLabel")
        filter_row.addWidget(browse_label)
        self.filter_combo = ThemedComboBox()
        self.filter_combo.setObjectName("browseSelector")
        self.filter_combo.addItems([name.title() for name in NAV_FILTERS])
        self.filter_combo.setToolTip("Which images to browse with Prev/Next")
        self.filter_combo.setMinimumContentsLength(10)
        self.filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.filter_combo.setMinimumWidth(0)
        self.filter_combo.setMinimumHeight(34)
        self.filter_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        _popup(self.filter_combo, max_items=8)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        layer_row = QHBoxLayout()
        layer_row.setSpacing(6)
        layer_label = QLabel("Layer")
        layer_label.setObjectName("fieldLabel")
        layer_row.addWidget(layer_label)
        self.layer_selector = ThemedComboBox()
        self.layer_selector.setObjectName("workflowSelector")
        self.layer_selector.addItem("Keypoints Layer", LAYER_KEYPOINTS)
        self.layer_selector.addItem("Segmentation Layer", LAYER_SEGMENTATION)
        self.layer_selector.addItem("Depth Layer", LAYER_DEPTH)
        self.layer_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.layer_selector.setMinimumContentsLength(18)
        self.layer_selector.setMinimumWidth(0)
        self.layer_selector.setMinimumHeight(34)
        self.layer_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.layer_selector.setToolTip(
            "Choose the annotation layer to edit. Each layer keeps its own labels, "
            "model, dataset, and analysis context."
        )
        _popup(self.layer_selector, max_items=6)
        layer_row.addWidget(self.layer_selector, 1)
        layout.addLayout(layer_row)

        visibility_row = QHBoxLayout()
        visibility_row.setSpacing(8)
        visibility_label = QLabel("Layers")
        visibility_label.setObjectName("fieldLabel")
        visibility_row.addWidget(visibility_label)
        self.visibility_buttons: dict[str, QPushButton] = {}
        for layer_id, text in (
            (LAYER_KEYPOINTS, "● Keypoints"),
            (LAYER_SEGMENTATION, "● Segmentation"),
            (LAYER_DEPTH, "● Depth"),
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setProperty("layerVisibilityPill", True)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            button.toggled.connect(
                lambda visible, selected_layer=layer_id: self.callbacks.visibility_changed(
                    selected_layer,
                    visible,
                )
            )
            visibility_row.addWidget(button)
            self.visibility_buttons[layer_id] = button
        visibility_row.addStretch(1)
        layout.addLayout(visibility_row)
        self.keypoints_visibility_btn = self.visibility_buttons[LAYER_KEYPOINTS]
        self.segmentation_visibility_btn = self.visibility_buttons[LAYER_SEGMENTATION]
        self.depth_visibility_btn = self.visibility_buttons[LAYER_DEPTH]

        self.nav_grid = QGridLayout()
        self.nav_grid.setHorizontalSpacing(6)
        self.nav_grid.setVerticalSpacing(6)
        self.previous_btn = _button("◀ Prev", self.callbacks.previous)
        self.next_btn = _button("Next ▶", self.callbacks.next)
        self.complete_btn = _button(
            "Complete",
            self.callbacks.complete,
            "Save and jump to next unlabeled image",
        )
        self.skip_btn = _button(
            "Skip",
            self.callbacks.skip,
            "Jump to next unlabeled image",
        )
        self.save_btn = _button("Save", self.callbacks.save)
        self.delete_image_btn = _button(
            "Delete Image",
            self.callbacks.delete_image,
            "Delete the current image after confirmation",
        )
        self.nav_grid.addWidget(self.previous_btn, 0, 0)
        self.nav_grid.addWidget(self.next_btn, 0, 1)
        self.nav_grid.addWidget(self.complete_btn, 0, 2)
        self.nav_grid.addWidget(self.skip_btn, 1, 0)
        self.nav_grid.addWidget(self.save_btn, 1, 1)
        self.nav_grid.addWidget(self.delete_image_btn, 1, 2)
        layout.addLayout(self.nav_grid)

        self.set_filter(active_filter, emit=False)
        self.set_active_layer(active_layer, emit=False)
        self.set_visibility(layer_visibility or {}, emit=False)
        self.filter_combo.currentTextChanged.connect(
            lambda text: self.callbacks.filter_changed(text.strip().lower())
        )
        self.layer_selector.currentIndexChanged.connect(self._emit_layer)

    def set_filter(self, value: str, *, emit: bool = False) -> None:
        normalized = str(value).strip().lower()
        index = NAV_FILTERS.index(normalized) if normalized in NAV_FILTERS else 0
        self.filter_combo.blockSignals(True)
        self.filter_combo.setCurrentIndex(index)
        self.filter_combo.blockSignals(False)
        if emit:
            self.callbacks.filter_changed(NAV_FILTERS[index])

    def set_active_layer(self, layer_id: str, *, emit: bool = False) -> None:
        index = self.layer_selector.findData(str(layer_id))
        if index < 0:
            index = self.layer_selector.findData(LAYER_KEYPOINTS)
        self.layer_selector.blockSignals(True)
        self.layer_selector.setCurrentIndex(index)
        self.layer_selector.blockSignals(False)
        for candidate, button in self.visibility_buttons.items():
            button.setProperty("activeLayer", candidate == str(layer_id))
            button.style().unpolish(button)
            button.style().polish(button)
        self.set_depth_mode(str(layer_id) == LAYER_DEPTH)
        if emit:
            self.callbacks.layer_changed(str(self.layer_selector.currentData()))

    def set_visibility(
        self,
        values: Mapping[str, bool],
        *,
        emit: bool = False,
    ) -> None:
        for layer_id, button in self.visibility_buttons.items():
            visible = bool(values.get(layer_id, True))
            button.blockSignals(True)
            button.setChecked(visible)
            button.blockSignals(False)
            if emit:
                self.callbacks.visibility_changed(layer_id, visible)

    def set_depth_mode(self, enabled: bool) -> None:
        self.save_btn.setEnabled(not enabled)
        self.complete_btn.setEnabled(not enabled)
        self.save_btn.setText("Save")
        self.save_btn.setToolTip(
            "Save labels for current frame" if not enabled else "Depth maps save after prediction"
        )

    def _emit_layer(self, _index: int) -> None:
        self.callbacks.layer_changed(str(self.layer_selector.currentData()))


__all__ = ["NAV_FILTERS", "NavigationPanel", "NavigationPanelCallbacks"]
