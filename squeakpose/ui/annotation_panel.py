"""Cohesive annotation controls with explicit, main-window-free callbacks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


def _ignore_mode(_mode: str) -> None:
    pass


def _ignore_class(_class_id: int) -> None:
    pass


def _ignore_action() -> None:
    pass


@dataclass(frozen=True, slots=True)
class AnnotationPanelCallbacks:
    mode_changed: Callable[[str], None] = _ignore_mode
    class_changed: Callable[[int], None] = _ignore_class
    manage_classes: Callable[[], None] = _ignore_action


def _panel_button(text: str, *, tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setMinimumWidth(116)
    button.setMinimumHeight(28)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if tooltip:
        button.setToolTip(tooltip)
    return button


class AnnotationPanel(QFrame):
    """Mode, class, and progress controls for the active annotation layer."""

    def __init__(
        self,
        classes: Sequence[str] = (),
        *,
        active_layer: str = LAYER_KEYPOINTS,
        active_mode: str = "panzoom",
        selected_class_id: int = 0,
        callbacks: AnnotationPanelCallbacks | None = None,
        embedded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.callbacks = callbacks or AnnotationPanelCallbacks()
        self.active_layer = str(active_layer)
        self.active_mode = str(active_mode)
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
            title = QLabel("Annotation")
            title.setObjectName("panelTitle")
            layout.addWidget(title)

        mode_label = QLabel("Mode")
        mode_label.setObjectName("sectionLabel")
        layout.addWidget(mode_label)
        self.mode_grid = QGridLayout()
        self.mode_grid.setHorizontalSpacing(5)
        self.mode_grid.setVerticalSpacing(5)
        layout.addLayout(self.mode_grid)

        self.panzoom_btn = _panel_button("Pan/Zoom (1)")
        self.bbox_btn = _panel_button("BBox (2)")
        self.keypoint_btn = _panel_button("Keypoint (3)")
        self.predict_btn = _panel_button("Predict (4)")
        self.segment_btn = _panel_button(
            "Segment (2)",
            tooltip="Segmentation click prompts (left=positive, right=negative)",
        )
        self.seg_edit_btn = _panel_button(
            "Edit Mask (E)",
            tooltip="Manual mask edit mode using brush add/erase.",
        )
        self.mode_buttons = {
            "panzoom": self.panzoom_btn,
            "bbox": self.bbox_btn,
            "keypoint": self.keypoint_btn,
            "predict": self.predict_btn,
            "segment": self.segment_btn,
            "segedit": self.seg_edit_btn,
        }
        for mode, button in self.mode_buttons.items():
            button.clicked.connect(
                lambda _checked=False, selected_mode=mode: self._choose_mode(selected_mode)
            )

        self.class_controls_frame = QFrame()
        class_layout = QVBoxLayout(self.class_controls_frame)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(5)
        self.class_label = QLabel("Class")
        self.class_label.setObjectName("fieldLabel")
        class_layout.addWidget(self.class_label)
        self.class_selector = ThemedComboBox()
        self.class_selector.setObjectName("classSelector")
        self.class_selector.setToolTip("Choose the active class to label")
        self.class_selector.setMinimumContentsLength(12)
        self.class_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.class_selector.setMinimumWidth(0)
        self.class_selector.setMinimumHeight(34)
        self.class_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.class_selector.setMaxVisibleItems(8)
        popup = QListView(self.class_selector)
        popup.setUniformItemSizes(True)
        popup.setSpacing(2)
        popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        style_combo_popup(popup)
        self.class_selector.setView(popup)
        class_layout.addWidget(self.class_selector)
        self.manage_classes_btn = _panel_button(
            "Classes…",
            tooltip="Manage classes and per-class keypoints",
        )
        self.manage_classes_btn.setMinimumHeight(30)
        self.manage_classes_btn.clicked.connect(
            lambda _checked=False: self.callbacks.manage_classes()
        )
        class_layout.addWidget(self.manage_classes_btn)
        layout.addWidget(self.class_controls_frame)

        progress_row = QHBoxLayout()
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progressBadge")
        progress_row.addWidget(self.progress_label)
        progress_row.addStretch(1)
        layout.addLayout(progress_row)

        self.set_classes(classes, selected_class_id=selected_class_id, emit=False)
        self.class_selector.currentIndexChanged.connect(self.callbacks.class_changed)
        self.set_layer(active_layer)
        self.set_active_mode(active_mode)

    def set_classes(
        self,
        classes: Sequence[str],
        *,
        selected_class_id: int = 0,
        emit: bool = False,
    ) -> None:
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.addItems([str(name) for name in classes])
        if self.class_selector.count():
            selected = min(max(0, int(selected_class_id)), self.class_selector.count() - 1)
            self.class_selector.setCurrentIndex(selected)
        self.class_selector.blockSignals(False)
        if emit:
            self.callbacks.class_changed(self.class_selector.currentIndex())

    def set_layer(self, layer_id: str) -> None:
        self.active_layer = str(layer_id)
        is_pose = self.active_layer == LAYER_KEYPOINTS
        is_segmentation = self.active_layer == LAYER_SEGMENTATION
        is_depth = self.active_layer == LAYER_DEPTH
        self.class_controls_frame.setVisible(not is_depth)
        self.bbox_btn.setVisible(is_pose)
        self.bbox_btn.setEnabled(is_pose)
        self.keypoint_btn.setVisible(is_pose)
        self.keypoint_btn.setEnabled(is_pose)
        self.segment_btn.setVisible(is_segmentation)
        self.segment_btn.setEnabled(is_segmentation)
        self.seg_edit_btn.setVisible(is_segmentation)
        self.seg_edit_btn.setEnabled(is_segmentation)
        self.predict_btn.setToolTip(
            "Run the Keypoints layer model on the current image"
            if is_pose
            else (
                "Run the Segmentation layer model on the current image"
                if is_segmentation
                else "Estimate and save a dense depth map for the current image"
            )
        )
        self.manage_classes_btn.setToolTip(
            "Depth maps do not use classes"
            if is_depth
            else (
                "Manage classes and per-class keypoints"
                if is_pose
                else "Manage segmentation classes"
            )
        )
        self._reflow_modes()

    def set_active_mode(self, mode: str) -> None:
        self.active_mode = str(mode)
        for name, button in self.mode_buttons.items():
            button.setProperty("activeMode", name == self.active_mode)
            button.style().unpolish(button)
            button.style().polish(button)

    def set_progress(self, text: str) -> None:
        self.progress_label.setText(str(text))

    def _choose_mode(self, mode: str) -> None:
        self.set_active_mode(mode)
        self.callbacks.mode_changed(mode)

    def _reflow_modes(self) -> None:
        is_pose = self.active_layer == LAYER_KEYPOINTS
        is_depth = self.active_layer == LAYER_DEPTH
        self.panzoom_btn.setText("Pan/Zoom (1)" if is_pose else "Pan (1)")
        self.segment_btn.setText("Segment (2)" if is_pose else "Segment Prompt (2)")
        if is_depth:
            self.mode_grid.addWidget(self.panzoom_btn, 0, 0)
            self.mode_grid.addWidget(self.predict_btn, 0, 1)
            self.mode_grid.addWidget(self.bbox_btn, 2, 0)
            self.mode_grid.addWidget(self.keypoint_btn, 2, 1)
            self.mode_grid.addWidget(self.segment_btn, 3, 0)
            self.mode_grid.addWidget(self.seg_edit_btn, 3, 1)
        elif is_pose:
            self.mode_grid.addWidget(self.panzoom_btn, 0, 0)
            self.mode_grid.addWidget(self.bbox_btn, 0, 1)
            self.mode_grid.addWidget(self.keypoint_btn, 1, 0)
            self.mode_grid.addWidget(self.predict_btn, 1, 1)
            self.mode_grid.addWidget(self.segment_btn, 2, 0, 1, 2)
            self.mode_grid.addWidget(self.seg_edit_btn, 3, 0, 1, 2)
        else:
            self.mode_grid.addWidget(self.panzoom_btn, 0, 0)
            self.mode_grid.addWidget(self.segment_btn, 0, 1)
            self.mode_grid.addWidget(self.seg_edit_btn, 1, 0)
            self.mode_grid.addWidget(self.predict_btn, 1, 1)
            self.mode_grid.addWidget(self.bbox_btn, 2, 0)
            self.mode_grid.addWidget(self.keypoint_btn, 2, 1)


@dataclass(frozen=True, slots=True)
class SegmentationToolsCallbacks:
    load_model: Callable[[], None] = _ignore_action
    run: Callable[[], None] = _ignore_action
    accept: Callable[[], None] = _ignore_action
    reset: Callable[[], None] = _ignore_action


class SegmentationToolsPanel(QFrame):
    """SAM prompt and brush controls, independent of model ownership."""

    def __init__(
        self,
        *,
        callbacks: SegmentationToolsCallbacks | None = None,
        brush_radius: int = 8,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.callbacks = callbacks or SegmentationToolsCallbacks()
        self.setObjectName("ToolPanel")
        self.setStyleSheet(sidebar_stylesheet())
        apply_panel_shadow(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(6)
        title = QLabel("Segmentation Tools")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        brush_row = QHBoxLayout()
        brush_label = QLabel("Brush")
        brush_label.setObjectName("fieldLabel")
        brush_row.addWidget(brush_label)
        self.brush_size_label = QLabel("")
        self.brush_size_label.setObjectName("brushSizeBadge")
        brush_row.addWidget(self.brush_size_label)
        brush_row.addStretch(1)
        layout.addLayout(brush_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.load_btn = _panel_button(
            "Load SAM",
            tooltip="Load a SAM model file for segmentation prompts",
        )
        self.run_btn = _panel_button(
            "Run (G)",
            tooltip="Run SAM using current positive/negative prompts",
        )
        self.accept_btn = _panel_button(
            "Accept",
            tooltip="Commit the current SAM mask preview to this class",
        )
        self.accept_btn.setObjectName("samAcceptButton")
        self.reset_btn = _panel_button(
            "Reset",
            tooltip="Remove prompt points and the current SAM preview",
        )
        for button, callback in (
            (self.load_btn, self.callbacks.load_model),
            (self.run_btn, self.callbacks.run),
            (self.accept_btn, self.callbacks.accept),
            (self.reset_btn, self.callbacks.reset),
        ):
            button.setMinimumHeight(30)
            button.clicked.connect(lambda _checked=False, action=callback: action())
        grid.addWidget(self.load_btn, 0, 0)
        grid.addWidget(self.run_btn, 0, 1)
        grid.addWidget(self.accept_btn, 1, 0)
        grid.addWidget(self.reset_btn, 1, 1)
        layout.addLayout(grid)
        self.helper_label = QLabel("")
        self.helper_label.setWordWrap(True)
        self.helper_label.setObjectName("samHelper")
        layout.addWidget(self.helper_label)
        self.set_brush_radius(brush_radius)
        self.set_state(model_loaded=False, prompt_count=0, has_preview=False)

    def set_brush_radius(self, radius: int) -> None:
        self.brush_size_label.setText(f"Brush: {max(1, int(radius))}px")

    def set_state(
        self,
        *,
        model_loaded: bool,
        prompt_count: int,
        has_preview: bool,
        busy: bool = False,
    ) -> None:
        count = max(0, int(prompt_count))
        self.load_btn.setEnabled(not busy)
        self.run_btn.setEnabled(model_loaded and count > 0 and not busy)
        self.accept_btn.setEnabled(has_preview and not busy)
        self.reset_btn.setEnabled((count > 0 or has_preview) and not busy)
        if busy:
            text = "Running segmentation model…"
        elif has_preview:
            text = "Mask preview ready. Accept it or reset the prompts."
        elif not model_loaded:
            text = "Load a SAM model to use positive and negative point prompts."
        elif count:
            text = f"{count} prompt(s) ready. Run SAM to create a mask preview."
        else:
            text = "Add positive or negative prompts on the image."
        self.helper_label.setText(text)


__all__ = [
    "AnnotationPanel",
    "AnnotationPanelCallbacks",
    "SegmentationToolsCallbacks",
    "SegmentationToolsPanel",
]
