"""Reusable right-sidebar operation panels with explicit callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from squeakpose.project.layers import LAYER_DEPTH, LAYER_KEYPOINTS, layer_definition
from squeakpose.ui.style import apply_panel_shadow, sidebar_stylesheet


def _ignore_action() -> None:
    pass


@dataclass(frozen=True, slots=True)
class OperationCallbacks:
    video_review: Callable[[], None] = _ignore_action
    analysis: Callable[[], None] = _ignore_action
    validate_labels: Callable[[], None] = _ignore_action
    export_dataset: Callable[[], None] = _ignore_action
    project_health: Callable[[], None] = _ignore_action
    train: Callable[[], None] = _ignore_action
    distill: Callable[[], None] = _ignore_action
    project_models: Callable[[], None] = _ignore_action
    inference: Callable[[], None] = _ignore_action
    apply_template: Callable[[], None] = _ignore_action
    save_template: Callable[[], None] = _ignore_action


def _button(text: str, callback: Callable[[], None], tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setMinimumHeight(30)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(lambda _checked=False: callback())
    return button


class _OperationFrame(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolPanel")
        self.setStyleSheet(sidebar_stylesheet())
        apply_panel_shadow(self)
        self.panel_layout = QVBoxLayout(self)
        self.panel_layout.setContentsMargins(12, 11, 12, 11)
        self.panel_layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        self.panel_layout.addWidget(self.title_label)


class VideoOperationsPanel(_OperationFrame):
    def __init__(
        self,
        callback: Callable[[], None] = _ignore_action,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Video", parent)
        self.panel_layout.setContentsMargins(12, 11, 12, 14)
        self.review_btn = _button(
            "Video Reviewer",
            callback,
            "Run configured project models over a video and review their overlays",
        )
        self.review_btn.setMinimumHeight(34)
        self.panel_layout.addWidget(self.review_btn)
        self.panel_layout.addSpacing(2)


class AnalysisOperationsPanel(_OperationFrame):
    def __init__(
        self,
        callback: Callable[[], None] = _ignore_action,
        *,
        layer_id: str = LAYER_KEYPOINTS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Analysis", parent)
        self.panel_layout.setContentsMargins(12, 11, 12, 14)
        self.analysis_btn = _button(
            "Run Analysis",
            callback,
            "Analyze inference results for the active layer",
        )
        self.analysis_btn.setMinimumHeight(34)
        self.panel_layout.addWidget(self.analysis_btn)
        self.set_layer(layer_id)

    def set_layer(self, layer_id: str) -> None:
        is_depth = str(layer_id) == LAYER_DEPTH
        self.title_label.setText(f"{layer_definition(layer_id).display_name} Analysis")
        self.analysis_btn.setEnabled(not is_depth)
        self.setVisible(not is_depth)


class DatasetOperationsPanel(_OperationFrame):
    def __init__(
        self,
        callbacks: OperationCallbacks | None = None,
        *,
        layer_id: str = LAYER_KEYPOINTS,
        parent: QWidget | None = None,
    ) -> None:
        self.callbacks = callbacks or OperationCallbacks()
        super().__init__("Dataset & Training", parent)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)
        self.validate_btn = _button(
            "Validate Labels",
            self.callbacks.validate_labels,
            "Rewrite labels_all files and ensure matching images exist in images_all",
        )
        self.export_btn = _button(
            "Export Dataset",
            self.callbacks.export_dataset,
            "Split images_all/labels_all into train/val and regenerate dataset.yaml",
        )
        self.health_btn = _button(
            "Project Health",
            self.callbacks.project_health,
            "Report orphan labels, duplicate stems, and stale transaction files",
        )
        self.train_btn = _button(
            "Train Model",
            self.callbacks.train,
            "Launch a training run for a selected dataset",
        )
        self.distillation_btn = _button(
            "Distillation",
            self.callbacks.distill,
            "Create an unlabeled image corpus and distill a DINO-backed pose model",
        )
        self.grid.addWidget(self.validate_btn, 0, 0)
        self.grid.addWidget(self.export_btn, 0, 1)
        self.grid.addWidget(self.health_btn, 1, 0)
        self.grid.addWidget(self.train_btn, 1, 1)
        self.grid.addWidget(self.distillation_btn, 2, 0, 1, 2)
        self.panel_layout.addLayout(self.grid)
        self.set_layer(layer_id)

    def set_layer(self, layer_id: str) -> None:
        is_depth = str(layer_id) == LAYER_DEPTH
        is_pose = str(layer_id) == LAYER_KEYPOINTS
        self.title_label.setText(
            "Project Tools"
            if is_depth
            else f"{layer_definition(layer_id).display_name} Dataset & Training"
        )
        for button in (self.validate_btn, self.export_btn, self.train_btn):
            button.setEnabled(not is_depth)
            button.setVisible(not is_depth)
        self.distillation_btn.setEnabled(is_pose)
        self.distillation_btn.setVisible(not is_depth)
        self.grid.addWidget(self.health_btn, 0 if is_depth else 1, 0, 1, 2 if is_depth else 1)


class ModelOperationsPanel(_OperationFrame):
    def __init__(
        self,
        callbacks: OperationCallbacks | None = None,
        *,
        layer_id: str = LAYER_KEYPOINTS,
        parent: QWidget | None = None,
    ) -> None:
        self.callbacks = callbacks or OperationCallbacks()
        super().__init__("Project Models & Inference", parent)
        self.status_label = QLabel("")
        self.status_label.setObjectName("fieldLabel")
        self.status_label.setWordWrap(True)
        self.panel_layout.addWidget(self.status_label)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)
        self.models_btn = _button("Project Models…", self.callbacks.project_models)
        self.inference_btn = _button(
            "Inference",
            self.callbacks.inference,
            "Select a video and run every configured project prediction model into "
            "layer-specific CSV outputs",
        )
        self.apply_template_btn = _button(
            "Apply Template",
            self.callbacks.apply_template,
            "Apply the saved template for the selected class",
        )
        self.save_template_btn = _button(
            "Save Template",
            self.callbacks.save_template,
            "Capture the current annotation as the class template",
        )
        self.grid.addWidget(self.models_btn, 0, 0)
        self.grid.addWidget(self.inference_btn, 0, 1)
        self.grid.addWidget(self.apply_template_btn, 1, 0)
        self.grid.addWidget(self.save_template_btn, 1, 1)
        self.panel_layout.addLayout(self.grid)
        self.set_layer(layer_id)

    def set_layer(self, layer_id: str) -> None:
        is_depth = str(layer_id) == LAYER_DEPTH
        is_pose = str(layer_id) == LAYER_KEYPOINTS
        self.title_label.setText("Project Inference" if is_depth else "Project Models & Inference")
        self.models_btn.setVisible(not is_depth)
        self.status_label.setVisible(not is_depth)
        self.apply_template_btn.setVisible(is_pose)
        self.apply_template_btn.setEnabled(is_pose)
        self.save_template_btn.setVisible(is_pose)
        self.save_template_btn.setEnabled(is_pose)
        self.inference_btn.setText("Run Inference")
        self.grid.addWidget(self.inference_btn, 0, 0 if is_depth else 1, 1, 2 if is_depth else 1)

    def set_model_status(self, text: str, *, tooltip: str = "") -> None:
        self.status_label.setText(str(text))
        self.status_label.setToolTip(str(tooltip))


__all__ = [
    "AnalysisOperationsPanel",
    "DatasetOperationsPanel",
    "ModelOperationsPanel",
    "OperationCallbacks",
    "VideoOperationsPanel",
]
