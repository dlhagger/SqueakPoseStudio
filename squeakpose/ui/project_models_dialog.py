"""Project-level prediction model configuration dialog."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from squeakpose.project.layers import (
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
)


class ProjectModelsDialog(QDialog):
    """Edit the prediction model assigned to each project layer."""

    def __init__(
        self,
        parent,
        model_paths: dict[str, str],
        *,
        active_layer: str = LAYER_KEYPOINTS,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Models")
        self.setMinimumWidth(760)
        self._paths = {
            layer_id: str(model_paths.get(layer_id) or "")
            for layer_id in (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
        }
        self._path_edits: dict[str, QLineEdit] = {}
        self._status_labels: dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("Project prediction models")
        title.setStyleSheet("font-size: 14pt; font-weight: 800;")
        root.addWidget(title)

        hint = QLabel(
            "Assign trained models to the Keypoints and Segmentation layers. "
            "Predict, video inference, and Video Reviewer automatically use the "
            "matching model. SAM and Depth remain separate labeling assistants."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aeb9c4;")
        root.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(QLabel("Layer"), 0, 0)
        grid.addWidget(QLabel("Prediction model"), 0, 1)
        grid.addWidget(QLabel("Expected"), 0, 2)
        grid.addWidget(QLabel("Status"), 0, 3)

        for row, layer_id in enumerate(
            (LAYER_KEYPOINTS, LAYER_SEGMENTATION), start=1
        ):
            layer = layer_definition(layer_id)
            layer_label = QLabel(layer.display_name)
            if layer_id == active_layer:
                layer_label.setText(f"{layer.display_name} · active")
                layer_label.setStyleSheet("font-weight: 800;")
            grid.addWidget(layer_label, row, 0)

            path_row = QHBoxLayout()
            path_row.setSpacing(6)
            path_edit = QLineEdit(self._paths[layer_id])
            path_edit.setReadOnly(True)
            path_edit.setPlaceholderText("Not configured")
            path_edit.setToolTip(self._paths[layer_id])
            self._path_edits[layer_id] = path_edit
            path_row.addWidget(path_edit, 1)

            choose_btn = QPushButton("Choose…")
            choose_btn.clicked.connect(
                lambda _checked=False, lid=layer_id: self._choose_model(lid)
            )
            path_row.addWidget(choose_btn)

            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(
                lambda _checked=False, lid=layer_id: self._set_path(lid, "")
            )
            path_row.addWidget(clear_btn)
            grid.addLayout(path_row, row, 1)

            grid.addWidget(QLabel(layer.model_task), row, 2)
            status = QLabel("")
            self._status_labels[layer_id] = status
            grid.addWidget(status, row, 3)
            self._refresh_status(layer_id)

        root.addLayout(grid)

        validation_note = QLabel(
            "The model task is verified when prediction starts. Pose and segmentation "
            "models can only run their matching project layer. Depth uses its separate "
            "assistant controls in the main window."
        )
        validation_note.setWordWrap(True)
        validation_note.setStyleSheet("color: #84909b; font-size: 9pt;")
        root.addWidget(validation_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(
            "Save Models"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @property
    def model_paths(self) -> dict[str, str]:
        return dict(self._paths)

    def _choose_model(self, layer_id: str) -> None:
        current = self._paths.get(layer_id, "")
        start_dir = os.path.dirname(current) if current else ""
        layer = layer_definition(layer_id)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {layer.display_name} prediction model",
            start_dir,
            "Model Files (*.pt *.yaml *.onnx)",
        )
        if path:
            self._set_path(layer_id, path)

    def _set_path(self, layer_id: str, path: str) -> None:
        normalized = str(path or "")
        if normalized:
            normalized = os.path.abspath(normalized)
        self._paths[layer_id] = normalized
        edit = self._path_edits[layer_id]
        edit.setText(normalized)
        edit.setToolTip(normalized)
        self._refresh_status(layer_id)

    def _refresh_status(self, layer_id: str) -> None:
        status = self._status_labels[layer_id]
        path = self._paths.get(layer_id, "")
        if not path:
            status.setText("Not configured")
            status.setStyleSheet("color: #aeb9c4;")
        elif not os.path.isfile(path):
            status.setText("File missing")
            status.setStyleSheet("color: #ff9b9b; font-weight: 700;")
        else:
            status.setText("Configured")
            status.setStyleSheet("color: #8fd3aa; font-weight: 700;")
