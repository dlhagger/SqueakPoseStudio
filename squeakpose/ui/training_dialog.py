"""YOLO training configuration and worker process dialog."""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from typing import Optional

import yaml
from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squeakpose.project.distillation import (
    discover_distillation_exports as _discover_distillation_exports,
    distillation_export_search_roots as _distillation_export_search_roots,
)
from squeakpose_core import atomic_write_text, infer_dataset_task
from layer_ops import layer_definition, normalize_layer_id
from squeakpose.workers.process import (
    remove_file_quietly as _remove_file_quietly,
    request_qprocess_stop,
    shutdown_qprocess as _shutdown_qprocess,
)
from squeakpose.workers.protocol import WorkerProtocolError, parse_event_line
from ui_style import (
    ThemedComboBox,
    style_combo_popup,
    train_dialog_stylesheet,
)

APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch as _torch
except Exception:
    _torch = None


def _auto_device() -> str:
    try:
        if _torch is not None:
            if hasattr(_torch, "cuda") and _torch.cuda.is_available():
                return "cuda"
            if hasattr(_torch, "backends") and hasattr(_torch.backends, "mps"):
                mps = _torch.backends.mps
                if (
                    getattr(mps, "is_built", lambda: False)()
                    and getattr(mps, "is_available", lambda: False)()
                ):
                    return "mps"
    except Exception:
        pass
    return "cpu"


def _refresh_qt_style(widget: QWidget | None) -> None:
    if widget is None:
        return
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class TrainDialog(QDialog):
    """Dialog for launching YOLO training in a child process."""

    ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    MODEL_OPTIONS = {
        "YOLOv26n (nano)": "yolo26n.yaml",
        "YOLOv26s (small)": "yolo26s.yaml",
        "YOLOv26m (medium)": "yolo26m.yaml",
        "YOLOv26l (large)": "yolo26l.yaml",
        "YOLOv26x (xlarge)": "yolo26x.yaml",
    }

    def __init__(
        self,
        parent,
        default_dataset: str,
        default_task: Optional[str] = None,
        layer_id: str = "",
    ):
        super().__init__(parent)
        self.layer_id = normalize_layer_id(layer_id or default_task)
        self.layer = layer_definition(self.layer_id)
        self.setWindowTitle(f"Train {self.layer.display_name} Layer Model")
        self.resize(1100, 720)
        self.setMinimumSize(760, 520)

        self.default_dataset = default_dataset
        self.default_task = (default_task or "").strip().lower() or None
        self.app_base_dir = os.path.abspath(getattr(parent, "app_base_dir", APP_BASE_DIR))
        self.project_root = os.path.abspath(getattr(parent, "project_root", self.app_base_dir))
        self.project_runs_dir = os.path.join(self.project_root, "runs")
        self.distillation_search_roots = _distillation_export_search_roots(self.project_root)
        os.makedirs(self.project_runs_dir, exist_ok=True)
        self.dino_exports: list[tuple[str, str]] = []
        self.dino_manual_path: Optional[str] = None
        self.resume_exports: list[tuple[str, str]] = []
        self.resume_manual_path: Optional[str] = None
        self.device = _auto_device()
        self.training_running = False
        self.train_process: Optional[QProcess] = None
        self.train_stdout_buffer = ""
        self.train_stderr_buffer = ""
        self.train_result_event: Optional[dict] = None
        self.train_config_path: Optional[str] = None
        self.train_cancel_requested = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        settings_panel = QFrame()
        settings_panel.setObjectName("TrainSettingsPanel")
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(12, 12, 12, 10)
        settings_layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("Training Setup")
        title.setObjectName("TrainPanelTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.train_status_label = QLabel("Idle")
        self.train_status_label.setObjectName("TrainStatusLabel")
        self.train_status_label.setProperty("tone", "idle")
        header.addWidget(self.train_status_label)
        settings_layout.addLayout(header)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)

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

        # Backbone source selection
        self.source_combo = ThemedComboBox()
        self.source_combo.addItem("Standard YOLO backbone")
        self.dino_source_index = self.source_combo.count()
        self.source_combo.addItem("DINO distillation export")
        self.checkpoint_source_index = self.source_combo.count()
        self.source_combo.addItem("Continue from YOLO checkpoint")
        self.resume_source_index = self.source_combo.count()
        self.source_combo.addItem("Resume YOLO run (exact)")
        self.source_combo.currentIndexChanged.connect(self._update_source_controls)
        form.addRow("Backbone source:", self.source_combo)

        # Model choice
        self.model_combo = ThemedComboBox()
        self.model_combo.addItems(self.MODEL_OPTIONS.keys())
        self.model_row = QWidget()
        model_layout = QHBoxLayout(self.model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_combo)
        self.model_form_label = QLabel("YOLO model:")
        form.addRow(self.model_form_label, self.model_row)

        # DINO export selection
        self.dino_row = QWidget()
        dino_layout = QVBoxLayout(self.dino_row)
        dino_layout.setContentsMargins(0, 0, 0, 0)
        dino_top = QHBoxLayout()
        self.dino_combo = ThemedComboBox()
        self.dino_combo.currentIndexChanged.connect(self._on_dino_combo_changed)
        dino_top.addWidget(self.dino_combo, 1)
        self.dino_refresh_btn = QPushButton("Refresh")
        self.dino_refresh_btn.clicked.connect(self._refresh_dino_list)
        dino_top.addWidget(self.dino_refresh_btn)
        self.dino_browse_btn = QPushButton("Browse…")
        self.dino_browse_btn.clicked.connect(self._browse_dino_file)
        dino_top.addWidget(self.dino_browse_btn)
        dino_layout.addLayout(dino_top)
        self.dino_path_edit = QLineEdit()
        self.dino_path_edit.setReadOnly(True)
        self.dino_path_edit.setPlaceholderText("No distillation export selected")
        dino_layout.addWidget(self.dino_path_edit)
        self.dino_form_label = QLabel("Distilled export:")
        form.addRow(self.dino_form_label, self.dino_row)
        self.dino_row.hide()
        self.dino_form_label.hide()
        self._refresh_dino_list()

        # Resume YOLO selection
        self.resume_row = QWidget()
        resume_layout = QVBoxLayout(self.resume_row)
        resume_layout.setContentsMargins(0, 0, 0, 0)
        resume_top = QHBoxLayout()
        self.resume_combo = ThemedComboBox()
        self.resume_combo.currentIndexChanged.connect(self._on_resume_combo_changed)
        resume_top.addWidget(self.resume_combo, 1)
        self.resume_refresh_btn = QPushButton("Refresh")
        self.resume_refresh_btn.clicked.connect(self._refresh_resume_list)
        resume_top.addWidget(self.resume_refresh_btn)
        self.resume_browse_btn = QPushButton("Browse…")
        self.resume_browse_btn.clicked.connect(self._browse_resume_file)
        resume_top.addWidget(self.resume_browse_btn)
        resume_layout.addLayout(resume_top)
        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setReadOnly(True)
        self.resume_path_edit.setPlaceholderText("No previous run selected")
        resume_layout.addWidget(self.resume_path_edit)
        self.resume_form_label = QLabel("Checkpoint:")
        form.addRow(self.resume_form_label, self.resume_row)
        self.resume_row.hide()
        self.resume_form_label.hide()
        self._refresh_resume_list()

        # Device info
        self.device_label = QLabel(self.device.upper())
        form.addRow("Device:", self.device_label)

        # Task selection
        self.task_combo = ThemedComboBox()
        if self.layer.id == "segmentation":
            self.task_combo.addItem("Segmentation")
        else:
            self.task_combo.addItem("Keypoints (YOLO Pose)")
        self.task_combo.setEnabled(False)
        self.task_combo.setToolTip(
            "Training task is determined by the active project layer."
        )
        form.addRow("Training task:", self.task_combo)
        train_combos = (
            self.source_combo,
            self.model_combo,
            self.dino_combo,
            self.resume_combo,
            self.task_combo,
        )
        for combo in train_combos:
            style_combo_popup(combo.view())

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
        self.batch_hint.setObjectName("TrainHintLabel")
        form.addRow("", self.batch_hint)

        settings_layout.addLayout(form)
        layout.addWidget(settings_panel, 0)

        output_panel = QFrame()
        output_panel.setObjectName("TrainOutputPanel")
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.setSpacing(8)
        output_header = QHBoxLayout()
        output_title = QLabel("Training Output")
        output_title.setObjectName("TrainPanelTitle")
        output_header.addWidget(output_title)
        output_header.addStretch(1)
        output_layout.addLayout(output_header)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("TrainLogView")
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Training output will appear here.")
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setMaximumBlockCount(12000)
        terminal_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        terminal_font.setPointSize(11)
        self.log_view.setFont(terminal_font)
        output_layout.addWidget(self.log_view, 1)
        layout.addWidget(output_panel, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)

        self.run_btn = QPushButton("Start Training")
        self.run_btn.clicked.connect(self._start_training)
        button_box.addButton(self.run_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self.cancel_train_btn = QPushButton("Cancel Training")
        self.cancel_train_btn.clicked.connect(self._cancel_training_process)
        self.cancel_train_btn.setEnabled(False)
        button_box.addButton(self.cancel_train_btn, QDialogButtonBox.ButtonRole.ActionRole)

        layout.addWidget(button_box)

        self.setStyleSheet(train_dialog_stylesheet())

        self._update_source_controls()
        self._configure_batch_controls()

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select dataset directory",
            self.dataset_edit.text() or self.default_dataset,
        )
        if path:
            self.dataset_edit.setText(path)

    def _update_source_controls(self):
        idx = self.source_combo.currentIndex()
        use_dino = idx == self.dino_source_index
        use_checkpoint_continue = idx == self.checkpoint_source_index
        use_exact_resume = idx == self.resume_source_index
        use_resume = use_checkpoint_continue or use_exact_resume
        self.model_row.setVisible(idx == 0)
        self.model_form_label.setVisible(idx == 0)
        self.dino_row.setVisible(use_dino)
        self.dino_form_label.setVisible(use_dino)
        self.resume_row.setVisible(use_resume)
        self.resume_form_label.setVisible(use_resume)
        if use_exact_resume:
            self.resume_path_edit.setPlaceholderText("Select weights/last.pt from a prior run")
        else:
            self.resume_path_edit.setPlaceholderText("No previous run selected")
        if use_dino and not self.dino_exports:
            self._refresh_dino_list()
        if use_resume and not self.resume_exports:
            self._refresh_resume_list()

    def _refresh_dino_list(self):
        self.dino_combo.blockSignals(True)
        self.dino_combo.clear()
        self.dino_combo.blockSignals(False)
        self.dino_exports = _discover_distillation_exports(
            getattr(self, "distillation_search_roots", []),
            task=self.layer.model_task,
        )
        exports = self.dino_exports
        if not exports:
            self.dino_combo.addItem(f"No {self.layer.display_name} exports found", "")
            self.dino_combo.setEnabled(False)
        else:
            self.dino_combo.setEnabled(True)
            for label, path in exports:
                self.dino_combo.addItem(label, path)
        self.dino_manual_path = None
        self._on_dino_combo_changed(self.dino_combo.currentIndex())

    def _on_dino_combo_changed(self, index: int):
        if self.dino_manual_path and index >= 0:
            # User selected a listed export → clear manual override
            self.dino_manual_path = None
        path = self.dino_combo.itemData(index) if index >= 0 else ""
        if not path:
            self.dino_path_edit.clear()
        else:
            self.dino_path_edit.setText(path)

    def _browse_dino_file(self):
        start_dir = next(
            (root for _label, root in getattr(self, "distillation_search_roots", []) if os.path.isdir(root)),
            os.getcwd(),
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select distillation checkpoint (.pt)",
            start_dir,
            "PyTorch weights (*.pt)",
        )
        if path:
            self.dino_manual_path = path
            self.dino_path_edit.setText(path)

    def _selected_dino_path(self) -> str:
        if self.dino_manual_path:
            return self.dino_manual_path
        idx = self.dino_combo.currentIndex()
        if idx < 0:
            return ""
        data = self.dino_combo.itemData(idx)
        return data or ""

    def _refresh_resume_list(self):
        self.resume_combo.blockSignals(True)
        self.resume_combo.clear()
        self.resume_combo.blockSignals(False)
        exports: list[tuple[str, str]] = []
        runs_root = getattr(self, "project_runs_dir", "")
        if runs_root and os.path.isdir(runs_root):
            try:
                for dirpath, _, _ in os.walk(runs_root):
                    if "weights" not in dirpath:
                        continue
                    for name in ("last.pt", "best.pt"):
                        candidate = os.path.join(dirpath, name)
                        if os.path.isfile(candidate):
                            label = os.path.relpath(candidate, runs_root)
                            exports.append((label, candidate))
                exports.sort(key=lambda pair: os.path.getmtime(pair[1]), reverse=True)
            except Exception:
                exports = []
        self.resume_exports = exports
        if not exports:
            self.resume_combo.addItem("No checkpoints found", "")
            self.resume_combo.setEnabled(False)
        else:
            self.resume_combo.setEnabled(True)
            for label, path in exports:
                self.resume_combo.addItem(label, path)
        self.resume_manual_path = None
        self._on_resume_combo_changed(self.resume_combo.currentIndex())

    def _on_resume_combo_changed(self, index: int):
        if self.resume_manual_path and index >= 0:
            self.resume_manual_path = None
        path = self.resume_combo.itemData(index) if index >= 0 else ""
        if not path:
            self.resume_path_edit.clear()
        else:
            self.resume_path_edit.setText(path)

    def _browse_resume_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO checkpoint (.pt)",
            self.project_runs_dir if os.path.isdir(self.project_runs_dir) else os.getcwd(),
            "PyTorch weights (*.pt)",
        )
        if path:
            self.resume_manual_path = path
            self.resume_path_edit.setText(path)
            self.resume_combo.setCurrentIndex(-1)

    def _selected_resume_path(self) -> str:
        if self.resume_manual_path:
            return self.resume_manual_path
        idx = self.resume_combo.currentIndex()
        if idx < 0:
            return ""
        data = self.resume_combo.itemData(idx)
        return data or ""

    def _run_name_from_model(self, model_spec: str, use_dino: bool) -> str:
        if use_dino or model_spec.lower().endswith((".pt", ".pth", ".yaml", ".yml")):
            base = os.path.splitext(os.path.basename(model_spec))[0]
        else:
            base = os.path.basename(model_spec)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
        return safe or "model"

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

    def _set_training_status(self, text: str, tone: str = "idle"):
        self.train_status_label.setText(text)
        tone = tone if tone in {"idle", "running", "complete", "failed", "canceled"} else "idle"
        self.train_status_label.setProperty("tone", tone)
        _refresh_qt_style(self.train_status_label)

    def _clean_training_output(self, text: str) -> str:
        cleaned = self.ANSI_ESCAPE_RE.sub("", text)
        cleaned = cleaned.replace("\x08", "")
        return cleaned.replace("\r", "\n").replace("\x1b", "")

    def _write_training_terminal_output(self, text: str):
        cleaned = self._clean_training_output(text)
        if cleaned:
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)
            self.log_view.insertPlainText(cleaned)
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)
            self.log_view.ensureCursorVisible()
        QApplication.processEvents()

    def _flush_training_terminal_output(self):
        self.log_view.ensureCursorVisible()

    def _log(self, message: str):
        cleaned = self._clean_training_output(str(message))
        if not cleaned:
            return
        self.log_view.appendPlainText(cleaned.rstrip())
        self.log_view.ensureCursorVisible()
        QApplication.processEvents()

    def closeEvent(self, event):
        if self.training_running:
            answer = QMessageBox.question(
                self,
                "Cancel training?",
                "Training is still running. Cancel it and close this dialog?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            _shutdown_qprocess(self.train_process)
            _remove_file_quietly(self.train_config_path)
            self.train_process = None
            self.train_config_path = None
            self.training_running = False
        super().closeEvent(event)

    def _resolve_model_config(self, base_cfg: str, task_value: Optional[str]) -> tuple[str, Optional[str]]:
        cfg = base_cfg
        notice = None
        if not task_value:
            return cfg, notice

        has_yaml_ext = cfg.lower().endswith(".yaml")
        stem = cfg[:-5] if has_yaml_ext else cfg
        stem_clean = re.sub(r"-(pose|seg)$", "", stem, flags=re.IGNORECASE)

        if task_value == "pose":
            target = f"{stem_clean}-pose"
            cfg = f"{target}.yaml" if has_yaml_ext else target
            if cfg != base_cfg:
                notice = "Pose task detected → switched to pose variant of the model config."
        elif task_value == "segment":
            target = f"{stem_clean}-seg"
            cfg = f"{target}.yaml" if has_yaml_ext else target
            if cfg != base_cfg:
                notice = "Segmentation task detected → switched to segmentation variant of the model config."
        elif task_value == "detect":
            cfg = f"{stem_clean}.yaml" if has_yaml_ext else stem_clean
            if cfg != base_cfg:
                notice = "Detection task selected → using detection variant of the model config."
        return cfg, notice

    def _infer_task_from_yaml(self, yaml_path: str) -> Optional[str]:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return None
        return infer_dataset_task(data)

    def _start_training(self):
        if self.training_running:
            QMessageBox.information(self, "Training running", "A training session is already in progress.")
            return

        source_idx = self.source_combo.currentIndex()
        use_dino = source_idx == self.dino_source_index
        use_checkpoint_continue = source_idx == self.checkpoint_source_index
        use_exact_resume = source_idx == self.resume_source_index

        resolved: Optional[str] = None
        if not use_exact_resume:
            dataset_path = self.dataset_edit.text().strip()
            if not dataset_path:
                QMessageBox.warning(self, "Dataset required", "Select a dataset folder before starting training.")
                return
            if os.path.isdir(dataset_path):
                data_yaml = os.path.join(dataset_path, "dataset.yaml")
                if os.path.isfile(data_yaml):
                    resolved = data_yaml
                else:
                    QMessageBox.warning(
                        self,
                        "dataset.yaml missing",
                        "Could not find dataset.yaml in the selected folder.\n"
                        "Select the dataset root (contains dataset.yaml) or the YAML file directly."
                    )
                    return
            elif dataset_path.lower().endswith((".yaml", ".yml")) and os.path.isfile(dataset_path):
                resolved = dataset_path
            else:
                QMessageBox.warning(self, "Invalid dataset", f"Path not found:\n{dataset_path}")
                return

        model_label = self.model_combo.currentText()
        base_model_cfg = self.MODEL_OPTIONS[model_label]
        epochs = self.epoch_spin.value()
        batch = self.batch_spin.value()
        batch_display = "auto" if batch <= 0 else str(batch)
        distilled_path = ""
        resume_path = ""
        if use_dino:
            distilled_path = self._selected_dino_path()
            if not distilled_path or not os.path.isfile(distilled_path):
                QMessageBox.warning(
                    self,
                    "Checkpoint required",
                    "Select a valid DINO distillation export (.pt) before training."
                )
                return
        elif use_checkpoint_continue or use_exact_resume:
            resume_path = self._selected_resume_path()
            if not resume_path or not os.path.isfile(resume_path):
                QMessageBox.warning(
                    self,
                    "Checkpoint required",
                    "Select a valid YOLO checkpoint (.pt) before continuing."
                )
                return
            if use_exact_resume and os.path.basename(resume_path).lower() != "last.pt":
                QMessageBox.warning(
                    self,
                    "Exact resume requires last.pt",
                    "For exact run continuation, select a weights/last.pt checkpoint."
                )
                return

        if (not use_exact_resume) and self.device == 'mps' and batch <= 0:
            QMessageBox.warning(
                self,
                "Batch size required",
                "Automatic batch sizing is unavailable on Apple MPS.\n"
                "Set a positive batch size before starting training."
            )
            return

        task_selection = self.task_combo.currentText()
        if use_dino:
            task_value = self.layer.model_task
        elif use_exact_resume:
            task_value = None
        elif task_selection.startswith("Auto"):
            inferred_task = self._infer_task_from_yaml(resolved) if resolved else None
            if inferred_task in {"pose", "detect", "segment"}:
                task_value = inferred_task
            elif self.default_task in {"pose", "detect", "segment"}:
                task_value = self.default_task
            else:
                task_value = None
        elif task_selection.startswith("Detection"):
            task_value = "detect"
        elif task_selection.startswith("Segmentation"):
            task_value = "segment"
        else:
            task_value = "pose"

        dataset_task = self._infer_task_from_yaml(resolved) if resolved else None
        if task_value and dataset_task and task_value != dataset_task:
            QMessageBox.warning(
                self,
                "Dataset Task Mismatch",
                f"The selected dataset is '{dataset_task}', but the training task is '{task_value}'.\n\n"
                "Choose the matching task or select a different dataset.",
            )
            return

        model_cfg = (
            distilled_path
            if use_dino
            else (resume_path if (use_checkpoint_continue or use_exact_resume) else base_model_cfg)
        )
        cfg_notice = None
        self.log_view.clear()
        self._set_training_status("Preparing", "running")
        if not (use_dino or use_checkpoint_continue or use_exact_resume):
            model_cfg, cfg_notice = self._resolve_model_config(base_model_cfg, task_value)
            if cfg_notice:
                self._log(cfg_notice)

        if use_dino:
            self._log(f"Starting training from DINO export: {model_cfg}")
        elif use_checkpoint_continue:
            self._log(f"Continuing training from checkpoint: {model_cfg}")
            self._log("- mode: checkpoint fine-tune (uses selected dataset and settings)")
        elif use_exact_resume:
            self._log(f"Resuming exact run from checkpoint: {model_cfg}")
            self._log("- mode: exact resume (uses prior run args/state)")
        else:
            self._log(f"Starting training for {model_label} ({model_cfg})")
        if resolved:
            self._log(f"- dataset: {resolved}")
        else:
            self._log("- dataset: from resume checkpoint")
        self._log(f"- device: {self.device}")
        if use_exact_resume:
            self._log("- epochs: from resume checkpoint")
            self._log("- batch size: from resume checkpoint")
        else:
            self._log(f"- epochs: {epochs}")
            self._log(f"- batch size: {batch_display}")
        if task_value:
            self._log(f"- task: {task_value}")
        self._log("Running training in a child process.")
        self._log("")

        if use_exact_resume:
            params = {
                "resume": True,
                "device": self.device,
            }
        else:
            batch_param = -1 if batch <= 0 else int(batch)

            task_folder = task_value if task_value in ("pose", "detect", "segment") else ("pose" if use_dino else "auto")
            project_dir = os.path.join(self.project_runs_dir, "train", task_folder)
            try:
                os.makedirs(project_dir, exist_ok=True)
            except Exception as e:
                self._log(f"Warning: could not create runs directory at {project_dir}: {e}")

            if use_checkpoint_continue:
                checkpoint_run = os.path.basename(os.path.dirname(os.path.dirname(model_cfg)))
                run_name = self._run_name_from_model(checkpoint_run or model_cfg, use_dino=True)
                if not run_name.endswith("_continue"):
                    run_name = f"{run_name}_continue"
            else:
                run_name = self._run_name_from_model(model_cfg, use_dino)

            params = {
                "data": resolved,
                "epochs": epochs,
                "device": self.device,
                "exist_ok": False,
                "batch": batch_param,
                "project": project_dir,
                "name": run_name,
            }
            if task_value:
                params["task"] = task_value

        self._start_training_process(model_cfg=model_cfg, params=params)

    def _start_training_process(self, *, model_cfg: str, params: dict):
        if self.train_process is not None and self.train_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Training running", "A training session is already in progress.")
            return

        config = {
            "layer_id": self.layer_id,
            "model_cfg": model_cfg,
            "params": params,
        }
        run_root = os.path.join(self.project_runs_dir, "train")
        try:
            os.makedirs(run_root, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Training setup error", f"Could not create training run directory:\n{e}")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        config_path = os.path.join(run_root, f".train_config_{timestamp}.json")
        try:
            atomic_write_text(config_path, json.dumps(config, indent=2))
        except Exception as e:
            QMessageBox.warning(self, "Training setup error", f"Could not write training config:\n{config_path}\n\n{e}")
            return

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "train_worker", "--config", config_path])
        process.setWorkingDirectory(self.app_base_dir)
        process.readyReadStandardOutput.connect(self._read_training_process_stdout)
        process.readyReadStandardError.connect(self._read_training_process_stderr)
        process.finished.connect(self._finish_training_process)
        process.errorOccurred.connect(self._handle_training_process_error)

        self.train_process = process
        self.train_stdout_buffer = ""
        self.train_stderr_buffer = ""
        self.train_result_event = None
        self.train_config_path = config_path
        self.train_cancel_requested = False
        self.training_running = True
        self.run_btn.setEnabled(False)
        self.cancel_train_btn.setEnabled(True)
        self._set_training_status("Launching", "running")

        self._log("Launching training worker process...")
        process.start()
        if not process.waitForStarted(1000):
            self.train_stderr_buffer = process.errorString()
            self._finish_training_process(1, QProcess.ExitStatus.CrashExit)
            return

    def _read_training_process_stdout(self):
        process = self.train_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self.train_stdout_buffer += text
        lines = self.train_stdout_buffer.splitlines(keepends=True)
        self.train_stdout_buffer = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                self._handle_training_event_line(line.strip())
            else:
                self.train_stdout_buffer = line

    def _read_training_process_stderr(self):
        process = self.train_process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        if not text:
            return
        self.train_stderr_buffer += text
        self._write_training_terminal_output(text)

    def _handle_training_event_line(self, line: str):
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self._log(line)
            return
        event_type = event.get("event")
        if event_type == "started":
            self._log(f"Training worker loaded config: {event.get('model_cfg', '')}")
            self._set_training_status("Loading", "running")
        elif event_type == "training":
            self._log(str(event.get("message") or "Training started"))
            self._set_training_status("Running", "running")
        elif event_type == "result":
            self.train_result_event = event
        elif event_type == "error":
            self.train_result_event = {
                "event": "result",
                "canceled": False,
                "had_error": True,
                "error_message": str(event.get("error_message") or "Training worker error"),
                "save_dir": "",
            }

    def _cancel_training_process(self):
        process = self.train_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        self.train_cancel_requested = True
        self._set_training_status("Canceling", "canceled")
        self._log("Cancel requested. Stopping training worker process...")
        request_qprocess_stop(
            process,
            schedule=QTimer.singleShot,
            force_kill=self._kill_training_process_if_running,
            kill_after_ms=5000,
        )

    def _kill_training_process_if_running(self):
        process = self.train_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._log("Training worker did not stop after terminate; killing process.")
            process.kill()

    def _handle_training_process_error(self, _error):
        process = self.train_process
        if process is not None:
            self.train_stderr_buffer += process.errorString() + "\n"

    def _finish_training_process(self, exit_code: int, exit_status):
        if self.train_process is None and self.train_config_path is None:
            return
        self._flush_training_terminal_output()
        if self.train_stdout_buffer.strip():
            self._handle_training_event_line(self.train_stdout_buffer.strip())
            self.train_stdout_buffer = ""

        config_path = self.train_config_path
        _remove_file_quietly(config_path)

        event = self.train_result_event
        stderr_text = self.train_stderr_buffer.strip()
        cancel_requested = self.train_cancel_requested

        self.training_running = False
        self.run_btn.setEnabled(True)
        self.cancel_train_btn.setEnabled(False)
        self.train_process = None
        self.train_config_path = None
        self.train_result_event = None
        self.train_stdout_buffer = ""
        self.train_stderr_buffer = ""
        self.train_cancel_requested = False

        if cancel_requested and event is None:
            self._set_training_status("Canceled", "canceled")
            self._log("Training canceled.")
            QMessageBox.information(self, "Training canceled", "Training worker process was canceled.")
            return

        if event is None:
            detail = stderr_text or f"Process exited with code {exit_code}."
            self._set_training_status("Failed", "failed")
            self._log(f"Training worker failed: {detail}")
            QMessageBox.critical(self, "Training error", f"Training worker failed:\n{detail}")
            return

        had_error = bool(event.get("had_error"))
        canceled = bool(event.get("canceled")) or cancel_requested
        save_dir = str(event.get("save_dir") or "")
        error_message = str(event.get("error_message") or stderr_text or "Unknown training error")

        if canceled and not had_error:
            self._set_training_status("Canceled", "canceled")
            self._log("Training canceled.")
            QMessageBox.information(self, "Training canceled", "Training was canceled.")
            return

        if had_error or exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0:
            self._set_training_status("Failed", "failed")
            self._log(f"Training failed: {error_message}")
            QMessageBox.critical(self, "Training error", f"Training failed:\n{error_message}")
            return

        self._set_training_status("Complete", "complete")
        if save_dir:
            self._log(f"Training complete. Artifacts saved to: {save_dir}")
        else:
            self._log("Training complete.")
        QMessageBox.information(
            self,
            "Training complete",
            "YOLO training finished. Review the logs for metrics.",
        )
