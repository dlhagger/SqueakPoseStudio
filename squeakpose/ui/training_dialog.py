"""YOLO training configuration and worker process dialog."""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from squeakpose.project.distillation import (
    discover_distillation_exports as _discover_distillation_exports,
)
from squeakpose.project.distillation import (
    distillation_export_search_roots as _distillation_export_search_roots,
)
from squeakpose.project.layers import layer_definition, normalize_layer_id
from squeakpose.services.training import (
    TrainingConfigError,
    TrainingConsoleBuffer,
    build_training_run_plan,
    build_training_worker_config,
    infer_training_task_from_yaml,
    resolve_model_config,
    training_run_name,
)
from squeakpose.ui.style import (
    ThemedComboBox,
    style_combo_popup,
    train_dialog_stylesheet,
)
from squeakpose.workers.process import (
    WorkerJobController,
    WorkerJobResult,
    create_worker_config,
)
from squeakpose.workers.process import (
    remove_file_quietly as _remove_file_quietly,
)
from squeakpose.workers.process import (
    shutdown_qprocess as _shutdown_qprocess,
)
from squeakpose.workers.protocol import WorkerProtocolError, parse_event_line

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
        self.training_controller: Optional[WorkerJobController] = None
        self.train_process: Optional[QProcess] = None
        self.train_stdout_buffer = ""
        self.train_stderr_buffer = ""
        self.train_result_event: Optional[dict] = None
        self.train_config_path: Optional[str] = None
        self.train_cancel_requested = False
        self.training_console = TrainingConsoleBuffer()

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
        self.task_combo.setToolTip("Training task is determined by the active project layer.")
        form.addRow("Training task:", self.task_combo)

        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText("Optional — generated from the selected model")
        self.run_name_edit.setToolTip(
            "Passed to Ultralytics as name=. Spaces and path-unsafe punctuation are normalized; "
            "an existing name is automatically numbered rather than overwritten."
        )
        form.addRow("Run name:", self.run_name_edit)
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
        output_title = QLabel("Training Monitor")
        output_title.setObjectName("TrainPanelTitle")
        output_header.addWidget(output_title)
        output_header.addStretch(1)
        output_layout.addLayout(output_header)

        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("TrainOutputTabs")

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(8, 8, 8, 8)
        overview_layout.setSpacing(8)

        monitor_header = QHBoxLayout()
        self.phase_label = QLabel("Ready")
        self.phase_label.setObjectName("TrainPhaseLabel")
        monitor_header.addWidget(self.phase_label)
        monitor_header.addStretch(1)
        self.epoch_label = QLabel("Epoch — / —")
        self.epoch_label.setObjectName("TrainEpochLabel")
        monitor_header.addWidget(self.epoch_label)
        self.eta_label = QLabel("ETA —")
        self.eta_label.setObjectName("TrainEtaLabel")
        monitor_header.addWidget(self.eta_label)
        overview_layout.addLayout(monitor_header)

        progress_grid = QGridLayout()
        progress_grid.setHorizontalSpacing(8)
        progress_grid.setVerticalSpacing(5)
        progress_grid.addWidget(QLabel("Overall"), 0, 0)
        self.overall_progress = QProgressBar()
        self.overall_progress.setObjectName("TrainOverallProgress")
        self.overall_progress.setRange(0, 1000)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("Waiting to start")
        progress_grid.addWidget(self.overall_progress, 0, 1)
        progress_grid.addWidget(QLabel("Current epoch"), 1, 0)
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setObjectName("TrainEpochProgress")
        self.epoch_progress.setRange(0, 1000)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Waiting for batches")
        progress_grid.addWidget(self.epoch_progress, 1, 1)
        overview_layout.addLayout(progress_grid)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(8)
        self.metric_values: dict[str, QLabel] = {}
        primary_metric_title = (
            "Mask mAP50–95" if self.layer.model_task == "segment" else "Pose mAP50–95"
        )
        for key, title in (
            ("primary_map", primary_metric_title),
            ("map50", "mAP50"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("loss", "Train loss"),
        ):
            card = QFrame()
            card.setObjectName("TrainMetricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(9, 7, 9, 7)
            card_layout.setSpacing(2)
            caption = QLabel(title)
            caption.setObjectName("TrainMetricCaption")
            value = QLabel("—")
            value.setObjectName("TrainMetricValue")
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            cards_layout.addWidget(card, 1)
            self.metric_values[key] = value
        overview_layout.addLayout(cards_layout)
        self.loss_detail_label = QLabel("Loss components will appear when training begins.")
        self.loss_detail_label.setObjectName("TrainLossDetail")
        self.loss_detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        overview_layout.addWidget(self.loss_detail_label)

        history_title = QLabel("Epoch history")
        history_title.setObjectName("TrainHistoryTitle")
        overview_layout.addWidget(history_title)
        self.history_table = QTableWidget(0, 8)
        self.history_table.setObjectName("TrainHistoryTable")
        self.history_table.setHorizontalHeaderLabels(
            ["Epoch", "Time", "Loss", "Precision", "Recall", "mAP50", "mAP50–95", "Best"]
        )
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setAlternatingRowColors(False)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setMinimumHeight(150)
        overview_layout.addWidget(self.history_table, 1)

        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        console_layout.setContentsMargins(6, 6, 6, 6)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("TrainLogView")
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Training output will appear here.")
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setMaximumBlockCount(12000)
        terminal_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        terminal_font.setPointSize(11)
        self.log_view.setFont(terminal_font)
        console_layout.addWidget(self.log_view)

        self.output_tabs.addTab(overview_tab, "Overview")
        self.output_tabs.addTab(console_tab, "Console")
        output_layout.addWidget(self.output_tabs, 1)
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
        self.run_name_edit.setEnabled(not use_exact_resume)
        self.run_name_edit.setPlaceholderText(
            "Inherited from the resumed run"
            if use_exact_resume
            else "Optional — generated from the selected model"
        )
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
            (
                root
                for _label, root in getattr(self, "distillation_search_roots", [])
                if os.path.isdir(root)
            ),
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
        del use_dino  # retained in the compatibility signature
        return training_run_name(model_spec)

    def _configure_batch_controls(self):
        if self.device == "cuda":
            self.batch_spin.setValue(0)
            self.batch_spin.setEnabled(False)
            self.batch_hint.setText("CUDA detected → using automatic batch sizing.")
        elif self.device == "mps":
            default = max(1, self.batch_spin.value() or 16)
            self.batch_spin.setValue(default)
            self.batch_spin.setEnabled(True)
            self.batch_hint.setText("MPS detected → choose a manual batch size that fits memory.")
        else:
            default = self.batch_spin.value() or 16
            self.batch_spin.setValue(default)
            self.batch_spin.setEnabled(True)
            self.batch_hint.setText(
                "CPU detected → adjust batch size as needed (lower values use less memory)."
            )

    def _set_training_status(self, text: str, tone: str = "idle"):
        self.train_status_label.setText(text)
        tone = tone if tone in {"idle", "running", "complete", "failed", "canceled"} else "idle"
        self.train_status_label.setProperty("tone", tone)
        _refresh_qt_style(self.train_status_label)

    @staticmethod
    def _format_duration(seconds) -> str:
        try:
            total = max(0, int(float(seconds)))
        except (TypeError, ValueError):
            return "—"
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    @staticmethod
    def _set_fractional_progress(bar: QProgressBar, current: int, total: int, label: str):
        current_value = max(0, int(current or 0))
        total_value = max(0, int(total or 0))
        fraction = min(1.0, current_value / total_value) if total_value else 0.0
        bar.setValue(round(fraction * 1000))
        bar.setFormat(label)

    def _reset_training_monitor(self, epochs: int = 0):
        self.phase_label.setText("Preparing training")
        self.epoch_label.setText(f"Epoch — / {epochs}" if epochs else "Epoch — / —")
        self.eta_label.setText("ETA —")
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("0%")
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Waiting for batches")
        for label in self.metric_values.values():
            label.setText("—")
        self.loss_detail_label.setText("Loss components will appear when training begins.")
        self.history_table.setRowCount(0)
        self.training_console = TrainingConsoleBuffer()

    @staticmethod
    def _numeric_mapping(payload) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        result = {}
        for key, value in payload.items():
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    def _preferred_metric(self, metrics: dict[str, float], fragment: str) -> float | None:
        matches = [
            (key, value)
            for key, value in metrics.items()
            if fragment in key.lower() and not (fragment == "map50" and "map50-95" in key.lower())
        ]
        if not matches:
            return None
        preferred_suffix = "(m)" if self.layer.model_task == "segment" else "(p)"
        for key, value in matches:
            if preferred_suffix in key.lower():
                return value
        for key, value in matches:
            if "(b)" in key.lower():
                return value
        return matches[0][1]

    @staticmethod
    def _loss_total(losses: dict[str, float]) -> float | None:
        values = [value for key, value in losses.items() if "loss" in key.lower()]
        if not values:
            values = list(losses.values())
        return sum(values) if values else None

    @staticmethod
    def _metric_text(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    def _update_metric_cards(
        self,
        losses: dict[str, float],
        metrics: dict[str, float],
        *,
        memory_gb=None,
    ):
        primary_map = self._preferred_metric(metrics, "map50-95")
        map50 = self._preferred_metric(metrics, "map50")
        precision = self._preferred_metric(metrics, "precision")
        recall = self._preferred_metric(metrics, "recall")
        loss = self._loss_total(losses)
        for key, value in (
            ("primary_map", primary_map),
            ("map50", map50),
            ("precision", precision),
            ("recall", recall),
            ("loss", loss),
        ):
            if value is not None:
                self.metric_values[key].setText(self._metric_text(value))
        details = [
            f"{key.removeprefix('train/').replace('_', ' ')} {value:.4f}"
            for key, value in losses.items()
        ]
        try:
            if memory_gb is not None:
                details.append(f"device memory {float(memory_gb):.2f} GB")
        except (TypeError, ValueError):
            pass
        if details:
            self.loss_detail_label.setText("  •  ".join(details))

    def _append_epoch_history(self, event: dict):
        losses = self._numeric_mapping(event.get("losses"))
        metrics = self._numeric_mapping(event.get("metrics"))
        loss = self._loss_total(losses)
        precision = self._preferred_metric(metrics, "precision")
        recall = self._preferred_metric(metrics, "recall")
        map50 = self._preferred_metric(metrics, "map50")
        primary_map = self._preferred_metric(metrics, "map50-95")
        best = event.get("best_fitness")
        try:
            best_text = f"{float(best):.4f}" if best is not None else "—"
        except (TypeError, ValueError):
            best_text = "—"
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        values = (
            str(event.get("epoch") or "—"),
            self._format_duration(event.get("epoch_seconds")),
            self._metric_text(loss),
            self._metric_text(precision),
            self._metric_text(recall),
            self._metric_text(map50),
            self._metric_text(primary_map),
            best_text,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row, column, item)
        self.history_table.scrollToBottom()

    def _update_training_monitor(self, event: dict):
        event_type = str(event.get("event") or "")
        epoch = max(0, int(event.get("epoch") or 0))
        epochs = max(epoch, int(event.get("epochs") or epoch))
        if event_type == "training_setup":
            self.phase_label.setText("Initializing dataloaders and optimizer")
            self.epoch_label.setText(f"Epoch {epoch} / {epochs}")
            return
        if event_type == "epoch_start":
            self.phase_label.setText("Training")
            self.epoch_label.setText(f"Epoch {epoch} / {epochs}")
            self._set_fractional_progress(
                self.overall_progress,
                max(0, epoch - 1),
                epochs,
                f"{max(0, epoch - 1)}/{epochs} epochs complete",
            )
            self.epoch_progress.setValue(0)
            self.epoch_progress.setFormat("Starting epoch")
            return
        if event_type == "batch_progress":
            batch = max(0, int(event.get("batch") or 0))
            batches = max(batch, int(event.get("batches") or batch))
            self.phase_label.setText("Training batches")
            self.epoch_label.setText(f"Epoch {epoch} / {epochs}")
            self._set_fractional_progress(
                self.epoch_progress,
                batch,
                batches,
                f"{batch}/{batches} batches" if batches else f"Batch {batch}",
            )
            losses = self._numeric_mapping(event.get("losses"))
            self._update_metric_cards(losses, {}, memory_gb=event.get("memory_gb"))
            eta = event.get("eta_seconds")
            self.eta_label.setText(f"Epoch ETA {self._format_duration(eta)}")
            return
        if event_type == "epoch_end":
            self.phase_label.setText("Validation complete")
            self.epoch_label.setText(f"Epoch {epoch} / {epochs}")
            self._set_fractional_progress(
                self.overall_progress,
                epoch,
                epochs,
                f"{epoch}/{epochs} epochs complete",
            )
            self.epoch_progress.setValue(1000)
            self.epoch_progress.setFormat("Epoch complete")
            self.eta_label.setText(
                f"Training ETA {self._format_duration(event.get('eta_seconds'))}"
            )
            losses = self._numeric_mapping(event.get("losses"))
            metrics = self._numeric_mapping(event.get("metrics"))
            self._update_metric_cards(losses, metrics)
            self._append_epoch_history(event)

    def _clean_training_output(self, text: str) -> str:
        cleaned = self.ANSI_ESCAPE_RE.sub("", text)
        cleaned = cleaned.replace("\x08", "")
        return cleaned.replace("\r", "\n").replace("\x1b", "")

    def _write_training_terminal_output(self, text: str):
        for line in self.training_console.feed(text):
            self.log_view.appendPlainText(line)
        self.log_view.ensureCursorVisible()
        QApplication.processEvents()

    def _flush_training_terminal_output(self):
        for line in self.training_console.finish():
            self.log_view.appendPlainText(line)
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
            if self.training_controller is not None:
                try:
                    self.training_controller.terminal.disconnect(self._finish_training_job)
                except (TypeError, RuntimeError):
                    pass
                self.training_controller.shutdown()
                self.training_controller.deleteLater()
                self.training_controller = None
            else:
                _shutdown_qprocess(self.train_process)
                _remove_file_quietly(self.train_config_path)
            self.train_process = None
            self.train_config_path = None
            self.training_running = False
        super().closeEvent(event)

    def _resolve_model_config(
        self, base_cfg: str, task_value: Optional[str]
    ) -> tuple[str, Optional[str]]:
        return resolve_model_config(base_cfg, task_value)

    def _infer_task_from_yaml(self, yaml_path: str) -> Optional[str]:
        return infer_training_task_from_yaml(yaml_path)

    def _start_training(self):
        if self.training_running:
            QMessageBox.information(
                self, "Training running", "A training session is already in progress."
            )
            return

        source_idx = self.source_combo.currentIndex()
        use_dino = source_idx == self.dino_source_index
        use_checkpoint_continue = source_idx == self.checkpoint_source_index
        use_exact_resume = source_idx == self.resume_source_index

        model_label = self.model_combo.currentText()
        base_model_cfg = self.MODEL_OPTIONS[model_label]
        epochs = self.epoch_spin.value()
        batch = self.batch_spin.value()
        batch_display = "auto" if batch <= 0 else str(batch)
        checkpoint_path = ""
        if use_dino:
            checkpoint_path = self._selected_dino_path()
        elif use_checkpoint_continue or use_exact_resume:
            checkpoint_path = self._selected_resume_path()

        task_selection = self.task_combo.currentText()
        selected_task = "auto"
        if task_selection.startswith("Detection"):
            selected_task = "detect"
        elif task_selection.startswith("Segmentation"):
            selected_task = "segment"
        elif not task_selection.startswith("Auto"):
            selected_task = "pose"
        source_mode = (
            "dino"
            if use_dino
            else "checkpoint"
            if use_checkpoint_continue
            else "resume"
            if use_exact_resume
            else "scratch"
        )
        try:
            plan = build_training_run_plan(
                source_mode=source_mode,
                dataset_path=self.dataset_edit.text(),
                base_model_cfg=base_model_cfg,
                checkpoint_path=checkpoint_path,
                selected_task=selected_task,
                default_task=self.default_task,
                layer_task=self.layer.model_task,
                device=self.device,
                epochs=epochs,
                batch=batch,
                project_runs_dir=self.project_runs_dir,
                run_name=self.run_name_edit.text(),
            )
        except TrainingConfigError as exc:
            title = {
                "required": "Dataset required",
                "yaml_missing": "dataset.yaml missing",
                "checkpoint_required": "Checkpoint required",
                "resume_checkpoint": "Exact resume requires last.pt",
                "mps_batch": "Batch size required",
                "task_mismatch": "Dataset Task Mismatch",
                "run_name": "Run name required",
            }.get(exc.code, "Invalid training configuration")
            QMessageBox.warning(self, title, str(exc))
            return

        resolved = plan.dataset_yaml
        task_value = plan.task
        model_cfg = plan.model_cfg
        self.log_view.clear()
        self._reset_training_monitor(0 if use_exact_resume else epochs)
        self.output_tabs.setCurrentIndex(0)
        self._set_training_status("Preparing", "running")
        if plan.model_notice:
            self._log(plan.model_notice)

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
        if not use_exact_resume:
            self._log(f"- run name: {plan.params.get('name', '')}")
        self._log("Running training in a child process.")
        self._log("")

        params = plan.params
        if not use_exact_resume:
            project_dir = str(params["project"])
            try:
                os.makedirs(project_dir, exist_ok=True)
            except Exception as e:
                self._log(f"Warning: could not create runs directory at {project_dir}: {e}")

        self._start_training_process(model_cfg=model_cfg, params=params)

    def _start_training_process(self, *, model_cfg: str, params: dict):
        if self.training_controller is not None and self.training_controller.is_running:
            QMessageBox.information(
                self, "Training running", "A training session is already in progress."
            )
            return

        config = build_training_worker_config(
            layer_id=self.layer_id,
            model_cfg=model_cfg,
            params=params,
        ).as_dict()
        run_root = os.path.join(self.project_runs_dir, "train")
        try:
            os.makedirs(run_root, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(
                self, "Training setup error", f"Could not create training run directory:\n{e}"
            )
            return
        try:
            config_path = create_worker_config(
                self.project_root,
                run_root,
                "train",
                config,
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Training setup error",
                f"Could not write the training worker configuration.\n\n{e}",
            )
            return

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
        controller = WorkerJobController(self)
        self.training_controller = controller
        controller.event_received.connect(self._handle_training_event)
        controller.output_received.connect(self._log)
        controller.stderr_received.connect(self._write_training_terminal_output)
        controller.terminal.connect(self._finish_training_job)
        started = controller.start(
            sys.executable,
            ["-m", "train_worker", "--config", config_path],
            config_path=config_path,
            working_directory=self.app_base_dir,
            start_timeout_ms=1000,
        )
        self.train_process = controller.process if controller.terminal_result is None else None
        if not started:
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
        self._handle_training_event(event)

    def _handle_training_event(self, event: dict):
        event_type = event.get("event")
        if event_type == "started":
            self._log(f"Training worker loaded config: {event.get('model_cfg', '')}")
            self._set_training_status("Loading", "running")
        elif event_type == "training":
            self._log(str(event.get("message") or "Training started"))
            self._set_training_status("Running", "running")
        elif event_type in {"training_setup", "epoch_start", "batch_progress", "epoch_end"}:
            self._update_training_monitor(event)
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
        controller = self.training_controller
        if controller is None or not controller.is_running:
            return
        self.train_cancel_requested = True
        self._set_training_status("Canceling", "canceled")
        self._log("Cancel requested. Stopping training worker process...")
        controller.cancel(kill_after_ms=5000)

    def _kill_training_process_if_running(self):
        process = (
            self.training_controller.process if self.training_controller else self.train_process
        )
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._log("Training worker did not stop after terminate; killing process.")
            process.kill()

    def _handle_training_process_error(self, _error):
        process = (
            self.training_controller.process if self.training_controller else self.train_process
        )
        if process is not None:
            self.train_stderr_buffer += process.errorString() + "\n"

    def _finish_training_process(self, exit_code: int, exit_status):
        """Compatibility entry point for legacy callers and tests."""
        state = (
            "cancelled"
            if self.train_cancel_requested
            else ("finished" if int(exit_code) == 0 else "failed")
        )
        self._finish_training_job(
            WorkerJobResult(
                state=state,
                exit_code=int(exit_code),
                exit_status=exit_status,
                stderr=self.train_stderr_buffer,
            )
        )

    def _finish_training_job(self, result: WorkerJobResult):
        self._flush_training_terminal_output()

        event = self.train_result_event
        stderr_text = (result.stderr or self.train_stderr_buffer).strip()
        cancel_requested = self.train_cancel_requested or result.state == "cancelled"
        exit_code = result.exit_code if result.exit_code is not None else 1
        exit_status = result.exit_status

        controller = self.training_controller
        self.training_controller = None
        self.training_running = False
        self.run_btn.setEnabled(True)
        self.cancel_train_btn.setEnabled(False)
        self.train_process = None
        self.train_config_path = None
        self.train_result_event = None
        self.train_stdout_buffer = ""
        self.train_stderr_buffer = ""
        self.train_cancel_requested = False
        if controller is not None:
            controller.deleteLater()

        if cancel_requested and event is None:
            self._set_training_status("Canceled", "canceled")
            self.phase_label.setText("Training canceled")
            self._log("Training canceled.")
            QMessageBox.information(
                self, "Training canceled", "Training worker process was canceled."
            )
            return

        if event is None:
            detail = stderr_text or f"Process exited with code {exit_code}."
            self._set_training_status("Failed", "failed")
            self.phase_label.setText("Training failed")
            self._log(f"Training worker failed: {detail}")
            QMessageBox.critical(self, "Training error", f"Training worker failed:\n{detail}")
            return

        had_error = bool(event.get("had_error"))
        canceled = bool(event.get("canceled")) or cancel_requested
        save_dir = str(event.get("save_dir") or "")
        error_message = str(event.get("error_message") or stderr_text or "Unknown training error")

        if canceled and not had_error:
            self._set_training_status("Canceled", "canceled")
            self.phase_label.setText("Training canceled")
            self._log("Training canceled.")
            QMessageBox.information(self, "Training canceled", "Training was canceled.")
            return

        if (
            had_error
            or exit_status == QProcess.ExitStatus.CrashExit
            or exit_code != 0
            or result.state in {"failed", "start_failed"}
        ):
            self._set_training_status("Failed", "failed")
            self.phase_label.setText("Training failed")
            self._log(f"Training failed: {error_message}")
            QMessageBox.critical(self, "Training error", f"Training failed:\n{error_message}")
            return

        self._set_training_status("Complete", "complete")
        self.phase_label.setText("Training complete")
        self.eta_label.setText("Complete")
        self.overall_progress.setValue(1000)
        self.overall_progress.setFormat("Training complete")
        if save_dir:
            self._log(f"Training complete. Artifacts saved to: {save_dir}")
        else:
            self._log("Training complete.")
        QMessageBox.information(
            self,
            "Training complete",
            "YOLO training finished. Review the logs for metrics.",
        )
