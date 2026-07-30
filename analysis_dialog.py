"""PyQt dialog for running SqueakPose inference analysis."""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QPointF, QProcess, QRectF, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squeakpose.workers.process import remove_file_quietly, shutdown_qprocess
from squeakpose.workers.protocol import WorkerProtocolError, parse_event_line
from ui_style import analysis_dialog_stylesheet


def _remove_file_quietly(path: Optional[str]) -> None:
    remove_file_quietly(path)


def _shutdown_qprocess(process: Optional[QProcess]) -> bool:
    return shutdown_qprocess(process)


def _safe_stem(path: str) -> str:
    stem = Path(path).stem if path else "analysis"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("_")
    return cleaned or "analysis"


def _fmt_number(value: Any, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}"


class FrameAnnotationView(QWidget):
    """Frame viewer that supports clicked scale points and rectangular ROIs."""

    scaleDistanceChanged = pyqtSignal(float)
    scalePointsChanged = pyqtSignal(list)
    roiDrawn = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("AnalysisFrameView")
        self.setMinimumSize(480, 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._pixmap = QPixmap()
        self._image_width = 0.0
        self._image_height = 0.0
        self._mode = "scale"
        self._scale_points: list[tuple[float, float]] = []
        self._rois: list[dict[str, Any]] = []
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_current: Optional[tuple[float, float]] = None

    def set_mode(self, mode: str) -> None:
        self._mode = "roi" if mode == "roi" else "scale"
        self.update()

    def set_frame(self, pixmap: QPixmap, width: int, height: int) -> None:
        self._pixmap = QPixmap(pixmap)
        self._image_width = float(width)
        self._image_height = float(height)
        self.update()

    def set_scale_points(self, points: list[tuple[float, float]]) -> None:
        self._scale_points = [(float(x), float(y)) for x, y in points[:2]]
        self.update()

    def set_rois(self, rois: list[dict[str, Any]]) -> None:
        self._rois = [dict(roi) for roi in rois]
        self.update()

    def clear_preview_roi(self) -> None:
        self._drag_start = None
        self._drag_current = None
        self.update()

    def _content_rect(self) -> QRectF:
        if self._pixmap.isNull() or self._image_width <= 0 or self._image_height <= 0:
            return QRectF()
        target = QSize(int(self._image_width), int(self._image_height))
        target.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        width = float(target.width())
        height = float(target.height())
        return QRectF((self.width() - width) / 2.0, (self.height() - height) / 2.0, width, height)

    def _widget_to_image(self, point: QPointF) -> Optional[tuple[float, float]]:
        rect = self._content_rect()
        if rect.isNull() or not rect.contains(point):
            return None
        x = (point.x() - rect.x()) / rect.width() * self._image_width
        y = (point.y() - rect.y()) / rect.height() * self._image_height
        return (max(0.0, min(self._image_width, x)), max(0.0, min(self._image_height, y)))

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        rect = self._content_rect()
        return QPointF(rect.x() + x / self._image_width * rect.width(), rect.y() + y / self._image_height * rect.height())

    def _roi_to_widget_rect(self, roi: dict[str, Any]) -> QRectF:
        p1 = self._image_to_widget(float(roi["x1"]), float(roi["y1"]))
        p2 = self._image_to_widget(float(roi["x2"]), float(roi["y2"]))
        return QRectF(p1, p2).normalized()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return

        if self._mode == "scale":
            if len(self._scale_points) >= 2:
                self._scale_points = [image_point]
            else:
                self._scale_points.append(image_point)
            self.scalePointsChanged.emit(list(self._scale_points))
            if len(self._scale_points) == 2:
                (x1, y1), (x2, y2) = self._scale_points
                self.scaleDistanceChanged.emit(math.hypot(x2 - x1, y2 - y1))
            self.update()
            return

        self._drag_start = image_point
        self._drag_current = image_point
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._mode != "roi" or self._drag_start is None:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        self._drag_current = image_point
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._mode != "roi" or self._drag_start is None:
            return
        image_point = self._widget_to_image(event.position()) or self._drag_current
        if image_point is None:
            self.clear_preview_roi()
            return
        x1, y1 = self._drag_start
        x2, y2 = image_point
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        self.clear_preview_roi()
        if right - left < 5 or bottom - top < 5:
            return
        self.roiDrawn.emit({"type": "rect", "x1": left, "y1": top, "x2": right, "y2": bottom})

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f151a"))

        rect = self._content_rect()
        if self._pixmap.isNull() or rect.isNull():
            painter.setPen(QColor("#77838f"))
            painter.setFont(QFont("Arial", 12, QFont.Weight.DemiBold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a frame")
            return

        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen(QPen(QColor("#26313b"), 1))
        painter.drawRect(rect)

        roi_pen = QPen(QColor("#f5b942"), 2)
        roi_fill = QColor(245, 185, 66, 34)
        painter.setFont(QFont("Arial", 10, QFont.Weight.DemiBold))
        for roi in self._rois:
            try:
                roi_rect = self._roi_to_widget_rect(roi)
            except (KeyError, TypeError, ValueError):
                continue
            painter.setPen(roi_pen)
            painter.fillRect(roi_rect, roi_fill)
            painter.drawRect(roi_rect)
            label_rect = QRectF(roi_rect.x() + 4, roi_rect.y() + 4, min(roi_rect.width() - 8, 180), 20)
            painter.fillRect(label_rect, QColor(17, 24, 32, 190))
            painter.setPen(QColor("#f9d782"))
            painter.drawText(label_rect.adjusted(5, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, str(roi.get("name", "ROI")))

        if self._drag_start is not None and self._drag_current is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            preview = {
                "x1": min(x1, x2),
                "y1": min(y1, y2),
                "x2": max(x1, x2),
                "y2": max(y1, y2),
            }
            painter.setPen(QPen(QColor("#76c7ff"), 2, Qt.PenStyle.DashLine))
            painter.fillRect(self._roi_to_widget_rect(preview), QColor(118, 199, 255, 30))
            painter.drawRect(self._roi_to_widget_rect(preview))

        if self._scale_points:
            painter.setPen(QPen(QColor("#76c7ff"), 2))
            painter.setBrush(QColor("#76c7ff"))
            widget_points = [self._image_to_widget(x, y) for x, y in self._scale_points]
            if len(widget_points) == 2:
                painter.drawLine(widget_points[0], widget_points[1])
            for index, point in enumerate(widget_points, start=1):
                painter.drawEllipse(point, 5.0, 5.0)
                painter.drawText(point + QPointF(8, -8), str(index))


class AnalysisDialog(QDialog):
    """Dialog for configuring and running the analysis notebook workflow."""

    def __init__(self, parent, *, project_root: str, app_base_dir: str):
        super().__init__(parent)
        self.setWindowTitle("Analysis")
        self.resize(1240, 900)
        self.setMinimumSize(1040, 680)
        self.project_root = os.path.abspath(project_root)
        self.app_base_dir = os.path.abspath(app_base_dir)
        self.analysis_process: Optional[QProcess] = None
        self.analysis_config_path: Optional[str] = None
        self.analysis_stdout_buffer = ""
        self.analysis_stderr_buffer = ""
        self.last_output_dir = ""
        self.scale_points: list[tuple[float, float]] = []
        self.rois: list[dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(10)
        layout.addLayout(body, 1)

        left_column = QWidget()
        left_column.setMinimumWidth(500)
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        inputs_panel = QFrame()
        inputs_panel.setObjectName("AnalysisPanel")
        inputs_layout = QVBoxLayout(inputs_panel)
        inputs_layout.setContentsMargins(12, 12, 12, 10)
        inputs_layout.setSpacing(8)

        inputs_header = QHBoxLayout()
        inputs_title = QLabel("Inputs")
        inputs_title.setObjectName("AnalysisPanelTitle")
        inputs_header.addWidget(inputs_title)
        inputs_header.addStretch(1)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("AnalysisStatusLabel")
        inputs_header.addWidget(self.status_label)
        inputs_layout.addLayout(inputs_header)

        input_form = QFormLayout()
        input_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        input_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        input_form.setHorizontalSpacing(8)
        input_form.setVerticalSpacing(7)

        self.csv_edit = QLineEdit()
        self.csv_edit.setPlaceholderText("Select an inference CSV")
        self.csv_edit.textChanged.connect(self._refresh_default_output_dir)
        csv_browse = QPushButton("Browse...")
        csv_browse.clicked.connect(self._browse_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_edit, 1)
        csv_row.addWidget(csv_browse)
        input_form.addRow("Detections CSV:", csv_row)

        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Optional; auto-detected from CSV when available")
        video_browse = QPushButton("Browse...")
        video_browse.clicked.connect(self._browse_video)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_edit, 1)
        video_row.addWidget(video_browse)
        input_form.addRow("Video file:", video_row)

        inputs_layout.addLayout(input_form)
        left_layout.addWidget(inputs_panel, 0)

        tracking_panel = QFrame()
        tracking_panel.setObjectName("AnalysisPanel")
        tracking_layout = QVBoxLayout(tracking_panel)
        tracking_layout.setContentsMargins(12, 12, 12, 10)
        tracking_layout.setSpacing(8)

        tracking_title = QLabel("Tracking Settings")
        tracking_title.setObjectName("AnalysisPanelTitle")
        tracking_layout.addWidget(tracking_title)

        tracking_form = QFormLayout()
        tracking_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        tracking_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        tracking_form.setHorizontalSpacing(8)
        tracking_form.setVerticalSpacing(7)

        scale_row = QHBoxLayout()
        scale_row.setSpacing(8)
        self.pixel_distance_px = 0.0
        self.pixel_distance_label = QLabel("Draw scale")
        self.pixel_distance_label.setObjectName("AnalysisValuePill")
        self.pixel_distance_label.setMinimumWidth(128)
        self.real_distance_spin = QDoubleSpinBox()
        self.real_distance_spin.setRange(0.000001, 1_000_000.0)
        self.real_distance_spin.setDecimals(4)
        self.real_distance_spin.setValue(1.0)
        self.real_distance_spin.setMinimumWidth(130)
        self.real_distance_spin.valueChanged.connect(self._update_scale_label)
        scale_row.addWidget(self.pixel_distance_label, 0)
        scale_row.addWidget(QLabel("equals"))
        scale_row.addWidget(self.real_distance_spin, 1)
        scale_row.addWidget(QLabel("mm"))
        tracking_form.addRow("Scale bar:", scale_row)

        self.smooth_check = QCheckBox("OneEuro smooth centers")
        self.smooth_check.setChecked(True)
        tracking_form.addRow("Smoothing:", self.smooth_check)
        filter_row = QHBoxLayout()
        self.min_cutoff_spin = QDoubleSpinBox()
        self.min_cutoff_spin.setRange(0.0001, 100.0)
        self.min_cutoff_spin.setDecimals(3)
        self.min_cutoff_spin.setValue(1.0)
        self.min_cutoff_spin.setPrefix("min ")
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 100.0)
        self.beta_spin.setDecimals(3)
        self.beta_spin.setValue(0.0)
        self.beta_spin.setPrefix("beta ")
        filter_row.addWidget(self.min_cutoff_spin)
        filter_row.addWidget(self.beta_spin)
        filter_row.addStretch(1)
        tracking_form.addRow("Filter:", filter_row)

        tracking_layout.addLayout(tracking_form)
        left_layout.addWidget(tracking_panel, 0)

        output_panel = QFrame()
        output_panel.setObjectName("AnalysisPanel")
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(12, 12, 12, 10)
        output_layout.setSpacing(8)

        output_title = QLabel("Output Settings")
        output_title.setObjectName("AnalysisPanelTitle")
        output_layout.addWidget(output_title)

        output_form = QFormLayout()
        output_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        output_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        output_form.setHorizontalSpacing(8)
        output_form.setVerticalSpacing(7)

        self.output_edit = QLineEdit()
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(output_browse)
        output_form.addRow("Folder:", output_row)

        options_box = QVBoxLayout()
        options_box.setSpacing(8)
        self.plots_check = QCheckBox("Plots")
        self.plots_check.setChecked(True)
        self.annotated_video_check = QCheckBox("Annotated video")
        self.cluster_check = QCheckBox("UMAP/HDBSCAN")
        self.cluster_clips_check = QCheckBox("Cluster clips")
        options_box.addWidget(self.plots_check)
        options_box.addWidget(self.annotated_video_check)
        options_box.addWidget(self.cluster_check)

        self.umap_settings_panel = QFrame()
        self.umap_settings_panel.setObjectName("AnalysisSubPanel")
        umap_layout = QHBoxLayout(self.umap_settings_panel)
        umap_layout.setContentsMargins(10, 8, 10, 8)
        umap_layout.setSpacing(8)
        self.umap_neighbors_spin = QSpinBox()
        self.umap_neighbors_spin.setRange(0, 500)
        self.umap_neighbors_spin.setSpecialValueText("neighbors Auto")
        self.umap_neighbors_spin.setValue(0)
        self.umap_neighbors_spin.setPrefix("neighbors ")
        self.umap_neighbors_spin.setMinimumWidth(145)
        self.umap_min_dist_spin = QDoubleSpinBox()
        self.umap_min_dist_spin.setRange(0.0, 1.0)
        self.umap_min_dist_spin.setDecimals(3)
        self.umap_min_dist_spin.setSingleStep(0.05)
        self.umap_min_dist_spin.setValue(0.3)
        self.umap_min_dist_spin.setPrefix("min dist ")
        self.umap_min_dist_spin.setMinimumWidth(120)
        self.hdbscan_min_cluster_size_spin = QSpinBox()
        self.hdbscan_min_cluster_size_spin.setRange(0, 10_000)
        self.hdbscan_min_cluster_size_spin.setSpecialValueText("cluster min Auto")
        self.hdbscan_min_cluster_size_spin.setValue(0)
        self.hdbscan_min_cluster_size_spin.setPrefix("cluster min ")
        self.hdbscan_min_cluster_size_spin.setMinimumWidth(150)
        umap_layout.addWidget(self.umap_neighbors_spin)
        umap_layout.addWidget(self.umap_min_dist_spin)
        umap_layout.addWidget(self.hdbscan_min_cluster_size_spin)
        umap_layout.addStretch(1)
        options_box.addWidget(self.umap_settings_panel)

        self.cluster_clip_settings_panel = QFrame()
        self.cluster_clip_settings_panel.setObjectName("AnalysisSubPanel")
        cluster_clip_layout = QHBoxLayout(self.cluster_clip_settings_panel)
        cluster_clip_layout.setContentsMargins(8, 4, 8, 4)
        cluster_clip_layout.setSpacing(8)
        self.clip_length_spin = QDoubleSpinBox()
        self.clip_length_spin.setRange(0.25, 60.0)
        self.clip_length_spin.setDecimals(2)
        self.clip_length_spin.setPrefix("clip ")
        self.clip_length_spin.setSuffix(" s")
        self.clip_length_spin.setValue(2.0)
        self.clip_length_spin.setMinimumWidth(110)
        self.samples_per_cluster_spin = QSpinBox()
        self.samples_per_cluster_spin.setRange(1, 20)
        self.samples_per_cluster_spin.setPrefix("samples ")
        self.samples_per_cluster_spin.setValue(1)
        self.samples_per_cluster_spin.setMinimumWidth(110)
        cluster_clip_layout.addWidget(self.clip_length_spin)
        cluster_clip_layout.addWidget(self.samples_per_cluster_spin)
        cluster_clip_layout.addStretch(1)

        options_box.addWidget(self.cluster_clips_check)
        options_box.addWidget(self.cluster_clip_settings_panel)
        output_form.addRow("Outputs:", options_box)

        output_layout.addLayout(output_form)
        left_layout.addWidget(output_panel, 0)
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("AnalysisLeftScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(520)
        left_scroll.setMaximumWidth(580)
        left_scroll.setWidget(left_column)
        body.addWidget(left_scroll, 0)

        workspace_panel = QFrame()
        workspace_panel.setObjectName("AnalysisPanel")
        workspace_layout = QVBoxLayout(workspace_panel)
        workspace_layout.setContentsMargins(12, 12, 12, 10)
        workspace_layout.setSpacing(8)

        workspace_header = QHBoxLayout()
        workspace_title = QLabel("Interactive Setup")
        workspace_title.setObjectName("AnalysisPanelTitle")
        workspace_header.addWidget(workspace_title)
        workspace_header.addStretch(1)
        self.frame_info_label = QLabel("No frame loaded")
        self.frame_info_label.setObjectName("AnalysisHintLabel")
        workspace_header.addWidget(self.frame_info_label)
        workspace_layout.addLayout(workspace_header)

        toolbar = QHBoxLayout()
        self.load_frame_btn = QPushButton("Load Frame")
        self.load_frame_btn.clicked.connect(lambda: self._load_preview_frame(silent=False))
        toolbar.addWidget(self.load_frame_btn)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.scale_mode_btn = QPushButton("Scale")
        self.scale_mode_btn.setCheckable(True)
        self.scale_mode_btn.setChecked(True)
        self.roi_mode_btn = QPushButton("ROI")
        self.roi_mode_btn.setCheckable(True)
        self.mode_group.addButton(self.scale_mode_btn)
        self.mode_group.addButton(self.roi_mode_btn)
        self.scale_mode_btn.toggled.connect(lambda checked: checked and self._set_annotation_mode("scale"))
        self.roi_mode_btn.toggled.connect(lambda checked: checked and self._set_annotation_mode("roi"))
        toolbar.addWidget(self.scale_mode_btn)
        toolbar.addWidget(self.roi_mode_btn)

        self.clear_scale_btn = QPushButton("Clear Scale")
        self.clear_scale_btn.clicked.connect(self._clear_scale)
        toolbar.addWidget(self.clear_scale_btn)
        self.clear_rois_btn = QPushButton("Clear ROIs")
        self.clear_rois_btn.clicked.connect(self._clear_rois)
        toolbar.addWidget(self.clear_rois_btn)
        toolbar.addStretch(1)
        self.scale_status_label = QLabel("")
        self.scale_status_label.setObjectName("AnalysisStatusLabel")
        toolbar.addWidget(self.scale_status_label)
        workspace_layout.addLayout(toolbar)

        self.frame_view = FrameAnnotationView()
        self.frame_view.scaleDistanceChanged.connect(self._apply_scale_distance)
        self.frame_view.scalePointsChanged.connect(self._set_scale_points)
        self.frame_view.roiDrawn.connect(self._add_roi_from_canvas)
        workspace_layout.addWidget(self.frame_view, 1)

        roi_row = QHBoxLayout()
        roi_column = QVBoxLayout()
        roi_title = QLabel("ROIs")
        roi_title.setObjectName("AnalysisPanelTitle")
        roi_column.addWidget(roi_title)
        self.roi_list = QListWidget()
        self.roi_list.setObjectName("AnalysisRoiList")
        self.roi_list.setMaximumHeight(120)
        roi_column.addWidget(self.roi_list)
        roi_row.addLayout(roi_column, 1)
        roi_actions = QVBoxLayout()
        self.delete_roi_btn = QPushButton("Delete Selected")
        self.delete_roi_btn.clicked.connect(self._delete_selected_roi)
        roi_actions.addWidget(self.delete_roi_btn)
        self.roi_count_label = QLabel("0 ROIs")
        self.roi_count_label.setObjectName("AnalysisHintLabel")
        roi_actions.addWidget(self.roi_count_label)
        roi_actions.addStretch(1)
        roi_row.addLayout(roi_actions)
        workspace_layout.addLayout(roi_row)
        body.addWidget(workspace_panel, 1)

        progress_panel = QFrame()
        progress_panel.setObjectName("AnalysisPanel")
        progress_panel.setMinimumHeight(170)
        progress_panel.setMaximumHeight(230)
        progress_layout = QVBoxLayout(progress_panel)
        progress_layout.setContentsMargins(12, 12, 12, 10)
        progress_layout.setSpacing(8)
        progress_header = QHBoxLayout()
        progress_title = QLabel("Results")
        progress_title.setObjectName("AnalysisPanelTitle")
        progress_header.addWidget(progress_title)
        progress_header.addStretch(1)
        progress_layout.addLayout(progress_header)
        self.progress = QProgressBar()
        self.progress.setRange(0, 8)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)
        result_row = QHBoxLayout()
        self.summary_view = QPlainTextEdit()
        self.summary_view.setObjectName("AnalysisSummaryView")
        self.summary_view.setReadOnly(True)
        self.summary_view.setMaximumBlockCount(300)
        self.summary_view.setMaximumHeight(120)
        self.summary_view.setPlaceholderText("Summary metrics will appear here.")
        result_row.addWidget(self.summary_view, 1)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("AnalysisLogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setMaximumHeight(120)
        self.log_view.setPlaceholderText("Analysis progress will appear here.")
        result_row.addWidget(self.log_view, 2)
        progress_layout.addLayout(result_row, 1)
        layout.addWidget(progress_panel, 0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._start_analysis)
        buttons.addButton(self.run_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self.open_output_btn = QPushButton("Open Output")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        buttons.addButton(self.open_output_btn, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)

        self.setStyleSheet(analysis_dialog_stylesheet())
        self.cluster_check.toggled.connect(self._sync_output_settings)
        self.cluster_clips_check.toggled.connect(self._sync_output_settings)
        self._select_initial_csv()
        self._refresh_default_output_dir()
        self._update_scale_label()
        self._refresh_roi_list()
        self._sync_output_settings()
        self._load_preview_frame(silent=True)

    def _candidate_csv_dirs(self) -> list[str]:
        return [
            os.path.join(self.project_root, "inference outputs"),
            os.path.join(self.app_base_dir, "analysis_toolset", "inference outputs"),
        ]

    def _select_initial_csv(self) -> None:
        candidates: list[str] = []
        for folder in self._candidate_csv_dirs():
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    if name.lower().endswith(".csv"):
                        candidates.append(os.path.join(folder, name))
        if candidates:
            newest = max(candidates, key=lambda path: os.path.getmtime(path))
            self.csv_edit.setText(newest)

    def _default_output_dir_for_csv(self, csv_path: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(self.project_root, "analysis outputs", f"{_safe_stem(csv_path)}_{timestamp}")

    def _refresh_default_output_dir(self) -> None:
        if self.output_edit.text().strip():
            return
        self.output_edit.setText(self._default_output_dir_for_csv(self.csv_edit.text().strip()))

    def _browse_csv(self) -> None:
        start = os.path.dirname(self.csv_edit.text().strip())
        if not start or not os.path.isdir(start):
            start = next((folder for folder in self._candidate_csv_dirs() if os.path.isdir(folder)), self.project_root)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select inference CSV",
            start,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.csv_edit.setText(path)
            self.output_edit.setText(self._default_output_dir_for_csv(path))
            self._clear_annotations()
            self._load_preview_frame(silent=True)

    def _browse_video(self) -> None:
        start = os.path.dirname(self.video_edit.text().strip()) or self.project_root
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            start,
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if path:
            self.video_edit.setText(path)
            self._clear_annotations()
            self._load_preview_frame(silent=False)

    def _browse_output_dir(self) -> None:
        start = self.output_edit.text().strip() or os.path.join(self.project_root, "analysis outputs")
        path = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if path:
            self.output_edit.setText(path)

    def _video_path_from_csv(self) -> str:
        csv_path = self.csv_edit.text().strip()
        if not os.path.isfile(csv_path):
            return ""
        try:
            import pandas as pd

            raw = pd.read_csv(csv_path, nrows=1000)
        except Exception:
            return ""
        if "video_path" not in raw.columns:
            return ""
        for value in raw["video_path"].dropna().unique():
            path = str(value).strip()
            if path and os.path.isfile(path):
                return path
        return ""

    def _frame_dimensions_from_csv(self) -> tuple[int, int]:
        csv_path = self.csv_edit.text().strip()
        if not os.path.isfile(csv_path):
            return (1280, 720)
        try:
            import pandas as pd

            raw = pd.read_csv(csv_path, nrows=1000)
        except Exception:
            return (1280, 720)
        width = 0
        height = 0
        if "image_width" in raw.columns:
            width = int(pd.to_numeric(raw["image_width"], errors="coerce").dropna().max() or 0)
        if "image_height" in raw.columns:
            height = int(pd.to_numeric(raw["image_height"], errors="coerce").dropna().max() or 0)
        return (width or 1280, height or 720)

    def _blank_frame_pixmap(self, width: int, height: int) -> QPixmap:
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#10161c"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#24303a"), 1))
        x_step = max(width // 8, 1)
        y_step = max(height // 8, 1)
        for x in range(0, width, x_step):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, y_step):
            painter.drawLine(0, y, width, y)
        painter.setPen(QColor("#77838f"))
        painter.setFont(QFont("Arial", max(12, min(width, height) // 34), QFont.Weight.DemiBold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "CSV coordinate canvas")
        painter.end()
        return pixmap

    def _load_video_pixmap(self, video_path: str) -> Optional[tuple[QPixmap, int, int]]:
        if not video_path or not os.path.isfile(video_path):
            return None
        try:
            import cv2
        except Exception:
            return None
        cap = cv2.VideoCapture(video_path)
        try:
            ok, frame = cap.read()
            if not ok:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
            return (QPixmap.fromImage(image), width, height)
        finally:
            cap.release()

    def _load_preview_frame(self, *, silent: bool) -> None:
        video_path = self.video_edit.text().strip()
        if not video_path:
            video_path = self._video_path_from_csv()
            if video_path:
                self.video_edit.setText(video_path)

        loaded = self._load_video_pixmap(video_path)
        if loaded is not None:
            pixmap, width, height = loaded
            self.frame_view.set_frame(pixmap, width, height)
            self.frame_info_label.setText(f"{width} x {height} | {os.path.basename(video_path)}")
            return

        if video_path and not silent:
            QMessageBox.information(self, "Frame preview", "Could not read the selected video. Showing CSV coordinates instead.")

        width, height = self._frame_dimensions_from_csv()
        self.frame_view.set_frame(self._blank_frame_pixmap(width, height), width, height)
        self.frame_info_label.setText(f"{width} x {height} | CSV coordinates")

    def _set_annotation_mode(self, mode: str) -> None:
        self.frame_view.set_mode(mode)

    def _sync_output_settings(self) -> None:
        clustering_enabled = self.cluster_check.isChecked()
        if not clustering_enabled and self.cluster_clips_check.isChecked():
            self.cluster_clips_check.setChecked(False)
        self.umap_settings_panel.setVisible(clustering_enabled)
        self.cluster_clips_check.setEnabled(clustering_enabled)
        self.cluster_clip_settings_panel.setVisible(
            clustering_enabled and self.cluster_clips_check.isChecked()
        )

    def _set_scale_points(self, points: list[tuple[float, float]]) -> None:
        self.scale_points = [(float(x), float(y)) for x, y in points[:2]]
        self._update_scale_label()

    def _apply_scale_distance(self, distance: float) -> None:
        if distance > 0:
            self.pixel_distance_px = float(distance)
            self.pixel_distance_label.setText(f"{distance:.1f} px")
        self._update_scale_label()

    def _clear_scale(self) -> None:
        self.scale_points = []
        self.pixel_distance_px = 0.0
        self.pixel_distance_label.setText("Draw scale")
        self.frame_view.set_scale_points([])
        self._update_scale_label()

    def _clear_rois(self) -> None:
        self.rois = []
        self._refresh_roi_list()

    def _clear_annotations(self) -> None:
        self._clear_scale()
        self._clear_rois()

    def _update_scale_label(self) -> None:
        pixel_distance = self.pixel_distance_px
        real_distance = self.real_distance_spin.value()
        if pixel_distance > 0:
            mm_per_pixel = real_distance / pixel_distance
            self.scale_status_label.setText(f"{mm_per_pixel:.4f} mm/px | {len(self.scale_points)}/2")
        else:
            self.scale_status_label.setText(f"Scale unset | {len(self.scale_points)}/2")

    def _add_roi_from_canvas(self, roi: dict[str, Any]) -> None:
        default_name = f"ROI {len(self.rois) + 1}"
        name, accepted = QInputDialog.getText(self, "Name ROI", "ROI name:", text=default_name)
        if not accepted:
            return
        roi["name"] = name.strip() or default_name
        self.rois.append(roi)
        self._refresh_roi_list()

    def _delete_selected_roi(self) -> None:
        row = self.roi_list.currentRow()
        if 0 <= row < len(self.rois):
            del self.rois[row]
            self._refresh_roi_list()

    def _refresh_roi_list(self) -> None:
        self.roi_list.clear()
        for index, roi in enumerate(self.rois, start=1):
            width = float(roi["x2"]) - float(roi["x1"])
            height = float(roi["y2"]) - float(roi["y1"])
            item = QListWidgetItem(f"{index}. {roi.get('name', 'ROI')}  {width:.0f} x {height:.0f}px")
            self.roi_list.addItem(item)
        self.roi_count_label.setText(f"{len(self.rois)} ROI{'s' if len(self.rois) != 1 else ''}")
        self.frame_view.set_rois(self.rois)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text.rstrip() + "\n")
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.ensureCursorVisible()

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.load_frame_btn.setEnabled(not running)
        if running:
            self.status_label.setText("Running")
            self.progress.setRange(0, 8)
            self.progress.setValue(0)

    def _config_payload(self) -> dict[str, Any]:
        return {
            "detections_csv": self.csv_edit.text().strip(),
            "video_path": self.video_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "fps": 0.0,
            "pixel_distance": self.pixel_distance_px,
            "real_world_distance_mm": self.real_distance_spin.value(),
            "smooth": self.smooth_check.isChecked(),
            "min_cutoff": self.min_cutoff_spin.value(),
            "beta": self.beta_spin.value(),
            "d_cutoff": 1.0,
            "make_plots": self.plots_check.isChecked(),
            "make_annotated_video": self.annotated_video_check.isChecked(),
            "run_clustering": self.cluster_check.isChecked(),
            "export_cluster_clips": self.cluster_clips_check.isChecked(),
            "umap_neighbors": self.umap_neighbors_spin.value(),
            "umap_min_dist": self.umap_min_dist_spin.value(),
            "hdbscan_min_cluster_size": self.hdbscan_min_cluster_size_spin.value(),
            "cluster_clip_length_sec": self.clip_length_spin.value(),
            "samples_per_cluster": self.samples_per_cluster_spin.value(),
            "rois": [dict(roi) for roi in self.rois],
        }

    def _validate_inputs(self) -> bool:
        if not os.path.isfile(self.csv_edit.text().strip()):
            QMessageBox.warning(self, "CSV required", "Select a valid inference CSV before running analysis.")
            return False
        video_path = self.video_edit.text().strip()
        if video_path and not os.path.isfile(video_path):
            QMessageBox.warning(self, "Invalid video", f"Video file not found:\n{video_path}")
            return False
        if len(self.scale_points) < 2 or self.pixel_distance_px <= 0:
            QMessageBox.warning(
                self,
                "Scale required",
                "Draw a two-point scale bar before running analysis.",
            )
            return False
        if self.cluster_clips_check.isChecked() and not self.cluster_check.isChecked():
            QMessageBox.warning(self, "Clustering required", "Enable UMAP/HDBSCAN before exporting cluster clips.")
            return False
        if self.annotated_video_check.isChecked() and not video_path:
            QMessageBox.information(
                self,
                "Video optional",
                "No video was selected. The worker will try to use the video_path stored in the CSV.",
            )
        return True

    def _start_analysis(self) -> None:
        if self.analysis_process and self.analysis_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Analysis running", "An analysis job is already running.")
            return
        if not self._validate_inputs():
            return

        payload = self._config_payload()
        os.makedirs(payload["output_dir"], exist_ok=True)
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="squeakpose_analysis_", delete=False)
        with handle:
            json.dump(payload, handle, indent=2)
        self.analysis_config_path = handle.name
        self.last_output_dir = payload["output_dir"]
        self.open_output_btn.setEnabled(False)
        self.log_view.clear()
        self.summary_view.clear()
        self._append_log(f"Detections: {payload['detections_csv']}")
        self._append_log(f"Output: {payload['output_dir']}")
        self._append_log(f"ROIs: {len(self.rois)}")

        process = QProcess(self)
        self.analysis_process = process
        process.setProgram(sys.executable)
        process.setArguments([os.path.join(self.app_base_dir, "analysis_worker.py"), "--config", self.analysis_config_path])
        process.readyReadStandardOutput.connect(self._read_analysis_stdout)
        process.readyReadStandardError.connect(self._read_analysis_stderr)
        process.finished.connect(self._analysis_finished)
        self._set_running(True)
        process.start()
        if not process.waitForStarted(3000):
            self._set_running(False)
            self.status_label.setText("Failed")
            _remove_file_quietly(self.analysis_config_path)
            self.analysis_config_path = None
            self.analysis_process = None
            process.deleteLater()
            QMessageBox.warning(self, "Analysis failed", "Could not start the analysis worker.")

    def _read_analysis_stdout(self) -> None:
        if self.analysis_process is None:
            return
        data = bytes(self.analysis_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.analysis_stdout_buffer += data
        while "\n" in self.analysis_stdout_buffer:
            line, self.analysis_stdout_buffer = self.analysis_stdout_buffer.split("\n", 1)
            self._handle_worker_line(line.strip())

    def _read_analysis_stderr(self) -> None:
        if self.analysis_process is None:
            return
        data = bytes(self.analysis_process.readAllStandardError()).decode("utf-8", errors="replace")
        self.analysis_stderr_buffer += data
        while "\n" in self.analysis_stderr_buffer:
            line, self.analysis_stderr_buffer = self.analysis_stderr_buffer.split("\n", 1)
            if line.strip():
                self._append_log(line.strip())

    def _show_result_summary(self, event: dict[str, Any]) -> None:
        summary = event.get("summary") or {}
        lines = [
            f"Frames: {int(summary.get('frames') or 0)}",
            f"Duration: {_fmt_number(summary.get('duration_s'))} s",
            f"Scale: {_fmt_number(summary.get('mm_per_pixel'), 4)} mm/pixel",
            f"Distance: {_fmt_number(summary.get('total_distance_m'))} m",
            f"Average speed: {_fmt_number(summary.get('average_speed_mm_per_sec'))} mm/s",
            f"ROIs: {int(summary.get('roi_count') or 0)}",
        ]
        if summary.get("analysis_kind") == "segmentation":
            lines.extend(
                [
                    f"Detections: {int(summary.get('detections') or 0)}",
                    f"No-detection frames: {int(summary.get('no_detection_frames') or 0)}",
                    f"Multi-detection frames: {int(summary.get('multi_detection_frames') or 0)}",
                    f"Mean mask area: {_fmt_number(summary.get('mean_mask_area_px2'))} px^2",
                    f"Coverage: {_fmt_number(float(summary.get('detection_coverage_fraction') or 0.0) * 100.0)}%",
                ]
            )
        roi_summary = event.get("roi_summary") or summary.get("roi_summary") or []
        if roi_summary:
            lines.append("")
            lines.append("ROI Time")
            for row in roi_summary:
                label = str(row.get("roi_label") or "ROI")
                frames = int(float(row.get("frames") or 0))
                duration = _fmt_number(row.get("duration_s"))
                lines.append(f"{label}: {duration} s, {frames} frames")
        self.summary_view.setPlainText("\n".join(lines))

    def _handle_worker_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self._append_log(line)
            return

        kind = event.get("event")
        if kind == "started":
            self._append_log("Analysis started.")
        elif kind == "progress":
            step = int(event.get("step") or 0)
            total = int(event.get("total") or 8)
            self.progress.setRange(0, total)
            self.progress.setValue(step)
            self._append_log(str(event.get("message") or "Working"))
        elif kind == "result":
            self.progress.setValue(self.progress.maximum())
            self.status_label.setText("Complete")
            self.last_output_dir = str(event.get("output_dir") or self.last_output_dir)
            summary = event.get("summary") or {}
            self._append_log("Analysis complete.")
            if summary:
                self._append_log(
                    "Frames: {frames} | Distance: {distance:.2f} m | Avg speed: {speed:.2f} mm/s".format(
                        frames=int(summary.get("frames") or 0),
                        distance=float(summary.get("total_distance_m") or 0.0),
                        speed=float(summary.get("average_speed_mm_per_sec") or 0.0),
                    )
                )
            self._append_log(f"Feature CSV: {event.get('feature_csv', '')}")
            if event.get("segmentation_detections_csv"):
                self._append_log(f"Segmentation detections: {event.get('segmentation_detections_csv')}")
            if event.get("roi_summary_csv"):
                self._append_log(f"ROI summary: {event.get('roi_summary_csv')}")
            self._show_result_summary(event)
            self.open_output_btn.setEnabled(bool(self.last_output_dir))
        elif kind == "error":
            self.status_label.setText("Failed")
            self._append_log(f"Error: {event.get('error_message', '')}")
        else:
            self._append_log(line)

    def _analysis_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self.analysis_stdout_buffer.strip():
            self._handle_worker_line(self.analysis_stdout_buffer.strip())
        if self.analysis_stderr_buffer.strip():
            self._append_log(self.analysis_stderr_buffer.strip())
        self.analysis_stdout_buffer = ""
        self.analysis_stderr_buffer = ""
        _remove_file_quietly(self.analysis_config_path)
        self.analysis_config_path = None
        if exit_code and self.status_label.text() == "Running":
            self.status_label.setText("Failed")
        self._set_running(False)

    def _open_output_dir(self) -> None:
        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_dir))

    def closeEvent(self, event):
        _shutdown_qprocess(self.analysis_process)
        _remove_file_quietly(self.analysis_config_path)
        self.analysis_process = None
        self.analysis_config_path = None
        super().closeEvent(event)
