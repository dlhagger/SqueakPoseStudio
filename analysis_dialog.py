"""PyQt dialog for running SqueakPose inference analysis."""

from __future__ import annotations

import json
import math
import os
import sys
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
    QPolygonF,
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

from layer_ops import layer_definition, normalize_layer_id
from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.services.analysis import (
    DEFAULT_ONE_EURO_BETA,
    DEFAULT_ONE_EURO_MIN_CUTOFF,
    AnalysisConfigError,
    AnalysisRunConfig,
    ProjectAnalysisBundle,
    analysis_csv_matches_layer,
    build_analysis_job_config,
    default_analysis_output_dir,
    default_combined_analysis_output_dir,
    inspect_analysis_csv,
    load_pose_preview,
    load_segmentation_preview,
    project_analysis_bundles,
    safe_analysis_stem,
)
from squeakpose.services.analysis_state import AnalysisAnnotationState
from squeakpose.services.video_analysis_setup import (
    load_video_analysis_setup,
    save_video_analysis_setup,
)
from squeakpose.workers.process import (
    WorkerJobController,
    WorkerJobResult,
    create_worker_config,
    remove_file_quietly,
    shutdown_qprocess,
)
from squeakpose.workers.protocol import WorkerProtocolError, parse_event_line
from ui_style import ThemedComboBox, analysis_dialog_stylesheet

POSE_PREVIEW_CONNECTIONS = (
    ("nose", "head"),
    ("head", "left_ear"),
    ("head", "right_ear"),
    ("head", "back"),
    ("back", "tail_base"),
)


def _remove_file_quietly(path: Optional[str]) -> None:
    remove_file_quietly(path)


def _shutdown_qprocess(process: Optional[QProcess]) -> bool:
    return shutdown_qprocess(process)


def _safe_stem(path: str) -> str:
    return safe_analysis_stem(path)


def _fmt_number(value: Any, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}"


class FrameAnnotationView(QWidget):
    """Frame viewer that supports clicked scale points and polygonal ROIs."""

    scaleDistanceChanged = pyqtSignal(float)
    scalePointsChanged = pyqtSignal(list)
    roiDrawn = pyqtSignal(dict)
    polygonDraftChanged = pyqtSignal(int)
    zoomChanged = pyqtSignal(int)

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
        self._segmentation_polygons: list[list[tuple[float, float]]] = []
        self._tracking_bbox: tuple[float, float, float, float] = ()
        self._pose_keypoints: list[dict[str, Any]] = []
        self._polygon_points: list[tuple[float, float]] = []
        self._polygon_cursor: Optional[tuple[float, float]] = None
        self._selected_roi_index = -1
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_drag_start: Optional[QPointF] = None
        self._pan_drag_origin = QPointF()
        self.setToolTip("Mouse wheel to zoom; right-drag to pan")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_mode(self, mode: str) -> None:
        next_mode = "roi" if mode == "roi" else "scale"
        if self._mode == "roi" and next_mode != "roi":
            self.cancel_polygon()
        self._mode = next_mode
        self.update()

    def set_frame(self, pixmap: QPixmap, width: int, height: int) -> None:
        self._pixmap = QPixmap(pixmap)
        self._image_width = float(width)
        self._image_height = float(height)
        self.reset_zoom()
        self.update()

    def set_segmentation_polygons(self, polygons: list[list[tuple[float, float]]]) -> None:
        self._segmentation_polygons = [
            [(float(x), float(y)) for x, y in polygon] for polygon in polygons
        ]
        self.update()

    def set_pose_overlay(
        self,
        bbox: tuple[float, float, float, float],
        keypoints: list[dict[str, Any]],
    ) -> None:
        """Display pose points plus the caller-selected tracking box."""
        self._tracking_bbox = tuple(float(value) for value in bbox) if len(bbox) == 4 else ()
        self._pose_keypoints = [dict(keypoint) for keypoint in keypoints]
        self.update()

    def set_scale_points(self, points: list[tuple[float, float]]) -> None:
        self._scale_points = [(float(x), float(y)) for x, y in points[:2]]
        self.update()

    def set_rois(self, rois: list[dict[str, Any]]) -> None:
        self._rois = [dict(roi) for roi in rois]
        self.update()

    def set_selected_roi(self, index: int) -> None:
        self._selected_roi_index = int(index)
        self.update()

    def cancel_polygon(self) -> None:
        self._polygon_points = []
        self._polygon_cursor = None
        self.polygonDraftChanged.emit(0)
        self.update()

    def undo_polygon_vertex(self) -> None:
        if self._polygon_points:
            self._polygon_points.pop()
        if not self._polygon_points:
            self._polygon_cursor = None
        self.polygonDraftChanged.emit(len(self._polygon_points))
        self.update()

    def finish_polygon(self) -> bool:
        if len(self._polygon_points) < 3:
            return False
        twice_area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                self._polygon_points,
                self._polygon_points[1:] + self._polygon_points[:1],
            )
        )
        if math.isclose(twice_area, 0.0, abs_tol=1e-6):
            return False
        points = list(self._polygon_points)
        self.cancel_polygon()
        self.roiDrawn.emit({"type": "polygon", "points": points})
        return True

    @property
    def polygon_vertex_count(self) -> int:
        return len(self._polygon_points)

    def clear_preview_roi(self) -> None:
        """Compatibility alias for clearing an unfinished ROI."""
        self.cancel_polygon()

    def _near_first_polygon_vertex(self, point: QPointF) -> bool:
        if len(self._polygon_points) < 3:
            return False
        first = self._image_to_widget(*self._polygon_points[0])
        return math.hypot(point.x() - first.x(), point.y() - first.y()) <= 11.0

    def keyPressEvent(self, event) -> None:
        if self._mode == "roi":
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish_polygon()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                self.undo_polygon_vertex()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_polygon()
                event.accept()
                return
        super().keyPressEvent(event)
        self.update()

    def _base_content_size(self) -> QSize:
        if self._pixmap.isNull() or self._image_width <= 0 or self._image_height <= 0:
            return QSize()
        target = QSize(int(self._image_width), int(self._image_height))
        target.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return target

    def _pan_limits(self) -> tuple[float, float]:
        target = self._base_content_size()
        width = float(target.width()) * self._zoom
        height = float(target.height()) * self._zoom
        return (max(0.0, (width - self.width()) / 2.0), max(0.0, (height - self.height()) / 2.0))

    def _clamp_pan(self) -> None:
        limit_x, limit_y = self._pan_limits()
        self._pan = QPointF(
            max(-limit_x, min(limit_x, self._pan.x())),
            max(-limit_y, min(limit_y, self._pan.y())),
        )

    def _content_rect(self) -> QRectF:
        target = self._base_content_size()
        if target.isEmpty():
            return QRectF()
        width = float(target.width()) * self._zoom
        height = float(target.height()) * self._zoom
        return QRectF(
            (self.width() - width) / 2.0 + self._pan.x(),
            (self.height() - height) / 2.0 + self._pan.y(),
            width,
            height,
        )

    def set_zoom(self, zoom: float, anchor: Optional[QPointF] = None) -> None:
        old_zoom = self._zoom
        new_zoom = max(1.0, min(8.0, float(zoom)))
        if math.isclose(old_zoom, new_zoom):
            return

        anchor_point = (
            QPointF(anchor)
            if anchor is not None
            else QPointF(self.width() / 2.0, self.height() / 2.0)
        )
        image_point = self._widget_to_image(anchor_point)
        if image_point is None:
            image_point = (self._image_width / 2.0, self._image_height / 2.0)
            anchor_point = QPointF(self.width() / 2.0, self.height() / 2.0)

        self._zoom = new_zoom
        target = self._base_content_size()
        width = float(target.width()) * self._zoom
        height = float(target.height()) * self._zoom
        self._pan = QPointF(
            anchor_point.x()
            - (self.width() - width) / 2.0
            - image_point[0] / self._image_width * width,
            anchor_point.y()
            - (self.height() - height) / 2.0
            - image_point[1] / self._image_height * height,
        )
        self._clamp_pan()
        self.zoomChanged.emit(int(round(self._zoom * 100)))
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.25)

    def reset_zoom(self) -> None:
        changed = not math.isclose(self._zoom, 1.0) or not self._pan.isNull()
        self._zoom = 1.0
        self._pan = QPointF()
        if changed:
            self.zoomChanged.emit(100)
        self.update()

    def _widget_to_image(self, point: QPointF) -> Optional[tuple[float, float]]:
        rect = self._content_rect()
        if rect.isNull() or not rect.contains(point):
            return None
        x = (point.x() - rect.x()) / rect.width() * self._image_width
        y = (point.y() - rect.y()) / rect.height() * self._image_height
        return (max(0.0, min(self._image_width, x)), max(0.0, min(self._image_height, y)))

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        rect = self._content_rect()
        return QPointF(
            rect.x() + x / self._image_width * rect.width(),
            rect.y() + y / self._image_height * rect.height(),
        )

    def _roi_to_widget_rect(self, roi: dict[str, Any]) -> QRectF:
        p1 = self._image_to_widget(float(roi["x1"]), float(roi["y1"]))
        p2 = self._image_to_widget(float(roi["x2"]), float(roi["y2"]))
        return QRectF(p1, p2).normalized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._zoom > 1.0:
            self._pan_drag_start = event.position()
            self._pan_drag_origin = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
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

        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._near_first_polygon_vertex(event.position()):
            self.finish_polygon()
            return
        self._polygon_points.append(image_point)
        self._polygon_cursor = image_point
        self.polygonDraftChanged.emit(len(self._polygon_points))
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._pan_drag_start is not None:
            delta = event.position() - self._pan_drag_start
            self._pan = self._pan_drag_origin + delta
            self._clamp_pan()
            self.update()
            event.accept()
            return
        if self._mode != "roi" or not self._polygon_points:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        self._polygon_cursor = image_point
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._mode == "roi":
            if self.finish_polygon():
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._pan_drag_start is not None:
            self._pan_drag_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.set_zoom(self._zoom * factor, event.position())
        event.accept()

    def resizeEvent(self, event) -> None:
        self._clamp_pan()
        super().resizeEvent(event)

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

        mask_pen = QPen(QColor("#ffe066"), 2)
        mask_fill = QColor(0, 210, 255, 58)
        for polygon in self._segmentation_polygons:
            if len(polygon) < 3:
                continue
            widget_polygon = QPolygonF([self._image_to_widget(x, y) for x, y in polygon])
            painter.setPen(mask_pen)
            painter.setBrush(mask_fill)
            painter.drawPolygon(widget_polygon)

        pose_points: dict[str, QPointF] = {}
        for keypoint in self._pose_keypoints:
            try:
                name = str(keypoint["name"])
                pose_points[name] = self._image_to_widget(
                    float(keypoint["x"]), float(keypoint["y"])
                )
            except (KeyError, TypeError, ValueError):
                continue

        painter.setPen(QPen(QColor("#d98cff"), 2))
        for first_name, second_name in POSE_PREVIEW_CONNECTIONS:
            first = pose_points.get(first_name)
            second = pose_points.get(second_name)
            if first is not None and second is not None:
                painter.drawLine(first, second)

        if self._tracking_bbox:
            x1, y1, x2, y2 = self._tracking_bbox
            pose_bounds = QRectF(
                self._image_to_widget(x1, y1), self._image_to_widget(x2, y2)
            ).normalized()
            painter.setPen(QPen(QColor("#54d6ff"), 2))
            painter.setBrush(QColor(0, 0, 0, 0))
            painter.drawRect(pose_bounds)

        keypoint_colors = (
            QColor("#ff5d8f"),
            QColor("#ffca3a"),
            QColor("#8ac926"),
            QColor("#4cc9f0"),
            QColor("#b892ff"),
            QColor("#ff924c"),
        )
        painter.setFont(QFont("Arial", 8, QFont.Weight.DemiBold))
        for index, keypoint in enumerate(self._pose_keypoints):
            point = pose_points.get(str(keypoint.get("name") or ""))
            if point is None:
                continue
            confidence = keypoint.get("confidence")
            try:
                low_confidence = math.isfinite(float(confidence)) and float(confidence) < 0.25
            except (TypeError, ValueError):
                low_confidence = False
            color = keypoint_colors[index % len(keypoint_colors)]
            pen = QPen(color, 2)
            if low_confidence:
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setBrush(QColor(0, 0, 0, 0))
            else:
                painter.setBrush(color)
            painter.setPen(pen)
            painter.drawEllipse(point, 4.0, 4.0)

        if pose_points:
            legend_width = 126.0
            legend_height = 8.0 + 17.0 * len(pose_points)
            available_right = self.width() - rect.right()
            legend_x = (
                rect.right() + 10.0
                if available_right >= legend_width + 18.0
                else rect.right() - legend_width - 8.0
            )
            legend_rect = QRectF(
                legend_x,
                rect.top() + 8.0,
                legend_width,
                legend_height,
            )
            painter.setPen(QPen(QColor(84, 214, 255, 150), 1))
            painter.setBrush(QColor(12, 18, 24, 215))
            painter.drawRoundedRect(legend_rect, 4.0, 4.0)
            painter.setFont(QFont("Arial", 8, QFont.Weight.DemiBold))
            legend_row = 0
            for index, keypoint in enumerate(self._pose_keypoints):
                name = str(keypoint.get("name") or "")
                if name not in pose_points:
                    continue
                color = keypoint_colors[index % len(keypoint_colors)]
                center = QPointF(
                    legend_rect.left() + 10.0,
                    legend_rect.top() + 12.0 + legend_row * 17.0,
                )
                painter.setPen(QPen(color, 1))
                painter.setBrush(color)
                painter.drawEllipse(center, 3.0, 3.0)
                painter.drawText(center + QPointF(8.0, 3.0), name)
                legend_row += 1

        roi_pen = QPen(QColor("#f5b942"), 2)
        roi_fill = QColor(245, 185, 66, 34)
        painter.setFont(QFont("Arial", 10, QFont.Weight.DemiBold))
        for roi_index in range(len(self._rois) - 1, -1, -1):
            roi = self._rois[roi_index]
            try:
                roi_type = str(roi.get("type") or "rect")
                if roi_type == "polygon":
                    points = [
                        self._image_to_widget(float(point[0]), float(point[1]))
                        for point in roi.get("points", [])
                    ]
                    if len(points) < 3:
                        continue
                    polygon = QPolygonF(points)
                    bounds = polygon.boundingRect()
                else:
                    bounds = self._roi_to_widget_rect(roi)
            except (KeyError, TypeError, ValueError):
                continue
            selected = roi_index == self._selected_roi_index
            painter.setPen(QPen(QColor("#76c7ff") if selected else roi_pen.color(), 2))
            painter.setBrush(QColor(118, 199, 255, 42) if selected else roi_fill)
            if roi_type == "polygon":
                painter.drawPolygon(polygon)
            else:
                painter.drawRect(bounds)
            label_rect = QRectF(
                bounds.x() + 4, bounds.y() + 4, min(max(bounds.width() - 8, 40), 180), 20
            )
            painter.fillRect(label_rect, QColor(17, 24, 32, 190))
            painter.setPen(QColor("#f9d782"))
            painter.drawText(
                label_rect.adjusted(5, 0, -5, 0),
                Qt.AlignmentFlag.AlignVCenter,
                str(roi.get("name", "ROI")),
            )

        if self._polygon_points:
            widget_points = [self._image_to_widget(x, y) for x, y in self._polygon_points]
            painter.setPen(QPen(QColor("#76c7ff"), 2, Qt.PenStyle.DashLine))
            if len(widget_points) > 1:
                painter.drawPolyline(QPolygonF(widget_points))
            if self._polygon_cursor is not None:
                painter.drawLine(widget_points[-1], self._image_to_widget(*self._polygon_cursor))
            painter.setBrush(QColor("#76c7ff"))
            for index, point in enumerate(widget_points):
                radius = 6.0 if index == 0 and len(widget_points) >= 3 else 4.0
                painter.drawEllipse(point, radius, radius)

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

    def __init__(
        self,
        parent,
        *,
        project_root: str,
        app_base_dir: str,
        layer_id: str = "",
    ):
        super().__init__(parent)
        self.layer_id = normalize_layer_id(layer_id)
        self.layer = layer_definition(self.layer_id)
        self.setWindowTitle("Analysis — Project Video")
        self.setSizeGripEnabled(True)
        self.resize(1240, 900)
        self.setMinimumSize(1040, 680)
        self.project_root = os.path.abspath(project_root)
        self.app_base_dir = os.path.abspath(app_base_dir)
        self.analysis_controller: Optional[WorkerJobController] = None
        self.analysis_process: Optional[QProcess] = None
        self.analysis_config_path: Optional[str] = None
        self.last_output_dir = ""
        self.annotation_state = AnalysisAnnotationState()
        self.analysis_inputs = {LAYER_KEYPOINTS: "", LAYER_SEGMENTATION: ""}
        self._selected_bundle: Optional[ProjectAnalysisBundle] = None
        self._active_setup_video_name = ""
        self._suspend_setup_persistence = False

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

        # Keep the path fields as the worker-facing source of truth, but drive
        # them from one project-aware selector instead of two independent pickers.
        self.csv_edit = QLineEdit(self)
        self.csv_edit.hide()
        self.csv_edit.textChanged.connect(self._refresh_default_output_dir)
        self.video_edit = QLineEdit(self)
        self.video_edit.hide()

        self.project_video_combo = ThemedComboBox(self)
        self.project_video_combo.setObjectName("AnalysisProjectVideoCombo")
        self.project_video_combo.setMinimumWidth(260)
        self.project_video_combo.currentIndexChanged.connect(self._project_video_changed)
        self.other_inputs_btn = QPushButton("Other…", self)
        self.other_inputs_btn.setToolTip(
            "Choose an inference CSV outside the project; its video is detected automatically."
        )
        self.other_inputs_btn.clicked.connect(self._browse_other_inputs)
        project_video_row = QHBoxLayout()
        project_video_row.addWidget(self.project_video_combo, 1)
        project_video_row.addWidget(self.other_inputs_btn)
        input_form.addRow("Project video:", project_video_row)

        self.analysis_mode_combo = ThemedComboBox(self)
        self.analysis_mode_combo.setObjectName("AnalysisModeCombo")
        self.analysis_mode_combo.currentIndexChanged.connect(self._analysis_mode_changed)
        input_form.addRow("Analyze:", self.analysis_mode_combo)

        self.input_detail_label = QLabel(self)
        self.input_detail_label.setObjectName("AnalysisInputDetail")
        self.input_detail_label.setWordWrap(True)
        input_form.addRow("", self.input_detail_label)

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
        self.real_distance_spin.valueChanged.connect(self._real_distance_changed)
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
        self.min_cutoff_spin.setValue(DEFAULT_ONE_EURO_MIN_CUTOFF)
        self.min_cutoff_spin.setPrefix("min ")
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 100.0)
        self.beta_spin.setDecimals(3)
        self.beta_spin.setValue(DEFAULT_ONE_EURO_BETA)
        self.beta_spin.setPrefix("beta ")
        filter_row.addWidget(self.min_cutoff_spin)
        filter_row.addWidget(self.beta_spin)
        filter_row.addStretch(1)
        tracking_form.addRow("Filter:", filter_row)

        tracking_layout.addLayout(tracking_form)
        self.setup_persistence_label = QLabel(
            "Scale and ROIs save automatically for each project video."
        )
        self.setup_persistence_label.setObjectName("AnalysisInputDetail")
        self.setup_persistence_label.setWordWrap(True)
        tracking_layout.addWidget(self.setup_persistence_label)
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
        self.roi_mode_btn = QPushButton("Polygon ROI")
        self.roi_mode_btn.setCheckable(True)
        self.mode_group.addButton(self.scale_mode_btn)
        self.mode_group.addButton(self.roi_mode_btn)
        self.scale_mode_btn.toggled.connect(
            lambda checked: checked and self._set_annotation_mode("scale")
        )
        self.roi_mode_btn.toggled.connect(
            lambda checked: checked and self._set_annotation_mode("roi")
        )
        toolbar.addWidget(self.scale_mode_btn)

        self.clear_scale_btn = QPushButton("Clear Scale")
        self.clear_scale_btn.clicked.connect(self._clear_scale)
        toolbar.addWidget(self.clear_scale_btn)
        self.clear_rois_btn = QPushButton("Clear ROIs")
        self.clear_rois_btn.clicked.connect(self._clear_rois)
        toolbar.addStretch(1)

        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedWidth(34)
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.clicked.connect(lambda: self.frame_view.zoom_out())
        toolbar.addWidget(self.zoom_out_btn)
        self.zoom_status_label = QLabel("100%")
        self.zoom_status_label.setObjectName("AnalysisHintLabel")
        self.zoom_status_label.setMinimumWidth(38)
        self.zoom_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self.zoom_status_label)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(34)
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.clicked.connect(lambda: self.frame_view.zoom_in())
        toolbar.addWidget(self.zoom_in_btn)
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setFixedWidth(44)
        self.zoom_fit_btn.setToolTip("Fit the full frame")
        self.zoom_fit_btn.clicked.connect(lambda: self.frame_view.reset_zoom())
        toolbar.addWidget(self.zoom_fit_btn)

        self.scale_status_label = QLabel("")
        self.scale_status_label.setObjectName("AnalysisStatusLabel")
        toolbar.addWidget(self.scale_status_label)
        workspace_layout.addLayout(toolbar)

        polygon_tools = QHBoxLayout()
        polygon_tools.addWidget(self.roi_mode_btn)
        self.polygon_help_label = QLabel(
            "Polygon: click vertices, then click the first point or press Enter to finish."
        )
        self.polygon_help_label.setObjectName("AnalysisHintLabel")
        polygon_tools.addStretch(1)
        self.undo_vertex_btn = QPushButton("Undo")
        self.undo_vertex_btn.setToolTip("Remove the last polygon vertex (Backspace)")
        self.undo_vertex_btn.clicked.connect(lambda: self.frame_view.undo_polygon_vertex())
        self.finish_roi_btn = QPushButton("Finish")
        self.finish_roi_btn.setToolTip("Finish the polygon ROI (Enter)")
        self.finish_roi_btn.clicked.connect(lambda: self.frame_view.finish_polygon())
        self.cancel_roi_btn = QPushButton("Cancel")
        self.cancel_roi_btn.setToolTip("Discard the unfinished polygon (Escape)")
        self.cancel_roi_btn.clicked.connect(lambda: self.frame_view.cancel_polygon())
        polygon_tools.addWidget(self.undo_vertex_btn)
        polygon_tools.addWidget(self.finish_roi_btn)
        polygon_tools.addWidget(self.cancel_roi_btn)
        self.clear_rois_btn.setText("Clear All")
        self.clear_rois_btn.setToolTip("Delete every completed ROI")
        polygon_tools.addWidget(self.clear_rois_btn)
        workspace_layout.addLayout(polygon_tools)
        workspace_layout.addWidget(self.polygon_help_label)

        self.frame_view = FrameAnnotationView()
        self.frame_view.scaleDistanceChanged.connect(self._apply_scale_distance)
        self.frame_view.scalePointsChanged.connect(self._set_scale_points)
        self.frame_view.roiDrawn.connect(self._add_roi_from_canvas)
        self.frame_view.polygonDraftChanged.connect(self._update_polygon_draft_controls)
        self.frame_view.zoomChanged.connect(
            lambda percent: self.zoom_status_label.setText(f"{percent}%")
        )
        workspace_layout.addWidget(self.frame_view, 1)

        roi_row = QHBoxLayout()
        roi_column = QVBoxLayout()
        roi_title = QLabel("ROIs")
        roi_title.setObjectName("AnalysisPanelTitle")
        roi_column.addWidget(roi_title)
        self.roi_priority_hint = QLabel("Priority: top ROI wins wherever shapes overlap.")
        self.roi_priority_hint.setObjectName("AnalysisHintLabel")
        roi_column.addWidget(self.roi_priority_hint)
        self.roi_list = QListWidget()
        self.roi_list.setObjectName("AnalysisRoiList")
        self.roi_list.setMaximumHeight(120)
        self.roi_list.currentRowChanged.connect(self._roi_selection_changed)
        self.roi_list.itemDoubleClicked.connect(lambda _item: self._rename_selected_roi())
        roi_column.addWidget(self.roi_list)
        roi_row.addLayout(roi_column, 1)
        roi_actions = QVBoxLayout()
        self.raise_roi_btn = QPushButton("Higher Priority")
        self.raise_roi_btn.setToolTip("Move the selected ROI toward the top of the priority list")
        self.raise_roi_btn.clicked.connect(lambda: self._move_selected_roi(-1))
        roi_actions.addWidget(self.raise_roi_btn)
        self.lower_roi_btn = QPushButton("Lower Priority")
        self.lower_roi_btn.setToolTip(
            "Move the selected ROI toward the bottom of the priority list"
        )
        self.lower_roi_btn.clicked.connect(lambda: self._move_selected_roi(1))
        roi_actions.addWidget(self.lower_roi_btn)
        self.rename_roi_btn = QPushButton("Rename")
        self.rename_roi_btn.clicked.connect(self._rename_selected_roi)
        roi_actions.addWidget(self.rename_roi_btn)
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
        self._update_polygon_draft_controls(0)
        self._load_preview_frame(silent=True)

    @property
    def scale_points(self) -> list[tuple[float, float]]:
        """Compatibility view of the selected scale points."""
        return list(self.annotation_state.scale_points)

    @scale_points.setter
    def scale_points(self, points: list[tuple[float, float]]) -> None:
        self.annotation_state.set_scale_points(points)

    @property
    def pixel_distance_px(self) -> float:
        """Compatibility alias for the domain state's pixel distance."""
        return self.annotation_state.pixel_distance

    @pixel_distance_px.setter
    def pixel_distance_px(self, distance: float) -> None:
        self.annotation_state.set_pixel_distance(distance)

    @property
    def rois(self) -> list[dict[str, Any]]:
        """Compatibility export using the worker's existing ROI dictionaries."""
        return self.annotation_state.worker_rois()

    @rois.setter
    def rois(self, rois: list[dict[str, Any]]) -> None:
        self.annotation_state.replace_rois(rois)

    def _candidate_csv_dirs(self) -> list[str]:
        return [
            os.path.join(self.project_root, "inference outputs", LAYER_KEYPOINTS),
            os.path.join(self.project_root, "inference outputs", LAYER_SEGMENTATION),
            os.path.join(self.project_root, "inference outputs"),
            os.path.join(self.app_base_dir, "analysis_toolset", "inference outputs"),
        ]

    def _select_initial_csv(self) -> None:
        self._populate_project_video_selector()

    def _populate_project_video_selector(self) -> None:
        current_video = self.video_edit.text().strip()
        options = project_analysis_bundles(self.project_root)
        self.project_video_combo.blockSignals(True)
        self.project_video_combo.clear()
        self.project_video_combo.addItem("Choose a project video…", None)
        selected_index = 0
        newest_index = 0
        newest_created_at = ""
        ready_count = 0
        for option in options:
            if option.both_ready:
                suffix = "Pose + Segmentation"
                ready_count += 1
            elif option.keypoints_csv:
                suffix = "Pose only"
                ready_count += 1
            elif option.segmentation_csv:
                suffix = "Segmentation only"
                ready_count += 1
            else:
                suffix = "No inference"
            self.project_video_combo.addItem(f"{option.video_name}  ·  {suffix}", option)
            index = self.project_video_combo.count() - 1
            self.project_video_combo.setItemData(
                index,
                f"{option.video_name} — {suffix}",
                Qt.ItemDataRole.ToolTipRole,
            )
            if current_video and os.path.realpath(option.video_path) == os.path.realpath(
                current_video
            ):
                selected_index = index
            created_at = max(option.keypoints_created_at, option.segmentation_created_at)
            if option.inference_ready and created_at >= newest_created_at:
                newest_created_at = created_at
                newest_index = index
        if selected_index == 0:
            selected_index = newest_index
        self.project_video_combo.setCurrentIndex(selected_index)
        self.project_video_combo.blockSignals(False)
        self._project_video_changed(selected_index)
        if not options:
            self.input_detail_label.setText(
                "No project videos found. Add videos from the Videos menu or use Other."
            )
        elif ready_count == 0 and selected_index == 0:
            self.input_detail_label.setText("No project videos have analysis-ready inference yet.")

    def _project_video_changed(self, index: int) -> None:
        self._save_analysis_setup()
        self._active_setup_video_name = ""
        self._clear_annotations(persist=False)
        option = self.project_video_combo.itemData(index)
        if not isinstance(option, ProjectAnalysisBundle):
            self._selected_bundle = None
            self.analysis_inputs = {LAYER_KEYPOINTS: "", LAYER_SEGMENTATION: ""}
            self.analysis_mode_combo.clear()
            self.csv_edit.clear()
            self.video_edit.clear()
            if self.project_video_combo.count() > 1:
                self.input_detail_label.setText("Select a project video to load its inference.")
            self.setup_persistence_label.setText(
                "Select a project video to restore its saved scale and ROIs."
            )
            return
        self._selected_bundle = option
        self.analysis_inputs = {
            LAYER_KEYPOINTS: option.keypoints_csv,
            LAYER_SEGMENTATION: option.segmentation_csv,
        }
        self.video_edit.setText(option.video_path)
        self._populate_analysis_modes(option.available_layers)
        details: list[str] = []
        if option.keypoints_csv:
            details.append(f"Pose: {os.path.basename(option.keypoints_csv)}")
        if option.segmentation_csv:
            details.append(f"Segmentation: {os.path.basename(option.segmentation_csv)}")
        self.input_detail_label.setText(
            " · ".join(details) if details else "Run inference for this video before analysis."
        )
        self._load_preview_frame(silent=True)
        self._active_setup_video_name = option.video_name
        self._restore_analysis_setup()

    def _populate_analysis_modes(self, available_layers: tuple[str, ...]) -> None:
        self.analysis_mode_combo.blockSignals(True)
        self.analysis_mode_combo.clear()
        available = set(available_layers)
        if {LAYER_KEYPOINTS, LAYER_SEGMENTATION}.issubset(available):
            self.analysis_mode_combo.addItem("Both — Pose + Segmentation", "both")
        if LAYER_KEYPOINTS in available:
            self.analysis_mode_combo.addItem("Pose only", LAYER_KEYPOINTS)
        if LAYER_SEGMENTATION in available:
            self.analysis_mode_combo.addItem("Segmentation only", LAYER_SEGMENTATION)
        preferred_index = self.analysis_mode_combo.findData(self.layer_id)
        if self.analysis_mode_combo.findData("both") >= 0:
            preferred_index = self.analysis_mode_combo.findData("both")
        self.analysis_mode_combo.setCurrentIndex(max(preferred_index, 0))
        self.analysis_mode_combo.blockSignals(False)
        self._analysis_mode_changed(self.analysis_mode_combo.currentIndex())

    def _analysis_mode(self) -> str:
        return str(self.analysis_mode_combo.currentData() or self.layer_id)

    def _analysis_mode_changed(self, _index: int) -> None:
        mode = self._analysis_mode()
        csv_layer = LAYER_SEGMENTATION if mode == LAYER_SEGMENTATION else LAYER_KEYPOINTS
        csv_path = self.analysis_inputs.get(csv_layer) or next(
            (path for path in self.analysis_inputs.values() if path), ""
        )
        self.csv_edit.setText(csv_path)
        bundle = self._selected_bundle
        if mode == "both" and bundle is not None:
            output = default_combined_analysis_output_dir(self.project_root, bundle.video_name)
        elif csv_path:
            output = default_analysis_output_dir(
                self.project_root,
                csv_layer,
                csv_path,
                video_name=bundle.video_name if bundle is not None else self.video_edit.text(),
            )
        else:
            output = ""
        self.output_edit.setText(output)
        if hasattr(self, "frame_view"):
            self._load_preview_frame(silent=True)

    def _csv_matches_active_layer(self, path: str) -> bool:
        return analysis_csv_matches_layer(path, self.layer_id)

    def _default_output_dir_for_csv(self, csv_path: str) -> str:
        layer_id = (
            LAYER_SEGMENTATION
            if analysis_csv_matches_layer(csv_path, LAYER_SEGMENTATION)
            else LAYER_KEYPOINTS
        )
        return default_analysis_output_dir(
            self.project_root,
            layer_id,
            csv_path,
            video_name=self.video_edit.text().strip(),
        )

    def _refresh_default_output_dir(self) -> None:
        if self.output_edit.text().strip():
            return
        self.output_edit.setText(self._default_output_dir_for_csv(self.csv_edit.text().strip()))

    def _browse_csv(self) -> None:
        start = os.path.dirname(self.csv_edit.text().strip())
        if not start or not os.path.isdir(start):
            start = next(
                (folder for folder in self._candidate_csv_dirs() if os.path.isdir(folder)),
                self.project_root,
            )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select inference CSV",
            start,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self._save_analysis_setup()
            self._active_setup_video_name = ""
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
            self._save_analysis_setup()
            self._active_setup_video_name = ""
            self.video_edit.setText(path)
            self._clear_annotations()
            self._load_preview_frame(silent=False)

    def _browse_other_inputs(self) -> None:
        start = next(
            (folder for folder in self._candidate_csv_dirs() if os.path.isdir(folder)),
            self.project_root,
        )
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.layer.display_name.lower()} inference CSV",
            start,
            "CSV files (*.csv);;All files (*.*)",
        )
        if not csv_path:
            return
        if not self._csv_matches_active_layer(csv_path):
            QMessageBox.warning(
                self,
                "Wrong Inference Layer",
                f"That CSV is not a {self.layer.display_name.lower()} inference output.",
            )
            return
        video_path = inspect_analysis_csv(csv_path).video_path
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select matching video file",
                self.project_root,
                "Video files (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*.*)",
            )
            if not video_path:
                return
        self.project_video_combo.blockSignals(True)
        self.project_video_combo.setCurrentIndex(0)
        self.project_video_combo.blockSignals(False)
        self._save_analysis_setup()
        self._active_setup_video_name = ""
        self._selected_bundle = None
        self.analysis_inputs = {LAYER_KEYPOINTS: "", LAYER_SEGMENTATION: ""}
        self.analysis_inputs[self.layer_id] = csv_path
        self._populate_analysis_modes((self.layer_id,))
        self.csv_edit.setText(csv_path)
        self.video_edit.setText(video_path)
        self.output_edit.setText(self._default_output_dir_for_csv(csv_path))
        self.input_detail_label.setText(
            f"External files · {os.path.basename(video_path)} · {os.path.basename(csv_path)}"
        )
        self._clear_annotations()
        self._load_preview_frame(silent=False)
        self.setup_persistence_label.setText(
            "External inputs are not attached to a project-video setup."
        )

    def _browse_output_dir(self) -> None:
        start = self.output_edit.text().strip() or os.path.join(
            self.project_root, "analysis outputs"
        )
        path = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if path:
            self.output_edit.setText(path)

    def _video_path_from_csv(self) -> str:
        csv_path = next((path for path in self.analysis_inputs.values() if path), "")
        return inspect_analysis_csv(csv_path or self.csv_edit.text().strip()).video_path

    def _frame_dimensions_from_csv(self) -> tuple[int, int]:
        csv_path = (
            self.analysis_inputs.get(LAYER_SEGMENTATION)
            or self.analysis_inputs.get(LAYER_KEYPOINTS)
            or self.csv_edit.text().strip()
        )
        context = inspect_analysis_csv(csv_path)
        return (context.width, context.height)

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

    def _load_video_pixmap(
        self, video_path: str, frame_index: int = 0
    ) -> Optional[tuple[QPixmap, int, int]]:
        if not video_path or not os.path.isfile(video_path):
            return None
        try:
            import cv2
        except Exception:
            return None
        cap = cv2.VideoCapture(video_path)
        try:
            if frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(
                rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
            ).copy()
            return (QPixmap.fromImage(image), width, height)
        finally:
            cap.release()

    def _load_preview_frame(self, *, silent: bool) -> None:
        video_path = self.video_edit.text().strip()
        if not video_path:
            video_path = self._video_path_from_csv()
            if video_path:
                self.video_edit.setText(video_path)

        segmentation_csv = self.analysis_inputs.get(LAYER_SEGMENTATION) or ""
        segmentation_preview = (
            load_segmentation_preview(segmentation_csv) if segmentation_csv else None
        )
        polygons = (
            [list(polygon) for polygon in segmentation_preview.polygons]
            if segmentation_preview
            else []
        )
        pose_csv = self.analysis_inputs.get(LAYER_KEYPOINTS) or ""
        pose_preview = (
            load_pose_preview(
                pose_csv,
                frame_index=segmentation_preview.frame_index if polygons else None,
            )
            if pose_csv
            else None
        )
        preview_frame = (
            segmentation_preview.frame_index
            if polygons and segmentation_preview is not None
            else pose_preview.frame_index
            if pose_preview is not None
            else 0
        )
        pose_bbox = pose_preview.bbox if pose_preview is not None else ()
        segmentation_bbox = (
            segmentation_preview.primary_bbox if segmentation_preview is not None else ()
        )
        tracking_bbox = (
            segmentation_bbox
            if self._analysis_mode() in {"both", LAYER_SEGMENTATION} and segmentation_bbox
            else pose_bbox
        )
        pose_keypoints = (
            [
                {
                    "name": keypoint.name,
                    "x": keypoint.x,
                    "y": keypoint.y,
                    "confidence": keypoint.confidence,
                }
                for keypoint in pose_preview.keypoints
            ]
            if pose_preview is not None
            else []
        )

        loaded = self._load_video_pixmap(video_path, preview_frame)
        if loaded is not None:
            pixmap, width, height = loaded
            self.annotation_state.set_frame_dimensions(width, height)
            self.frame_view.set_frame(pixmap, width, height)
            self.frame_view.set_segmentation_polygons(polygons)
            self.frame_view.set_pose_overlay(tracking_bbox, pose_keypoints)
            overlay_parts = [f"frame {preview_frame}"]
            if polygons:
                overlay_parts.append(f"{len(polygons)} mask(s)")
            if pose_keypoints:
                overlay_parts.append(f"{len(pose_keypoints)} keypoints")
            self.frame_info_label.setText(
                f"{width} x {height} | {os.path.basename(video_path)} | "
                + " | ".join(overlay_parts)
            )
            return

        if video_path and not silent:
            QMessageBox.information(
                self,
                "Frame preview",
                "Could not read the selected video. Showing CSV coordinates instead.",
            )

        width, height = self._frame_dimensions_from_csv()
        self.annotation_state.set_frame_dimensions(width, height)
        self.frame_view.set_frame(self._blank_frame_pixmap(width, height), width, height)
        self.frame_view.set_segmentation_polygons(polygons)
        self.frame_view.set_pose_overlay(tracking_bbox, pose_keypoints)
        overlay_status = f" | frame {preview_frame}"
        if polygons:
            overlay_status += f" | {len(polygons)} mask(s)"
        if pose_keypoints:
            overlay_status += f" | {len(pose_keypoints)} keypoints"
        self.frame_info_label.setText(f"{width} x {height} | CSV coordinates{overlay_status}")

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
        self.annotation_state.set_scale_points(points)
        self._update_scale_label()
        self._save_analysis_setup()

    def _apply_scale_distance(self, distance: float) -> None:
        if distance > 0:
            self.annotation_state.set_pixel_distance(distance)
            self.pixel_distance_label.setText(f"{distance:.1f} px")
        self._update_scale_label()
        self._save_analysis_setup()

    def _clear_scale(self) -> None:
        self.annotation_state.clear_scale()
        self.pixel_distance_label.setText("Draw scale")
        self.frame_view.set_scale_points([])
        self._update_scale_label()
        self._save_analysis_setup()

    def _clear_rois(self) -> None:
        frame_view = getattr(self, "frame_view", None)
        if frame_view is not None:
            frame_view.cancel_polygon()
        self.annotation_state.clear_rois()
        self._refresh_roi_list()
        self._save_analysis_setup(rois_cleared=True)

    def _clear_annotations(self, *, persist: bool = True) -> None:
        previous = self._suspend_setup_persistence
        self._suspend_setup_persistence = True
        try:
            self._clear_scale()
            self._clear_rois()
        finally:
            self._suspend_setup_persistence = previous
        if persist:
            self._save_analysis_setup()

    def _real_distance_changed(self, _value: float) -> None:
        self._update_scale_label()
        self._save_analysis_setup()

    def _update_scale_label(self) -> None:
        real_distance = self.real_distance_spin.value()
        self.annotation_state.set_real_world_distance(real_distance)
        mm_per_pixel = self.annotation_state.mm_per_pixel
        if mm_per_pixel is not None:
            self.scale_status_label.setText(
                f"{mm_per_pixel:.4f} mm/px | {len(self.scale_points)}/2"
            )
        else:
            self.scale_status_label.setText(f"Scale unset | {len(self.scale_points)}/2")

    def _save_analysis_setup(self, *, rois_cleared: bool = False) -> None:
        video_name = str(self._active_setup_video_name or "")
        if self._suspend_setup_persistence or not video_name:
            return
        try:
            save_video_analysis_setup(
                self.project_root,
                video_name,
                frame_width=self.annotation_state.frame.width,
                frame_height=self.annotation_state.frame.height,
                scale_points=self.annotation_state.scale_points,
                real_world_distance_mm=self.real_distance_spin.value(),
                rois=self.annotation_state.worker_rois(),
                rois_cleared=rois_cleared,
            )
        except (OSError, TypeError, ValueError) as exc:
            self.setup_persistence_label.setText(f"Could not save video setup: {exc}")
            return
        self.setup_persistence_label.setText(f"Setup saved automatically for {video_name}.")

    def _restore_analysis_setup(self) -> None:
        video_name = str(self._active_setup_video_name or "")
        if not video_name:
            return
        try:
            setup = load_video_analysis_setup(self.project_root, video_name)
        except (OSError, TypeError, ValueError) as exc:
            self.setup_persistence_label.setText(f"Could not restore video setup: {exc}")
            return
        if setup is None:
            self.setup_persistence_label.setText(
                f"No saved setup for {video_name} yet · changes save automatically."
            )
            return
        current_width = self.annotation_state.frame.width
        current_height = self.annotation_state.frame.height
        saved_size = (setup.frame_width, setup.frame_height)
        current_size = (current_width, current_height)
        if all(saved_size) and all(current_size) and saved_size != current_size:
            self.setup_persistence_label.setText(
                "Saved setup was not loaded because the video's frame size changed "
                f"({setup.frame_width}×{setup.frame_height} → {current_width}×{current_height})."
            )
            return

        previous = self._suspend_setup_persistence
        self._suspend_setup_persistence = True
        try:
            self.annotation_state.clear()
            self.annotation_state.set_scale_points(setup.scale_points)
            self.annotation_state.replace_rois(setup.rois)
            self.real_distance_spin.setValue(setup.real_world_distance_mm)
            self.frame_view.set_scale_points(list(setup.scale_points))
            if len(setup.scale_points) == 2:
                self.pixel_distance_label.setText(f"{self.annotation_state.pixel_distance:.1f} px")
            else:
                self.pixel_distance_label.setText("Draw scale")
            self._update_scale_label()
            self._refresh_roi_list()
        finally:
            self._suspend_setup_persistence = previous
        self.setup_persistence_label.setText(
            f"Restored saved scale and {len(setup.rois)} ROI"
            f"{'s' if len(setup.rois) != 1 else ''} for {video_name}."
        )

    def _add_roi_from_canvas(self, roi: dict[str, Any]) -> None:
        default_name = f"ROI {len(self.rois) + 1}"
        try:
            self.annotation_state.add_roi(roi, name=default_name)
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid ROI", str(exc))
            return
        self._refresh_roi_list()
        self.roi_list.setCurrentRow(self.roi_list.count() - 1)
        self._save_analysis_setup()

    def _rename_selected_roi(self) -> None:
        row = self.roi_list.currentRow()
        if not 0 <= row < len(self.annotation_state.rois):
            return
        current_name = self.annotation_state.rois[row].name
        name, accepted = QInputDialog.getText(self, "Rename ROI", "ROI name:", text=current_name)
        if accepted and self.annotation_state.rename_roi(row, name):
            self._refresh_roi_list()
            self.roi_list.setCurrentRow(row)
            self._save_analysis_setup()

    def _move_selected_roi(self, offset: int) -> None:
        row = self.roi_list.currentRow()
        new_row = self.annotation_state.move_roi(row, offset)
        if new_row < 0:
            return
        self._refresh_roi_list()
        self.roi_list.setCurrentRow(new_row)
        self._save_analysis_setup()

    def _roi_selection_changed(self, row: int) -> None:
        self.frame_view.set_selected_roi(row)
        count = len(self.annotation_state.rois)
        valid = 0 <= row < count
        self.raise_roi_btn.setEnabled(valid and row > 0)
        self.lower_roi_btn.setEnabled(valid and row < count - 1)
        self.rename_roi_btn.setEnabled(valid)
        self.delete_roi_btn.setEnabled(valid)

    def _update_polygon_draft_controls(self, vertex_count: int) -> None:
        count = max(0, int(vertex_count))
        drawing = count > 0
        if hasattr(self, "undo_vertex_btn"):
            self.undo_vertex_btn.setEnabled(drawing)
            self.finish_roi_btn.setEnabled(count >= 3)
            self.cancel_roi_btn.setEnabled(drawing)
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(not drawing)
        if hasattr(self, "polygon_help_label"):
            if drawing:
                self.polygon_help_label.setText(
                    f"Drawing polygon · {count} vertex{'es' if count != 1 else ''} · "
                    "Enter/first point finishes · Backspace undoes · Esc cancels"
                )
            else:
                self.polygon_help_label.setText(
                    "Polygon: click vertices, then click the first point or press Enter to finish."
                )

    def _delete_selected_roi(self) -> None:
        row = self.roi_list.currentRow()
        if self.annotation_state.delete_roi(row):
            self._refresh_roi_list()
            self._save_analysis_setup()

    def _refresh_roi_list(self) -> None:
        selected_row = self.roi_list.currentRow()
        self.roi_list.clear()
        state_rois = self.annotation_state.rois
        for index, roi in enumerate(state_rois, start=1):
            if roi.type == "polygon":
                detail = f"Polygon · {len(roi.points)} vertices · {roi.area:.0f}px²"
            else:
                detail = f"Legacy rectangle · {roi.width:.0f} x {roi.height:.0f}px"
            item = QListWidgetItem(f"P{index} · {roi.name}  ·  {detail}")
            self.roi_list.addItem(item)
        count = len(state_rois)
        self.roi_count_label.setText(f"{count} ROI{'s' if count != 1 else ''}")
        self.frame_view.set_rois(self.annotation_state.worker_rois())
        if count:
            self.roi_list.setCurrentRow(min(max(selected_row, 0), count - 1))
        else:
            self.frame_view.set_selected_roi(-1)
            self._roi_selection_changed(-1)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text.rstrip() + "\n")
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.ensureCursorVisible()

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running and self.frame_view.polygon_vertex_count == 0)
        self.load_frame_btn.setEnabled(not running)
        if running:
            self.status_label.setText("Running")
            self.progress.setRange(0, 8)
            self.progress.setValue(0)

    def _config_payload(self) -> dict[str, Any]:
        return self._build_analysis_config().as_dict()

    def _build_analysis_config(self) -> AnalysisRunConfig:
        self.annotation_state.set_real_world_distance(self.real_distance_spin.value())
        pixel_distance = (
            self.annotation_state.pixel_distance
            if len(self.annotation_state.scale_points) >= 2
            else 0.0
        )
        return build_analysis_job_config(
            analysis_mode=self._analysis_mode(),
            analysis_inputs=self.analysis_inputs,
            video_path=self.video_edit.text(),
            output_dir=self.output_edit.text(),
            pixel_distance=pixel_distance,
            real_world_distance_mm=self.real_distance_spin.value(),
            smooth=self.smooth_check.isChecked(),
            min_cutoff=self.min_cutoff_spin.value(),
            beta=self.beta_spin.value(),
            make_plots=self.plots_check.isChecked(),
            make_annotated_video=self.annotated_video_check.isChecked(),
            run_clustering=self.cluster_check.isChecked(),
            export_cluster_clips=self.cluster_clips_check.isChecked(),
            umap_neighbors=self.umap_neighbors_spin.value(),
            umap_min_dist=self.umap_min_dist_spin.value(),
            hdbscan_min_cluster_size=self.hdbscan_min_cluster_size_spin.value(),
            cluster_clip_length_sec=self.clip_length_spin.value(),
            samples_per_cluster=self.samples_per_cluster_spin.value(),
            rois=self.annotation_state.worker_rois(),
        )

    def _validate_inputs(self) -> bool:
        if self.frame_view.polygon_vertex_count:
            QMessageBox.warning(
                self,
                "Finish ROI",
                "Finish or cancel the polygon currently being drawn before running analysis.",
            )
            return False
        try:
            config = self._build_analysis_config()
        except AnalysisConfigError as exc:
            QMessageBox.warning(self, exc.title, exc.message)
            return False
        self._validated_analysis_config = config
        if config.video_fallback_notice:
            QMessageBox.information(
                self,
                "Video optional",
                "No video was selected. The worker will try to use the video_path stored in the CSV.",
            )
        return True

    def _start_analysis(self) -> None:
        if self.analysis_controller is not None and self.analysis_controller.is_running:
            QMessageBox.information(self, "Analysis running", "An analysis job is already running.")
            return
        if not self._validate_inputs():
            return

        config = getattr(self, "_validated_analysis_config", None)
        if config is None:
            config = self._build_analysis_config()
        self._validated_analysis_config = None
        payload = config.as_dict()
        os.makedirs(payload["output_dir"], exist_ok=True)
        try:
            self.analysis_config_path = create_worker_config(
                self.project_root,
                os.path.join(self.project_root, "logs"),
                "analysis",
                payload,
            )
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Analysis failed",
                f"Could not create the analysis worker configuration.\n\n{exc}",
            )
            return
        self.last_output_dir = payload["output_dir"]
        self.open_output_btn.setEnabled(False)
        self.log_view.clear()
        self.summary_view.clear()
        self._append_log(f"Detections: {payload['detections_csv']}")
        self._append_log(f"Mode: {payload.get('analysis_mode', self._analysis_mode())}")
        for layer_id, csv_path in payload.get("analysis_inputs", {}).items():
            self._append_log(f"{layer_id}: {csv_path}")
        self._append_log(f"Output: {payload['output_dir']}")
        self._append_log(f"ROIs: {len(self.rois)}")

        controller = WorkerJobController(self)
        self.analysis_controller = controller
        controller.event_received.connect(self._handle_worker_event)
        controller.output_received.connect(self._append_log)
        controller.stderr_received.connect(self._append_log)
        controller.terminal.connect(self._analysis_job_finished)
        self._set_running(True)
        started = controller.start(
            sys.executable,
            [
                os.path.join(self.app_base_dir, "analysis_worker.py"),
                "--config",
                self.analysis_config_path,
            ],
            config_path=self.analysis_config_path,
        )
        self.analysis_process = controller.process if controller.terminal_result is None else None
        if not started:
            QMessageBox.warning(self, "Analysis failed", "Could not start the analysis worker.")

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
        elif summary.get("analysis_kind") in {"combined", "pose_and_segmentation"}:
            source_counts = summary.get("centroid_source_counts") or {}
            qc_counts = summary.get("prediction_qc_status_counts") or {}
            qc_reasons = summary.get("prediction_qc_reason_counts") or {}
            lines.extend(
                [
                    f"Pose-valid frames: {int(summary.get('pose_valid_frames') or 0)}",
                    "Segmentation-valid frames: "
                    f"{int(summary.get('segmentation_valid_frames') or 0)}",
                    "Centroid sources: "
                    + ", ".join(
                        f"{source}={int(count)}" for source, count in source_counts.items()
                    ),
                    "Prediction QC: "
                    + ", ".join(
                        f"{status}={int(qc_counts.get(status) or 0)}"
                        for status in ("good", "warning", "bad")
                    ),
                ]
            )
            if qc_reasons:
                lines.append(
                    "QC reasons: "
                    + ", ".join(f"{reason}={int(count)}" for reason, count in qc_reasons.items())
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

        self._handle_worker_event(event)

    def _handle_worker_event(self, event: dict[str, Any]) -> None:
        """Apply a parsed worker protocol event to the dialog UI."""

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
            for layer_id, result in (event.get("results_by_layer") or {}).items():
                self._append_log(f"{layer_id} features: {result.get('feature_csv', '')}")
            for layer_id, error in (event.get("errors_by_layer") or {}).items():
                self._append_log(f"{layer_id} warning: {error}")
            if event.get("segmentation_detections_csv"):
                self._append_log(
                    f"Segmentation detections: {event.get('segmentation_detections_csv')}"
                )
            if event.get("roi_summary_csv"):
                self._append_log(f"ROI summary: {event.get('roi_summary_csv')}")
            if event.get("prediction_qc_csv"):
                self._append_log(f"Prediction QC: {event.get('prediction_qc_csv')}")
            self._show_result_summary(event)
            self.open_output_btn.setEnabled(bool(self.last_output_dir))
        elif kind == "error":
            self.status_label.setText("Failed")
            self._append_log(f"Error: {event.get('error_message', '')}")
        else:
            self._append_log(json.dumps(event, sort_keys=True))

    def _analysis_job_finished(self, result: WorkerJobResult) -> None:
        controller = self.analysis_controller
        self.analysis_controller = None
        self.analysis_config_path = None
        self.analysis_process = None
        if not result.succeeded and self.status_label.text() == "Running":
            self.status_label.setText("Failed")
        self._set_running(False)
        if controller is not None:
            controller.deleteLater()

    def _open_output_dir(self) -> None:
        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_dir))

    def closeEvent(self, event):
        self._save_analysis_setup()
        if self.analysis_controller is not None:
            self.analysis_controller.shutdown()
        else:
            _shutdown_qprocess(self.analysis_process)
            _remove_file_quietly(self.analysis_config_path)
        self.analysis_controller = None
        self.analysis_process = None
        self.analysis_config_path = None
        super().closeEvent(event)
