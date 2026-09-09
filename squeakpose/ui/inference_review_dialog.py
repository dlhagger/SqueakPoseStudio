"""Project-wide triage of low-quality pose and segmentation inference frames."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from squeakpose.core import commit_staged_paths, remove_path, stable_path_id, staging_path_for
from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.project.paths import ProjectPaths
from squeakpose.services.inference_review import (
    RANK_KEYPOINT_CONFIDENCE,
    RANK_OVERALL,
    RANK_POSE_CONFIDENCE,
    RANK_SEGMENTATION_CONFIDENCE,
    InferenceReviewCandidate,
    InferenceReviewScanResult,
    candidate_reasons,
    rank_inference_review_candidates,
    scan_project_inference_quality,
)
from squeakpose.services.video_review import exported_frame_indices, plan_export_frame_path

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


POSE_SKELETON = (
    ("nose", "head"),
    ("head", "left_ear"),
    ("head", "right_ear"),
    ("head", "back"),
    ("back", "tail_base"),
)


class _InferenceReviewScanWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, project_root: str, layer_id: str) -> None:
        super().__init__()
        self.project_root = project_root
        self.layer_id = layer_id
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = scan_project_inference_quality(
                self.project_root,
                layer_id=self.layer_id,
                progress_callback=self.progress.emit,
                cancel_requested=lambda: self._cancel_requested,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class InferenceReviewDialog(QDialog):
    """Review the lowest-quality analyzed frames across an entire project."""

    def __init__(
        self,
        project_root: str,
        parent=None,
        *,
        auto_scan: bool = True,
        preferred_model_paths: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = os.path.abspath(project_root)
        self._scan_result: InferenceReviewScanResult | None = None
        self._shown_candidates: tuple[InferenceReviewCandidate, ...] = ()
        self._preview_frame = None
        self._preview_pixmap = QPixmap()
        self._scan_thread: QThread | None = None
        self._scan_worker: _InferenceReviewScanWorker | None = None
        self._close_after_scan = False
        self._preferred_model_paths = dict(preferred_model_paths or {})

        self.setWindowTitle("Inference Quality Review — SqueakPose Studio")
        self.setMinimumSize(1080, 680)
        self.resize(1480, 860)
        self.setSizeGripEnabled(True)
        self._build_ui()
        if auto_scan:
            QTimer.singleShot(0, self.start_scan)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(10)

        title = QLabel("Review inference quality across the project", self)
        title.setObjectName("InferencePickerTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Choose one model layer, review its associated output for every project video, "
            "and send suspicious original frames back to that model's labeler.",
            self,
        )
        subtitle.setObjectName("InferencePickerSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        filters_panel = QFrame(self)
        filters_panel.setObjectName("InferenceTrackingDefaults")
        filters = QHBoxLayout(filters_panel)
        filters.setContentsMargins(12, 9, 12, 9)
        filters.setSpacing(10)
        self.layer_combo = QComboBox(filters_panel)
        self.layer_combo.addItem("Keypoints / Pose", LAYER_KEYPOINTS)
        self.layer_combo.addItem("Segmentation", LAYER_SEGMENTATION)
        self.model_combo = QComboBox(filters_panel)
        self.model_combo.addItem("Detecting model outputs…", "")
        self.ranking_combo = QComboBox(filters_panel)
        self._populate_ranking_choices()
        self.status_combo = QComboBox(filters_panel)
        for label, value in (
            ("All priorities", ""),
            ("Problems only", "problem"),
            ("Bad only", "bad"),
            ("Warnings only", "warning"),
            ("Good only", "good"),
        ):
            self.status_combo.addItem(label, value)
        self.video_combo = QComboBox(filters_panel)
        self.video_combo.addItem("All videos", "")
        self.reason_combo = QComboBox(filters_panel)
        self.reason_combo.addItem("All reasons", "")
        self.limit_spin = _spin(1, 500, 25, filters_panel)
        self.per_video_spin = _spin(1, 100, 5, filters_panel)
        self.spacing_spin = _spin(0, 10000, 15, filters_panel)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(7)
        form.addRow("Output", self.layer_combo)
        form.addRow("Model", self.model_combo)
        filters.addLayout(form, 1)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(7)
        form.addRow("Rank", self.ranking_combo)
        form.addRow("Status", self.status_combo)
        filters.addLayout(form)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(7)
        form.addRow("Video", self.video_combo)
        form.addRow("Reason", self.reason_combo)
        filters.addLayout(form, 1)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(7)
        form.addRow("Show", self.limit_spin)
        form.addRow("Maximum/video", self.per_video_spin)
        filters.addLayout(form)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(7)
        form.addRow("Frame spacing", self.spacing_spin)
        self.rescan_button = QPushButton("Rescan Outputs", filters_panel)
        self.rescan_button.clicked.connect(self.start_scan)
        form.addRow("", self.rescan_button)
        filters.addLayout(form)
        root.addWidget(filters_panel)

        for widget in (
            self.ranking_combo,
            self.model_combo,
            self.status_combo,
            self.video_combo,
            self.reason_combo,
        ):
            widget.currentIndexChanged.connect(self.refresh_candidates)
        self.layer_combo.currentIndexChanged.connect(self._layer_changed)
        for widget in (self.limit_spin, self.per_video_spin, self.spacing_spin):
            widget.valueChanged.connect(self.refresh_candidates)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.table = QTableWidget(0, 9, splitter)
        self.table.setHorizontalHeaderLabels(
            (
                "Video",
                "Frame",
                "Time",
                "QC",
                "Reason",
                "Confidence",
                "Weakest KP",
                "Detections",
                "Tracks",
            )
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 66), (2, 82), (3, 76), (5, 68), (6, 78), (7, 78), (8, 88)):
            self.table.setColumnWidth(column, width)
        self.table.itemSelectionChanged.connect(self._show_selected_candidate)

        preview_panel = QWidget(splitter)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        self.preview_label = QLabel("Select a candidate frame", preview_panel)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(480, 400)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("background: #0d1217; color: #9aa8b5;")
        preview_layout.addWidget(self.preview_label, 1)
        self.detail_label = QLabel("", preview_panel)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        preview_layout.addWidget(self.detail_label)
        splitter.addWidget(self.table)
        splitter.addWidget(preview_panel)
        splitter.setSizes((780, 650))
        root.addWidget(splitter, 1)

        status_panel = QFrame(self)
        status_layout = QHBoxLayout(status_panel)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("Waiting to scan inference outputs…", status_panel)
        self.summary_label.setWordWrap(True)
        status_layout.addWidget(self.summary_label, 1)
        self.progress_bar = QProgressBar(status_panel)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(220)
        self.progress_bar.hide()
        status_layout.addWidget(self.progress_bar)
        root.addWidget(status_panel)

        actions = QHBoxLayout()
        self.export_selected_button = QPushButton("Export Selected to Labeler", self)
        self.export_shown_button = QPushButton("Export All Shown", self)
        self.export_selected_button.setEnabled(False)
        self.export_shown_button.setEnabled(False)
        self.export_selected_button.clicked.connect(self.export_selected)
        self.export_shown_button.clicked.connect(self.export_shown)
        actions.addWidget(self.export_selected_button)
        actions.addWidget(self.export_shown_button)
        actions.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        actions.addWidget(buttons)
        root.addLayout(actions)

    def _populate_ranking_choices(self) -> None:
        selected = self.ranking_combo.currentData() if self.ranking_combo.count() else None
        layer_id = str(self.layer_combo.currentData() or LAYER_KEYPOINTS)
        self.ranking_combo.blockSignals(True)
        self.ranking_combo.clear()
        self.ranking_combo.addItem("Overall QC priority", RANK_OVERALL)
        if layer_id == LAYER_KEYPOINTS:
            self.ranking_combo.addItem("Lowest detection confidence", RANK_POSE_CONFIDENCE)
            self.ranking_combo.addItem("Lowest keypoint confidence", RANK_KEYPOINT_CONFIDENCE)
        else:
            self.ranking_combo.addItem("Lowest detection confidence", RANK_SEGMENTATION_CONFIDENCE)
        _restore_combo_data(self.ranking_combo, selected)
        self.ranking_combo.blockSignals(False)

    @pyqtSlot()
    def _layer_changed(self) -> None:
        self._populate_ranking_choices()
        self._scan_result = None
        self._shown_candidates = ()
        self.table.setRowCount(0)
        self.model_combo.clear()
        self.model_combo.addItem("Detecting model outputs…", "")
        self.start_scan()

    def start_scan(self) -> None:
        if self._scan_thread is not None and self._scan_thread.isRunning():
            return
        self.rescan_button.setEnabled(False)
        self.layer_combo.setEnabled(False)
        self.export_selected_button.setEnabled(False)
        self.export_shown_button.setEnabled(False)
        layer_label = self.layer_combo.currentText()
        self.summary_label.setText(f"Scanning {layer_label.lower()} inference outputs…")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        thread = QThread(self)
        worker = _InferenceReviewScanWorker(
            self.project_root, str(self.layer_combo.currentData() or LAYER_KEYPOINTS)
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._scan_progress)
        worker.finished.connect(self._scan_finished)
        worker.failed.connect(self._scan_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._scan_thread_finished)
        self._scan_thread = thread
        self._scan_worker = worker
        thread.start()

    @pyqtSlot(int, int, str)
    def _scan_progress(self, completed: int, total: int, video_name: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(completed, max(1, total)))
        self.progress_bar.setFormat(f"{video_name} — %v/%m")

    @pyqtSlot(object)
    def _scan_finished(self, result: object) -> None:
        if not isinstance(result, InferenceReviewScanResult):
            self._scan_failed("The quality scan returned an invalid result.")
            return
        self._scan_result = result
        self.progress_bar.hide()
        self.rescan_button.setEnabled(True)
        self.layer_combo.setEnabled(True)
        self._populate_filter_choices()
        self.refresh_candidates()

    @pyqtSlot(str)
    def _scan_failed(self, message: str) -> None:
        self.progress_bar.hide()
        self.rescan_button.setEnabled(True)
        self.layer_combo.setEnabled(True)
        self.summary_label.setText("Could not scan project inference outputs.")
        QMessageBox.warning(self, "Inference Quality Review", message)

    @pyqtSlot()
    def _scan_thread_finished(self) -> None:
        thread = self._scan_thread
        self._scan_thread = None
        self._scan_worker = None
        if thread is not None:
            thread.deleteLater()
        if self._close_after_scan:
            self._close_after_scan = False
            QTimer.singleShot(0, self.reject)

    def _populate_filter_choices(self) -> None:
        if self._scan_result is None:
            return
        selected_video = self.video_combo.currentData()
        selected_reason = self.reason_combo.currentData()
        selected_model = self.model_combo.currentData()
        preferred_model = self._preferred_model_paths.get(self._scan_result.layer_id, "")
        self.model_combo.blockSignals(True)
        self.video_combo.blockSignals(True)
        self.reason_combo.blockSignals(True)
        self.model_combo.clear()
        for model_path in self._scan_result.model_paths:
            self.model_combo.addItem(_model_label(model_path), model_path)
            self.model_combo.setItemData(
                self.model_combo.count() - 1, model_path, Qt.ItemDataRole.ToolTipRole
            )
        if not self._scan_result.model_paths:
            self.model_combo.addItem("Model path not recorded", "")
        self.video_combo.clear()
        self.video_combo.addItem("All videos", "")
        for name in sorted(
            {candidate.video_name for candidate in self._scan_result.candidates}, key=str.casefold
        ):
            self.video_combo.addItem(name, name)
        self.reason_combo.clear()
        self.reason_combo.addItem("All reasons", "")
        for reason in candidate_reasons(self._scan_result.candidates):
            self.reason_combo.addItem(reason.replace("_", " ").title(), reason)
        _restore_combo_data(self.video_combo, selected_video)
        _restore_combo_data(self.reason_combo, selected_reason)
        if self.model_combo.findData(selected_model) >= 0:
            _restore_combo_data(self.model_combo, selected_model)
        elif self.model_combo.findData(preferred_model) >= 0:
            _restore_combo_data(self.model_combo, preferred_model)
        self.video_combo.blockSignals(False)
        self.reason_combo.blockSignals(False)
        self.model_combo.blockSignals(False)
        is_pose = self._scan_result.layer_id == LAYER_KEYPOINTS
        self.table.setColumnHidden(6, not is_pose)

    @pyqtSlot()
    def refresh_candidates(self) -> None:
        if self._scan_result is None:
            return
        status_value = str(self.status_combo.currentData() or "")
        candidates = self._scan_result.candidates
        if status_value == "problem":
            candidates = tuple(candidate for candidate in candidates if candidate.status != "good")
            status_value = ""
        self._shown_candidates = rank_inference_review_candidates(
            candidates,
            mode=str(self.ranking_combo.currentData() or RANK_OVERALL),
            limit=self.limit_spin.value(),
            per_video_limit=self.per_video_spin.value(),
            min_frame_spacing=self.spacing_spin.value(),
            video_name=str(self.video_combo.currentData() or ""),
            status=status_value,
            reason=str(self.reason_combo.currentData() or ""),
            model_path=str(self.model_combo.currentData() or ""),
        )
        self._populate_table()
        result = self._scan_result
        issue_text = f" • {len(result.issues)} issue(s)" if result.issues else ""
        self.summary_label.setText(
            f"{result.videos_with_inference}/{result.videos_total} videos with "
            f"{_layer_label(result.layer_id).lower()} inference • "
            f"{result.frames_scanned:,} frames scanned • {result.bad_frames:,} bad • "
            f"{result.warning_frames:,} warnings • showing {len(self._shown_candidates)}"
            f"{issue_text}"
        )
        if result.issues:
            self.summary_label.setToolTip("\n".join(result.issues))
        self.export_shown_button.setEnabled(bool(self._shown_candidates))

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for candidate in self._shown_candidates:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = (
                candidate.video_name,
                str(candidate.frame_index),
                _format_time(candidate.time_seconds),
                candidate.status.title(),
                ", ".join(reason.replace("_", " ") for reason in candidate.reasons) or "—",
                _format_metric(candidate.detection_confidence),
                _format_metric(candidate.keypoint_confidence),
                str(candidate.detections_in_frame),
                "—" if candidate.tracks_in_frame is None else str(candidate.tracks_in_frame),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, candidate)
                if column == 3:
                    color = {
                        "bad": QColor("#ef6b73"),
                        "warning": QColor("#f2bd5a"),
                        "good": QColor("#67c587"),
                    }.get(candidate.status)
                    if color is not None:
                        item.setForeground(color)
                self.table.setItem(row, column, item)
        if self.table.rowCount():
            self.table.selectRow(0)
        else:
            self.preview_label.setText("No frames match these filters")
            self.detail_label.clear()
            self.export_selected_button.setEnabled(False)

    def _candidate_for_row(self, row: int) -> InferenceReviewCandidate | None:
        item = self.table.item(row, 0)
        candidate = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return candidate if isinstance(candidate, InferenceReviewCandidate) else None

    def _selected_candidates(self) -> tuple[InferenceReviewCandidate, ...]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return tuple(
            candidate for row in rows if (candidate := self._candidate_for_row(row)) is not None
        )

    def _show_selected_candidate(self) -> None:
        selected = self._selected_candidates()
        self.export_selected_button.setEnabled(bool(selected))
        if not selected:
            return
        candidate = selected[0]
        if cv2 is None:
            self.preview_label.setText("OpenCV is unavailable")
            return
        capture = cv2.VideoCapture(candidate.video_path)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            self.preview_label.setText("Could not read this frame")
            return
        self._preview_frame = frame.copy()
        rendered = _draw_candidate_overlay(frame.copy(), candidate)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._preview_pixmap = QPixmap.fromImage(image)
        self._refresh_preview_pixmap()
        reasons = ", ".join(reason.replace("_", " ") for reason in candidate.reasons)
        self.detail_label.setText(
            f"{candidate.video_name} • frame {candidate.frame_index} • "
            f"{_format_time(candidate.time_seconds)}\n"
            f"{_layer_label(candidate.layer_id)} • {_model_label(candidate.model_path)}\n"
            f"QC: {candidate.status}{' — ' + reasons if reasons else ''}"
        )

    def _refresh_preview_pixmap(self) -> None:
        if self._preview_pixmap.isNull():
            return
        self.preview_label.setPixmap(
            self._preview_pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def export_selected(self) -> None:
        self._export_candidates(self._selected_candidates())

    def export_shown(self) -> None:
        self._export_candidates(self._shown_candidates)

    def _export_candidates(self, candidates: tuple[InferenceReviewCandidate, ...]) -> None:
        if not candidates:
            return
        if cv2 is None:
            QMessageBox.warning(self, "Export Error", "OpenCV is unavailable.")
            return
        destination = getattr(self.parent(), "image_dir_queue", None)
        if not destination:
            destination = ProjectPaths.from_root(self.project_root).images_to_label
        try:
            os.makedirs(destination, exist_ok=True)
            existing_names = os.listdir(destination)
        except OSError as exc:
            QMessageBox.warning(self, "Export Error", str(exc))
            return

        progress = QProgressDialog("Exporting review frames…", "Cancel", 0, len(candidates), self)
        progress.setWindowTitle("Exporting to Labeler")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        saved = 0
        skipped = 0
        failed: list[str] = []
        capture = None
        capture_path = ""
        try:
            for index, candidate in enumerate(candidates, start=1):
                if progress.wasCanceled():
                    break
                video_base = Path(candidate.video_path).stem
                source_id = stable_path_id(candidate.video_path)
                exported = exported_frame_indices(
                    existing_names, video_base=video_base, source_id=source_id
                )
                if candidate.frame_index in exported:
                    skipped += 1
                    progress.setValue(index)
                    continue
                if capture is None or capture_path != candidate.video_path:
                    if capture is not None:
                        capture.release()
                    capture = cv2.VideoCapture(candidate.video_path)
                    capture_path = candidate.video_path
                capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed.append(
                        f"{candidate.video_name} frame {candidate.frame_index}: read failed"
                    )
                else:
                    output_path = plan_export_frame_path(
                        destination,
                        video_base=video_base,
                        source_id=source_id,
                        frame_index=candidate.frame_index,
                    )
                    staged_path = staging_path_for(output_path)
                    try:
                        if not cv2.imwrite(staged_path, frame):
                            raise OSError("image encoder failed")
                        commit_staged_paths([(staged_path, output_path)])
                        existing_names.append(os.path.basename(output_path))
                        saved += 1
                    except Exception as exc:
                        remove_path(staged_path)
                        failed.append(
                            f"{candidate.video_name} frame {candidate.frame_index}: {exc}"
                        )
                progress.setValue(index)
                progress.setLabelText(f"{candidate.video_name} — frame {candidate.frame_index}")
                QApplication.processEvents()
        finally:
            if capture is not None:
                capture.release()
            progress.close()

        parent = self.parent()
        if saved and parent is not None and hasattr(parent, "refresh_image_list"):
            parent.refresh_image_list()
        if saved and parent is not None and hasattr(parent, "update_status_bar"):
            parent.update_status_bar(f"Exported {saved} inference-review frame(s) to labeler")
        message = f"Exported {saved} frame(s) to:\n{destination}"
        if skipped:
            message += f"\n\nSkipped {skipped} frame(s) already in the label queue."
        if failed:
            message += "\n\nIssues:\n" + "\n".join(failed[:10])
            if len(failed) > 10:
                message += f"\n…{len(failed) - 10} more"
        QMessageBox.information(self, "Inference Review Export", message)

    def reject(self) -> None:
        thread = self._scan_thread
        if thread is not None and thread.isRunning():
            if self._scan_worker is not None:
                self._scan_worker.cancel()
            self._close_after_scan = True
            self.summary_label.setText("Canceling inference scan…")
            self.rescan_button.setEnabled(False)
            return
        super().reject()

    def closeEvent(self, event) -> None:
        worker = self._scan_worker
        thread = self._scan_thread
        if thread is not None and thread.isRunning():
            if worker is not None:
                worker.cancel()
            self._close_after_scan = True
            self.summary_label.setText("Canceling inference scan…")
            event.ignore()
            return
        super().closeEvent(event)


def _draw_candidate_overlay(frame, candidate: InferenceReviewCandidate):
    if cv2 is None or np is None:
        return frame
    palette = ((255, 205, 70), (70, 190, 255), (210, 100, 255), (90, 220, 130))
    for overlay_index, overlay in enumerate(candidate.overlays):
        color = palette[overlay_index % len(palette)]
        if len(overlay.mask_polygon) >= 3:
            polygon = np.asarray(overlay.mask_polygon, dtype=np.int32).reshape((-1, 1, 2))
            fill = frame.copy()
            cv2.fillPoly(fill, [polygon], color, lineType=cv2.LINE_AA)
            cv2.addWeighted(fill, 0.26, frame, 0.74, 0, frame)
            cv2.polylines(frame, [polygon], True, color, 2, lineType=cv2.LINE_AA)
        if len(overlay.bbox) == 4:
            x1, y1, x2, y2 = (int(round(value)) for value in overlay.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                overlay.animal_id.replace("_", " ").title(),
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        points = {name: (x, y) for name, x, y, confidence in overlay.keypoints if confidence > 0}
        for first, second in POSE_SKELETON:
            if first in points and second in points:
                cv2.line(
                    frame,
                    tuple(int(round(value)) for value in points[first]),
                    tuple(int(round(value)) for value in points[second]),
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
        for name, x, y, confidence in overlay.keypoints:
            if confidence > 0 or math.isnan(confidence):
                cv2.circle(
                    frame,
                    (int(round(x)), int(round(y))),
                    4,
                    (255, 255, 255),
                    -1,
                    lineType=cv2.LINE_AA,
                )
    return frame


def _spin(minimum: int, maximum: int, value: int, parent) -> QSpinBox:
    spin = QSpinBox(parent)
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_time(value: float | None) -> str:
    if value is None:
        return "—"
    minutes, seconds = divmod(max(0.0, value), 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def _layer_label(layer_id: str) -> str:
    return "Keypoints / Pose" if layer_id == LAYER_KEYPOINTS else "Segmentation"


def _model_label(model_path: str) -> str:
    if not model_path:
        return "Model path not recorded"
    path = Path(model_path)
    parent = path.parent.name
    return f"{parent} / {path.name}" if parent else path.name


def _restore_combo_data(combo: QComboBox, value: object) -> None:
    index = combo.findData(value)
    combo.setCurrentIndex(max(0, index))


__all__ = ["InferenceReviewDialog"]
