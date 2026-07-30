"""Video review, prediction ranking, and frame export dialog."""

from __future__ import annotations

import datetime
import json
import os
import random
import re
import sys
from typing import Optional

from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontInfo,
    QKeySequence,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from prediction_ops import rank_prediction_frames
from squeakpose.annotation.video_view import VideoView
from squeakpose_core import (
    atomic_write_text,
    commit_staged_paths,
    effective_prediction_batch,
    remove_path,
    stable_path_id,
    staging_path_for,
)
from squeakpose.workers.process import (
    remove_file_quietly as _remove_file_quietly,
    request_qprocess_stop,
    shutdown_qprocess as _shutdown_qprocess,
)
from squeakpose.workers.protocol import WorkerProtocolError, parse_event_line

APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOW_POSE = "pose"
WORKFLOW_SEG = "segmentation"

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


def _ui_font(px: int) -> QFont:
    font = QFont()
    available = set(QFontDatabase.families())
    system_family = QFontDatabase.systemFont(
        QFontDatabase.SystemFont.GeneralFont
    ).family()
    for family in ("Fira Sans", system_family, "Segoe UI", "Arial", "Helvetica"):
        if family and family in available:
            font.setFamily(family)
            if QFontInfo(font).family() == family:
                break
    font.setPixelSize(px)
    return font


class VideoReviewDialog(QDialog):
    """
    Modal tool that:
      1) Loads a video
      2) Runs YOLO predict over a chosen frame range in a child process
      3) Lets you scrub a timeline and see prediction overlays and confidence
    """
    def __init__(
        self,
        parent,
        device: str,
        kp_names: list[str],
        classes: list[str],
        class_keypoints: Optional[dict[str, list[str]]] = None,
        workflow: str = WORKFLOW_POSE,
    ):
        super().__init__(parent)
        self.workflow = WORKFLOW_SEG if str(workflow).lower() == WORKFLOW_SEG else WORKFLOW_POSE
        workflow_title = "Segmentation" if self.workflow == WORKFLOW_SEG else "Pose"
        self.setWindowTitle(f"Video Review ({workflow_title})")
        self.resize(980, 700)

        self.device = device
        self.kp_names = kp_names
        self.classes = classes
        self.class_keypoints = class_keypoints or {}
        self.model_path = getattr(parent, 'predict_model_path', None)

        # runtime state
        self.cap = None
        self.path: Optional[str] = None
        self.base: str = ""
        self.video_source_id: str = ""
        self.total: int = 0
        self.fps: float = 0.0
        self.cur: int = 0
        self.preds: dict[int, dict] = {}
        self._last_frame_bgr = None  # holds the current raw frame for export
        self._review_process: Optional[QProcess] = None
        self._review_progress: Optional[QProgressDialog] = None
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_result_event: Optional[dict] = None
        self._review_partial_preds: dict[int, dict] = {}
        self._review_config_path: Optional[str] = None
        self._review_cancel_requested = False
        self._review_run_meta: Optional[dict] = None

        # build all widgets/layouts
        self._build_ui()

    def _is_seg_workflow(self) -> bool:
        return self.workflow == WORKFLOW_SEG

    def _build_ui(self):
        # --- UI ---
        top = QVBoxLayout(self)
        top.setContentsMargins(10, 10, 10, 10)
        top.setSpacing(8)

        # Header row: load + current-video summary
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self._choose_video)
        row.addWidget(self.btn_load)

        self.info = QLabel("")
        self.info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.info.setMinimumWidth(180)
        self.info.setWordWrap(False)
        self.info.setStyleSheet("padding-left: 4px;")
        self._info_full_text = "No video loaded"
        self._set_info_text(self._info_full_text)
        row.addWidget(self.info, 1)
        top.addLayout(row)

        # Control rows (split to keep dialog resizable at narrower widths)
        controls_row_1 = QHBoxLayout()
        controls_row_1.setSpacing(8)
        controls_row_1.addWidget(QLabel("Start"))
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 0)
        self.spin_start.setValue(0)
        self.spin_start.setMaximumWidth(110)
        controls_row_1.addWidget(self.spin_start)

        controls_row_1.addWidget(QLabel("End"))
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, 0)
        self.spin_end.setValue(0)
        self.spin_end.setMaximumWidth(110)
        controls_row_1.addWidget(self.spin_end)

        controls_row_1.addWidget(QLabel("Stride"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(1, 1000)
        self.spin_stride.setValue(5)
        self.spin_stride.setMaximumWidth(90)
        controls_row_1.addWidget(self.spin_stride)

        controls_row_1.addWidget(QLabel("Batch"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(0, 256)
        self.spin_batch.setSpecialValueText("Auto")
        self.spin_batch.setValue(0)
        self.spin_batch.setToolTip(
            "Auto uses batched inference on CUDA/MPS with memory fallback; CPU defaults to one frame."
        )
        self.spin_batch.setMaximumWidth(90)
        controls_row_1.addWidget(self.spin_batch)

        controls_row_1.addStretch(1)
        self.btn_predict = QPushButton("Predict Range")
        self.btn_predict.setEnabled(False)
        if self._is_seg_workflow():
            self.btn_predict.setToolTip("Run segmentation predictions over the selected frame range.")
        else:
            self.btn_predict.setToolTip("Run pose predictions over the selected frame range.")
        self.btn_predict.clicked.connect(self._start_range_prediction)
        controls_row_1.addWidget(self.btn_predict)
        top.addLayout(controls_row_1)

        controls_row_2 = QHBoxLayout()
        controls_row_2.setSpacing(8)
        controls_row_2.addWidget(QLabel("Conf≥"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setMaximumWidth(90)
        controls_row_2.addWidget(self.spin_conf)

        controls_row_2.addWidget(QLabel("IoU"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.0, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.50)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setMaximumWidth(90)
        controls_row_2.addWidget(self.spin_iou)

        # keypoint visibility threshold (pose workflow only)
        self.lbl_kpvis = QLabel("kp≥")
        controls_row_2.addWidget(self.lbl_kpvis)
        self.spin_kpvis = QDoubleSpinBox()
        self.spin_kpvis.setRange(0.0, 1.0)
        self.spin_kpvis.setSingleStep(0.05)
        self.spin_kpvis.setDecimals(2)
        self.spin_kpvis.setValue(0.50)  # >= → visible (red); < → occluded (yellow)
        self.spin_kpvis.setMaximumWidth(90)
        controls_row_2.addWidget(self.spin_kpvis)
        show_kp_controls = not self._is_seg_workflow()
        self.lbl_kpvis.setVisible(show_kp_controls)
        self.spin_kpvis.setVisible(show_kp_controls)
        controls_row_2.addStretch(1)
        top.addLayout(controls_row_2)

        # Graphics view (pan/zoom enabled)
        self.scene = QGraphicsScene()
        self.view = VideoView(self.scene)
        top.addWidget(self.view, 1)

        # Timeline
        bar2 = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        bar2.addWidget(self.slider)
        self.lbl_idx = QLabel("0/0")
        bar2.addWidget(self.lbl_idx)
        top.addLayout(bar2)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        self.btn_send = QPushButton("Send Frame")
        self.btn_send.setToolTip("Save current frame to the labeler's images_to_label folder")
        self.btn_send.setEnabled(False)
        self.btn_send.clicked.connect(self._export_current_frame_to_images)
        buttons.addButton(self.btn_send, QDialogButtonBox.ButtonRole.ActionRole)

        self.btn_send_low = QPushButton("Send Low…")
        self.btn_send_low.setToolTip(
            "Export lowest-confidence frames for one class or balanced across classes"
        )
        self.btn_send_low.setEnabled(False)
        self.btn_send_low.clicked.connect(self._export_low_confidence_frames)
        buttons.addButton(self.btn_send_low, QDialogButtonBox.ButtonRole.ActionRole)

        self.btn_send_high = QPushButton("Send High…")
        self.btn_send_high.setToolTip(
            "Export highest-confidence frames for one class or balanced across classes"
        )
        self.btn_send_high.setEnabled(False)
        self.btn_send_high.clicked.connect(self._export_high_confidence_frames)
        buttons.addButton(self.btn_send_high, QDialogButtonBox.ButtonRole.ActionRole)

        self.btn_send_random = QPushButton("Send Random…")
        self.btn_send_random.setToolTip("Export N random frames to the labeler for fresh labeling")
        self.btn_send_random.setEnabled(False)
        self.btn_send_random.clicked.connect(self._export_random_frames)
        buttons.addButton(self.btn_send_random, QDialogButtonBox.ButtonRole.ActionRole)

        # Shortcut: Shift+E exports N lowest (asks for N)
        self._exportN_shortcut = QShortcut(QKeySequence("Shift+E"), self)
        self._exportN_shortcut.activated.connect(self._export_low_confidence_frames)

        # Shortcut: Shift+H exports N highest (asks for N)
        self._exportNhigh_shortcut = QShortcut(QKeySequence("Shift+H"), self)
        self._exportNhigh_shortcut.activated.connect(self._export_high_confidence_frames)

        # Shortcut: Shift+R exports N random frames
        self._exportNrandom_shortcut = QShortcut(QKeySequence("Shift+R"), self)
        self._exportNrandom_shortcut.activated.connect(self._export_random_frames)

        top.addWidget(buttons)

        # overlay items
        self._overlay_items: list[QGraphicsItem] = []

        # Arrow-key timeline stepping (Left/Right = ±1 frame)
        self._left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._left_shortcut.setAutoRepeat(True)
        self._left_shortcut.activated.connect(lambda: self._step(-1))

        self._right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._right_shortcut.setAutoRepeat(True)
        self._right_shortcut.activated.connect(lambda: self._step(+1))

        self._export_shortcut = QShortcut(QKeySequence("E"), self)
        self._export_shortcut.activated.connect(self._export_current_frame_to_images)

        # Zoom shortcuts for the view
        self._zoom_in_sc = QShortcut(QKeySequence("+"), self)
        self._zoom_in_sc.activated.connect(lambda: self.view.scale(1.05, 1.05))
        self._zoom_out_sc = QShortcut(QKeySequence("-"), self)
        self._zoom_out_sc.activated.connect(lambda: self.view.scale(1/1.05, 1/1.05))
        self._zoom_reset_sc = QShortcut(QKeySequence("R"), self)
        self._zoom_reset_sc.activated.connect(self.view.reset_view)

    def _set_info_text(self, text: str):
        self._info_full_text = text or ""
        self._refresh_info_label()

    def _refresh_info_label(self):
        if not hasattr(self, "info"):
            return
        full = getattr(self, "_info_full_text", "") or ""
        # Elide in the middle so filename prefix/suffix stay visible.
        width = max(140, int(self.info.width()) - 8)
        elided = self.info.fontMetrics().elidedText(
            full,
            Qt.TextElideMode.ElideMiddle,
            width,
        )
        self.info.setText(elided)
        self.info.setToolTip(full)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_info_label()

    def _labeler_image_dir(self) -> Optional[str]:
        """Return the parent labeler's queue folder for exported frames."""
        parent = self.parent()
        if parent is None:
            return None
        queue_dir = getattr(parent, "image_dir_queue", None)
        if queue_dir:
            return queue_dir
        # Backward compatibility with older attribute naming.
        legacy_dir = getattr(parent, "image_dir", None)
        if legacy_dir:
            return legacy_dir
        return None

    # ---------- caching ----------
    def _cache_path(self) -> Optional[str]:
        if not self.path:
            return None
        return os.path.abspath(self.path) + ".sqp_preds.json"

    def _video_signature(self) -> dict:
        try:
            return {
                "path": os.path.abspath(self.path) if self.path else "",
                "size": int(os.path.getsize(self.path)) if self.path else 0,
                "mtime": float(os.path.getmtime(self.path)) if self.path else 0.0,
                "total": int(self.total),
                "fps": float(self.fps),
            }
        except Exception:
            return {
                "path": os.path.abspath(self.path) if self.path else "",
                "size": 0,
                "mtime": 0.0,
                "total": int(self.total),
                "fps": float(self.fps),
            }

    def _load_cache_if_valid(self) -> bool:
        fp = self._cache_path()
        if not fp or not os.path.exists(fp):
            return False
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            vid = meta.get("video", {})
            cur = self._video_signature()

            # Same file check (path/size) and mtime within a couple seconds
            if (vid.get("path") != cur.get("path")) or (int(vid.get("size", -1)) != int(cur.get("size", -2))):
                return False
            if abs(float(vid.get("mtime", 0.0)) - float(cur.get("mtime", 0.0))) > 2.0:
                return False

            # Optional: require the same model if both are known
            mp_saved = meta.get("model_path")
            if mp_saved and self.model_path and (mp_saved != self.model_path):
                return False
            saved_workflow = str(meta.get("workflow", WORKFLOW_POSE)).strip().lower()
            if saved_workflow != self.workflow:
                return False

            preds = data.get("preds", {})
            self.preds = {int(k): v for k, v in preds.items()}
            return bool(self.preds)
        except Exception:
            return False

    def _save_cache(self, meta: dict):
        fp = self._cache_path()
        if not fp:
            return
        data = {
            "meta": meta,
            "preds": {str(k): v for k, v in self.preds.items()},
        }
        try:
            atomic_write_text(fp, json.dumps(data))
        except Exception:
            pass

    # ---------- video load ----------
    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
        if not path:
            return
        self._open_video(path)

    def _open_video(self, path: str):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None

        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies.")
            return

        cap = _cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            QMessageBox.warning(self, "Video Error", "Failed to open video.")
            return

        self.cap = cap
        self.path = path
        self.base = os.path.splitext(os.path.basename(path))[0]
        self.video_source_id = stable_path_id(path)
        self.total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)

        w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._set_info_text(f"{self.base} — {w}x{h} @ {self.fps:.2f} fps — {self.total} frames")
        self.spin_start.setRange(0, max(0, self.total - 1)); self.spin_start.setValue(0)
        self.spin_end.setRange(0, max(0, self.total - 1)); self.spin_end.setValue(max(0, self.total - 1))
        self.slider.setRange(0, max(0, self.total - 1))
        self.btn_predict.setEnabled(True)
        if hasattr(self, "btn_send"):
            self.btn_send.setEnabled(True)
        if hasattr(self, "btn_send_low"):
            self.btn_send_low.setEnabled(False)
        if hasattr(self, "btn_send_high"):
            self.btn_send_high.setEnabled(False)
        if hasattr(self, "btn_send_random"):
            self.btn_send_random.setEnabled(True)
        # Try to load cached predictions; if present, enable timeline immediately
        if self._load_cache_if_valid():
            self.slider.setEnabled(True)
            if hasattr(self, "btn_send_low"):
                self.btn_send_low.setEnabled(bool(self.preds))
            if hasattr(self, "btn_send_high"):
                self.btn_send_high.setEnabled(bool(self.preds))
            cached_keys = sorted(self.preds.keys())
            self._seek(cached_keys[0] if cached_keys else 0, show_only=False, fit_view=True)
        else:
            # Timeline scrubbing should work even without predictions.
            self.slider.setEnabled(True)
            self._seek(0, show_only=True, fit_view=True)

        # Reset pan/zoom whenever a new video is opened
        if hasattr(self, "view") and hasattr(self.view, "reset_view"):
            self.view.reset_view()

    # ---------- prediction ----------
    def _start_range_prediction(self):
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        if not self.model_path:
            QMessageBox.information(self, "No model", "Click 'Load Model' in the main window first.")
            return
        if self._review_process is not None and self._review_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Prediction running", "Video prediction is already running.")
            return

        start = int(self.spin_start.value())
        end = int(self.spin_end.value())
        stride = max(1, int(self.spin_stride.value()))
        conf = float(self.spin_conf.value())
        iou = float(self.spin_iou.value())
        requested_batch = int(self.spin_batch.value()) if hasattr(self, "spin_batch") else 0
        effective_batch = effective_prediction_batch(requested_batch, self.device)
        imgsz = 640
        kpvis = float(self.spin_kpvis.value()) if (hasattr(self, "spin_kpvis") and not self._is_seg_workflow()) else None

        if end < start:
            QMessageBox.warning(self, "Range Error", "End must be ≥ Start.")
            return

        steps = max(1, ((end - start) // stride) + 1)
        prog = QProgressDialog("Running prediction…", "Cancel", 0, steps, self)
        prog.setWindowTitle("Predicting")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        prog.canceled.connect(self._cancel_review_prediction_process)

        meta = {
            "video": self._video_signature(),
            "model_path": self.model_path,
            "workflow": self.workflow,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "kpvis": kpvis,
            "start": start,
            "end": end,
            "stride": stride,
            "batch": requested_batch,
            "initial_effective_batch": effective_batch,
            "total": self.total,
            "fps": self.fps,
            "classes": self.classes,
            "kp_names": self.kp_names,
        }
        config = {
            "model_path": self.model_path,
            "video_path": self.path,
            "workflow": self.workflow,
            "device": self.device,
            "start": start,
            "end": end,
            "stride": stride,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "batch": requested_batch,
            "effective_batch": effective_batch,
        }

        parent = self.parent()
        parent_log_path = getattr(parent, "_log_path", "") if parent is not None else ""
        config_dir = os.path.dirname(parent_log_path) if parent_log_path else os.path.join(APP_BASE_DIR, "logs")
        try:
            os.makedirs(config_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config_path = os.path.join(config_dir, f".video_review_predict_{stamp}.json")
            atomic_write_text(config_path, json.dumps(config, indent=2))
        except Exception as e:
            prog.close()
            QMessageBox.warning(self, "Prediction Error", f"Could not write video prediction config:\n{e}")
            return

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "video_review_worker", "--config", config_path])
        process.setWorkingDirectory(APP_BASE_DIR)
        process.readyReadStandardOutput.connect(self._read_review_prediction_stdout)
        process.readyReadStandardError.connect(self._read_review_prediction_stderr)
        process.finished.connect(self._finish_review_prediction_process)
        process.errorOccurred.connect(self._handle_review_prediction_error)

        self._review_process = process
        self._review_progress = prog
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_result_event = None
        self._review_partial_preds = {}
        self._review_config_path = config_path
        self._review_cancel_requested = False
        self._review_run_meta = meta
        self.preds.clear()
        self.btn_predict.setEnabled(False)
        if hasattr(self, "btn_send_low"):
            self.btn_send_low.setEnabled(False)
        if hasattr(self, "btn_send_high"):
            self.btn_send_high.setEnabled(False)

        prog.show()
        process.start()
        if not process.waitForStarted(1000):
            self._review_stderr = process.errorString()
            self._finish_review_prediction_process(1, QProcess.ExitStatus.CrashExit)
            return

    def _read_review_prediction_stdout(self):
        process = self._review_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._review_stdout_buffer += text
        lines = self._review_stdout_buffer.splitlines(keepends=True)
        self._review_stdout_buffer = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                self._handle_review_prediction_event_line(line.strip())
            else:
                self._review_stdout_buffer = line

    def _read_review_prediction_stderr(self):
        process = self._review_process
        if process is None:
            return
        self._review_stderr += bytes(process.readAllStandardError()).decode("utf-8", errors="replace")

    def _handle_review_prediction_event_line(self, line: str):
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self._review_stderr += line + "\n"
            return

        event_type = event.get("event")
        if event_type == "started":
            progress = self._review_progress
            if progress is not None:
                progress.setLabelText("Loading model in video prediction process…")
        elif event_type == "progress":
            streamed_predictions = event.get("predictions")
            if isinstance(streamed_predictions, dict):
                for raw_idx, prediction in streamed_predictions.items():
                    try:
                        frame_idx = int(raw_idx)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(prediction, dict):
                        self._review_partial_preds[frame_idx] = prediction
            try:
                frame_idx = int(event.get("frame_idx"))
                prediction = event.get("prediction")
                if isinstance(prediction, dict):
                    self._review_partial_preds[frame_idx] = prediction
            except (TypeError, ValueError):
                pass
            progress = self._review_progress
            if progress is not None:
                processed = int(event.get("processed") or 0)
                total = int(event.get("total") or progress.maximum())
                progress.setMaximum(max(1, total))
                progress.setValue(min(processed, max(1, total)))
                progress.setLabelText(str(event.get("message") or f"Predicting {processed}/{total}"))
            QApplication.processEvents()
        elif event_type == "batch_adjusted":
            effective_batch = int(event.get("effective_batch") or 1)
            if self._review_run_meta is not None:
                self._review_run_meta["final_effective_batch"] = effective_batch
            progress = self._review_progress
            if progress is not None:
                progress.setLabelText(
                    str(event.get("message") or f"Reducing inference batch to {effective_batch}")
                )
            QApplication.processEvents()
        elif event_type == "result":
            self._review_result_event = event
        elif event_type == "error":
            self._review_result_event = {
                "event": "result",
                "canceled": False,
                "had_error": True,
                "error_message": str(event.get("error_message") or "Video prediction worker error"),
                "preds": {},
            }

    def _cancel_review_prediction_process(self):
        process = self._review_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        self._review_cancel_requested = True
        progress = self._review_progress
        if progress is not None:
            progress.setLabelText("Canceling prediction process…")
        request_qprocess_stop(
            process,
            schedule=QTimer.singleShot,
            force_kill=self._kill_review_prediction_if_running,
            kill_after_ms=5000,
        )

    def _kill_review_prediction_if_running(self):
        process = self._review_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_review_prediction_error(self, _error):
        process = self._review_process
        if process is not None:
            self._review_stderr += process.errorString() + "\n"

    def _finish_review_prediction_process(self, exit_code: int, exit_status):
        if self._review_process is None and self._review_config_path is None:
            return
        if self._review_stdout_buffer.strip():
            self._handle_review_prediction_event_line(self._review_stdout_buffer.strip())
            self._review_stdout_buffer = ""

        progress = self._review_progress
        if progress is not None:
            progress.close()

        config_path = self._review_config_path
        _remove_file_quietly(config_path)

        event = self._review_result_event
        partial_preds = dict(self._review_partial_preds)
        stderr_text = self._review_stderr.strip()
        cancel_requested = self._review_cancel_requested
        run_meta = self._review_run_meta or {}

        self._review_process = None
        self._review_progress = None
        self._review_config_path = None
        self._review_result_event = None
        self._review_partial_preds = {}
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_cancel_requested = False
        self._review_run_meta = None
        self.btn_predict.setEnabled(self.cap is not None)

        if cancel_requested and event is None:
            event = {
                "event": "result",
                "canceled": True,
                "had_error": False,
                "error_message": "",
                "preds": {},
            }

        if event is None:
            detail = stderr_text or f"Process exited with code {exit_code}."
            QMessageBox.critical(self, "Prediction Error", f"Video prediction failed:\n{detail}")
            return

        raw_preds = event.get("preds") or {}
        self.preds = {}
        for key, value in partial_preds.items():
            self.preds[key] = value
        if isinstance(raw_preds, dict):
            for key, value in raw_preds.items():
                try:
                    self.preds[int(key)] = value if isinstance(value, dict) else {"ok": False}
                except Exception:
                    continue

        canceled = bool(event.get("canceled")) or cancel_requested
        had_error = bool(event.get("had_error")) or (
            not canceled and (exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0)
        )
        error_message = str(event.get("error_message") or stderr_text or "Unknown video prediction error")

        if self.preds:
            try:
                self._save_cache(run_meta)
            except Exception:
                pass
            self.slider.setEnabled(True)
            if hasattr(self, "btn_send_low"):
                self.btn_send_low.setEnabled(True)
            if hasattr(self, "btn_send_high"):
                self.btn_send_high.setEnabled(True)
            first_idx = min(self.preds.keys())
            self._seek(first_idx, show_only=False)

        if had_error:
            if self.preds:
                QMessageBox.warning(
                    self,
                    "Prediction Error",
                    f"Video prediction stopped with an error, but partial predictions were kept:\n{error_message}",
                )
            else:
                QMessageBox.critical(self, "Prediction Error", f"Video prediction failed:\n{error_message}")
            return

        if canceled:
            if self.preds:
                QMessageBox.information(self, "Prediction canceled", "Video prediction was canceled; partial predictions were kept.")
            else:
                QMessageBox.information(self, "Prediction canceled", "Video prediction was canceled before results were generated.")
            return

        if not self.preds:
            QMessageBox.information(self, "No predictions", "Video prediction completed without generating predictions.")

    # ---------- timeline / overlay ----------
    def _step(self, delta: int):
        """Jump the timeline by `delta` frames (negative for left, positive for right)."""
        if self.cap is None or self.total <= 0:
            return
        new_idx = max(0, min(self.total - 1, self.cur + int(delta)))
        if new_idx == self.cur:
            return
        # Update the slider (for UI) and seek the frame (even if slider is disabled)
        try:
            self.slider.setValue(new_idx)
        except Exception:
            pass
        self._seek(new_idx, show_only=False)

    def _on_slider(self, idx: int):
        self._seek(int(idx), show_only=False)

    def _seek(self, frame_idx: int, show_only: bool, fit_view: bool = False):
        if self.cap is None:
            return
        self.cur = frame_idx
        self.lbl_idx.setText(f"{self.cur+1}/{self.total}")

        self.cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        # remember the raw BGR frame so we can export it to the labeler
        self._last_frame_bgr = frame.copy()
        pix = self._cv_to_qpix(frame)
        self.scene.clear()
        self.scene.setSceneRect(0, 0, pix.width(), pix.height())
        self.scene.addPixmap(pix)
        if fit_view:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        if not show_only:
            self._draw_overlay_for(frame_idx)

    def _export_current_frame_to_images(self):
        """Export the *currently displayed* frame into the labeler's images_to_label folder.
        Skips if this frame index has already been exported (dedupe across restarts)."""
        parent = self.parent()
        dest_dir = self._labeler_image_dir()
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label directory.")
            return
        if self._last_frame_bgr is None:
            QMessageBox.information(self, "No frame", "Load a video and seek to a frame first.")
            return

        # Filesystem-based dedupe: do not export if this frame has already been exported
        existing = self._existing_export_indices()
        if self.cur in existing:
            QMessageBox.information(
                self,
                "Already exported",
                f"Frame {self.cur} is already in images_to_label.\nSkipping duplicate export."
            )
            return

        try:
            os.makedirs(dest_dir, exist_ok=True)
            base_name = f"{self.base}_{self.video_source_id}_f{self.cur:06d}.png"
            out_path = os.path.join(dest_dir, base_name)

            if _cv2 is None:
                QMessageBox.warning(self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies.")
                return

            if not self._write_frame_image(out_path, self._last_frame_bgr):
                QMessageBox.warning(self, "Export Error", "cv2.imwrite failed to save the image.")
                return

            # Refresh the labeler file list if available
            if hasattr(parent, "refresh_image_list"):
                try:
                    parent.refresh_image_list()
                except Exception:
                    pass

            QMessageBox.information(self, "Exported", f"Saved: {out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export frame:\n{e}")

    def _write_frame_image(self, out_path: str, frame) -> bool:
        staged_path = staging_path_for(out_path)
        try:
            if not _cv2.imwrite(staged_path, frame):
                remove_path(staged_path)
                return False
            commit_staged_paths([(staged_path, out_path)])
            return True
        except Exception:
            try:
                remove_path(staged_path)
            except Exception:
                pass
            return False
            
    def _export_random_frames(self):
        """Export N random frames from the loaded video for fresh labeling."""
        if self.cap is None or self.total <= 0:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        parent = self.parent()
        dest_dir = self._labeler_image_dir()
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label directory.")
            return
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies.")
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not create destination folder:\n{e}")
            return

        available = list(range(self.total))
        existing = self._existing_export_indices()
        if existing:
            available = [idx for idx in available if idx not in existing]
        if not available:
            QMessageBox.information(self, "Nothing to export", "Every frame from this video is already in images_to_label.")
            return

        max_n = len(available)
        default_n = min(25, max_n)
        n, ok = QInputDialog.getInt(
            self,
            "Export Random Frames",
            "How many random frames should I send to the labeler?",
            default_n,
            1,
            max_n,
            1,
        )
        if not ok or n <= 0:
            return

        count = min(n, len(available))
        selected = random.sample(available, count)
        selected.sort()

        prog = QProgressDialog("Saving frames…", "Cancel", 0, len(selected), self)
        prog.setWindowTitle("Exporting")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        saved = 0
        failed: list[tuple[int, str]] = []
        cur_pos = int(self.cur)

        for i, fi in enumerate(selected, start=1):
            if prog.wasCanceled():
                break

            self.cap.set(_cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failed.append((fi, "read-failed"))
            else:
                base_name = f"{self.base}_{self.video_source_id}_f{fi:06d}.png"
                dest_path = os.path.join(dest_dir, base_name)
                suffix = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(
                        dest_dir,
                        f"{self.base}_{self.video_source_id}_f{fi:06d}_{suffix}.png",
                    )
                    suffix += 1
                if self._write_frame_image(dest_path, frame):
                    saved += 1
                else:
                    failed.append((fi, "write-failed"))

            prog.setValue(i)
            prog.setLabelText(f"Exporting frame {fi}")
            QApplication.processEvents()

        canceled = prog.wasCanceled()
        prog.close()

        try:
            self._seek(cur_pos, show_only=False)
        except Exception:
            pass

        if hasattr(parent, "refresh_image_list"):
            try:
                parent.refresh_image_list()
                parent.update_status_bar(f"Exported {saved} random frame(s) to images_to_label")
            except Exception:
                pass

        if saved > 0:
            msg = f"Saved {saved} random frame(s) to:\n{dest_dir}"
            if canceled and saved < len(selected):
                msg += "\n\nExport canceled before completing all requested frames."
            QMessageBox.information(self, "Export complete", msg)
        else:
            title = "Export canceled" if canceled else "No frames saved"
            detail = "Export was canceled." if canceled else "Nothing was written."
            if failed:
                detail += "\n\nIssues:\n" + "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
                if len(failed) > 10:
                    detail += f"\n…{len(failed) - 10} more"
            QMessageBox.information(self, title, detail)

        if failed:
            msg = "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
            more = "" if len(failed) <= 10 else f"\n…{len(failed) - 10} more"
            QMessageBox.warning(self, "Some exports failed", f"{saved} succeeded, {len(failed)} failed.\n\n{msg}{more}")
            
    def _existing_export_indices(self) -> set[int]:
        """Scan the labeler's images_to_label folder for frames already exported for this video."""
        out: set[int] = set()
        dest_dir = self._labeler_image_dir()
        if not dest_dir or not os.path.isdir(dest_dir):
            return out
        try:
            import re
            prefix = f"{self.base}_{self.video_source_id}_f"
            pat = re.compile(
                rf"^{re.escape(prefix)}(\d{{6}})(?:_.*)?\.(?:png|jpg|jpeg|bmp|webp)$",
                re.IGNORECASE,
            )
            for fn in os.listdir(dest_dir):
                m = pat.match(fn)
                if m:
                    try:
                        out.add(int(m.group(1)))
                    except Exception:
                        pass
        except Exception:
            pass
        return out
    
    def _export_low_confidence_frames(self):
        self._export_predictions_by_confidence(order="low")

    def _export_high_confidence_frames(self):
        self._export_predictions_by_confidence(order="high")

    def _export_predictions_by_confidence(self, order: str):
        order_key = (order or "low").lower()
        if order_key not in {"low", "high"}:
            order_key = "low"

        if not self.preds:
            QMessageBox.information(self, "No predictions", "Run Predict Range first to generate predictions.")
            return
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        mode_choices = ["Balanced by class"] + [
            self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            for class_id in range(len(self.classes))
        ]
        ranking_choice, choice_ok = QInputDialog.getItem(
            self,
            "Confidence Ranking",
            "Rank frames for which class?",
            mode_choices,
            0,
            False,
        )
        if not choice_ok:
            return

        balanced = ranking_choice == "Balanced by class"
        if balanced:
            ranking_class_ids = list(range(len(self.classes)))
            ranking_label = "balanced by class"
        else:
            try:
                ranking_class_ids = [mode_choices.index(ranking_choice) - 1]
            except ValueError:
                return
            ranking_label = ranking_choice

        candidates = rank_prediction_frames(
            self.preds,
            class_ids=ranking_class_ids,
            order=order_key,
            balanced=balanced,
        )
        if not candidates:
            QMessageBox.information(
                self,
                "No predictions",
                f"No predictions are available for the {ranking_label} ranking.",
            )
            return

        if order_key == "low":
            order_label = "lowest"
            dialog_title = "Export Lowest Confidence"
        else:
            order_label = "highest"
            dialog_title = "Export Highest Confidence"

        already = self._existing_export_indices()
        pending = [candidate for candidate in candidates if candidate[0] not in already]
        if not pending:
            QMessageBox.information(
                self,
                "Nothing to export",
                f"All {order_label}-confidence frames for {ranking_label} are already exported.",
            )
            return

        max_n = len(pending)
        default_n = min(25, max_n)
        n, ok = QInputDialog.getInt(
            self,
            dialog_title,
            f"How many {ranking_label} frames should I send to the labeler?",
            default_n,
            1,
            max_n,
            1,
        )
        if not ok or n <= 0:
            return

        selected = pending[:min(n, len(pending))]

        parent = self.parent()
        dest_dir = self._labeler_image_dir()
        if not dest_dir:
            QMessageBox.warning(self, "Export Error", "Could not locate the labeler's images_to_label directory.")
            return
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies.")
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not create destination folder:\n{e}")
            return

        prog = QProgressDialog("Saving frames…", "Cancel", 0, len(selected), self)
        prog.setWindowTitle("Exporting")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        saved = 0
        saved_rankings: list[tuple[float, int]] = []
        failed: list[tuple[int, str]] = []
        cur_pos = int(self.cur)

        for i, (fi, ranking_conf, ranking_class_id) in enumerate(selected, start=1):
            if prog.wasCanceled():
                break

            self.cap.set(_cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failed.append((fi, "read-failed"))
            else:
                base_name = f"{self.base}_{self.video_source_id}_f{fi:06d}.png"
                dest_path = os.path.join(dest_dir, base_name)
                suffix = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(
                        dest_dir,
                        f"{self.base}_{self.video_source_id}_f{fi:06d}_{suffix}.png",
                    )
                    suffix += 1
                if self._write_frame_image(dest_path, frame):
                    saved += 1
                    saved_rankings.append((ranking_conf, ranking_class_id))
                else:
                    failed.append((fi, "write-failed"))

            prog.setValue(i)
            prog.setLabelText(f"Exporting frame {fi}")
            QApplication.processEvents()

        canceled = prog.wasCanceled()
        prog.close()

        try:
            self._seek(cur_pos, show_only=False)
        except Exception:
            pass

        if hasattr(parent, "refresh_image_list"):
            try:
                parent.refresh_image_list()
                parent.update_status_bar(f"Exported {saved} frame(s) to images_to_label")
            except Exception:
                pass

        if saved > 0:
            msg = f"Saved {saved} frame(s) to:\n{dest_dir}"
            if saved_rankings:
                msg += f"\nRanking: {order_label} confidence, {ranking_label}"
                for class_id in ranking_class_ids:
                    class_confs = [
                        confidence
                        for confidence, ranked_class_id in saved_rankings
                        if ranked_class_id == class_id
                    ]
                    if not class_confs:
                        continue
                    class_name = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
                    msg += (
                        f"\n{class_name}: {min(class_confs):.2f}–{max(class_confs):.2f} "
                        f"({len(class_confs)} frame(s))"
                    )
            if canceled and saved < len(selected):
                msg += "\n\nExport canceled before completing all requested frames."
            QMessageBox.information(self, "Export complete", msg)
        else:
            title = "Export canceled" if canceled else "No frames saved"
            detail = "Export was canceled." if canceled else "Nothing was written."
            if failed:
                detail += "\n\nIssues:\n" + "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
                if len(failed) > 10:
                    detail += f"\n…{len(failed) - 10} more"
            QMessageBox.information(self, title, detail)

        if failed:
            msg = "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
            more = "" if len(failed) <= 10 else f"\n…{len(failed) - 10} more"
            QMessageBox.warning(self, "Some exports failed", f"{saved} succeeded, {len(failed)} failed.\n\n{msg}{more}")

    def _draw_overlay_for(self, frame_idx: int):
        # clear old
        for it in getattr(self, "_overlay_items", []):
            try:
                owner_scene = it.scene()
            except Exception:
                owner_scene = None
            if owner_scene is not None:
                try:
                    owner_scene.removeItem(it)
                except Exception:
                    pass
        self._overlay_items = []

        p = self.preds.get(frame_idx)
        if not p or not p.get("ok"):
            return

        detections = p.get("detections")
        if isinstance(detections, list) and detections:
            for detection in detections:
                if isinstance(detection, dict) and detection.get("ok"):
                    self._draw_prediction_overlay(detection)
            return
        self._draw_prediction_overlay(p)

    def _draw_prediction_overlay(self, p: dict):
        cls_id = int(p.get("cls", 0))
        class_name = self.classes[cls_id] if 0 <= cls_id < len(self.classes) else str(cls_id)
        if self._is_seg_workflow():
            seg_points_raw = p.get("segments", []) or []
            seg_points: list[tuple[float, float]] = []
            for pair in seg_points_raw:
                try:
                    x = float(pair[0])
                    y = float(pair[1])
                    seg_points.append((x, y))
                except Exception:
                    continue

            color = QColor.fromHsv(int((cls_id * 47) % 360), 210, 245, 255)
            label_x = 6.0
            label_y = 6.0

            if len(seg_points) >= 3:
                path = QPainterPath()
                path.moveTo(seg_points[0][0], seg_points[0][1])
                for x, y in seg_points[1:]:
                    path.lineTo(x, y)
                path.closeSubpath()

                seg_item = QGraphicsPathItem(path)
                seg_pen = QPen(color)
                seg_pen.setWidth(3)
                seg_pen.setCosmetic(True)
                seg_item.setPen(seg_pen)
                seg_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 72)))
                seg_item.setZValue(5)
                self.scene.addItem(seg_item)
                self._overlay_items.append(seg_item)

                bbox = path.boundingRect()
                label_x = bbox.left() + 3.0
                label_y = bbox.top() + 3.0
            elif p.get("xyxy"):
                x1, y1, x2, y2 = p["xyxy"]
                rect_item = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
                rect_pen = QPen(color)
                rect_pen.setWidth(3)
                rect_pen.setCosmetic(True)
                rect_pen.setStyle(Qt.PenStyle.DashLine)
                rect_item.setPen(rect_pen)
                rect_item.setBrush(QBrush(Qt.GlobalColor.transparent))
                rect_item.setZValue(5)
                self.scene.addItem(rect_item)
                self._overlay_items.append(rect_item)
                label_x = x1 + 2.0
                label_y = y1 + 2.0

            label_item = QGraphicsSimpleTextItem(f"{class_name} {p.get('conf', 0.0):.2f}")
            label_item.setFont(_ui_font(24))
            label_item.setBrush(QBrush(color))
            label_item.setPos(label_x, label_y)
            label_item.setZValue(6)
            self.scene.addItem(label_item)
            self._overlay_items.append(label_item)
            return

        class_kp_names = self.class_keypoints.get(class_name, self.kp_names)

        # ---- Bounding box (blue, thicker) ----
        if p.get("xyxy"):
            x1, y1, x2, y2 = p["xyxy"]
            r = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(Qt.GlobalColor.blue); pen.setWidth(3); pen.setCosmetic(True)
            r.setPen(pen); r.setZValue(5)
            self.scene.addItem(r); self._overlay_items.append(r)

            # class + confidence (bigger, blue)
            t = QGraphicsSimpleTextItem(f"{class_name} {p.get('conf', 0.0):.2f}")
            t.setFont(_ui_font(24))
            t.setBrush(QBrush(Qt.GlobalColor.blue))
            t.setPos(x1 + 2, y1 + 2); t.setZValue(6)
            self.scene.addItem(t); self._overlay_items.append(t)

        # ---- Keypoints (map kp conf → visibility) ----
        thr = float(self.spin_kpvis.value()) if hasattr(self, "spin_kpvis") else 0.5
        for i, kp in enumerate(p.get("kps", [])):
            if i >= len(self.kp_names):
                break
            name = self.kp_names[i]
            if name not in class_kp_names:
                continue
            x, y, conf = kp
            vis = 2 if conf >= thr else 1  # 2=visible(red), 1=occluded(yellow)

            if vis == 2:
                color = Qt.GlobalColor.red; fill = QBrush(color); style = Qt.PenStyle.SolidLine
            elif vis == 1:
                color = Qt.GlobalColor.yellow; fill = QBrush(color); style = Qt.PenStyle.SolidLine
            else:
                color = Qt.GlobalColor.lightGray; fill = QBrush(Qt.GlobalColor.transparent); style = Qt.PenStyle.DashLine

            dot = QGraphicsEllipseItem(-4, -4, 8, 8)  # slightly larger dot
            dot.setPos(x, y)
            pen = QPen(color); pen.setCosmetic(True); pen.setWidth(2); pen.setStyle(style)
            dot.setPen(pen); dot.setBrush(fill); dot.setZValue(7)
            self.scene.addItem(dot); self._overlay_items.append(dot)

            # label next to kp
            lbl = QGraphicsSimpleTextItem(name)
            lbl.setFont(_ui_font(18))
            lbl.setBrush(QBrush(color))
            lbl.setPos(x + 8, y - 16); lbl.setZValue(8)
            lbl.setVisible(vis != 0)  # hide if invisible
            self.scene.addItem(lbl); self._overlay_items.append(lbl)

    @staticmethod
    def _cv_to_qpix(frame_bgr) -> QPixmap:
        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        from PyQt6.QtGui import QImage
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def reject(self):
        if self._review_process is not None and self._review_process.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(
                self,
                "Cancel prediction?",
                "Video prediction is still running. Cancel it and close the reviewer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            _shutdown_qprocess(self._review_process)
            _remove_file_quietly(self._review_config_path)
            if self._review_progress is not None:
                self._review_progress.close()
            self._review_process = None
            self._review_progress = None
            self._review_config_path = None
        # cleanup
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().reject()
