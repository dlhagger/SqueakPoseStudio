"""Video review, prediction ranking, and frame export dialog."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

from PyQt6.QtCore import QProcess, Qt, QTimer
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
    QCheckBox,
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
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from squeakpose.annotation.video_view import VideoView
from squeakpose.core import (
    atomic_write_text,
    commit_staged_paths,
    effective_prediction_batch,
    remove_path,
    stable_path_id,
    staging_path_for,
)
from squeakpose.json_io import read_json_file
from squeakpose.project.layers import (
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    normalize_layer_id,
)
from squeakpose.services.video_library import list_project_videos
from squeakpose.services.video_review import (
    MAX_VIDEO_CACHE_BYTES,
    available_export_frame_indices,
    build_video_review_cache_payload,
    build_video_review_pass_config,
    build_video_signature,
    complete_video_review_pass,
    decide_video_review_cache,
    exported_frame_indices,
    plan_confidence_export,
    plan_export_frame_path,
    plan_video_review_run,
    select_random_export_frames,
    video_review_cache_path,
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
WORKFLOW_POSE = "pose"
WORKFLOW_SEG = "segmentation"
logger = logging.getLogger(__name__)

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


def _ui_font(px: int) -> QFont:
    font = QFont()
    available = set(QFontDatabase.families())
    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
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
        layer_id: str = "",
        model_paths: Optional[dict[str, str]] = None,
        layer_schemas: Optional[dict[str, dict]] = None,
    ):
        super().__init__(parent)
        self.layer_id = normalize_layer_id(layer_id or workflow)
        self.workflow = WORKFLOW_SEG if str(workflow).lower() == WORKFLOW_SEG else WORKFLOW_POSE
        self.project_root = os.path.abspath(getattr(parent, "project_root", APP_BASE_DIR))

        self.device = device
        self.kp_names = kp_names
        self.classes = classes
        self.class_keypoints = class_keypoints or {}
        inherited_model = getattr(parent, "predict_model_path", None)
        self.model_paths = {
            LAYER_KEYPOINTS: str((model_paths or {}).get(LAYER_KEYPOINTS) or ""),
            LAYER_SEGMENTATION: str((model_paths or {}).get(LAYER_SEGMENTATION) or ""),
        }
        if not any(self.model_paths.values()) and inherited_model:
            self.model_paths[self.layer_id] = str(inherited_model)
        self.layer_schemas = dict(layer_schemas or {})
        self.layer_schemas.setdefault(
            self.layer_id,
            {
                "classes": list(classes),
                "kp_names": list(kp_names),
                "class_keypoints": dict(self.class_keypoints),
            },
        )
        ordered_layers = [
            self.layer_id,
            LAYER_KEYPOINTS,
            LAYER_SEGMENTATION,
        ]
        self.review_layers = [
            candidate
            for candidate in dict.fromkeys(ordered_layers)
            if self.model_paths.get(candidate)
        ]
        if self.review_layers and self.layer_id not in self.review_layers:
            self.layer_id = self.review_layers[0]
            self.workflow = layer_definition(self.layer_id).worker_mode
            primary_schema = self.layer_schemas.get(self.layer_id, {})
            self.classes = list(primary_schema.get("classes") or [])
            self.kp_names = list(primary_schema.get("kp_names") or [])
            self.class_keypoints = dict(primary_schema.get("class_keypoints") or {})
        self.model_path = self.model_paths.get(self.layer_id) or None

        if len(self.review_layers) > 1:
            self.setWindowTitle("Video Review (Project Models)")
        else:
            layer_title = "Segmentation" if self._is_seg_workflow() else "Keypoints"
            self.setWindowTitle(f"Video Review ({layer_title} Layer)")
        self.resize(1080, 760)

        # runtime state
        self.cap = None
        self.path: Optional[str] = None
        self.base: str = ""
        self.video_source_id: str = ""
        self.total: int = 0
        self.fps: float = 0.0
        self.cur: int = 0
        self.preds_by_layer: dict[str, dict[int, dict]] = {
            layer: {} for layer in (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
        }
        self.preds: dict[int, dict] = self.preds_by_layer[self.layer_id]
        self._last_frame_bgr = None  # holds the current raw frame for export
        self._review_job: Optional[WorkerJobController] = None
        self._review_process: Optional[QProcess] = None
        self._review_progress: Optional[QProgressDialog] = None
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_result_event: Optional[dict] = None
        self._review_partial_preds: dict[int, dict] = {}
        self._review_config_path: Optional[str] = None
        self._review_cancel_requested = False
        self._review_run_meta: Optional[dict] = None
        self._review_job_queue: list[str] = []
        self._review_current_layer = self.layer_id
        self._review_run_errors: list[str] = []
        self._review_run_canceled = False
        self._review_steps_per_pass = 1
        self._review_pass_index = 0
        self._review_pass_total = 0
        self._review_settings: dict = {}
        self._review_closing = False

        # build all widgets/layouts
        self._build_ui()

    def _is_seg_workflow(self, layer_id: Optional[str] = None) -> bool:
        candidate = normalize_layer_id(layer_id or self.layer_id)
        return candidate == LAYER_SEGMENTATION

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

        self.btn_project_video = QPushButton("Project Videos…")
        self.btn_project_video.setToolTip("Choose a video from this project's videos library")
        self.btn_project_video.clicked.connect(self._choose_project_video)
        row.addWidget(self.btn_project_video)

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
        self.predict_layer_actions = {}
        if not self.review_layers:
            self.btn_predict.setToolTip(
                "Configure a Keypoints or Segmentation project model to enable predictions."
            )
        elif len(self.review_layers) > 1:
            self.btn_predict.setText("Predict Layers…")
            self.btn_predict.setToolTip(
                "Choose which configured model to run over the selected frame range."
            )
            predict_menu = QMenu(self.btn_predict)
            choices = (
                ("keypoints", "Predict Keypoints", (LAYER_KEYPOINTS,)),
                ("segmentation", "Predict Segmentation", (LAYER_SEGMENTATION,)),
                (
                    "both",
                    "Predict Both Layers",
                    (LAYER_KEYPOINTS, LAYER_SEGMENTATION),
                ),
            )
            for action_id, label, layers in choices:
                if action_id == "both":
                    predict_menu.addSeparator()
                action = predict_menu.addAction(label)
                action.triggered.connect(
                    lambda _checked=False, selected=layers: self._start_range_prediction(selected)
                )
                self.predict_layer_actions[action_id] = action
            self.btn_predict.setMenu(predict_menu)
        elif self._is_seg_workflow():
            self.btn_predict.setToolTip(
                "Run segmentation predictions over the selected frame range."
            )
        else:
            self.btn_predict.setToolTip("Run pose predictions over the selected frame range.")
        if len(self.review_layers) <= 1:
            self.btn_predict.clicked.connect(
                lambda _checked=False: self._start_range_prediction(self.review_layers)
            )
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
        show_kp_controls = LAYER_KEYPOINTS in self.review_layers
        self.lbl_kpvis.setVisible(show_kp_controls)
        self.spin_kpvis.setVisible(show_kp_controls)
        controls_row_2.addStretch(1)
        top.addLayout(controls_row_2)

        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(8)
        overlay_row.addWidget(QLabel("Overlays"))
        self.keypoints_overlay_check = QCheckBox("Keypoints")
        self.segmentation_overlay_check = QCheckBox("Segmentation")
        self.keypoints_overlay_check.setChecked(LAYER_KEYPOINTS in self.review_layers)
        self.segmentation_overlay_check.setChecked(LAYER_SEGMENTATION in self.review_layers)
        self.keypoints_overlay_check.setEnabled(LAYER_KEYPOINTS in self.review_layers)
        self.segmentation_overlay_check.setEnabled(LAYER_SEGMENTATION in self.review_layers)
        self.keypoints_overlay_check.toggled.connect(
            lambda _checked: self._seek(self.cur, show_only=False) if self.cap is not None else None
        )
        self.segmentation_overlay_check.toggled.connect(
            lambda _checked: self._seek(self.cur, show_only=False) if self.cap is not None else None
        )
        overlay_row.addWidget(self.keypoints_overlay_check)
        overlay_row.addWidget(self.segmentation_overlay_check)
        overlay_row.addStretch(1)
        configured_text = []
        for configured_layer in self.review_layers:
            configured_text.append(
                f"{layer_definition(configured_layer).display_name}: "
                f"{os.path.basename(self.model_paths[configured_layer])}"
            )
        self.model_summary_label = QLabel(
            "  ·  ".join(configured_text) if configured_text else "No models · browse/export only"
        )
        self.model_summary_label.setToolTip(
            "\n".join(configured_text)
            if configured_text
            else "Configure project models later to enable prediction overlays."
        )
        overlay_row.addWidget(self.model_summary_label)
        top.addLayout(overlay_row)

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
            "Choose a prediction layer, then export its lowest-confidence frames by class or balanced across classes"
        )
        self.btn_send_low.setEnabled(False)
        self.btn_send_low.clicked.connect(self._export_low_confidence_frames)
        buttons.addButton(self.btn_send_low, QDialogButtonBox.ButtonRole.ActionRole)

        self.btn_send_high = QPushButton("Send High…")
        self.btn_send_high.setToolTip(
            "Choose a prediction layer, then export its highest-confidence frames by class or balanced across classes"
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
        self._zoom_out_sc.activated.connect(lambda: self.view.scale(1 / 1.05, 1 / 1.05))
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
        return video_review_cache_path(self.project_root, self.path)

    def _video_signature(self) -> dict:
        return build_video_signature(self.path, total=self.total, fps=self.fps)

    def _load_cache_if_valid(self) -> bool:
        try:
            fp = self._cache_path()
        except (OSError, ValueError):
            logger.warning(
                "Rejected unsafe video prediction cache path",
                exc_info=True,
                extra={
                    "event": "video_cache_path_rejected",
                    "operation": "load_video_cache",
                    "project_root": self.project_root,
                    "source_path": self.path,
                },
            )
            return False
        if not fp or not os.path.exists(fp):
            return False
        try:
            data = read_json_file(
                fp,
                max_bytes=MAX_VIDEO_CACHE_BYTES,
                require_object=True,
            )
            active_layer = getattr(self, "layer_id", getattr(self, "workflow", WORKFLOW_POSE))
            decision = decide_video_review_cache(
                data,
                current_video=self._video_signature(),
                review_layers=getattr(self, "review_layers", []),
                model_paths=getattr(self, "model_paths", {}),
                layer_id=active_layer,
                model_path=getattr(self, "model_path", None),
                workflow=getattr(self, "workflow", WORKFLOW_POSE),
            )
            if decision is None:
                return False
            if hasattr(self, "preds_by_layer"):
                self.preds_by_layer.update(decision.predictions_by_layer)
                self.preds = self.preds_by_layer.get(active_layer, {})
            else:
                self.preds = decision.predictions_by_layer.get(normalize_layer_id(active_layer), {})
            return decision.has_predictions
        except (OSError, UnicodeError, TypeError, ValueError, AttributeError):
            logger.warning(
                "Ignored invalid video prediction cache",
                exc_info=True,
                extra={
                    "event": "video_cache_invalid",
                    "operation": "load_video_cache",
                    "project_root": self.project_root,
                    "source_path": fp,
                },
            )
            return False

    def _save_cache(self, meta: dict):
        try:
            fp = self._cache_path()
        except (OSError, ValueError):
            logger.warning(
                "Rejected unsafe video prediction cache path",
                exc_info=True,
                extra={
                    "event": "video_cache_path_rejected",
                    "operation": "save_video_cache",
                    "project_root": self.project_root,
                    "source_path": self.path,
                },
            )
            return
        if not fp:
            return
        data = build_video_review_cache_payload(meta, self.preds_by_layer)
        try:
            serialized = json.dumps(data)
            if len(serialized.encode("utf-8")) > MAX_VIDEO_CACHE_BYTES:
                logger.warning(
                    "Skipped oversized video prediction cache",
                    extra={
                        "event": "video_cache_oversized",
                        "operation": "save_video_cache",
                        "project_root": self.project_root,
                        "target_path": fp,
                    },
                )
                return
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            atomic_write_text(fp, serialized)
        except (OSError, TypeError, ValueError):
            logger.warning(
                "Could not save video prediction cache",
                exc_info=True,
                extra={
                    "event": "video_cache_save_failed",
                    "operation": "save_video_cache",
                    "project_root": self.project_root,
                    "target_path": fp,
                },
            )
            return

    # ---------- video load ----------
    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            os.path.join(self.project_root, "videos"),
            "Videos (*.mp4 *.mov *.avi *.mkv)",
        )
        if not path:
            return
        self._open_video(path)

    def _choose_project_video(self):
        videos_dir = os.path.join(self.project_root, "videos")
        entries = list_project_videos(videos_dir)
        available = [entry for entry in entries if entry.target_exists]
        missing_count = len(entries) - len(available)
        if not available:
            detail = (
                "All project video links have missing sources."
                if entries
                else "No videos have been added to this project yet."
            )
            QMessageBox.information(
                self,
                "No Project Videos Available",
                f"{detail}\n\nUse Videos > Add Video Links… in the main window.",
            )
            return
        prompt = f"Choose a project video ({len(available)} available"
        if missing_count:
            prompt += f", {missing_count} missing"
        prompt += "):"
        names = [entry.name for entry in available]
        selected, accepted = QInputDialog.getItem(
            self,
            "Select Project Video",
            prompt,
            names,
            0,
            False,
        )
        if not accepted:
            return
        entry = next((candidate for candidate in available if candidate.name == selected), None)
        if entry is not None:
            self._open_video(entry.path)

    def _open_video(self, path: str):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies."
            )
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
        self.spin_start.setRange(0, max(0, self.total - 1))
        self.spin_start.setValue(0)
        self.spin_end.setRange(0, max(0, self.total - 1))
        self.spin_end.setValue(max(0, self.total - 1))
        self.slider.setRange(0, max(0, self.total - 1))
        self.btn_predict.setEnabled(bool(self.review_layers))
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
                self.btn_send_low.setEnabled(any(self.preds_by_layer.values()))
            if hasattr(self, "btn_send_high"):
                self.btn_send_high.setEnabled(any(self.preds_by_layer.values()))
            cached_keys = sorted(
                {
                    frame_idx
                    for predictions in self.preds_by_layer.values()
                    for frame_idx in predictions
                }
            )
            self._seek(cached_keys[0] if cached_keys else 0, show_only=False, fit_view=True)
        else:
            # Timeline scrubbing should work even without predictions.
            self.slider.setEnabled(True)
            self._seek(0, show_only=True, fit_view=True)

        # Reset pan/zoom whenever a new video is opened
        if hasattr(self, "view") and hasattr(self.view, "reset_view"):
            self.view.reset_view()

    # ---------- prediction ----------
    def _review_prediction_is_running(self) -> bool:
        job = self._review_job
        if job is not None:
            return job.is_running
        process = self._review_process
        return process is not None and process.state() != QProcess.ProcessState.NotRunning

    def _create_review_job_controller(self) -> WorkerJobController:
        return WorkerJobController(self)

    def _start_range_prediction(self, requested_layers=None):
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        if not self.review_layers:
            QMessageBox.information(
                self,
                "No project models",
                "Configure a Keypoints or Segmentation model in Project Models first.",
            )
            return
        if self._review_prediction_is_running():
            QMessageBox.information(
                self, "Prediction running", "Video prediction is already running."
            )
            return

        if requested_layers is None:
            prediction_layers = list(self.review_layers)
        else:
            requested = {normalize_layer_id(layer) for layer in requested_layers}
            prediction_layers = [layer for layer in self.review_layers if layer in requested]
        if not prediction_layers:
            QMessageBox.information(
                self,
                "Model not configured",
                "Configure the selected prediction model in Project Models first.",
            )
            return

        start = int(self.spin_start.value())
        end = int(self.spin_end.value())
        stride = max(1, int(self.spin_stride.value()))
        conf = float(self.spin_conf.value())
        iou = float(self.spin_iou.value())
        requested_batch = int(self.spin_batch.value()) if hasattr(self, "spin_batch") else 0
        effective_batch = effective_prediction_batch(requested_batch, self.device)
        imgsz = 640
        kpvis = (
            float(self.spin_kpvis.value())
            if hasattr(self, "spin_kpvis") and LAYER_KEYPOINTS in self.review_layers
            else None
        )

        if end < start:
            QMessageBox.warning(self, "Range Error", "End must be ≥ Start.")
            return

        run_plan = plan_video_review_run(
            video_signature=self._video_signature(),
            model_paths=self.model_paths,
            review_layers=prediction_layers,
            layer_schemas=self.layer_schemas,
            start=start,
            end=end,
            stride=stride,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            kpvis=kpvis,
            requested_batch=requested_batch,
            effective_batch=effective_batch,
            total=self.total,
            fps=self.fps,
        )
        preparing = (
            f"Preparing {layer_definition(prediction_layers[0]).display_name} model…"
            if len(prediction_layers) == 1
            else "Preparing project models…"
        )
        prog = QProgressDialog(preparing, "Cancel", 0, run_plan.total_steps, self)
        prog.setWindowTitle("Project Video Review")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        prog.canceled.connect(self._cancel_review_prediction_process)

        self._review_run_meta = run_plan.meta
        self._review_settings = run_plan.settings
        self._review_progress = prog
        self._review_cancel_requested = False
        self._review_run_canceled = False
        self._review_run_errors = []
        self._review_job_queue = list(prediction_layers)
        self._review_pass_total = len(prediction_layers)
        self._review_pass_index = 0
        self._review_steps_per_pass = run_plan.steps_per_pass
        for layer in prediction_layers:
            self.preds_by_layer[layer] = {}
        self.preds = self.preds_by_layer[self.layer_id]
        self.btn_predict.setEnabled(False)
        if hasattr(self, "btn_send_low"):
            self.btn_send_low.setEnabled(False)
        if hasattr(self, "btn_send_high"):
            self.btn_send_high.setEnabled(False)

        prog.show()
        self._start_next_review_prediction_pass()

    def _start_next_review_prediction_pass(self) -> None:
        if not self._review_job_queue:
            self._finish_project_review_prediction()
            return
        layer_id = self._review_job_queue.pop(0)
        self._review_current_layer = layer_id
        self._review_pass_index += 1
        config = build_video_review_pass_config(
            layer_id=layer_id,
            model_path=self.model_paths.get(layer_id) or "",
            video_path=self.path,
            device=self.device,
            settings=self._review_settings,
        )

        config_dir = os.path.join(self.project_root, "logs")
        try:
            config_path = create_worker_config(
                self.project_root,
                config_dir,
                f"video_review_{layer_id}",
                config,
            )
        except Exception as e:
            self._review_run_errors.append(f"{layer_definition(layer_id).display_name}: {e}")
            QTimer.singleShot(0, self._start_next_review_prediction_pass)
            return

        job = self._create_review_job_controller()
        job.event_received.connect(self._handle_review_prediction_event)
        job.output_received.connect(self._handle_review_prediction_output)
        job.stderr_received.connect(self._handle_review_prediction_stderr)
        job.terminal.connect(self._finish_review_prediction_job)

        self._review_job = job
        self._review_process = None
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_result_event = None
        self._review_partial_preds = {}
        self._review_config_path = config_path
        progress = self._review_progress
        if progress is not None:
            progress.setLabelText(
                f"Pass {self._review_pass_index}/{self._review_pass_total} · "
                f"Loading {layer_definition(layer_id).display_name} model…"
            )
        started = job.start(
            sys.executable,
            ["-m", "video_review_worker", "--config", config_path],
            config_path=config_path,
            working_directory=APP_BASE_DIR,
            start_timeout_ms=1000,
        )
        if started and self._review_job is job:
            self._review_process = job.process

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
        self._review_stderr += bytes(process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )

    def _handle_review_prediction_output(self, line: str) -> None:
        if line:
            self._review_stderr += line.rstrip("\n") + "\n"

    def _handle_review_prediction_stderr(self, line: str) -> None:
        if line:
            self._review_stderr += line.rstrip("\n") + "\n"

    def _handle_review_prediction_event_line(self, line: str):
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self._handle_review_prediction_output(line)
            return

        self._handle_review_prediction_event(event)

    def _handle_review_prediction_event(self, event: dict) -> None:
        event_type = event.get("event")
        if event_type == "started":
            progress = self._review_progress
            if progress is not None:
                progress.setLabelText(
                    f"Pass {self._review_pass_index}/{self._review_pass_total} · "
                    f"Loading {layer_definition(self._review_current_layer).display_name} model…"
                )
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
                total = int(event.get("total") or self._review_steps_per_pass)
                completed_before = (self._review_pass_index - 1) * self._review_steps_per_pass
                progress.setMaximum(
                    max(
                        1,
                        self._review_steps_per_pass * self._review_pass_total,
                    )
                )
                progress.setValue(completed_before + min(processed, max(1, total)))
                detail = str(event.get("message") or f"Predicting {processed}/{total}")
                progress.setLabelText(
                    f"Pass {self._review_pass_index}/{self._review_pass_total} · "
                    f"{layer_definition(self._review_current_layer).display_name}\n{detail}"
                )
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
        self._review_run_canceled = True
        job = self._review_job
        if job is None or not job.is_running:
            self._review_job_queue.clear()
            self._finish_project_review_prediction()
            return
        self._review_cancel_requested = True
        progress = self._review_progress
        if progress is not None:
            progress.setLabelText("Canceling prediction process…")
        job.cancel(kill_after_ms=5000)

    def _kill_review_prediction_if_running(self):
        job = self._review_job
        process = job.process if job is not None else self._review_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_review_prediction_error(self, _error):
        job = self._review_job
        process = job.process if job is not None else self._review_process
        if process is not None:
            self._review_stderr += process.errorString() + "\n"

    def _finish_review_prediction_process(self, exit_code: int, exit_status) -> None:
        """Compatibility adapter for callers using the former QProcess callback."""
        if self._review_process is None and self._review_config_path is None:
            return
        if self._review_stdout_buffer.strip():
            self._handle_review_prediction_event_line(self._review_stdout_buffer.strip())
            self._review_stdout_buffer = ""
        _remove_file_quietly(self._review_config_path)
        state = (
            "cancelled"
            if self._review_cancel_requested
            else "finished"
            if int(exit_code) == 0
            else "failed"
        )
        self._finish_review_prediction_job(
            WorkerJobResult(
                state=state,
                exit_code=int(exit_code),
                exit_status=exit_status,
                stderr=self._review_stderr,
            )
        )

    def _finish_review_prediction_job(self, result: WorkerJobResult) -> None:
        event = self._review_result_event
        partial_preds = dict(self._review_partial_preds)
        stderr_text = self._review_stderr.strip() or result.stderr.strip()
        cancel_requested = self._review_cancel_requested
        layer_id = self._review_current_layer

        self._review_job = None
        self._review_process = None
        self._review_config_path = None
        self._review_result_event = None
        self._review_partial_preds = {}
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_cancel_requested = False

        if self._review_closing:
            return

        completion = complete_video_review_pass(
            partial_predictions=partial_preds,
            result_event=event,
            cancel_requested=cancel_requested,
            worker_state=result.state,
            exit_code=result.exit_code,
            crashed=result.exit_status == QProcess.ExitStatus.CrashExit,
            worker_error=result.error_message,
            stderr=stderr_text,
        )
        self.preds_by_layer[layer_id] = completion.predictions
        self.preds = self.preds_by_layer.get(self.layer_id, {})

        if completion.had_error:
            self._review_run_errors.append(
                f"{layer_definition(layer_id).display_name}: {completion.error_message}"
            )

        if completion.canceled:
            self._review_run_canceled = True
            self._review_job_queue.clear()

        if self._review_job_queue:
            QTimer.singleShot(0, self._start_next_review_prediction_pass)
            return
        self._finish_project_review_prediction()

    def _finish_project_review_prediction(self) -> None:
        progress = self._review_progress
        self._review_progress = None
        if progress is not None:
            # QProgressDialog.close() emits canceled(), even when the dialog
            # already reached its maximum and the worker finished normally.
            # Disconnect the user-cancel handler before programmatic cleanup so
            # a successful run is not reported as canceled.
            try:
                progress.canceled.disconnect(self._cancel_review_prediction_process)
            except (TypeError, RuntimeError):
                pass
            progress.close()
        self._review_job = None
        self._review_process = None
        self._review_config_path = None
        self.btn_predict.setEnabled(self.cap is not None and bool(self.review_layers))

        self.preds = self.preds_by_layer.get(self.layer_id, {})
        has_predictions = any(self.preds_by_layer.values())
        if has_predictions:
            try:
                self._save_cache(self._review_run_meta or {})
            except Exception:
                pass
            self.slider.setEnabled(True)
            if hasattr(self, "btn_send_low"):
                self.btn_send_low.setEnabled(any(self.preds_by_layer.values()))
            if hasattr(self, "btn_send_high"):
                self.btn_send_high.setEnabled(any(self.preds_by_layer.values()))
            frame_keys = sorted(
                {
                    frame_idx
                    for predictions in self.preds_by_layer.values()
                    for frame_idx in predictions
                }
            )
            if frame_keys:
                self._seek(frame_keys[0], show_only=False)

        errors = list(self._review_run_errors)
        canceled = bool(self._review_run_canceled)
        self._review_job_queue = []
        self._review_run_errors = []
        self._review_run_canceled = False
        self._review_run_meta = None

        if errors:
            prefix = (
                "Some model passes failed, but successful predictions were kept."
                if has_predictions
                else "Video prediction failed."
            )
            QMessageBox.warning(
                self,
                "Project Prediction Finished",
                prefix + "\n\n" + "\n".join(errors),
            )
        elif canceled:
            QMessageBox.information(
                self,
                "Prediction canceled",
                "Prediction was canceled; completed layer results were kept."
                if has_predictions
                else "Prediction was canceled before results were generated.",
            )
        elif not has_predictions:
            QMessageBox.information(
                self,
                "No predictions",
                "The configured models completed without generating predictions.",
            )

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
        self.lbl_idx.setText(f"{self.cur + 1}/{self.total}")

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
            QMessageBox.warning(
                self, "Export Error", "Could not locate the labeler's images_to_label directory."
            )
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
                f"Frame {self.cur} is already in images_to_label.\nSkipping duplicate export.",
            )
            return

        try:
            os.makedirs(dest_dir, exist_ok=True)
            out_path = plan_export_frame_path(
                dest_dir,
                video_base=self.base,
                source_id=self.video_source_id,
                frame_index=self.cur,
                avoid_collisions=False,
            )

            if _cv2 is None:
                QMessageBox.warning(
                    self,
                    "OpenCV missing",
                    "Run `uv sync --locked` to restore project dependencies.",
                )
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
            QMessageBox.warning(
                self, "Export Error", "Could not locate the labeler's images_to_label directory."
            )
            return
        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies."
            )
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not create destination folder:\n{e}")
            return

        existing = self._existing_export_indices()
        available = available_export_frame_indices(
            self.total,
            already_exported=existing,
        )
        if not available:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Every frame from this video is already in images_to_label.",
            )
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

        selected = select_random_export_frames(
            self.total,
            already_exported=existing,
            count=n,
        )

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
                dest_path = plan_export_frame_path(
                    dest_dir,
                    video_base=self.base,
                    source_id=self.video_source_id,
                    frame_index=fi,
                )
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
                detail += "\n\nIssues:\n" + "\n".join(
                    f"frame {fi}: {reason}" for fi, reason in failed[:10]
                )
                if len(failed) > 10:
                    detail += f"\n…{len(failed) - 10} more"
            QMessageBox.information(self, title, detail)

        if failed:
            msg = "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
            more = "" if len(failed) <= 10 else f"\n…{len(failed) - 10} more"
            QMessageBox.warning(
                self,
                "Some exports failed",
                f"{saved} succeeded, {len(failed)} failed.\n\n{msg}{more}",
            )

    def _existing_export_indices(self) -> set[int]:
        """Scan the labeler's images_to_label folder for frames already exported for this video."""
        dest_dir = self._labeler_image_dir()
        if not dest_dir or not os.path.isdir(dest_dir):
            return set()
        try:
            return exported_frame_indices(
                os.listdir(dest_dir),
                video_base=self.base,
                source_id=self.video_source_id,
            )
        except Exception:
            return set()

    def _export_low_confidence_frames(self):
        self._export_predictions_by_confidence(order="low")

    def _export_high_confidence_frames(self):
        self._export_predictions_by_confidence(order="high")

    def _export_predictions_by_confidence(self, order: str):
        order_key = (order or "low").lower()
        if order_key not in {"low", "high"}:
            order_key = "low"

        ranking_layers = [
            layer_id for layer_id in self.review_layers if self.preds_by_layer.get(layer_id)
        ]
        if not ranking_layers:
            QMessageBox.information(
                self, "No predictions", "Run Predict Range first to generate predictions."
            )
            return
        if self.cap is None or not self.path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        if len(ranking_layers) > 1:
            layer_choices = [layer_definition(layer_id).display_name for layer_id in ranking_layers]
            layer_choice, layer_ok = QInputDialog.getItem(
                self,
                "Confidence Ranking Layer",
                "Use predictions from which layer?",
                layer_choices,
                0,
                False,
            )
            if not layer_ok:
                return
            try:
                ranking_layer_id = ranking_layers[layer_choices.index(layer_choice)]
            except ValueError:
                return
        else:
            ranking_layer_id = ranking_layers[0]

        ranking_layer_name = layer_definition(ranking_layer_id).display_name
        ranking_predictions = self.preds_by_layer[ranking_layer_id]
        schema = self.layer_schemas.get(ranking_layer_id, {})
        ranking_classes = list(schema.get("classes") or self.classes)

        mode_choices = ["Balanced by class"] + [
            ranking_classes[class_id] if class_id < len(ranking_classes) else str(class_id)
            for class_id in range(len(ranking_classes))
        ]
        ranking_choice, choice_ok = QInputDialog.getItem(
            self,
            f"{ranking_layer_name} Confidence Ranking",
            f"Rank {ranking_layer_name} frames for which class?",
            mode_choices,
            0,
            False,
        )
        if not choice_ok:
            return

        balanced = ranking_choice == "Balanced by class"
        if balanced:
            ranking_class_ids = list(range(len(ranking_classes)))
            ranking_label = "balanced by class"
        else:
            try:
                ranking_class_ids = [mode_choices.index(ranking_choice) - 1]
            except ValueError:
                return
            ranking_label = ranking_choice

        export_plan = plan_confidence_export(
            ranking_predictions,
            class_ids=ranking_class_ids,
            order=order_key,
            balanced=balanced,
            already_exported=self._existing_export_indices(),
        )
        if not export_plan.candidates:
            QMessageBox.information(
                self,
                "No predictions",
                f"No {ranking_layer_name} predictions are available for the {ranking_label} ranking.",
            )
            return

        if order_key == "low":
            order_label = "lowest"
            dialog_title = f"Export Lowest {ranking_layer_name} Confidence"
        else:
            order_label = "highest"
            dialog_title = f"Export Highest {ranking_layer_name} Confidence"

        pending = export_plan.pending
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
            f"How many {ranking_layer_name} / {ranking_label} frames should I send to the labeler?",
            default_n,
            1,
            max_n,
            1,
        )
        if not ok or n <= 0:
            return

        selected = plan_confidence_export(
            ranking_predictions,
            class_ids=ranking_class_ids,
            order=order_key,
            balanced=balanced,
            already_exported=self._existing_export_indices(),
            count=n,
        ).selected

        parent = self.parent()
        dest_dir = self._labeler_image_dir()
        if not dest_dir:
            QMessageBox.warning(
                self, "Export Error", "Could not locate the labeler's images_to_label directory."
            )
            return
        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies."
            )
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
                dest_path = plan_export_frame_path(
                    dest_dir,
                    video_base=self.base,
                    source_id=self.video_source_id,
                    frame_index=fi,
                )
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
                msg += f"\nRanking: {ranking_layer_name}, {order_label} confidence, {ranking_label}"
                for class_id in ranking_class_ids:
                    class_confs = [
                        confidence
                        for confidence, ranked_class_id in saved_rankings
                        if ranked_class_id == class_id
                    ]
                    if not class_confs:
                        continue
                    class_name = (
                        ranking_classes[class_id]
                        if class_id < len(ranking_classes)
                        else str(class_id)
                    )
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
                detail += "\n\nIssues:\n" + "\n".join(
                    f"frame {fi}: {reason}" for fi, reason in failed[:10]
                )
                if len(failed) > 10:
                    detail += f"\n…{len(failed) - 10} more"
            QMessageBox.information(self, title, detail)

        if failed:
            msg = "\n".join(f"frame {fi}: {reason}" for fi, reason in failed[:10])
            more = "" if len(failed) <= 10 else f"\n…{len(failed) - 10} more"
            QMessageBox.warning(
                self,
                "Some exports failed",
                f"{saved} succeeded, {len(failed)} failed.\n\n{msg}{more}",
            )

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

        for layer_id in self.review_layers:
            if layer_id == LAYER_KEYPOINTS:
                visible = self.keypoints_overlay_check.isChecked()
            else:
                visible = self.segmentation_overlay_check.isChecked()
            if not visible:
                continue
            p = self.preds_by_layer.get(layer_id, {}).get(frame_idx)
            if not p or not p.get("ok"):
                continue
            detections = p.get("detections")
            if isinstance(detections, list) and detections:
                for detection in detections:
                    if isinstance(detection, dict) and detection.get("ok"):
                        self._draw_prediction_overlay(detection, layer_id=layer_id)
                continue
            self._draw_prediction_overlay(p, layer_id=layer_id)

    def _draw_prediction_overlay(self, p: dict, *, layer_id: Optional[str] = None):
        layer_id = normalize_layer_id(layer_id or self.layer_id)
        schema = self.layer_schemas.get(layer_id, {})
        classes = list(schema.get("classes") or self.classes)
        kp_names = list(schema.get("kp_names") or self.kp_names)
        class_keypoints = dict(schema.get("class_keypoints") or self.class_keypoints)
        cls_id = int(p.get("cls", 0))
        class_name = classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)
        if self._is_seg_workflow(layer_id):
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
            frame_color = QColor(32, 78, 255)
            label_x = 6.0
            label_y = 6.0
            frame_rect = None

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
                frame_rect = bbox
                label_x = bbox.left() + 2.0
                label_y = bbox.top() + 2.0
            elif p.get("xyxy"):
                x1, y1, x2, y2 = p["xyxy"]
                frame_rect = (x1, y1, x2 - x1, y2 - y1)
                label_x = x1 + 2.0
                label_y = y1 + 2.0

            if frame_rect is not None:
                if isinstance(frame_rect, tuple):
                    rect_item = QGraphicsRectItem(*frame_rect)
                else:
                    rect_item = QGraphicsRectItem(frame_rect)
                rect_pen = QPen(frame_color)
                rect_pen.setWidth(3)
                rect_pen.setCosmetic(True)
                rect_item.setPen(rect_pen)
                rect_item.setBrush(QBrush(Qt.GlobalColor.transparent))
                rect_item.setZValue(5.5)
                self.scene.addItem(rect_item)
                self._overlay_items.append(rect_item)

            label_item = QGraphicsSimpleTextItem(f"{class_name} {p.get('conf', 0.0):.2f}")
            label_item.setFont(_ui_font(24))
            label_item.setBrush(QBrush(frame_color))
            label_item.setPos(label_x, label_y)
            label_item.setZValue(6)
            self.scene.addItem(label_item)
            self._overlay_items.append(label_item)
            return

        class_kp_names = class_keypoints.get(class_name, kp_names)

        # ---- Bounding box (blue, thicker) ----
        if p.get("xyxy"):
            x1, y1, x2, y2 = p["xyxy"]
            r = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(Qt.GlobalColor.blue)
            pen.setWidth(3)
            pen.setCosmetic(True)
            r.setPen(pen)
            r.setZValue(5)
            self.scene.addItem(r)
            self._overlay_items.append(r)

            # class + confidence (bigger, blue)
            t = QGraphicsSimpleTextItem(f"{class_name} {p.get('conf', 0.0):.2f}")
            t.setFont(_ui_font(24))
            t.setBrush(QBrush(Qt.GlobalColor.blue))
            t.setPos(x1 + 2, y1 + 2)
            t.setZValue(6)
            self.scene.addItem(t)
            self._overlay_items.append(t)

        # ---- Keypoints (map kp conf → visibility) ----
        thr = float(self.spin_kpvis.value()) if hasattr(self, "spin_kpvis") else 0.5
        for i, kp in enumerate(p.get("kps", [])):
            if i >= len(kp_names):
                break
            name = kp_names[i]
            if name not in class_kp_names:
                continue
            x, y, conf = kp
            vis = 2 if conf >= thr else 1  # 2=visible(red), 1=occluded(yellow)

            if vis == 2:
                color = Qt.GlobalColor.red
                fill = QBrush(color)
                style = Qt.PenStyle.SolidLine
            elif vis == 1:
                color = Qt.GlobalColor.yellow
                fill = QBrush(color)
                style = Qt.PenStyle.SolidLine
            else:
                color = Qt.GlobalColor.lightGray
                fill = QBrush(Qt.GlobalColor.transparent)
                style = Qt.PenStyle.DashLine

            dot = QGraphicsEllipseItem(-4, -4, 8, 8)  # slightly larger dot
            dot.setPos(x, y)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(style)
            dot.setPen(pen)
            dot.setBrush(fill)
            dot.setZValue(7)
            self.scene.addItem(dot)
            self._overlay_items.append(dot)

            # label next to kp
            lbl = QGraphicsSimpleTextItem(name)
            lbl.setFont(_ui_font(18))
            lbl.setBrush(QBrush(color))
            lbl.setPos(x + 8, y - 16)
            lbl.setZValue(8)
            lbl.setVisible(vis != 0)  # hide if invisible
            self.scene.addItem(lbl)
            self._overlay_items.append(lbl)

    @staticmethod
    def _cv_to_qpix(frame_bgr) -> QPixmap:
        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        from PyQt6.QtGui import QImage

        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def reject(self):
        if self._review_prediction_is_running():
            answer = QMessageBox.question(
                self,
                "Cancel prediction?",
                "Video prediction is still running. Cancel it and close the reviewer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._review_closing = True
            self._review_run_canceled = True
            self._review_job_queue.clear()
            job = self._review_job
            if job is not None:
                job.shutdown()
            else:
                _shutdown_qprocess(self._review_process)
                _remove_file_quietly(self._review_config_path)
            if self._review_progress is not None:
                self._review_progress.close()
            self._review_job = None
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
