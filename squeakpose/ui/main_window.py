#!/usr/bin/env python3
"""Main-window and feature dialogs for SqueakPose Studio.

The repository-level ``squeakpose_studio.py`` remains a compatibility launcher.
"""

import datetime
import json
import logging
import os
import platform
import random
import re
import shlex
import shutil
import sys
from dataclasses import replace
from typing import List, Optional

import yaml
from PyQt6.QtCore import QLibraryInfo, QPoint, QPointF, QProcess, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontDatabase,
    QFontInfo,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis_dialog import AnalysisDialog
from squeakpose import __version__
from squeakpose.annotation.depth import load_depth_artifacts, plan_depth_artifacts
from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    SegmentationAnnotationDocument,
)
from squeakpose.annotation.graphics import BoxItem, KeypointItem, LabelView
from squeakpose.annotation.models import (
    Annotation,
    BoundingBox,
    Keypoint,
    KeypointEntry,
)
from squeakpose.annotation.pose import PoseEditState
from squeakpose.annotation.segmentation import (
    SegmentationEditState,
    apply_brush_stroke,
    clamp_point_to_image,
    polygon_to_mask,
    segmentation_mask_shape,
)
from squeakpose.annotation.segmentation_assistant import (
    SamPromptRequest,
    discover_sam_weight_candidates,
    select_existing_sam_weight,
)
from squeakpose.annotation.serialization import (
    parse_pose_label_line,
    parse_segmentation_label_line,
    pose_annotation_to_line,
    segmentation_annotation_to_line,
)
from squeakpose.annotation.video_view import VideoView
from squeakpose.core import (
    atomic_write_text,
    atomic_write_text_files,
    commit_staged_paths,
    effective_prediction_batch,
    filter_image_stem_collisions,
    find_duplicate_names,
    remove_path,
    stable_path_id,
    staging_path_for,
)
from squeakpose.depth_ops import DepthMapError, keypoint_depth_label, sample_depth_map
from squeakpose.diagnostics import configure_project_logging, project_log_path
from squeakpose.json_io import read_json_file
from squeakpose.project.distillation import (
    discover_distillation_exports,
    distillation_export_search_roots,
    distillation_sample_count,
    preferred_distillation_export,
)
from squeakpose.project.health import (
    cleanup_project_temporary_paths,
    format_project_health_summary,
    scan_project_health,
)
from squeakpose.project.layers import (
    LAYER_DEFINITIONS,
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    layer_worker_mode,
    normalize_layer_id,
    normalize_layer_settings,
)
from squeakpose.project.metadata import ProjectMetadataStore
from squeakpose.project.paths import (
    ProjectPaths,
    default_projects_root,
    ensure_project_structure,
    load_last_project,
    project_window_title,
    save_last_project,
)
from squeakpose.project.recovery import (
    cleanup_transaction_staging,
    restore_missing_transaction_targets,
    scan_transaction_artifacts,
)
from squeakpose.project.safety import (
    ProjectLock,
    ProjectLockedError,
    ProjectPathError,
    break_stale_project_lock,
    canonical_path,
)
from squeakpose.project.session import ProjectSession
from squeakpose.services.annotation_save import save_annotation_transaction
from squeakpose.services.dataset import export_dataset_transaction
from squeakpose.services.dataset_ops import (
    DATASET_POSE,
    DATASET_SEGMENT,
    backup_label_dir,
    dataset_dirs_have_files,
    dataset_export_paths,
    format_dataset_export_summary,
    format_label_normalization_summary,
    label_file_has_usable_rows,
    list_image_files,
    list_label_files,
    normalize_label_directory,
    partition_images_by_usable_labels,
    split_train_val_images,
)
from squeakpose.services.frame_annotations import (
    SegmentationBoxUnavailableError,
    build_pose_save_request,
    build_segmentation_save_request,
    load_pose_document,
    load_segmentation_document,
    plan_segmentation_box_transfer,
)
from squeakpose.services.image_queue import (
    ImageDeletionPlan,
    ImageQueueNavigator,
    image_label_path,
    image_stem_conflicts,
    next_unlabeled_index,
    plan_image_deletion,
    queue_progress,
    scan_image_queue,
)
from squeakpose.services.inference import (
    InferenceJobPlan,
    InferencePassResult,
    InferenceRunSummary,
    configured_inference_layers,
    plan_inference_run,
    prepare_inference_run,
)
from squeakpose.services.inference_runtime import probe_video_metadata
from squeakpose.services.prediction import (
    DepthPredictionTargets,
    plan_prediction_application,
    validate_prediction_identity,
)
from squeakpose.services.prediction_serialization import rank_prediction_frames
from squeakpose.ui.annotation_panel import (
    AnnotationPanel,
    AnnotationPanelCallbacks,
    SegmentationToolsCallbacks,
    SegmentationToolsPanel,
)
from squeakpose.ui.canvas_presentation import CanvasHudPresenter
from squeakpose.ui.canvas_scene_presenter import (
    CanvasScenePresenter,
    PoseReferenceKeypoint,
)
from squeakpose.ui.class_manager import (
    AddClassDialog,
    ClassManagerDialog,
)
from squeakpose.ui.depth_controller import DepthAssistantController
from squeakpose.ui.depth_panel import (
    DepthDisplayCallbacks,
    DepthDisplayPanel,
    DepthModelCallbacks,
    DepthModelPanel,
    DepthRangePanel,
)
from squeakpose.ui.depth_presentation import DepthPreviewPresenter
from squeakpose.ui.dialog_launch import (
    DialogUnavailableError,
    plan_analysis_dialog,
    plan_training_dialog,
    plan_video_review_dialog,
    require_dialog_support,
)
from squeakpose.ui.distillation_dialog import DistillationDialog
from squeakpose.ui.inference_controller import InferenceController
from squeakpose.ui.navigation_panel import NavigationPanel, NavigationPanelCallbacks
from squeakpose.ui.operation_panel import (
    AnalysisOperationsPanel,
    DatasetOperationsPanel,
    ModelOperationsPanel,
    OperationCallbacks,
    VideoOperationsPanel,
)
from squeakpose.ui.pose_controller import PoseAnnotationController
from squeakpose.ui.prediction_controller import PredictionController
from squeakpose.ui.project_launcher import (
    ProjectLauncherDialog,
    choose_project_root,
    create_project_root,
)
from squeakpose.ui.project_models_dialog import ProjectModelsDialog
from squeakpose.ui.sam_assistant_controller import SamAssistantController
from squeakpose.ui.segmentation_controller import SegmentationAnnotationController
from squeakpose.ui.style import (
    apply_panel_shadow,
    sidebar_stylesheet,
    train_dialog_stylesheet,
)
from squeakpose.ui.training_dialog import TrainDialog
from squeakpose.ui.video_reviewer import VideoReviewDialog
from squeakpose.workers.process import remove_file_quietly, shutdown_qprocess

APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = ["mouse"]
DEFAULT_KEYPOINT_NAMES = ["nose", "head", "left_ear", "right_ear", "back", "tail_base"]
DEFAULT_SAM3_WEIGHTS = "sam3.pt"
# Worker-facing compatibility values. Project and UI state use layer ids.
WORKFLOW_POSE = "pose"
WORKFLOW_SEG = "segmentation"
WORKFLOW_DEPTH = "depth"


def _remove_file_quietly(path: Optional[str]) -> None:
    remove_file_quietly(path)


def _shutdown_qprocess(
    process: Optional[QProcess],
    *,
    terminate_timeout_ms: int = 2000,
    kill_timeout_ms: int = 1000,
) -> bool:
    """Synchronously stop a child process before its owning window closes."""
    return shutdown_qprocess(
        process,
        terminate_timeout_ms=terminate_timeout_ms,
        kill_timeout_ms=kill_timeout_ms,
    )


def _qt_app_instance():
    return QApplication.instance()


def _retain_main_window(window) -> None:
    app = _qt_app_instance()
    if app is not None:
        app._squeakpose_main_window = window


def _acquire_project_lock_for_ui(
    project_root: str,
    *,
    parent: Optional[QWidget] = None,
) -> Optional[ProjectLock]:
    """Acquire a project writer lock, prompting only for a proven stale lock."""

    try:
        lock = ProjectLock(project_root, version=__version__)
        return lock.acquire()
    except (OSError, ProjectPathError) as exc:
        QMessageBox.critical(
            parent,
            "Project Lock Error",
            f"Could not create a safe project writer lock.\n\n{exc}",
        )
        return None
    except ProjectLockedError as exc:
        if not exc.stale:
            QMessageBox.warning(
                parent,
                "Project Already Open",
                f"This project already has a writer lock and cannot be opened for editing.\n\n"
                f"{exc}\n\nLock file:\n{exc.lock_path}",
            )
            return None
        decision = QMessageBox.question(
            parent,
            "Stale Project Lock",
            f"The previous SqueakPose Studio process is no longer running.\n\n{exc}\n\n"
            "Remove the stale lock and open the project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return None
        try:
            break_stale_project_lock(project_root)
            return lock.acquire()
        except (OSError, ProjectLockedError) as retry_error:
            QMessageBox.critical(
                parent,
                "Project Lock Error",
                f"Could not acquire the project writer lock.\n\n{retry_error}",
            )
            return None


def _recover_project_transactions_for_ui(
    project_root: str,
    *,
    parent: Optional[QWidget] = None,
) -> None:
    """Safely recover interrupted writes after the project lock is held."""

    recovery = restore_missing_transaction_targets(project_root)
    if recovery.restored_paths:
        restored = "\n".join(
            os.path.relpath(path, project_root) for path in recovery.restored_paths[:8]
        )
        if len(recovery.restored_paths) > 8:
            restored += f"\n...{len(recovery.restored_paths) - 8} more"
        QMessageBox.information(
            parent,
            "Interrupted Write Recovered",
            "SqueakPose restored missing project data from an interrupted "
            f"transaction:\n\n{restored}",
        )
    if recovery.errors:
        QMessageBox.warning(
            parent,
            "Transaction Recovery Incomplete",
            "Some missing transaction targets could not be restored. Their backups "
            "were left in place:\n\n" + "\n".join(recovery.errors[:8]),
        )

    report = scan_transaction_artifacts(project_root)
    if report.preserved_backups:
        preserved = "\n".join(
            os.path.relpath(item.backup_path, project_root) for item in report.preserved_backups[:8]
        )
        if len(report.preserved_backups) > 8:
            preserved += f"\n...{len(report.preserved_backups) - 8} more"
        QMessageBox.warning(
            parent,
            "Transaction Backups Need Review",
            "Transaction backups conflict with existing data or are ambiguous. "
            "SqueakPose did not modify or delete them:\n\n" + preserved,
        )

    if not report.staging_paths:
        return
    answer = QMessageBox.question(
        parent,
        "Remove Interrupted Transaction Files?",
        (
            f"Found {len(report.staging_paths)} recognized staging file(s) or "
            "export folder(s) from an interrupted transaction. Remove them?\n\n"
            "Transaction backups and project data will not be removed."
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if answer != QMessageBox.StandardButton.Yes:
        return
    cleanup = cleanup_transaction_staging(project_root)
    if cleanup.errors:
        QMessageBox.warning(
            parent,
            "Transaction Cleanup Incomplete",
            "Some staging paths could not be removed:\n\n" + "\n".join(cleanup.errors[:8]),
        )


def _refresh_qt_style(widget: Optional[QWidget]) -> None:
    if widget is None:
        return
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _project_paths(project_root: str) -> ProjectPaths:
    """Compatibility wrapper for callers importing the legacy helper."""
    return ProjectPaths.from_root(project_root)


def _project_window_title(project_root: str) -> str:
    return project_window_title(project_root)


def _distillation_export_search_roots(project_root: str) -> list[tuple[str, str]]:
    return distillation_export_search_roots(project_root)


def _preferred_distillation_export(run_dir: str) -> str:
    return preferred_distillation_export(run_dir)


def _discover_distillation_exports(search_roots: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return discover_distillation_exports(search_roots)


def _distillation_sample_count(total_frames: int, stride: int, max_frames: int = 0) -> int:
    return distillation_sample_count(total_frames, stride, max_frames)


def _ensure_project_structure(project_root: str) -> ProjectPaths:
    """Compatibility wrapper for project creation."""
    return ensure_project_structure(
        project_root,
        default_segmentation_classes=tuple(DEFAULT_CLASS_NAMES),
    )


def _load_last_project() -> Optional[str]:
    try:
        return load_last_project()
    except OSError:
        return None


def _save_last_project(project_root: str):
    try:
        save_last_project(project_root)
    except OSError:
        pass


def _default_projects_root() -> str:
    return default_projects_root()


def _choose_project_root(default_dir: str, parent: Optional[QWidget] = None) -> Optional[str]:
    return choose_project_root(default_dir, parent)


def _create_project_root(default_dir: str, parent: Optional[QWidget] = None) -> Optional[str]:
    return create_project_root(default_dir, parent)


def _ensure_qt_plugin_paths() -> None:
    """Validate and set Qt plugin env vars for the active interpreter.

    Some launchers (`uv run`, IDE shells, conda activation hooks) leave stale
    `QT_*` env vars behind, so Qt searches an old/incompatible plugins tree and
    aborts before the app window opens.
    """

    def _expected_platform_plugins() -> tuple[str, ...]:
        if sys.platform == "darwin":
            return ("libqcocoa.dylib",)
        if sys.platform.startswith("win"):
            return ("qwindows.dll",)
        if sys.platform.startswith("linux"):
            # xcb is typical; keep wayland/offscreen as valid fallbacks.
            return ("libqxcb.so", "libqwayland-egl.so", "libqoffscreen.so")
        return tuple()

    def _split_env_paths(value: str) -> list[str]:
        if not value:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for part in value.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            norm = os.path.abspath(part)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
        return out

    def _has_platform_plugin(root: str) -> bool:
        if not root:
            return False
        platform_dir = os.path.join(root, "platforms")
        if not os.path.isdir(platform_dir):
            return False

        expected = _expected_platform_plugins()
        if expected:
            for name in expected:
                if os.path.isfile(os.path.join(platform_dir, name)):
                    return True
            return False

        try:
            return any(name.lower().startswith(("libq", "q")) for name in os.listdir(platform_dir))
        except Exception:
            return False

    # Candidate roots in priority order:
    # 1) QLibraryInfo (active interpreter), 2) PyQt6 package fallback,
    # 3) existing QT_PLUGIN_PATH entries.
    candidates: list[str] = []
    seen: set[str] = set()

    def _add_candidate(path: str):
        path = os.path.abspath(path or "")
        if not path or path in seen:
            return
        seen.add(path)
        candidates.append(path)

    primary_error: Optional[Exception] = None
    try:
        _add_candidate(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath) or "")
    except Exception as exc:
        primary_error = exc

    try:
        import PyQt6  # local fallback resolution

        _add_candidate(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins"))
    except Exception:
        pass

    for existing in _split_env_paths(os.environ.get("QT_PLUGIN_PATH", "")):
        _add_candidate(existing)

    valid_roots = [root for root in candidates if _has_platform_plugin(root)]
    chosen_root = valid_roots[0] if valid_roots else ""
    if not chosen_root:
        if primary_error is not None:
            print(
                f"[Qt bootstrap] Warning: unable to resolve Qt plugins path: {primary_error}",
                file=sys.stderr,
            )
        elif candidates:
            print(
                "[Qt bootstrap] Warning: no candidate Qt plugin root contains a valid platform plugin.",
                file=sys.stderr,
            )
        return

    platform_dir = os.path.join(chosen_root, "platforms")

    prev_plugin = os.environ.get("QT_PLUGIN_PATH", "")
    if prev_plugin and os.path.abspath(prev_plugin) != chosen_root:
        print(
            f"[Qt bootstrap] Replacing stale QT_PLUGIN_PATH:\n  old={prev_plugin}\n  new={chosen_root}",
            file=sys.stderr,
        )
    os.environ["QT_PLUGIN_PATH"] = chosen_root

    prev_platform = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if prev_platform and os.path.abspath(prev_platform) != platform_dir:
        print(
            f"[Qt bootstrap] Replacing stale QT_QPA_PLATFORM_PLUGIN_PATH:\n  old={prev_platform}\n  new={platform_dir}",
            file=sys.stderr,
        )
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_dir


# --- cross-platform UI font helper ---
def _ui_font(px: int) -> QFont:
    f = QFont()
    available = set(QFontDatabase.families())
    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    ordered = ["Fira Sans", system_family, "Segoe UI", "Arial", "Helvetica"]
    ignored = {"Sans Serif", "SansSerif", "sans"}
    seen = set()
    for family in ordered:
        if not family or family in ignored or family in seen:
            continue
        seen.add(family)
        if family in available:
            f.setFamily(family)
            if QFontInfo(f).family() == family:
                break
    f.setPixelSize(px)
    return f


# CV2

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None

try:
    import numpy as _np
except Exception:
    _np = None

# Preferred device order: CUDA → MPS → CPU
try:
    import torch as _torch
except Exception:
    _torch = None


def _auto_device() -> str:
    try:
        if _torch is not None:
            if hasattr(_torch, "cuda") and _torch.cuda.is_available():
                return "cuda"
            # On macOS, MPS can be present but not fully usable; check both built and available
            if hasattr(_torch, "backends") and hasattr(_torch.backends, "mps"):
                mps = _torch.backends.mps
                if (
                    getattr(mps, "is_built", lambda: False)()
                    and getattr(mps, "is_available", lambda: False)()
                ):
                    return "mps"
        return "cpu"
    except Exception:
        return "cpu"


# =========================
# Graphics Items
# =========================


class CongratsPopup(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎉 SqueakPose Studio")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        emoji = QLabel("🐭🧀🎉")
        emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emoji.setStyleSheet("font-size: 48px;")

        message = QLabel("All images have been labeled!\nAmazing work, Squeaker!")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet("font-size: 18px; padding: 10px;")

        ok_btn = QPushButton("Let's Go!")
        ok_btn.setStyleSheet("padding: 8px 16px; font-size: 14px;")
        ok_btn.clicked.connect(self.accept)

        layout.addWidget(emoji)
        layout.addWidget(message)
        layout.addWidget(ok_btn)
        self.setLayout(layout)
        self.setFixedSize(350, 300)


# =========================
# View
# =========================


# =========================
# Video Review Pan/Zoom View
# =========================

# =========================
# Main Application
# =========================


class LabelingApp(QMainWindow):
    @property
    def seg_edit_state(self) -> SegmentationEditState:
        """Authoritative segmentation state, created lazily for legacy test seams."""
        state = self.__dict__.get("_seg_edit_state")
        if state is None:
            state = SegmentationEditState()
            self.__dict__["_seg_edit_state"] = state
        return state

    @seg_edit_state.setter
    def seg_edit_state(self, state: SegmentationEditState) -> None:
        self.__dict__["_seg_edit_state"] = state
        controller = self.__dict__.get("_segmentation_controller")
        if controller is not None:
            controller.state = state

    @property
    def active_workflow(self) -> str:
        """Compatibility view of the active layer for existing workers/dialogs."""

        return layer_worker_mode(getattr(self, "active_layer", LAYER_KEYPOINTS))

    @active_workflow.setter
    def active_workflow(self, value: str) -> None:
        layer_id = normalize_layer_id(value)
        session = getattr(self, "_project_session", None)
        if session is not None:
            session.transition_workflow(layer_id)
        self.active_layer = layer_id

    @property
    def seg_prompt_points(self) -> list[tuple[float, float, int]]:
        """Compatibility view backed by the segmentation edit state."""
        return self.seg_edit_state.prompt_points

    @seg_prompt_points.setter
    def seg_prompt_points(self, points: list[tuple[float, float, int]]) -> None:
        self.seg_edit_state.prompt_points = [
            (float(x), float(y), int(label)) for x, y, label in points
        ]

    @property
    def seg_prompt_labels(self) -> list[int]:
        return [int(label) for _, _, label in self.seg_edit_state.prompt_points]

    @property
    def seg_preview_points(self) -> list[tuple[float, float]]:
        """Compatibility view backed by the segmentation edit state."""
        return self.seg_edit_state.preview_points

    @seg_preview_points.setter
    def seg_preview_points(self, points: list[tuple[float, float]]) -> None:
        self.seg_edit_state.set_preview(points, self.seg_edit_state.preview_score)

    @property
    def seg_preview_score(self) -> float:
        return self.seg_edit_state.preview_score

    @seg_preview_score.setter
    def seg_preview_score(self, score: float) -> None:
        self.seg_edit_state.preview_score = float(score)

    def _active_layer_definition(self):
        return layer_definition(getattr(self, "active_layer", LAYER_KEYPOINTS))

    def _is_keypoints_layer(self) -> bool:
        return getattr(self, "active_layer", LAYER_KEYPOINTS) == LAYER_KEYPOINTS

    def _is_segmentation_layer(self) -> bool:
        return getattr(self, "active_layer", LAYER_KEYPOINTS) == LAYER_SEGMENTATION

    def _is_depth_layer(self) -> bool:
        state = getattr(self, "__dict__", {})
        return state.get("active_layer", LAYER_KEYPOINTS) == LAYER_DEPTH

    # Compatibility helpers for code paths and integrations using the former
    # workflow terminology.
    def _is_pose_workflow(self) -> bool:
        return self._is_keypoints_layer()

    def _is_seg_workflow(self) -> bool:
        return self._is_segmentation_layer()

    def _is_depth_workflow(self) -> bool:
        return self._is_depth_layer()

    def _workflow_label(self) -> str:
        return self._active_layer_definition().display_name

    def _depth_view_mode(self) -> str:
        combo = getattr(self, "depth_display_combo", None)
        if combo is not None:
            value = str(combo.currentData() or "").strip().lower()
            if value in {"original", "depth", "overlay"}:
                return value
        settings = getattr(self, "layer_settings", {}).get(LAYER_DEPTH, {})
        value = str(settings.get("display_mode") or "depth").strip().lower()
        return value if value in {"original", "depth", "overlay"} else "depth"

    def _on_depth_view_changed(self, _index: int) -> None:
        mode = self._depth_view_mode()
        controller = getattr(self, "_depth_controller", None)
        if controller is not None:
            mode = controller.set_view_mode(mode)
        self.layer_settings = normalize_layer_settings(getattr(self, "layer_settings", {}))
        self.layer_settings[LAYER_DEPTH]["display_mode"] = mode
        self._save_project_preferences()
        if self._is_depth_layer() and getattr(self, "images", None):
            self.load_image()

    def _update_depth_range_label(self, image_stem: str) -> None:
        label = getattr(self, "depth_range_label", None)
        if label is None:
            return
        metadata_path = os.path.join(self.depth_image_dir, f"{image_stem}_depth.json")
        if not os.path.isfile(metadata_path):
            controller = getattr(self, "_depth_controller", None)
            if controller is not None:
                controller.state.set_metadata(None)
                label.setText(controller.state.range_text())
            else:
                label.setText("No saved depth range · Near = bright")
            return
        try:
            metadata = read_json_file(
                metadata_path,
                max_bytes=1024 * 1024,
                require_object=True,
            )
            low = float(metadata["p02_depth"])
            high = float(metadata["p98_depth"])
            median = float(metadata["median_depth"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            controller = getattr(self, "_depth_controller", None)
            if controller is not None:
                controller.state.set_metadata({})
                label.setText(controller.state.range_text())
            else:
                label.setText("Depth range unavailable · Near = bright")
            return
        controller = getattr(self, "_depth_controller", None)
        if controller is not None:
            controller.state.set_metadata(metadata)
            label.setText(controller.state.range_text())
        else:
            label.setText(
                f"Range (2–98%): {low:.3f}–{high:.3f} m · median {median:.3f} m · Near = bright"
            )

    def _refresh_depth_probe_label(self) -> None:
        presenter = self.__dict__.get("_depth_preview_presenter")
        controller = getattr(self, "_depth_controller", None)
        if presenter is not None and controller is not None:
            self._depth_probes = [probe.as_mapping() for probe in controller.state.probes]
            presenter.present_state(controller.state)
            return
        label = getattr(self, "depth_probe_label", None)
        if label is None:
            return
        if controller is not None:
            probes = [probe.as_mapping() for probe in controller.state.probes]
            self._depth_probes = probes
            label.setText(controller.state.probe_text())
        else:
            probes = list(getattr(self, "_depth_probes", []))
            error = str(getattr(self, "_depth_probe_error", "") or "")
            label.setText(error if error else "Right-click the image to sample raw depth.")
        button = getattr(self, "depth_clear_probes_btn", None)
        if button is not None:
            button.setEnabled(bool(probes))

    def _clear_depth_probe_items(self) -> None:
        presenter = self.__dict__.get("_depth_preview_presenter")
        if presenter is not None:
            presenter.clear_probe_markers()

    def _clear_depth_probes(self, _checked: bool = False) -> None:
        self._clear_depth_probe_items()
        controller = getattr(self, "_depth_controller", None)
        if controller is not None:
            controller.clear_probes()
        self._depth_probes = []
        self._refresh_depth_probe_label()

    def _render_depth_probes(self) -> None:
        presenter = self.__dict__.get("_depth_preview_presenter")
        if presenter is None:
            return
        presenter.present_probe_markers(
            self._depth_probes,
            active_depth_layer=self._is_depth_layer(),
        )

    def _probe_depth_at(self, scene_pos: QPointF) -> bool:
        if not self._is_depth_layer():
            return False
        controller = getattr(self, "_depth_controller", None)
        if controller is None or controller.depth_map is None or _np is None:
            self.update_status_bar("No aligned raw depth map is available for pixel sampling.")
            return True
        attempt = controller.probe(float(scene_pos.x()), float(scene_pos.y()))
        if not attempt.accepted or attempt.probe is None:
            self.update_status_bar(attempt.error)
            return True
        probe = attempt.probe.as_mapping()
        self._depth_probes = [item.as_mapping() for item in controller.state.probes]
        self._render_depth_probes()
        self._refresh_depth_probe_label()
        if probe["valid"]:
            self.update_status_bar(
                f"Depth at ({probe['x']}, {probe['y']}): {float(probe['depth']):.3f} m"
            )
        else:
            self.update_status_bar(f"Depth at ({probe['x']}, {probe['y']}): invalid")
        return True

    def _preserve_invalid_schema_file(self, path: str, error: Exception) -> str:
        """Move an unreadable schema aside before creating safe defaults."""
        if not os.path.exists(path):
            raise RuntimeError(f"Project schema file disappeared while reading: {path}") from error

        stem, extension = os.path.splitext(path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = f"{stem}.corrupt-{timestamp}{extension}"
        suffix = 1
        while os.path.exists(backup_path):
            backup_path = f"{stem}.corrupt-{timestamp}-{suffix}{extension}"
            suffix += 1
        try:
            os.replace(path, backup_path)
        except OSError as backup_error:
            raise RuntimeError(
                f"Could not read or preserve project schema file '{path}': {error}"
            ) from backup_error

        recoveries = getattr(self, "_schema_recoveries", None)
        if recoveries is None:
            recoveries = []
            self._schema_recoveries = recoveries
        recoveries.append((path, backup_path, str(error)))
        return backup_path

    def _read_schema_lines(self, path: str) -> list[str]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
        except (OSError, UnicodeError) as error:
            self._preserve_invalid_schema_file(path, error)
            return []

    def _show_schema_recoveries(self) -> None:
        recoveries = list(getattr(self, "_schema_recoveries", []) or [])
        if not recoveries:
            return
        self._schema_recoveries = []
        details = []
        for original, backup, error in recoveries:
            details.append(f"{original}\nPreserved at: {backup}\n{error}")
        QMessageBox.warning(
            self,
            "Project Schema Recovered",
            "One or more project schema files could not be read. The originals "
            "were preserved and replacement defaults were created. Review the "
            "recovered schema before editing existing labels.\n\n" + "\n\n".join(details),
        )

    def _ensure_classes_file(self, class_file: str, defaults: list[str]) -> tuple[list[str], bool]:
        created_any = False
        project_root = os.path.dirname(class_file) if class_file else os.getcwd()
        if not class_file:
            class_file = os.path.join(project_root, "classes.txt")

        cf_dir = os.path.dirname(class_file)
        if cf_dir:
            os.makedirs(cf_dir, exist_ok=True)

        classes = self._read_schema_lines(class_file)

        if not classes:
            classes = defaults[:] or DEFAULT_CLASS_NAMES[:]
            atomic_write_text(class_file, "".join(f"{name}\n" for name in classes))
            created_any = True

        return classes, created_any

    def _project_meta_path(self) -> str:
        return ProjectMetadataStore(self.project_root).path

    def _read_project_meta(self) -> dict:
        try:
            result = ProjectMetadataStore(self.project_root).read()
        except OSError:
            logger.warning(
                "Could not read project metadata",
                exc_info=True,
                extra={
                    "event": "metadata_read_failed",
                    "operation": "read_metadata",
                    "project_root": getattr(self, "project_root", ""),
                    "source_path": self._project_meta_path(),
                },
            )
            return {}
        if result.recovery_error:
            self._project_meta_recovery = (
                result.recovery_path,
                result.recovery_error,
            )
        return result.data

    def _write_project_meta(self, updates: dict):
        if not isinstance(updates, dict):
            return
        try:
            result = ProjectMetadataStore(self.project_root).update(updates)
        except OSError:
            logger.warning(
                "Could not update project metadata",
                exc_info=True,
                extra={
                    "event": "metadata_update_failed",
                    "operation": "update_metadata",
                    "project_root": self.project_root,
                    "target_path": self._project_meta_path(),
                },
            )
        else:
            if result.recovery_error:
                self._project_meta_recovery = (
                    result.recovery_path,
                    result.recovery_error,
                )

    def _show_project_meta_recovery(self):
        recovery = getattr(self, "_project_meta_recovery", None)
        if not recovery:
            return
        backup_path, detail = recovery
        self._project_meta_recovery = None
        if backup_path:
            message = (
                "The project metadata file was invalid and has been replaced with defaults.\n\n"
                f"The original file was preserved at:\n{backup_path}\n\n{detail}"
            )
        else:
            message = (
                f"The project metadata file is invalid and could not be backed up.\n\n{detail}"
            )
        QMessageBox.warning(self, "Project Metadata Recovered", message)

    def _meta_normalize_path(self, path: str) -> str:
        return ProjectMetadataStore(self.project_root).resolve_path(path)

    def _meta_store_path(self, path: str) -> str:
        return ProjectMetadataStore(self.project_root).store_path(path)

    @staticmethod
    def _is_builtin_model_reference(path: str) -> bool:
        return str(path or "").lower() in {f"yolo26{size}-depth.pt" for size in "nslmx"}

    def _resolve_model_reference(self, path: str) -> str:
        if self._is_builtin_model_reference(path):
            return str(path)
        return self._meta_normalize_path(path)

    def _store_model_reference(self, path: str) -> str:
        if self._is_builtin_model_reference(path):
            return str(path)
        return self._meta_store_path(path)

    def _load_project_preferences(self):
        meta = self._read_project_meta()
        project_paths = replace(
            ProjectPaths.from_root(self.project_root),
            images_to_label=self.image_dir_queue,
            images_all=self.image_dir_all,
            labels_all=self.pose_label_dir,
            labels_seg_all=self.seg_label_dir,
            depth_images=self.depth_image_dir,
            classes_file=self.pose_class_file,
            keypoints_file=self.pose_keypoint_file,
            class_keypoints_file=self.pose_class_keypoints_path,
            classes_seg_file=self.seg_class_file,
        )
        session = ProjectSession.from_preferences(
            self.project_root,
            meta,
            paths=project_paths,
            pose_classes=self.pose_classes,
            pose_keypoints=self.pose_kp_names,
            pose_class_keypoints=self.pose_class_keypoints,
            segmentation_classes=self.seg_classes,
        )
        if session.assistant_model_path:
            self.sam_model_path = session.assistant_model_path
        else:
            session.assistant_model_path = str(getattr(self, "sam_model_path", "") or "")
        self._project_session = session
        self._sync_project_session_legacy_fields()

    def _save_project_preferences(self):
        session = getattr(self, "_project_session", None)
        if session is None:
            return
        self._persist_active_layer_state()
        session.layer_settings = normalize_layer_settings(getattr(self, "layer_settings", {}))
        for layer_id in LAYER_DEFINITIONS:
            session.set_model_path(
                layer_id,
                str(getattr(self, "layer_model_paths", {}).get(layer_id) or ""),
            )
            session.set_layer_visibility(
                layer_id,
                bool(getattr(self, "layer_visibility", {}).get(layer_id, True)),
            )
        session.assistant_model_path = str(getattr(self, "sam_model_path", "") or "")
        payload = session.to_preferences()
        self.layer_settings = normalize_layer_settings(session.layer_settings)
        self.layer_visibility = dict(session.layer_visibility)
        self._write_project_meta(payload)

    def _sync_project_session_legacy_fields(self) -> None:
        session = getattr(self, "_project_session", None)
        if session is None:
            return
        snapshot = session.snapshot()
        self.active_layer = snapshot.active_layer
        self.layer_settings = normalize_layer_settings(session.layer_settings)
        self.layer_visibility = dict(snapshot.layer_visibility)
        self.layer_model_paths = {layer.layer_id: layer.model_path for layer in snapshot.layers}
        pose = snapshot.layer(LAYER_KEYPOINTS)
        self.pose_classes = list(pose.classes)
        self.pose_kp_names = list(pose.keypoints)
        self.pose_class_keypoints = pose.class_keypoint_mapping()
        segmentation = snapshot.layer(LAYER_SEGMENTATION)
        self.seg_classes = list(segmentation.classes)

    def _safe_remove_scene_item(self, item: Optional[QGraphicsItem]):
        if item is None:
            return
        try:
            owner_scene = item.scene()
        except Exception:
            owner_scene = None
        if owner_scene is None:
            return
        try:
            owner_scene.removeItem(item)
        except Exception:
            pass

    def _persist_active_layer_state(self):
        session = getattr(self, "_project_session", None)
        if session is not None:
            selected = getattr(self, "_active_class_id", -1)
            snapshot = session.capture_active_state(
                classes=getattr(self, "classes", ()),
                keypoints=getattr(self, "kp_names", ()),
                class_keypoints=getattr(self, "class_keypoints", {}),
                selected_class_id=selected,
                model_path=str(getattr(self, "predict_model_path", "") or ""),
            )
            self.layer_model_paths[snapshot.layer_id] = snapshot.model_path
            if snapshot.layer_id == LAYER_KEYPOINTS:
                self.pose_classes = list(snapshot.classes)
                self.pose_kp_names = list(snapshot.keypoints)
                self.pose_class_keypoints = snapshot.class_keypoint_mapping()
            elif snapshot.layer_id == LAYER_SEGMENTATION:
                self.seg_classes = list(snapshot.classes)
            return
        if self._is_keypoints_layer():
            self.pose_classes = self.classes[:]
            self.pose_kp_names = self.kp_names[:]
            self.pose_class_keypoints = {
                name: self.class_keypoints.get(name, [])[:] for name in self.classes
            }
        elif self._is_segmentation_layer():
            self.seg_classes = self.classes[:]
        if hasattr(self, "layer_model_paths"):
            self.layer_model_paths[self.active_layer] = str(
                getattr(self, "predict_model_path", "") or ""
            )

    def _bind_layer_state(self, layer_id: str):
        layer_id = normalize_layer_id(layer_id)
        session = getattr(self, "_project_session", None)
        if session is not None:
            session.transition_to(layer_id)
            state = session.active_state
            active_paths = session.active_paths
            self.active_layer = state.layer_id
            self.label_dir = active_paths.label_dir
            self.class_file = active_paths.class_file
            self.keypoint_file = active_paths.keypoint_file
            self.class_keypoints_path = active_paths.class_keypoints_file
            self.classes = list(state.classes)
            self.kp_names = list(state.keypoints)
            self.class_keypoints = state.class_keypoint_mapping()
            self._active_class_id = state.selected_class_id
            self._schema_locked = True if layer_id == LAYER_DEPTH else self._detect_schema_locked()
            self.predict_model_path = state.model_path or None
            self.layer_visibility = dict(session.layer_visibility)
            self._refresh_kp_index_lookup()
            return
        if layer_id == LAYER_KEYPOINTS:
            self.active_layer = LAYER_KEYPOINTS
            self.label_dir = self.pose_label_dir
            self.class_file = self.pose_class_file
            self.keypoint_file = self.pose_keypoint_file
            self.class_keypoints_path = self.pose_class_keypoints_path
            self.classes = self.pose_classes[:]
            self.kp_names = self.pose_kp_names[:]
            self.class_keypoints = {
                name: self.pose_class_keypoints.get(name, [])[:] for name in self.classes
            }
            if self._sync_canonical_keypoints_from_class_map():
                self.pose_kp_names = self.kp_names[:]
            self._schema_locked = self._detect_schema_locked()
        elif layer_id == LAYER_SEGMENTATION:
            self.active_layer = LAYER_SEGMENTATION
            self.label_dir = self.seg_label_dir
            self.class_file = self.seg_class_file
            self.keypoint_file = ""
            self.class_keypoints_path = ""
            self.classes = self.seg_classes[:]
            self.kp_names = []
            self.class_keypoints = {}
            self._schema_locked = self._detect_schema_locked()
        else:
            self.active_layer = LAYER_DEPTH
            self.label_dir = self.depth_image_dir
            self.class_file = ""
            self.keypoint_file = ""
            self.class_keypoints_path = ""
            self.classes = []
            self.kp_names = []
            self.class_keypoints = {}
            self._schema_locked = True

        self.predict_model_path = getattr(self, "layer_model_paths", {}).get(layer_id) or None
        self._refresh_kp_index_lookup()

    def _bind_workflow_state(self, workflow: str):
        self._bind_layer_state(workflow)

    def _refresh_class_selector_for_workflow(self):
        current = self.class_selector.currentText() if hasattr(self, "class_selector") else ""
        session = getattr(self, "_project_session", None)
        selected_id = session.active_state.selected_class_id if session is not None else -1
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.addItems(self.classes)
        self._fit_class_selector_to_items()
        if 0 <= selected_id < len(self.classes):
            self.class_selector.setCurrentIndex(selected_id)
        elif current in self.classes:
            self.class_selector.setCurrentIndex(self.classes.index(current))
        elif self.classes:
            self.class_selector.setCurrentIndex(0)
        self.class_selector.blockSignals(False)
        self._active_class_id = self.class_selector.currentIndex()
        if session is not None:
            session.select_class(self._active_class_id)
        if hasattr(self, "seg_edit_state"):
            selected = self._active_class_id
            self.seg_edit_state.select_target(
                selected if self._is_seg_workflow() and selected >= 0 else None
            )

    def _fit_class_selector_to_items(self):
        if not hasattr(self, "class_selector"):
            return
        longest = 14
        for class_name in self.classes:
            longest = max(longest, min(len(str(class_name)), 36))
        self.class_selector.setMinimumContentsLength(longest)

    def _ensure_layer_selector_items(self):
        if not hasattr(self, "workflow_selector"):
            return
        expected = [
            ("Keypoints Layer", LAYER_KEYPOINTS),
            ("Segmentation Layer", LAYER_SEGMENTATION),
            ("Depth Layer", LAYER_DEPTH),
        ]
        needs_reset = self.workflow_selector.count() != len(expected)
        if not needs_reset:
            for idx, (text, data) in enumerate(expected):
                if (
                    self.workflow_selector.itemText(idx) != text
                    or str(self.workflow_selector.itemData(idx)) != data
                ):
                    needs_reset = True
                    break
        if not needs_reset:
            return

        current_layer = getattr(self, "active_layer", LAYER_KEYPOINTS)
        self.workflow_selector.blockSignals(True)
        self.workflow_selector.clear()
        for text, data in expected:
            self.workflow_selector.addItem(text, data)
        layer_index = [item[1] for item in expected].index(current_layer)
        self.workflow_selector.setCurrentIndex(layer_index)
        self.workflow_selector.blockSignals(False)

    def _ensure_workflow_selector_items(self):
        self._ensure_layer_selector_items()

    def _segmentation_labels_exist(self) -> bool:
        labels_dir = getattr(self, "seg_label_dir", "")
        if not labels_dir or not os.path.isdir(labels_dir):
            return False
        try:
            for name in os.listdir(labels_dir):
                if not name.lower().endswith(".txt"):
                    continue
                path = os.path.join(labels_dir, name)
                try:
                    if os.path.getsize(path) > 0:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _maybe_prompt_seg_class_manager_initial(self):
        if getattr(self, "_seg_setup_prompted", False):
            return
        if not self._is_seg_workflow():
            return
        self._seg_setup_prompted = True

        has_real_seg_labels = self._segmentation_labels_exist()
        default_classes = self.seg_classes == DEFAULT_CLASS_NAMES
        if has_real_seg_labels or not default_classes:
            return

        decision = QMessageBox.question(
            self,
            "Define Segmentation Classes",
            "Before labeling in the Segmentation layer, define what objects/classes you want to segment.\n\n"
            "Open Segmentation Classes now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if decision == QMessageBox.StandardButton.Yes:
            self._open_seg_class_manager()
        else:
            self.update_status_bar(
                "Using default segmentation class ('mouse'). Edit via Seg Classes… anytime."
            )

    def _reflow_mode_grid(self, is_pose: bool):
        if not hasattr(self, "mode_grid"):
            return
        panel = getattr(self, "__dict__", {}).get("annotation_panel")
        if panel is not None:
            panel.set_layer(getattr(self, "active_layer", LAYER_KEYPOINTS))
            return
        self.panzoom_btn.setText("Pan/Zoom (1)" if is_pose else "Pan (1)")
        self.segment_btn.setText("Segment (2)" if is_pose else "Segment Prompt (2)")

        if is_pose:
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
            # Hidden in seg workflow, but keep deterministic positions.
            self.mode_grid.addWidget(self.bbox_btn, 2, 0)
            self.mode_grid.addWidget(self.keypoint_btn, 2, 1)

    def _sync_layer_visibility_controls(self) -> None:
        instance_state = getattr(self, "__dict__", {})
        checks = {
            LAYER_KEYPOINTS: getattr(self, "keypoints_visibility_check", None),
            LAYER_SEGMENTATION: getattr(self, "segmentation_visibility_check", None),
            LAYER_DEPTH: getattr(self, "depth_visibility_check", None),
        }
        active_layer = getattr(self, "active_layer", LAYER_KEYPOINTS)
        session = getattr(self, "_project_session", None)
        if session is not None:
            session.set_layer_visibility(active_layer, True)
            self.layer_visibility = dict(session.layer_visibility)
        else:
            self.layer_visibility[active_layer] = True
        navigation_panel = instance_state.get("navigation_panel")
        if navigation_panel is not None:
            navigation_panel.set_active_layer(active_layer, emit=False)
            navigation_panel.set_visibility(self.layer_visibility, emit=False)
        for layer_id, check in checks.items():
            if check is None:
                continue
            is_active = layer_id == active_layer
            is_visible = bool(self.layer_visibility.get(layer_id, True))
            check.blockSignals(True)
            check.setChecked(is_visible)
            check.setEnabled(not is_active)
            check.setProperty("activeLayer", is_active)
            check.setText(
                f"{layer_definition(layer_id).display_name} · "
                f"{'View' if layer_id == LAYER_DEPTH else 'Edit'}"
                if is_active
                else (
                    f"● {layer_definition(layer_id).display_name}"
                    if is_visible
                    else f"○ {layer_definition(layer_id).display_name}"
                )
            )
            check.setToolTip(
                "The layer being edited is always visible."
                if is_active
                else f"Show the saved {layer_definition(layer_id).display_name} layer as a read-only reference."
            )
            _refresh_qt_style(check)
            check.blockSignals(False)
        self._refresh_layer_context_hud()

    def _refresh_layer_context_hud(self) -> None:
        if not hasattr(self, "layer_editing_label"):
            return
        active_layer = getattr(self, "active_layer", LAYER_KEYPOINTS)
        active_name = layer_definition(active_layer).display_name
        visible_references = [
            layer_definition(layer_id).display_name
            for layer_id in LAYER_DEFINITIONS
            if layer_id != active_layer and self.layer_visibility.get(layer_id, True)
        ]
        editing_text = (
            f"{active_name.upper()} · {'VIEW' if active_layer == LAYER_DEPTH else 'EDITING'}"
        )
        reference_text = (
            "● " + " + ".join(visible_references) + " references visible"
            if visible_references
            else "○ Reference layers hidden"
        )
        presenter = getattr(self, "__dict__", {}).get("canvas_hud_presenter")
        if presenter is not None:
            presenter.set_context(
                editing=editing_text,
                references=reference_text,
            )
        else:
            self.layer_editing_label.setText(editing_text)
            self.layer_reference_label.setText(reference_text)
            self.layer_context_frame.adjustSize()
        self._layout_hot_corners()

    def _on_layer_visibility_changed(self, layer_id: str, visible: bool) -> None:
        layer_id = normalize_layer_id(layer_id)
        session = getattr(self, "_project_session", None)
        if layer_id == getattr(self, "active_layer", LAYER_KEYPOINTS):
            if session is not None:
                session.set_layer_visibility(layer_id, True)
                self.layer_visibility = dict(session.layer_visibility)
            else:
                self.layer_visibility[layer_id] = True
            self._sync_layer_visibility_controls()
            return
        if session is not None:
            session.set_layer_visibility(layer_id, visible)
            self.layer_visibility = dict(session.layer_visibility)
        else:
            self.layer_visibility[layer_id] = bool(visible)
        self._save_project_preferences()
        self._sync_layer_visibility_controls()
        if hasattr(self, "scene"):
            self._refresh_reference_layer_overlay()

    def _update_layer_ui_state(self):
        is_pose = self._is_keypoints_layer()
        is_segmentation = self._is_segmentation_layer()
        is_depth = self._is_depth_layer()
        instance_state = getattr(self, "__dict__", {})
        navigation_panel = instance_state.get("navigation_panel")
        if navigation_panel is not None:
            navigation_panel.set_active_layer(self.active_layer, emit=False)
        annotation_panel = instance_state.get("annotation_panel")
        if annotation_panel is not None:
            annotation_panel.set_layer(self.active_layer)
        analysis_panel = instance_state.get("analysis_frame")
        if isinstance(analysis_panel, AnalysisOperationsPanel):
            analysis_panel.set_layer(self.active_layer)
        dataset_panel = instance_state.get("bottom_left_frame")
        if isinstance(dataset_panel, DatasetOperationsPanel):
            dataset_panel.set_layer(self.active_layer)
        model_panel = instance_state.get("bottom_right_frame")
        if isinstance(model_panel, ModelOperationsPanel):
            model_panel.set_layer(self.active_layer)
        self._ensure_layer_selector_items()

        self.save_btn.setEnabled(not is_depth)
        self.complete_btn.setEnabled(not is_depth)
        self.bbox_btn.setEnabled(is_pose)
        self.segment_btn.setEnabled(is_segmentation)
        self.keypoint_btn.setEnabled(is_pose)
        self.predict_btn.setEnabled(True)
        self.seg_edit_btn.setEnabled(is_segmentation)
        self.sam_load_btn.setEnabled(is_segmentation)
        self.sam_run_btn.setEnabled(is_segmentation)
        self.sam_accept_btn.setEnabled(is_segmentation)
        self.sam_clear_btn.setEnabled(is_segmentation)
        self.template_apply_btn.setEnabled(is_pose)
        self.template_save_btn.setEnabled(is_pose)
        self.inference_btn.setEnabled(True)
        self.normalize_btn.setEnabled(not is_depth)
        self.export_dataset_btn.setEnabled(not is_depth)
        self.train_btn.setEnabled(not is_depth)
        self.distillation_btn.setEnabled(is_pose)
        self.normalize_btn.setVisible(not is_depth)
        self.export_dataset_btn.setVisible(not is_depth)
        self.train_btn.setVisible(not is_depth)
        self.distillation_btn.setVisible(not is_depth)
        if hasattr(self, "training_grid"):
            self.training_grid.addWidget(
                self.project_health_btn,
                0 if is_depth else 1,
                0,
                1,
                2 if is_depth else 1,
            )
        if hasattr(self, "analysis_btn"):
            self.analysis_btn.setEnabled(not is_depth)
        if hasattr(self, "analysis_frame"):
            self.analysis_frame.setVisible(not is_depth)
        if hasattr(self, "delete_image_btn"):
            self.delete_image_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        active_layer = self._active_layer_definition()
        self.load_model_btn.setText("Project Models…")
        self.load_model_btn.setToolTip(
            "Configure the Keypoints and Segmentation prediction models for this project"
        )
        self.inference_btn.setText("Run Inference")
        self.train_btn.setText("Train Model")
        if hasattr(self, "model_inference_title"):
            self.model_inference_title.setText(
                "Project Inference" if is_depth else "Project Models & Inference"
            )
        if hasattr(self, "model_status_label"):
            configured = []
            for layer_id in (LAYER_KEYPOINTS, LAYER_SEGMENTATION):
                path = self.layer_model_paths.get(layer_id) or ""
                state = os.path.basename(path) if path else "not configured"
                configured.append(f"{layer_definition(layer_id).display_name}: {state}")
            self.model_status_label.setText("  ·  ".join(configured))
            self.model_status_label.setToolTip("\n".join(configured))
            self.model_status_label.setVisible(not is_depth)
        self.load_model_btn.setVisible(not is_depth)
        self.template_apply_btn.setVisible(is_pose)
        self.template_save_btn.setVisible(is_pose)
        if hasattr(self, "inference_grid"):
            self.inference_grid.addWidget(
                self.inference_btn,
                0,
                0 if is_depth else 1,
                1,
                2 if is_depth else 1,
            )
        if hasattr(self, "dataset_training_title"):
            self.dataset_training_title.setText(
                "Project Tools" if is_depth else f"{active_layer.display_name} Dataset & Training"
            )
        if hasattr(self, "analysis_title"):
            self.analysis_title.setText(f"{active_layer.display_name} Analysis")
        if hasattr(self, "analysis_btn"):
            self.analysis_btn.setText("Run Analysis")

        self.manage_classes_btn.setEnabled(not is_depth)
        self.manage_classes_btn.setToolTip(
            "Depth maps do not use classes"
            if is_depth
            else (
                "Manage classes and per-class keypoints"
                if is_pose
                else "Manage segmentation classes"
            )
        )
        self.manage_classes_btn.setText("Classes…")
        if hasattr(self, "class_controls_frame"):
            self.class_controls_frame.setVisible(not is_depth)
        if hasattr(self, "class_label_widget"):
            self.class_label_widget.setText("Class")
        if hasattr(self, "bbox_btn"):
            self.bbox_btn.setVisible(is_pose)
        if hasattr(self, "keypoint_btn"):
            self.keypoint_btn.setVisible(is_pose)
        if hasattr(self, "predict_btn"):
            self.predict_btn.setVisible(True)
            self.predict_btn.setToolTip(
                "Run the Keypoints layer model on the current image"
                if is_pose
                else (
                    "Run the Segmentation layer model on the current image"
                    if is_segmentation
                    else "Estimate and save a dense depth map for the current image"
                )
            )
        if hasattr(self, "segment_btn"):
            self.segment_btn.setVisible(is_segmentation)
        if hasattr(self, "seg_edit_btn"):
            self.seg_edit_btn.setVisible(is_segmentation)
        if hasattr(self, "seg_tools_frame"):
            self.seg_tools_frame.setVisible(is_segmentation)
        if hasattr(self, "depth_display_frame"):
            self.depth_display_frame.setVisible(is_depth)
        if hasattr(self, "depth_range_frame"):
            self.depth_range_frame.setVisible(is_depth)
        if hasattr(self, "depth_assistant_frame"):
            self.depth_assistant_frame.setVisible(is_depth)
            self._refresh_depth_assistant_controls()
        self._reflow_mode_grid(is_pose=is_pose)
        if is_depth:
            self.mode_grid.addWidget(self.panzoom_btn, 0, 0)
            self.mode_grid.addWidget(self.predict_btn, 0, 1)
        self.save_btn.setText("Save")
        self.save_btn.setToolTip(
            "Save labels for current frame"
            if is_pose
            else "Save segmentation masks for current frame"
        )

        if is_pose and self.mode in {"segment", "segedit"}:
            self.mode = "panzoom"
        if is_segmentation and self.mode not in {"panzoom", "segment", "segedit"}:
            self.mode = "segment"
        if is_depth:
            self.mode = "panzoom"

        self._clear_seg_edit_handles()
        self._refresh_seg_brush_size_badge()
        if hasattr(self, "view") and hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()

        self._update_status()
        self._update_progress_label()
        self._refresh_sam_controls()
        self._sync_layer_visibility_controls()
        self._layout_hot_corners()
        self._layout_overlays()

    def _update_workflow_ui_state(self):
        self._update_layer_ui_state()

    def _switch_layer(self, layer_id: str):
        layer_id = normalize_layer_id(layer_id)
        previous_layer = getattr(self, "active_layer", LAYER_KEYPOINTS)
        if layer_id == previous_layer:
            return
        if previous_layer == LAYER_SEGMENTATION:
            self._stop_sam_assistant()
        if getattr(self, "_predict_busy", False):
            self._cancel_prediction_process()
            self._predict_busy = False
            self._prediction_pending_request = None
            self._prediction_current_request_id = None
            self._prediction_image_path = None
        self._ensure_layer_selector_items()
        if hasattr(self, "workflow_selector"):
            idx = list(LAYER_DEFINITIONS).index(layer_id)
            if self.workflow_selector.currentIndex() != idx:
                self.workflow_selector.blockSignals(True)
                self.workflow_selector.setCurrentIndex(idx)
                self.workflow_selector.blockSignals(False)
        self._persist_active_layer_state()
        self._bind_layer_state(layer_id)
        self._save_project_preferences()
        if layer_id == LAYER_SEGMENTATION:
            self.mode = "segment"
        elif self.mode == "segment":
            self.mode = "panzoom"
        self._refresh_class_selector_for_workflow()
        self.annotation_cache.clear()
        self.seg_edit_state.reset(
            selected_target=(
                self.class_selector.currentIndex() if layer_id == LAYER_SEGMENTATION else None
            )
        )
        self._clear_seg_prompt_state()
        self._update_layer_ui_state()
        self.load_image()
        if self._is_segmentation_layer():
            loading_now, loaded_path = self._try_autoload_sam_model_from_project_root()
            self._refresh_sam_controls()
            if loading_now:
                self.update_status_bar(
                    "Segmentation layer selected. Loading SAM model: "
                    f"{os.path.basename(loaded_path)}"
                )
            elif self._sam_model_ready:
                self.update_status_bar("Segmentation layer selected. SAM model ready.")
            else:
                self.update_status_bar(
                    "Segmentation layer selected. Use Segment mode and SAM prompts."
                )
            QTimer.singleShot(0, self._maybe_prompt_seg_class_manager_initial)
        elif self._is_keypoints_layer():
            self.update_status_bar("Keypoints layer selected.")
        else:
            self.update_status_bar("Depth layer selected. Predict to create a saved depth map.")
        if self.predict_model_path:
            self._restart_prediction_worker(warm=True)
        else:
            self._restart_prediction_worker(warm=False)

    def _switch_workflow(self, workflow: str):
        self._switch_layer(workflow)

    def _on_layer_changed(self, _index: int):
        self._ensure_layer_selector_items()
        layer_id = self.workflow_selector.currentData()
        self._switch_layer(str(layer_id))

    def _on_workflow_changed(self, index: int):
        self._on_layer_changed(index)

    def _ensure_label_files(
        self, class_file: str, keypoint_file: str
    ) -> tuple[list[str], list[str], bool]:
        """
        Ensure class and keypoint name files exist WITHOUT any UI prompts.
        - If either path is empty, place files next to the labels directory.
        - If files are missing/empty, backfill sensible defaults.
        - Returns (classes, kp_names, created_flag).
        """
        created_any = False

        # Resolve fallback locations if the provided paths are empty.
        # Do not rely on self.label_dir here because this helper is called
        # during early __init__ before workflow-bound paths are fully wired.
        project_root = (
            os.path.dirname(class_file or "")
            or os.path.dirname(keypoint_file or "")
            or getattr(self, "project_root", "")
            or os.getcwd()
        )
        if not class_file:
            class_file = os.path.join(project_root, "classes.txt")
        if not keypoint_file:
            keypoint_file = os.path.join(project_root, "keypoints.txt")

        cf_dir = os.path.dirname(class_file)
        kf_dir = os.path.dirname(keypoint_file)
        if cf_dir:
            os.makedirs(cf_dir, exist_ok=True)
        if kf_dir and kf_dir != cf_dir:
            os.makedirs(kf_dir, exist_ok=True)

        classes = self._read_schema_lines(class_file)
        kp_names = self._read_schema_lines(keypoint_file)

        # Backfill defaults so the app is always usable even if initial setup is skipped.
        if not classes:
            classes = DEFAULT_CLASS_NAMES[:]
            atomic_write_text(class_file, "".join(f"{name}\n" for name in classes))
            created_any = True
        if not kp_names:
            kp_names = DEFAULT_KEYPOINT_NAMES[:]
            atomic_write_text(keypoint_file, "".join(f"{name}\n" for name in kp_names))
            created_any = True

        return classes, kp_names, created_any

    def _load_class_keypoints(self) -> dict[str, list[str]]:
        data: dict[str, list[str]] = {}
        if os.path.exists(self.class_keypoints_path):
            try:
                raw = read_json_file(
                    self.class_keypoints_path,
                    max_bytes=4 * 1024 * 1024,
                    require_object=True,
                )
                for name, lst in raw.items():
                    if not isinstance(name, str) or not isinstance(lst, list):
                        raise ValueError("class_keypoints.json contains an invalid class entry")
                    cleaned = [str(item) for item in lst if isinstance(item, str)]
                    if cleaned:
                        data[name] = cleaned
            except (OSError, UnicodeError, ValueError, TypeError) as error:
                self._preserve_invalid_schema_file(self.class_keypoints_path, error)
                data = {}
        # ensure each known class has an entry
        known_classes = getattr(self, "classes", None)
        if not known_classes:
            known_classes = getattr(self, "pose_classes", [])
        default_kp = getattr(self, "kp_names", None)
        if default_kp is None:
            default_kp = getattr(self, "pose_kp_names", DEFAULT_KEYPOINT_NAMES[:])

        for name in known_classes:
            if name not in data or not data[name]:
                data[name] = default_kp[:]
        return data

    def _sync_canonical_keypoints_from_class_map(self) -> bool:
        """Ensure keypoints.txt covers every per-class keypoint name."""
        class_map = getattr(self, "class_keypoints", {}) or {}
        if not class_map:
            return False

        canonical: list[str] = []
        seen: set[str] = set()
        for raw in getattr(self, "kp_names", []) or []:
            name = str(raw).strip()
            if name and name not in seen:
                canonical.append(name)
                seen.add(name)

        classes = getattr(self, "classes", []) or list(class_map.keys())
        for class_name in classes:
            for raw in class_map.get(class_name, []) or []:
                name = str(raw).strip()
                if name and name not in seen:
                    canonical.append(name)
                    seen.add(name)

        if canonical == (getattr(self, "kp_names", []) or []):
            return False

        self.kp_names = canonical
        if hasattr(self, "_refresh_kp_index_lookup"):
            self._refresh_kp_index_lookup()
        keypoint_file = getattr(self, "keypoint_file", "") or getattr(
            self, "pose_keypoint_file", ""
        )
        if keypoint_file:
            self._write_list_file(keypoint_file, self.kp_names)
        return True

    def _save_class_keypoints(self):
        try:
            atomic_write_text(self.class_keypoints_path, json.dumps(self.class_keypoints, indent=2))
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Could not write class keypoints file:\n{e}")

    def _kp_names_for_class(self, class_name: str) -> list[str]:
        return self.class_keypoints.get(class_name, self.kp_names)

    def _kp_names_for_index(self, idx: int) -> list[str]:
        if idx < 0 or idx >= len(self.classes):
            return self.kp_names
        return self._kp_names_for_class(self.classes[idx])

    def _active_kp_names(self) -> list[str]:
        return self._kp_names_for_index(self.class_selector.currentIndex())

    def _update_class_keypoints(self, class_name: str, kp_list: list[str]):
        if not kp_list:
            return
        self.class_keypoints[class_name] = kp_list[:]
        self._save_class_keypoints()

    def _refresh_kp_index_lookup(self):
        self._kp_index_lookup = {name: idx for idx, name in enumerate(self.kp_names)}

    def _ensure_canonical_name(self, name: str) -> int:
        if name in self._kp_index_lookup:
            return self._kp_index_lookup[name]
        self.kp_names.append(name)
        self._refresh_kp_index_lookup()
        self._write_list_file(self.keypoint_file, self.kp_names)
        return self._kp_index_lookup[name]

    def _label_file_is_usable(self, label_file: str) -> bool:
        state = getattr(self, "__dict__", {})
        layer_id = normalize_layer_id(state.get("active_layer"))
        if layer_id == LAYER_DEPTH:
            map_path = os.path.splitext(label_file)[0] + ".npy"
            return os.path.isfile(map_path) and os.path.getsize(map_path) > 0
        mode = DATASET_SEGMENT if layer_id == LAYER_SEGMENTATION else DATASET_POSE
        return label_file_has_usable_rows(
            label_file,
            mode=mode,
            class_count=len(state.get("classes", []) or []),
            keypoint_count=len(state.get("kp_names", []) or []),
        )

    def _count_labeled_images(self, images: list[str], label_dir: str) -> tuple[int, int]:
        progress = queue_progress(
            images,
            label_dir,
            label_is_usable=self._label_file_is_usable,
        )
        return progress.labeled, progress.total

    def _count_labeled_frames(self) -> tuple[int, int]:
        return self._count_labeled_images(self.images_queue, self.label_dir)

    def _detect_schema_locked(self) -> bool:
        """Schema is considered locked once any non-empty label file exists."""
        if not self.label_dir or not os.path.isdir(self.label_dir):
            return False
        try:
            for name in os.listdir(self.label_dir):
                if not name.lower().endswith(".txt"):
                    continue
                path = os.path.join(self.label_dir, name)
                try:
                    if os.path.getsize(path) > 0:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _schema_is_locked(self) -> bool:
        if getattr(self, "_schema_locked", False):
            return True
        locked = self._detect_schema_locked()
        self._schema_locked = bool(locked)
        return self._schema_locked

    def _validate_locked_schema_changes(
        self,
        classes_clean: list[str],
        normalized_map: dict[str, list[str]],
    ) -> tuple[bool, str]:
        """When schema is locked, allow only additive edits."""
        existing_classes = self.classes[:]
        if len(classes_clean) < len(existing_classes):
            return False, "Cannot remove classes after labeled data exists."
        if classes_clean[: len(existing_classes)] != existing_classes:
            return (
                False,
                "Existing class names/order are locked.\nOnly append new classes at the end.",
            )
        for class_name in existing_classes:
            old_kp = self.class_keypoints.get(class_name, [])[:]
            new_kp = normalized_map.get(class_name, [])[:]
            if len(new_kp) < len(old_kp):
                return False, f"Cannot remove keypoints from class '{class_name}'."
            if new_kp[: len(old_kp)] != old_kp:
                return (
                    False,
                    f"Class '{class_name}' keypoints are locked.\n"
                    "Only append new keypoints at the end.",
                )
        return True, ""

    def _update_progress_label(self):
        if not hasattr(self, "progress_label"):
            return
        queue_labeled, queue_total = self._count_labeled_frames()
        noun = "depth maps" if LabelingApp._is_depth_layer(self) else "labeled"
        text = f"Queue: {queue_labeled}/{queue_total} {noun}"
        panel = getattr(self, "__dict__", {}).get("annotation_panel")
        if panel is not None:
            panel.set_progress(text)
        else:
            self.progress_label.setText(text)

    def _maybe_prompt_class_manager(self):
        if getattr(self, "_prompted_class_manager", False):
            return
        self._prompted_class_manager = True
        force_setup = getattr(self, "_force_initial_setup", False)
        created = getattr(self, "_created_label_files", False)
        missing_info = not (self.classes and self.kp_names)
        if not (force_setup or created or missing_info):
            return
        QTimer.singleShot(200, self._launch_class_manager_initial)

    def _launch_class_manager_initial(self):
        dlg = ClassManagerDialog(
            self.classes,
            self.class_keypoints,
            self.kp_names,
            self,
            schema_locked=self._schema_is_locked(),
        )
        if getattr(self, "_force_initial_setup", False):
            dlg.setWindowTitle("New Project Setup — Add Classes & Keypoints")
            msg = "Define classes and keypoints for this new project."
        else:
            dlg.setWindowTitle("Initial Setup — Add Classes & Keypoints")
            msg = "Please define your classes and keypoints before labeling."
        dlg_label = QLabel(msg)
        dlg.layout().insertWidget(0, dlg_label)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            classes, keypoints, kp_map = dlg.get_results()
            if self._apply_class_manager_results(classes, keypoints, kp_map):
                self.class_selector.setCurrentIndex(0)
                self.update_status_bar("Initial setup complete")
        else:
            if getattr(self, "_force_initial_setup", False):
                confirm = QMessageBox.question(
                    self,
                    "Use Defaults?",
                    "Project setup was canceled.\nContinue with default class/keypoints?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    QTimer.singleShot(0, self._launch_class_manager_initial)
                    return
            self.update_status_bar("Setup skipped; using defaults until edited.")

    def _write_list_file(self, path: str, items: list[str]):
        try:
            atomic_write_text(path, "".join(f"{item}\n" for item in items))
        except Exception as e:
            QMessageBox.warning(self, "File write error", f"Could not write {path}:\n{e}")

    def _backup_label_dir(self, labels_dir: str) -> str:
        return backup_label_dir(labels_dir)

    def _track_scene_item(self, item: QGraphicsItem):
        if item not in self._item_refs:
            self._item_refs.append(item)

    def _untrack_scene_item(self, item: QGraphicsItem):
        try:
            self._item_refs.remove(item)
        except ValueError:
            pass

    def _is_reference_layer_item(self, item: QGraphicsItem) -> bool:
        return bool(getattr(item, "reference_layer_id", ""))

    def _clear_reference_layer_items(self) -> None:
        presenter = self.__dict__.get("_scene_presenter")
        if presenter is not None:
            presenter.clear_references()

    def _refresh_reference_layer_overlay(self) -> None:
        self._clear_reference_layer_items()
        if not getattr(self, "images", None):
            return
        active_layer = getattr(self, "active_layer", LAYER_KEYPOINTS)
        base = os.path.splitext(self.images[self.current_idx])[0]
        for reference_layer in LAYER_DEFINITIONS:
            if reference_layer == active_layer:
                continue
            if not self.layer_visibility.get(reference_layer, True):
                continue
            if reference_layer == LAYER_DEPTH:
                self._add_depth_reference_overlay(base)
            elif reference_layer == LAYER_SEGMENTATION:
                self._add_segmentation_reference_overlay(base)
            else:
                self._add_keypoints_reference_overlay(base)

    def _add_depth_reference_overlay(self, base: str) -> None:
        preview_path = os.path.join(self.depth_preview_dir, f"{base}_depth.png")
        if not os.path.isfile(preview_path):
            return
        depth_pixmap = QPixmap(preview_path)
        if depth_pixmap.isNull():
            return
        self._scene_presenter.add_depth_reference(
            depth_pixmap,
            layer_id=LAYER_DEPTH,
            image_width=self.img_w,
            image_height=self.img_h,
        )

    def _add_segmentation_reference_overlay(self, base: str) -> None:
        reference_color = QColor(104, 164, 207)
        label_file = os.path.join(self.seg_label_dir, f"{base}.txt")
        if not os.path.isfile(label_file):
            return
        document = load_segmentation_document(
            label_file,
            classes_count=len(self.seg_classes),
            image_width=self.img_w,
            image_height=self.img_h,
        )
        for cid, entry in document.snapshot().items():
            points = []
            for pair in entry.get("segments", []):
                try:
                    points.append((float(pair[0]), float(pair[1])))
                except Exception:
                    continue
            label = self.seg_classes[cid] if 0 <= cid < len(self.seg_classes) else f"class_{cid}"
            self._scene_presenter.add_segmentation_reference(
                cid,
                points,
                label_text=label,
                layer_id=LAYER_SEGMENTATION,
                color=reference_color,
            )

    def _add_keypoints_reference_overlay(self, base: str) -> None:
        reference_color = QColor(104, 164, 207)
        depth_map = getattr(self, "_active_depth_map", None)
        show_depth_labels = self._is_depth_layer() and depth_map is not None and _np is not None
        label_file = os.path.join(self.pose_label_dir, f"{base}.txt")
        if not os.path.isfile(label_file):
            return
        class_lookup = [
            self.pose_class_keypoints.get(name, self.pose_kp_names) for name in self.pose_classes
        ]
        load_result = load_pose_document(
            label_file,
            classes_count=len(self.pose_classes),
            canonical_names=self.pose_kp_names,
            class_keypoint_lookup=class_lookup,
            image_width=self.img_w,
            image_height=self.img_h,
        )
        for cid, entry in load_result.document.snapshot().items():
            bbox_data = entry.get("bbox", {})
            bbox = BoundingBox(
                bbox_data.get("x", 0.0),
                bbox_data.get("y", 0.0),
                bbox_data.get("w", 0.0),
                bbox_data.get("h", 0.0),
                cid,
            )
            class_name = self.pose_classes[cid] if 0 <= cid < len(self.pose_classes) else str(cid)
            keypoint_references = []
            for kp_info in entry.get("keypoints", []):
                name = str(kp_info.get("name") or "")
                kp = Keypoint(
                    kp_info.get("x", 0.0),
                    kp_info.get("y", 0.0),
                    cid,
                    name,
                )
                visibility = int(kp_info.get("vis", 2))
                display_name = ""
                if show_depth_labels and visibility > 0:
                    try:
                        display_name = keypoint_depth_label(
                            name,
                            depth_map,
                            x=kp.x,
                            y=kp.y,
                            numpy_module=_np,
                        )
                    except DepthMapError:
                        display_name = f"{name} · unavailable"
                keypoint_references.append(
                    PoseReferenceKeypoint(
                        kp,
                        visibility=visibility,
                        label_text=display_name,
                    )
                )
            self._scene_presenter.add_pose_reference(
                bbox,
                keypoint_references,
                class_name=class_name,
                layer_id=LAYER_KEYPOINTS,
                keypoint_radius=self.kp_pixel_radius,
                keypoint_font_px=self.kp_font_px,
                show_keypoint_labels=show_depth_labels,
                color=reference_color,
            )

    # ---------- Annotation helpers ----------

    def _seg_class_color(self, class_id: int, alpha: int = 255) -> QColor:
        return CanvasScenePresenter.segmentation_color(class_id, alpha)

    def _seg_class_name(self, class_id: int) -> str:
        if 0 <= int(class_id) < len(self.classes):
            return self.classes[int(class_id)]
        return f"class_{int(class_id)}"

    def _is_seg_mask_item(self, item: QGraphicsItem) -> bool:
        return isinstance(item, QGraphicsPathItem) and hasattr(item, "seg_class_id")

    def _class_seg_mask_item(self, class_id: int) -> Optional[QGraphicsPathItem]:
        for item in self.scene.items():
            if (
                self._is_seg_mask_item(item)
                and not self._is_reference_layer_item(item)
                and not bool(getattr(item, "seg_preview", False))
                and int(getattr(item, "seg_class_id", -1)) == class_id
            ):
                return item
        return None

    def _polygon_path(self, points: list[tuple[float, float]]) -> Optional[QPainterPath]:
        return CanvasScenePresenter.polygon_path(points)

    def _extract_seg_item_points(
        self, item: Optional[QGraphicsPathItem]
    ) -> list[tuple[float, float]]:
        if item is None:
            return []
        raw = getattr(item, "seg_points", [])
        out: list[tuple[float, float]] = []
        for pair in raw:
            try:
                x = float(pair[0])
                y = float(pair[1])
                out.append((x, y))
            except Exception:
                continue
        return out

    def _clamp_scene_xy(self, x: float, y: float) -> tuple[int, int]:
        return clamp_point_to_image(x, y, self.img_w, self.img_h)

    def _seg_mask_shape(self) -> tuple[int, int]:
        return segmentation_mask_shape(self.img_w, self.img_h)

    def _seg_mask_from_points(self, points: list[tuple[float, float]]) -> Optional[object]:
        return polygon_to_mask(
            points,
            image_width=self.img_w,
            image_height=self.img_h,
            numpy_module=_np,
            cv2_module=_cv2,
        )

    def _seg_edit_target_item(self) -> Optional[QGraphicsPathItem]:
        if not self._is_seg_workflow():
            return None
        cid = self.class_selector.currentIndex() if hasattr(self, "class_selector") else -1
        if cid < 0:
            return None
        preview = self.seg_preview_item
        if (
            preview is not None
            and int(getattr(preview, "seg_class_id", -1)) == cid
            and len(self.seg_preview_points) >= 3
        ):
            return preview
        return self._class_seg_mask_item(cid)

    def _seg_update_item_geometry(
        self, item: Optional[QGraphicsPathItem], points: list[tuple[float, float]]
    ) -> bool:
        return self._scene_presenter.update_segmentation_geometry(item, points)

    def _clear_seg_edit_handles(self):
        self.seg_brush_active = False
        self.seg_brush_mask = None
        self.seg_brush_item = None
        self.seg_brush_anchor_points = []

    def _is_seg_edit_tool_brush(self) -> bool:
        return True

    def _refresh_seg_brush_size_badge(self):
        if not hasattr(self, "seg_brush_size_label"):
            return
        radius = int(getattr(self, "seg_brush_radius", 8))
        self.seg_brush_size_label.setText(f"Brush: {radius}px")
        if self._is_seg_workflow():
            self.seg_brush_size_label.setVisible(True)
        else:
            self.seg_brush_size_label.setVisible(False)

    def _start_seg_brush(self, scene_pos: QPointF, add: bool) -> bool:
        if not self._is_seg_workflow() or self.mode != "segedit":
            return False
        if not self._is_seg_edit_tool_brush():
            return False
        if _cv2 is None or _np is None:
            QMessageBox.warning(
                self,
                "Brush edit unavailable",
                "Brush-based mask editing requires OpenCV and NumPy in this environment.",
            )
            return False
        item = self._seg_edit_target_item()
        points = self._extract_seg_item_points(item)
        if item is None or len(points) < 3:
            self.update_status_bar("No segmentation mask available to brush-edit.")
            return False
        self._clear_seg_edit_handles()
        self.seg_brush_mask = self._seg_mask_from_points(points)
        self.seg_brush_item = item
        self.seg_brush_anchor_points = [(float(x), float(y)) for x, y in points]
        if self.seg_brush_mask is None:
            return False
        self.seg_brush_active = True
        self._apply_seg_brush(scene_pos, add=add, prev_scene_pos=None)
        return True

    def _finish_seg_brush(self):
        self.seg_brush_active = False
        self.seg_brush_mask = None
        self.seg_brush_item = None
        self.seg_brush_anchor_points = []
        self._clear_seg_edit_handles()
        self._refresh_sam_controls()

    def _apply_seg_brush(
        self, scene_pos: QPointF, add: bool, prev_scene_pos: Optional[QPointF] = None
    ) -> bool:
        if not self._is_seg_workflow() or self.mode != "segedit":
            return False
        if not self._is_seg_edit_tool_brush():
            return False
        if _cv2 is None or _np is None:
            return False

        item = self._seg_edit_target_item()
        if item is None:
            return False
        if self.img_w <= 0 or self.img_h <= 0:
            return False

        points = self._extract_seg_item_points(item)
        if len(points) < 3:
            return False

        # Keep a persistent raster mask through the brush stroke. This avoids
        # repeated polygon->mask->polygon conversions that can drift vertices.
        active_mask = getattr(self, "seg_brush_mask", None)
        active_item = getattr(self, "seg_brush_item", None)
        if active_mask is None or active_item is not item:
            active_mask = self._seg_mask_from_points(points)
            if active_mask is None:
                return False
            self.seg_brush_mask = active_mask
            self.seg_brush_item = item
            self.seg_brush_anchor_points = [(float(x), float(y)) for x, y in points]

        start = None
        if prev_scene_pos is not None:
            start = (prev_scene_pos.x(), prev_scene_pos.y())
        result = apply_brush_stroke(
            active_mask,
            end=(scene_pos.x(), scene_pos.y()),
            start=start,
            radius=getattr(self, "seg_brush_radius", 8),
            add=add,
            image_width=self.img_w,
            image_height=self.img_h,
            cv2_module=_cv2,
            anchor_points=getattr(self, "seg_brush_anchor_points", None) or points,
            max_points=1200,
        )
        if result is None:
            return False
        if result.erased:
            if item is self.seg_preview_item:
                self._clear_seg_preview()
            else:
                cid = int(getattr(item, "seg_class_id", self.class_selector.currentIndex()))
                self._clear_class_items(cid, drop_cache=True)
            self.seg_brush_mask = None
            self.seg_brush_item = None
            self.seg_brush_anchor_points = []
            self.update_status_bar("Brush erased mask.")
            return True

        new_points = result.points
        if len(new_points) < 3:
            return False

        if not self._seg_update_item_geometry(item, new_points):
            return False
        if item is self.seg_preview_item:
            self.seg_preview_points = [(float(x), float(y)) for x, y in new_points]
        else:
            cid = int(getattr(item, "seg_class_id", self.class_selector.currentIndex()))
            self._cache_active_annotation(cid)
        self._refresh_sam_controls()
        return True

    def _add_seg_mask_item(
        self, class_id: int, points: list[tuple[float, float]], preview: bool = False
    ):
        return self._scene_presenter.add_segmentation_mask(
            class_id,
            points,
            label_text=self._seg_class_name(class_id),
            preview=preview,
        )

    def _clear_seg_prompt_items(self):
        self._scene_presenter.clear_prompts()

    def _clear_seg_preview(self, *, clear_state: bool = True):
        self._clear_seg_edit_handles()
        if self.seg_preview_item is not None:
            self._safe_remove_scene_item(self.seg_preview_item)
            self._untrack_scene_item(self.seg_preview_item)
            self.seg_preview_item = None
        if clear_state:
            controller = getattr(self, "_segmentation_controller", None)
            if controller is not None and controller.state is self.seg_edit_state:
                controller.state.clear_preview()
            else:
                self.seg_edit_state.clear_preview()
        self._refresh_sam_controls()

    def _clear_seg_prompt_state(self):
        controller = getattr(self, "_segmentation_controller", None)
        if controller is not None and controller.state is self.seg_edit_state:
            controller.clear_prompts()
        else:
            self.seg_edit_state.clear_prompt_state()
        self._clear_seg_prompt_items()
        self._clear_seg_preview()
        self._refresh_sam_controls()

    def _refresh_sam_controls(self):
        if not hasattr(self, "sam_helper_label"):
            return
        if not self._is_seg_workflow():
            self.sam_helper_label.setText("")
            return

        total_prompts = len(self.seg_prompt_points)
        pos_prompts = sum(1 for _, _, lb in self.seg_prompt_points if int(lb) == 1)
        neg_prompts = total_prompts - pos_prompts
        has_preview = len(self.seg_preview_points) >= 3
        has_image = bool(self.images)
        in_segment_mode = self.mode == "segment"
        model_loaded = bool(self.__dict__.get("_sam_model_ready", False))
        model_loading = bool(self.__dict__.get("_sam_model_loading", False))
        sam_controller = self.__dict__.get("_sam_assistant_controller")
        model_busy = bool(sam_controller is not None and sam_controller.is_busy)

        cid = self.class_selector.currentIndex() if hasattr(self, "class_selector") else -1
        class_name = self._seg_class_name(cid) if cid >= 0 else "class"
        has_mask = False
        if cid >= 0:
            entry = self.annotation_cache.get(cid, {})
            has_mask = len(entry.get("segments", [])) >= 3 or (
                self._class_seg_mask_item(cid) is not None
            )
        completed = sum(1 for idx in range(len(self.classes)) if self._class_is_complete(idx))
        run_enabled = (
            has_image and in_segment_mode and total_prompts > 0 and model_loaded and not model_busy
        )
        accept_enabled = has_preview and not model_busy
        clear_enabled = (has_preview or total_prompts > 0) and not model_busy
        load_enabled = not model_loaded and not model_loading and not model_busy
        if hasattr(self, "sam_load_btn"):
            self.sam_load_btn.setEnabled(load_enabled)
            self.sam_load_btn.setText(
                "SAM Ready" if model_loaded else ("Loading SAM…" if model_loading else "Load SAM")
            )
            self.sam_load_btn.setProperty("tone", "load" if load_enabled else "")
            _refresh_qt_style(self.sam_load_btn)
        self.sam_run_btn.setEnabled(run_enabled)
        self.sam_accept_btn.setEnabled(accept_enabled)
        self.sam_clear_btn.setEnabled(clear_enabled)
        self.sam_run_btn.setProperty("tone", "run" if (run_enabled and not has_preview) else "")
        self.sam_accept_btn.setProperty("tone", "accept" if accept_enabled else "")
        self.sam_clear_btn.setProperty("tone", "clear" if clear_enabled else "")
        _refresh_qt_style(self.sam_run_btn)
        _refresh_qt_style(self.sam_accept_btn)
        _refresh_qt_style(self.sam_clear_btn)
        if model_busy:
            self.sam_run_btn.setToolTip("SAM segmentation is running.")
        elif model_loading:
            self.sam_run_btn.setToolTip("SAM model is loading in the worker.")
        elif not model_loaded:
            self.sam_run_btn.setToolTip("Load SAM model first, then add prompts and run.")
        elif not has_image:
            self.sam_run_btn.setToolTip("No image loaded.")
        elif not total_prompts:
            self.sam_run_btn.setToolTip("Add prompt points first (left=positive, right=negative).")
        else:
            self.sam_run_btn.setToolTip("Run SAM using current positive/negative prompts.")

        if self.mode == "segedit":
            brush_px = int(getattr(self, "seg_brush_radius", 8))
            tool_text = "Brush"
            edit_text = f"left add, right erase (brush {brush_px}px; ,/. size)."
            if has_preview or has_mask:
                action = f"Mask edit ({tool_text}): {edit_text}"
            else:
                action = "No mask yet. Run SAM, then accept or edit."
        elif not model_loaded:
            action = "Load SAM, add prompts, then Run (G)."
        elif not has_image:
            action = "Open an image to segment."
        elif not in_segment_mode:
            action = "Press 2. Left=positive, right=negative."
        elif not total_prompts and not has_preview:
            action = "Left=positive, right=negative."
        elif total_prompts and not has_preview:
            action = "Run SAM (G) for preview."
        else:
            action = "Accept (Shift+Enter) to save mask."

        mask_text = "saved" if has_mask else "none"
        preview_text = "ready" if has_preview else "none"
        model_text = "ready" if model_loaded else ("loading" if model_loading else "missing")
        self.sam_helper_label.setText(
            f"Class {class_name} | Done {completed}/{len(self.classes)} | Model {model_text} | Mask {mask_text}\n"
            f"Prompts +{pos_prompts}/-{neg_prompts} | Preview {preview_text}\n"
            f"{action}"
        )

    def _load_sam_model_interactive(self):
        if not self._is_seg_workflow():
            self._switch_workflow(WORKFLOW_SEG)
        self._ensure_sam_model_loaded()
        self._refresh_sam_controls()

    def _draw_seg_prompt_marker(self, x: float, y: float, positive: bool):
        self._scene_presenter.add_prompt_marker(x, y, positive=positive)

    def _refresh_seg_prompt_markers(self):
        self._clear_seg_prompt_items()
        for x, y, label in self.seg_prompt_points:
            self._draw_seg_prompt_marker(float(x), float(y), positive=bool(label))
        self._refresh_sam_controls()

    def _add_seg_prompt(self, scene_pos: QPointF, positive: bool = True):
        if not self._is_seg_workflow() or self.mode != "segment":
            return
        sam_controller = self.__dict__.get("_sam_assistant_controller")
        if sam_controller is not None and sam_controller.is_busy:
            self.update_status_bar("Wait for the current SAM segmentation to finish.")
            return
        if not self.images:
            return
        x = max(0.0, min(float(self.img_w - 1), float(scene_pos.x())))
        y = max(0.0, min(float(self.img_h - 1), float(scene_pos.y())))
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            controller.add_prompt(x, y, positive=positive)
        else:
            self.seg_edit_state.add_prompt(x, y, positive=positive)
        self._draw_seg_prompt_marker(x, y, positive=positive)
        self.update_status_bar(
            f"Added {'positive' if positive else 'negative'} prompt ({len(self.seg_prompt_points)} total)."
        )
        self._refresh_sam_controls()

    def _sam3_model_candidates_in_project_root(self) -> list[str]:
        return list(
            discover_sam_weight_candidates(
                self.__dict__.get("project_root", ""),
                default_filename=DEFAULT_SAM3_WEIGHTS,
            )
        )

    def _try_autoload_sam_model_from_project_root(self) -> tuple[bool, str]:
        """Queue non-interactive SAM warm-up from a project-local weight file."""
        if self._sam_model_ready:
            return False, self._sam_worker_model_path
        candidates = self._sam3_model_candidates_in_project_root()
        model_path = select_existing_sam_weight(candidates)
        if model_path:
            self._warm_sam_model(model_path)
            return True, model_path
        return False, ""

    def _ensure_sam_model_loaded(self) -> bool:
        if self._sam_model_ready and self._sam_worker_model_path:
            return True
        if self._sam_model_loading:
            return False

        candidate_paths = []
        if self.sam_model_path:
            candidate_paths.append(self.sam_model_path)
        candidate_paths.extend(self._sam3_model_candidates_in_project_root())
        candidate_paths.extend(
            [
                os.path.join(self.project_root, DEFAULT_SAM3_WEIGHTS),
                os.path.join(os.getcwd(), DEFAULT_SAM3_WEIGHTS),
            ]
        )

        model_path = select_existing_sam_weight(candidate_paths) or ""

        if not model_path:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select SAM model file",
                self.project_root,
                "Model Files (*.pt *.pth)",
            )
            if not selected:
                return False
            model_path = selected
        return self._warm_sam_model(model_path)

    def _warm_sam_model(self, model_path: str) -> bool:
        normalized = os.path.abspath(str(model_path or ""))
        if not normalized:
            return False
        if self._sam_model_ready and normalized == self._sam_worker_model_path:
            return True
        self.sam_model_path = normalized
        self._sam_worker_model_path = normalized
        self._sam_model_ready = False
        self._sam_model_loading = True
        self._save_project_preferences()
        self._sam_assistant_controller.restart_model(
            model_path=normalized,
            device=self._device,
            warm=True,
        )
        self.update_status_bar(f"Loading SAM model: {os.path.basename(normalized)}")
        self._refresh_sam_controls()
        return False

    def _stop_sam_assistant(self) -> None:
        controller = self.__dict__.get("_sam_assistant_controller")
        self._pending_sam_prompt_request = None
        self._pending_sam_class_id = None
        self._sam_request_class_id = None
        self._sam_model_ready = False
        self._sam_model_loading = False
        self._sam_worker_model_path = ""
        if controller is not None:
            if controller.is_busy:
                controller.cancel()
            else:
                controller.shutdown()
        self._restore_sam_wait_cursor()

    def _restore_sam_wait_cursor(self) -> None:
        if not self.__dict__.get("_sam_cursor_active", False):
            return
        self._sam_cursor_active = False
        QApplication.restoreOverrideCursor()

    def _sam_worker_busy_changed(self, busy: bool) -> None:
        if busy and not self._sam_cursor_active:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._sam_cursor_active = True
        elif not busy:
            self._restore_sam_wait_cursor()
        self._refresh_sam_controls()

    def _handle_sam_worker_event(self, event: dict) -> None:
        event_type = str(event.get("event") or "")
        event_model_path = os.path.abspath(str(event.get("model_path") or ""))
        if event_type == "loading":
            self._sam_model_ready = False
            self._sam_model_loading = True
        elif event_type == "loaded" and (
            not event_model_path or event_model_path == self._sam_worker_model_path
        ):
            self._sam_model_ready = True
            self._sam_model_loading = False
            pending = self._pending_sam_prompt_request
            pending_class_id = self._pending_sam_class_id
            self._pending_sam_prompt_request = None
            self._pending_sam_class_id = None
            if pending is not None and pending_class_id is not None:
                QTimer.singleShot(
                    0,
                    lambda request=pending, class_id=pending_class_id: self._submit_sam_prompt(
                        request, class_id
                    ),
                )
        elif event_type == "error" and not self._sam_assistant_controller.is_busy:
            self._sam_model_ready = False
            self._sam_model_loading = False
            self._pending_sam_prompt_request = None
            self._pending_sam_class_id = None
        self._refresh_sam_controls()

    def _handle_sam_worker_terminal(self, result) -> None:
        self._sam_model_ready = False
        restarting = str(getattr(result, "state", "")) == "stopped" and self._sam_model_loading
        if not restarting:
            self._sam_model_loading = False
            self._sam_worker_model_path = ""
            self._pending_sam_prompt_request = None
            self._pending_sam_class_id = None
        self._restore_sam_wait_cursor()
        self._refresh_sam_controls()

    def _submit_sam_prompt(self, request: SamPromptRequest, class_id: int) -> None:
        if not self._is_seg_workflow() or not self._sam_model_ready:
            return
        try:
            self._sam_request_class_id = int(class_id)
            self._sam_assistant_controller.submit_prompt(
                model_path=self._sam_worker_model_path,
                prompt=request,
                device=self._device,
            )
        except (RuntimeError, ValueError) as exc:
            self._sam_request_class_id = None
            QMessageBox.warning(self, "SAM inference error", f"SAM segmentation failed:\n{exc}")

    def _handle_sam_worker_decision(self, decision) -> None:
        action = str(getattr(decision, "action", ""))
        if action == "background_error":
            self._sam_model_ready = False
            self._sam_model_loading = False
            QMessageBox.warning(
                self,
                "SAM load error",
                f"Could not load SAM model:\n{decision.error_message}",
            )
            self._refresh_sam_controls()
            return
        class_id = self._sam_request_class_id
        self._sam_request_class_id = None
        if action == "cancel":
            if self._is_seg_workflow():
                self.update_status_bar("SAM segmentation canceled.")
            return
        if action == "discard":
            self.update_status_bar("Discarded SAM result because the displayed image changed.")
            return
        if action == "error":
            QMessageBox.warning(
                self,
                "SAM inference error",
                f"SAM segmentation failed:\n{decision.error_message}",
            )
            return
        if action != "apply":
            return
        if not self._is_seg_workflow() or class_id is None:
            self.update_status_bar("Discarded SAM result because the annotation layer changed.")
            return
        if self.class_selector.currentIndex() != class_id:
            self.update_status_bar("Discarded SAM result because the active class changed.")
            return
        if decision.failure == "no_masks":
            QMessageBox.information(
                self, "No masks", "SAM did not return any segmentation mask for these prompts."
            )
            return
        if decision.result is None:
            QMessageBox.information(
                self, "No polygon", "SAM returned a mask without a usable contour polygon."
            )
            return
        points_xy = list(decision.result.points)
        self._clear_seg_preview(clear_state=False)
        self.seg_preview_item = self._add_seg_mask_item(class_id, points_xy, preview=True)
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            controller.select_target(class_id, clear_prompts=False)
            controller.set_preview(points_xy, score=decision.result.score)
        else:
            self.seg_edit_state.set_preview(points_xy, decision.result.score)
        if self.seg_preview_item is None:
            QMessageBox.information(self, "No polygon", "Unable to render SAM mask preview.")
            return
        self.update_status_bar("SAM mask preview ready. Click Accept Mask to commit.")
        self._refresh_sam_controls()

    def _run_sam_segmentation(self):
        if not self._is_seg_workflow():
            return
        if self.mode != "segment":
            self.set_mode("segment")
        if not self.images:
            self.update_status_bar("No image loaded.")
            return
        if not self.seg_prompt_points:
            QMessageBox.information(
                self, "No prompts", "Add at least one prompt point before running SAM."
            )
            return
        if self._sam_assistant_controller.is_busy:
            self.update_status_bar("SAM segmentation is already running.")
            return

        img_source = self.current_image_path or os.path.join(
            self.active_image_dir, self.images[self.current_idx]
        )
        cid = self.class_selector.currentIndex()
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            controller.select_target(cid, clear_prompts=False)
            request = controller.build_prompt_request(img_source)
        else:
            request = SamPromptRequest(
                source=img_source,
                class_id=cid,
                prompts=tuple(self.seg_prompt_points),
            )

        if not self._sam_model_ready:
            self._pending_sam_prompt_request = request
            self._pending_sam_class_id = cid
            if not self._ensure_sam_model_loaded() and not self._sam_model_loading:
                self._pending_sam_prompt_request = None
                self._pending_sam_class_id = None
            self._refresh_sam_controls()
            return
        self._submit_sam_prompt(request, cid)
        self._refresh_sam_controls()

    def _accept_segmentation_preview(self):
        if not self._is_seg_workflow():
            return
        if len(self.seg_preview_points) < 3:
            QMessageBox.information(
                self, "No preview", "Run SAM first to create a segmentation mask preview."
            )
            return

        cid = self.class_selector.currentIndex()
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            controller.select_target(cid, clear_prompts=False)
            accepted_entry = controller.accept_preview()
        else:
            self.seg_edit_state.select_target(cid)
            accepted_entry = self.seg_edit_state.accept_preview()
        if accepted_entry is None:
            return
        self._clear_class_items(cid, drop_cache=False)
        self._set_segmentation_cache_entry(cid, accepted_entry)
        self._restore_annotation_for_class(cid)
        self._clear_seg_prompt_state()
        self._update_item_editability()
        if self.mode == "segedit":
            self._clear_seg_edit_handles()
        self._jump_to_next_pending_class()
        self.update_status_bar("Segmentation mask accepted for current class.")
        self._refresh_sam_controls()

    def _class_box_item(self, class_id: int) -> Optional[BoxItem]:
        for item in self.scene.items():
            if (
                isinstance(item, BoxItem)
                and not self._is_reference_layer_item(item)
                and item.bbox.class_id == class_id
            ):
                return item
        return None

    def _class_keypoint_items(self, class_id: int) -> list[KeypointItem]:
        return [
            item
            for item in self.scene.items()
            if (
                isinstance(item, KeypointItem)
                and not self._is_reference_layer_item(item)
                and item.kp.class_id == class_id
            )
        ]

    def _clear_class_items(self, class_id: int, drop_cache: bool = False):
        removed = False
        for item in list(self.scene.items()):
            if self._is_reference_layer_item(item):
                continue
            if isinstance(item, BoxItem) and item.bbox.class_id == class_id:
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                removed = True
            elif isinstance(item, KeypointItem) and item.kp.class_id == class_id:
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                removed = True
            elif (
                self._is_seg_mask_item(item) and int(getattr(item, "seg_class_id", -1)) == class_id
            ):
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                removed = True
        if drop_cache:
            if self._is_seg_workflow():
                self._drop_segmentation_cache_entry(class_id)
            else:
                self.annotation_cache.delete_annotation(class_id)
                if self.pose_edit_state.active_class_id == class_id:
                    self.pose_edit_state.clear()
        if class_id == self.class_selector.currentIndex():
            self.bboxes.clear()
            self.kps.clear()
            self.current_box = None
            self.current_kps = self.kps
            self.current_class_id = class_id if not self._is_seg_workflow() else None
            self.current_kp_idx = 0
            if self._is_seg_workflow():
                self._clear_seg_prompt_state()
            if removed:
                self._update_status()

    def _clear_all_annotation_items(self):
        self._clear_seg_edit_handles()
        for item in list(self.scene.items()):
            if self._is_reference_layer_item(item):
                continue
            if isinstance(item, (BoxItem, KeypointItem)) or self._is_seg_mask_item(item):
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
        self.bboxes.clear()
        self.kps.clear()
        self.current_box = None
        self.current_kps = self.kps
        self.current_class_id = None
        self.current_kp_idx = 0
        if hasattr(self, "pose_edit_state"):
            self.pose_edit_state.select_class(None)
        self._clear_seg_prompt_state()

    def _sync_active_class_state(self):
        if self._is_seg_workflow():
            self.bboxes = []
            self.kps = []
            self.current_box = None
            self.current_kps = self.kps
            self.current_class_id = None
            self.current_kp_idx = 0
            return
        cid = self.class_selector.currentIndex()
        self._sync_pose_state_from_scene(cid)
        self._sync_pose_legacy_mirrors()

    def _update_item_editability(self):
        active_cid = self.class_selector.currentIndex()
        for item in self.scene.items():
            if self._is_reference_layer_item(item):
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                continue
            if isinstance(item, BoxItem):
                editable = item.bbox.class_id == active_cid
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
                item.setOpacity(1.0 if editable else 0.4)
            elif isinstance(item, KeypointItem):
                editable = item.kp.class_id == active_cid
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
                item.setOpacity(1.0 if editable else 0.4)
            elif self._is_seg_mask_item(item):
                editable = int(getattr(item, "seg_class_id", -1)) == active_cid
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
                item.setOpacity(1.0 if editable else 0.35)

    def _class_is_complete(self, class_id: int) -> bool:
        if self._is_seg_workflow():
            entry = self.seg_edit_state.accepted_masks.get(class_id)
            if not entry:
                return False
            seg = entry.get("segments", [])
            return len(seg) >= 3
        return self.annotation_cache.is_complete(
            class_id,
            required_keypoints=self._kp_names_for_index(class_id),
        )

    def _jump_to_next_pending_class(self):
        total = len(self.classes)
        if total <= 1:
            return
        current = self.class_selector.currentIndex()
        pending = []
        for idx in range(total):
            if not self._class_is_complete(idx):
                pending.append(idx)
        if not pending:
            return
        for offset in range(1, total + 1):
            nxt = (current + offset) % total
            if nxt in pending:
                self.class_selector.setCurrentIndex(nxt)
                self.update_status_bar(f"Next class: {self.classes[nxt]}")
                return

    def _maybe_autoadvance(self):
        cid = self.class_selector.currentIndex()
        names = self._active_kp_names()
        if not names:
            if self._class_box_item(cid) and self._cache_active_annotation():
                self._update_item_editability()
                self._jump_to_next_pending_class()
            return
        if self.current_kp_idx >= len(names):
            if self._cache_active_annotation():
                self._update_item_editability()
                self._jump_to_next_pending_class()

    def _cycle_class(self, direction: int = 1):
        if not self.classes:
            return
        idx = self.class_selector.currentIndex()
        new_idx = (idx + direction) % len(self.classes)
        if new_idx != idx:
            self.class_selector.setCurrentIndex(new_idx)

    def refresh_image_list(self):
        """Reload queue file list (used after exporting a frame from video)."""
        self._refresh_queue_images()
        self.images = self.images_queue[:]
        if self.current_idx >= len(self.images):
            self.current_idx = max(0, len(self.images) - 1)
        self._update_progress_label()

    def _warn_image_stem_collisions(self):
        collisions = getattr(self, "_queue_stem_collisions", {}) or {}
        if not collisions:
            self._reported_stem_collision_signature = None
            return
        signature = tuple((stem, tuple(names)) for stem, names in sorted(collisions.items()))
        if signature == getattr(self, "_reported_stem_collision_signature", None):
            return
        self._reported_stem_collision_signature = signature
        groups = [" / ".join(names) for names in collisions.values()]
        preview = "\n".join(groups[:10])
        if len(groups) > 10:
            preview += f"\n...{len(groups) - 10} more"
        QMessageBox.warning(
            self,
            "Duplicate Image Names",
            "Images that share the same filename stem cannot have independent YOLO labels. "
            "The following files are excluded until they are renamed:\n\n" + preview,
        )

    def _refresh_queue_images(self, *, show_warning: bool = True):
        try:
            scan = scan_image_queue(self.image_dir_queue)
            self.images_queue = list(scan.images)
            self._queue_stem_collisions = scan.collisions
        except OSError:
            logger.warning(
                "Could not refresh project image queue",
                exc_info=True,
                extra={
                    "event": "image_queue_refresh_failed",
                    "operation": "refresh_image_queue",
                    "project_root": self.project_root,
                    "source_path": self.image_dir_queue,
                },
            )
            self.images_queue = []
            self._queue_stem_collisions = {}
        if not self._queue_stem_collisions:
            self._reported_stem_collision_signature = None
        elif show_warning:
            QTimer.singleShot(0, self._warn_image_stem_collisions)

    def __init__(
        self,
        image_dir: Optional[str],
        label_dir: Optional[str],
        class_file: Optional[str],
        keypoint_file: Optional[str],
        project_root: Optional[str] = None,
        force_initial_setup: bool = False,
        project_lock: Optional[ProjectLock] = None,
    ):
        super().__init__()
        self.resize(1400, 860)
        self.setMinimumSize(1180, 700)
        self.app_base_dir = APP_BASE_DIR
        inferred_root = project_root or os.path.dirname(image_dir or "") or os.getcwd()
        self.project_root = os.path.abspath(inferred_root)
        if project_lock is None:
            project_lock = ProjectLock(self.project_root, version=__version__).acquire()
        elif canonical_path(project_lock.project_root) != canonical_path(self.project_root):
            raise ValueError("project lock does not belong to the selected project")
        elif not project_lock.acquired:
            raise ValueError("project lock must be acquired before opening the project")
        self._project_lock = project_lock
        self._log_path = project_log_path(self.project_root)
        try:
            self._log_path = configure_project_logging(self.project_root)
        except OSError:
            logger.exception(
                "Could not configure project logging",
                extra={
                    "event": "logging_configuration_failed",
                    "operation": "configure_logging",
                    "project_root": self.project_root,
                    "target_path": self._log_path,
                },
            )
        logger.info(
            "Project window initializing",
            extra={
                "event": "project_window_initializing",
                "operation": "open_project",
                "project_root": self.project_root,
            },
        )
        self._force_initial_setup = bool(force_initial_setup)
        self._schema_recoveries: list[tuple[str, str, str]] = []

        self.image_dir_queue = image_dir or os.path.join(self.project_root, "images_to_label")
        # Backward-compatible alias used by some dialogs/tools.
        self.image_dir = self.image_dir_queue
        self.image_dir_all = os.path.join(self.project_root, "images_all")
        self.pose_label_dir = label_dir or os.path.join(self.project_root, "labels_all")
        self.seg_label_dir = os.path.join(self.project_root, "labels_seg_all")
        self.depth_map_dir = os.path.join(self.project_root, "depth maps")
        self.depth_image_dir = os.path.join(self.depth_map_dir, "images")
        self.depth_preview_dir = os.path.join(self.depth_map_dir, "previews")
        os.makedirs(self.image_dir_queue, exist_ok=True)
        os.makedirs(self.pose_label_dir, exist_ok=True)
        os.makedirs(self.seg_label_dir, exist_ok=True)
        os.makedirs(self.image_dir_all, exist_ok=True)
        os.makedirs(self.depth_image_dir, exist_ok=True)
        os.makedirs(self.depth_preview_dir, exist_ok=True)
        self.pose_class_file = class_file or os.path.join(self.project_root, "classes.txt")
        self.pose_keypoint_file = keypoint_file or os.path.join(self.project_root, "keypoints.txt")
        self.pose_class_keypoints_path = os.path.join(self.project_root, "class_keypoints.json")
        self.seg_class_file = os.path.join(self.project_root, "classes_seg.txt")
        self.base_dir = self.project_root

        self._queue_stem_collisions: dict[str, list[str]] = {}
        self._reported_stem_collision_signature = None
        self._refresh_queue_images(show_warning=False)
        self.images = self.images_queue[:]
        self.active_image_dir = self.image_dir_queue
        self.current_image_path = ""
        self.current_idx = 0
        self._queue_current_idx = 0

        # Keypoints layer resources
        self.pose_classes, self.pose_kp_names, self._created_label_files = self._ensure_label_files(
            self.pose_class_file, self.pose_keypoint_file
        )
        self.classes = self.pose_classes[:]
        self.kp_names = self.pose_kp_names[:]
        self.class_keypoints_path = self.pose_class_keypoints_path
        self.class_keypoints = self._load_class_keypoints()
        if self._sync_canonical_keypoints_from_class_map():
            self.pose_kp_names = self.kp_names[:]
        self._save_class_keypoints()
        self.pose_class_keypoints = {
            name: self.class_keypoints.get(name, [])[:] for name in self.pose_classes
        }

        # Segmentation layer resources
        self.seg_classes, self._created_seg_class_file = self._ensure_classes_file(
            self.seg_class_file, DEFAULT_CLASS_NAMES
        )

        self.active_layer = LAYER_KEYPOINTS
        self.classes = self.pose_classes[:]
        self.kp_names = self.pose_kp_names[:]
        self.class_keypoints = {
            name: self.pose_class_keypoints.get(name, [])[:] for name in self.pose_classes
        }
        self.label_dir = self.pose_label_dir
        self.class_file = self.pose_class_file
        self.keypoint_file = self.pose_keypoint_file
        self._schema_locked = self._detect_schema_locked()
        self._kp_index_lookup: dict[str, int] = {}
        self._refresh_kp_index_lookup()
        self.annotation_cache = PoseAnnotationDocument()
        self.template_dir = os.path.join(self.project_root, "templates")
        os.makedirs(self.template_dir, exist_ok=True)

        self.mode = "panzoom"
        self.bboxes: List[BoundingBox] = []
        self.kps: List[Keypoint] = []
        self.current_kp_idx = 0
        self._item_refs: list[QGraphicsItem] = []
        self.layer_settings = normalize_layer_settings({})
        self.layer_model_paths: dict[str, str] = {layer_id: "" for layer_id in LAYER_DEFINITIONS}
        self.layer_visibility: dict[str, bool] = {layer_id: True for layer_id in LAYER_DEFINITIONS}
        self.predict_model_path: Optional[str] = None
        self.sam_model_path = os.path.join(self.project_root, DEFAULT_SAM3_WEIGHTS)
        self._sam_worker_model_path = ""
        self._sam_model_ready = False
        self._sam_model_loading = False
        self._sam_cursor_active = False
        self._pending_sam_prompt_request: SamPromptRequest | None = None
        self._pending_sam_class_id: int | None = None
        self._sam_request_class_id: int | None = None
        self._pose_controller = PoseAnnotationController(
            self.annotation_cache,
            keypoint_order_for=self._kp_names_for_index,
            canonical_names=self.kp_names,
        )
        self.pose_edit_state = self._pose_controller.state
        self.current_box: Optional[BoundingBox] = None
        self.current_kps: list[Keypoint] = []
        self.current_class_id: Optional[int] = None
        self._segmentation_document = SegmentationAnnotationDocument()
        self._segmentation_controller = SegmentationAnnotationController(
            self._segmentation_document
        )
        self.seg_edit_state = self._segmentation_controller.state
        self.seg_preview_item: Optional[QGraphicsPathItem] = None
        self.seg_brush_radius: int = 8
        self.seg_brush_active: bool = False
        self.seg_brush_mask: Optional[object] = None
        self.seg_brush_item: Optional[QGraphicsPathItem] = None
        self.seg_brush_anchor_points: list[tuple[float, float]] = []
        self._seg_setup_prompted = False
        self.nav_filter = "all"  # 'all' | 'labeled' | 'unlabeled'

        self._project_meta_recovery: Optional[tuple[str, str]] = None
        self._load_project_preferences()
        self._bind_layer_state(self.active_layer)
        self._save_project_preferences()

        # keypoint display (screen-space)
        self.kp_pixel_radius = 4
        self.kp_font_px = 10
        self._precision_active = False

        self._predict_busy = False
        self._inference_progress: Optional[QProgressDialog] = None
        self._prediction_depth_targets: Optional[dict[str, str]] = None
        self._active_depth_map = None
        self._depth_probes: list[dict] = []
        self._depth_probe_image_name = ""
        self._depth_probe_error = ""
        self._depth_controller = DepthAssistantController(
            sampler=lambda depth_map, *, x, y: sample_depth_map(
                depth_map,
                x=x,
                y=y,
                numpy_module=_np,
            )
        )
        # Auto-select device once at startup
        self._device = _auto_device()
        self._sam_assistant_controller = SamAssistantController(
            self,
            displayed_image_path=self._displayed_image_path,
            working_directory=APP_BASE_DIR,
        )
        self._sam_assistant_controller.status_changed.connect(self.update_status_bar)
        self._sam_assistant_controller.busy_changed.connect(self._sam_worker_busy_changed)
        self._sam_assistant_controller.event_received.connect(self._handle_sam_worker_event)
        self._sam_assistant_controller.decision_ready.connect(self._handle_sam_worker_decision)
        self._sam_assistant_controller.terminal.connect(self._handle_sam_worker_terminal)
        self._prediction_coordinator = PredictionController(
            self,
            displayed_image_path=self._displayed_image_path,
            working_directory=APP_BASE_DIR,
        )
        self._prediction_coordinator.status_changed.connect(self.update_status_bar)
        self._prediction_coordinator.busy_changed.connect(self._prediction_controller_busy_changed)
        self._prediction_coordinator.decision_ready.connect(
            self._handle_prediction_controller_decision
        )
        self._inference_coordinator = InferenceController(
            self,
            discard_outputs=self._discard_inference_outputs,
            working_directory=APP_BASE_DIR,
        )
        self._inference_coordinator.busy_changed.connect(self._inference_controller_busy_changed)
        self._inference_coordinator.job_started.connect(self._inference_controller_job_started)
        self._inference_coordinator.progress.connect(self._inference_controller_progress)
        self._inference_coordinator.pass_finished.connect(self._inference_controller_pass_finished)
        self._inference_coordinator.completed.connect(self._inference_controller_completed)
        print(f"🧠 Inference device: {self._device}")
        # Build UI and load first image
        self._setup_ui()
        self._depth_preview_presenter = DepthPreviewPresenter(
            self.scene,
            range_view=self.depth_range_frame,
            track_item=self._track_scene_item,
        )
        self._update_layer_ui_state()
        self.load_image()
        self._update_progress_label()
        if self._queue_stem_collisions:
            QTimer.singleShot(0, self._warn_image_stem_collisions)
        if self._project_meta_recovery:
            QTimer.singleShot(0, self._show_project_meta_recovery)
        if self._schema_recoveries:
            QTimer.singleShot(0, self._show_schema_recoveries)
        if self._is_seg_workflow():
            QTimer.singleShot(0, self._maybe_prompt_seg_class_manager_initial)
        else:
            QTimer.singleShot(0, self._maybe_prompt_class_manager)

    def closeEvent(self, event):
        inference_coordinator = getattr(self, "_inference_coordinator", None)
        if inference_coordinator is not None and inference_coordinator.is_busy:
            try:
                inference_coordinator.completed.disconnect(self._inference_controller_completed)
            except (TypeError, RuntimeError):
                pass
            inference_coordinator.shutdown()
        prediction_coordinator = getattr(self, "_prediction_coordinator", None)
        if prediction_coordinator is not None:
            prediction_coordinator.shutdown()
        sam_coordinator = self.__dict__.get("_sam_assistant_controller")
        if sam_coordinator is not None:
            sam_coordinator.shutdown()
            session = sam_coordinator.session
            process = getattr(session, "process", None) if session is not None else None
            if process is not None:
                _shutdown_qprocess(process, terminate_timeout_ms=1000, kill_timeout_ms=1000)
        self._restore_sam_wait_cursor()
        self._cleanup_prediction_depth_staging()
        self._project_lock.release()
        super().closeEvent(event)

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_project_action = file_menu.addAction("Open Project…")
        open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        open_project_action.triggered.connect(self.open_project_command)

        close_project_action = file_menu.addAction("Close Project")
        close_project_action.setShortcut(QKeySequence.StandardKey.Close)
        close_project_action.triggered.connect(self.close_project_command)

        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(QApplication.instance().quit)

    def _confirm_project_change(self, message: str) -> bool:
        decision = QMessageBox.question(
            self,
            "Switch Project",
            f"{message}\n\nUnsaved edits in the current view may be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return decision == QMessageBox.StandardButton.Yes

    def _switch_to_project_root(self, project_root: str, force_initial_setup: bool = False):
        target_root = os.path.abspath(project_root)
        if target_root == self.project_root and not force_initial_setup:
            self.update_status_bar("That project is already open.")
            return

        project_lock = _acquire_project_lock_for_ui(target_root, parent=self)
        if project_lock is None:
            self.update_status_bar("Project switch canceled because the project is locked.")
            return
        try:
            _recover_project_transactions_for_ui(target_root, parent=self)
            paths = _ensure_project_structure(target_root)
        except (OSError, ProjectPathError) as exc:
            project_lock.release()
            logger.exception(
                "Selected project has an invalid managed path",
                extra={
                    "event": "project_structure_invalid",
                    "operation": "switch_project",
                    "project_root": target_root,
                },
            )
            QMessageBox.critical(
                self,
                "Invalid Project Structure",
                f"The selected project contains an unsafe or unavailable managed path.\n\n{exc}",
            )
            return
        _save_last_project(target_root)

        try:
            new_window = LabelingApp(
                paths["images_to_label"],
                paths["labels_all"],
                paths["classes_file"],
                paths["keypoints_file"],
                project_root=paths["root"],
                force_initial_setup=force_initial_setup,
                project_lock=project_lock,
            )
        except Exception as exc:
            project_lock.release()
            logger.exception(
                "Could not initialize switched project",
                extra={
                    "event": "project_switch_failed",
                    "operation": "switch_project",
                    "project_root": paths.root,
                },
            )
            QMessageBox.critical(
                self,
                "Open Project Failed",
                f"Could not initialize the selected project.\n\n{exc}",
            )
            return
        _retain_main_window(new_window)
        new_window.setWindowTitle(_project_window_title(paths["root"]))
        new_window.setGeometry(self.geometry())
        new_window.show()
        new_window.raise_()
        new_window.activateWindow()
        new_window.update_status_bar(f"Project loaded: {paths['root']}")
        self.close()

    def open_project_command(self):
        if not self._confirm_project_change("Open a different project?"):
            return
        default_dir = (
            os.path.dirname(self.project_root) if self.project_root else _default_projects_root()
        )
        project_root = _choose_project_root(default_dir, parent=self)
        if not project_root:
            return
        self._switch_to_project_root(project_root, force_initial_setup=False)

    def close_project_command(self):
        if not self._confirm_project_change(
            "Close this project and return to the project launcher?"
        ):
            return
        default_dir = (
            os.path.dirname(self.project_root) if self.project_root else _default_projects_root()
        )
        launcher = ProjectLauncherDialog(
            default_dir, os.path.join(self.app_base_dir, "squeakpose_studio_logo.png"), self
        )
        if launcher.exec() != QDialog.DialogCode.Accepted:
            self.update_status_bar("Close project canceled.")
            return
        project_root = launcher.project_root
        if not project_root:
            self.update_status_bar("No project selected.")
            return
        self._switch_to_project_root(
            project_root, force_initial_setup=(launcher.selection_mode == "create")
        )

    # ---------- UI Setup ----------

    def _setup_ui(self):
        self.setWindowTitle("SqueakPose Studio")
        self._setup_menu()
        central = QWidget()
        self.setCentralWidget(central)

        self.scene = QGraphicsScene()
        self._scene_presenter = CanvasScenePresenter(
            self.scene,
            track_item=self._track_scene_item,
            untrack_item=self._untrack_scene_item,
        )
        self.view = LabelView(self.scene, self)

        # Main layout: reserve permanent tool space around a clean canvas.
        panel_style = sidebar_stylesheet()
        central.setStyleSheet(panel_style)
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.left_sidebar_content = QWidget()
        self.left_sidebar_content.setObjectName("SidebarContent")
        self.left_sidebar_layout = QVBoxLayout(self.left_sidebar_content)
        self.left_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.left_sidebar_layout.setSpacing(10)

        self.left_sidebar = QScrollArea()
        self.left_sidebar.setWidgetResizable(True)
        self.left_sidebar.setFrameShape(QFrame.Shape.NoFrame)
        self.left_sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_sidebar.setWidget(self.left_sidebar_content)
        self.left_sidebar.setFixedWidth(360)

        self.right_sidebar_content = QWidget()
        self.right_sidebar_content.setObjectName("SidebarContent")
        self.right_sidebar_layout = QVBoxLayout(self.right_sidebar_content)
        self.right_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.right_sidebar_layout.setSpacing(10)

        self.right_sidebar = QScrollArea()
        self.right_sidebar.setWidgetResizable(True)
        self.right_sidebar.setFrameShape(QFrame.Shape.NoFrame)
        self.right_sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_sidebar.setWidget(self.right_sidebar_content)
        self.right_sidebar.setFixedWidth(360)

        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.left_sidebar)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.right_sidebar)
        central.setLayout(layout)

        # Shared widgets/state
        self.annotation_panel = AnnotationPanel(
            self.classes,
            active_layer=self.active_layer,
            active_mode=self.mode,
            callbacks=AnnotationPanelCallbacks(
                mode_changed=self.set_mode,
                class_changed=self._on_class_changed,
                manage_classes=self.open_class_manager,
                use_segmentation_box=self._use_segmentation_box_for_pose,
            ),
            embedded=True,
            parent=self.left_sidebar_content,
        )
        self.mode_grid = self.annotation_panel.mode_grid
        self.panzoom_btn = self.annotation_panel.panzoom_btn
        self.bbox_btn = self.annotation_panel.bbox_btn
        self.segment_btn = self.annotation_panel.segment_btn
        self.seg_edit_btn = self.annotation_panel.seg_edit_btn
        self.keypoint_btn = self.annotation_panel.keypoint_btn
        self.predict_btn = self.annotation_panel.predict_btn
        self.use_segmentation_box_btn = self.annotation_panel.use_segmentation_box_btn
        self.class_controls_frame = self.annotation_panel.class_controls_frame
        self.class_label_widget = self.annotation_panel.class_label
        self.class_selector = self.annotation_panel.class_selector
        self.manage_classes_btn = self.annotation_panel.manage_classes_btn
        self.progress_label = self.annotation_panel.progress_label
        self._fit_class_selector_to_items()
        self._active_class_id = self.class_selector.currentIndex()
        self.navigation_panel = NavigationPanel(
            active_filter=self.nav_filter,
            active_layer=self.active_layer,
            layer_visibility=self.layer_visibility,
            callbacks=NavigationPanelCallbacks(
                filter_changed=self._set_nav_filter,
                layer_changed=self._switch_layer,
                visibility_changed=self._on_layer_visibility_changed,
                previous=self.prev_index,
                next=self.next_index,
                complete=self.complete_and_next_unlabeled,
                skip=self.skip_to_next_unlabeled,
                save=self.save_labels,
                delete_image=self.delete_current_image,
            ),
            embedded=True,
            parent=self.left_sidebar_content,
        )
        self.filter_combo = self.navigation_panel.filter_combo
        self.workflow_selector = self.navigation_panel.layer_selector
        self.keypoints_visibility_check = self.navigation_panel.keypoints_visibility_btn
        self.segmentation_visibility_check = self.navigation_panel.segmentation_visibility_btn
        self.depth_visibility_check = self.navigation_panel.depth_visibility_btn
        self.complete_btn = self.navigation_panel.complete_btn
        self.skip_btn = self.navigation_panel.skip_btn
        self.save_btn = self.navigation_panel.save_btn
        self.delete_image_btn = self.navigation_panel.delete_image_btn

        # -----------------------------
        # Top-left: navigation + labeling
        # -----------------------------
        self.top_left_frame = QFrame(self.left_sidebar_content)
        self.top_left_frame.setObjectName("ToolPanel")
        self.top_left_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.top_left_frame)
        top_left_layout = QVBoxLayout(self.top_left_frame)
        top_left_layout.setContentsMargins(10, 9, 10, 9)
        top_left_layout.setSpacing(6)

        top_left_title = QLabel("Navigation & Labeling")
        top_left_title.setObjectName("panelTitle")
        top_left_layout.addWidget(top_left_title)

        top_left_layout.addWidget(self.navigation_panel)
        top_left_layout.addWidget(self.annotation_panel)
        self.left_sidebar_layout.addWidget(self.top_left_frame)

        initial_depth_mode = (
            str(
                getattr(self, "layer_settings", {})
                .get(LAYER_DEPTH, {})
                .get("display_mode", "depth")
            )
            .strip()
            .lower()
        )
        if initial_depth_mode not in {"original", "depth", "overlay"}:
            initial_depth_mode = "depth"
        depth_display_callbacks = DepthDisplayCallbacks(
            mode_changed=lambda _mode: self._on_depth_view_changed(0),
            clear_probes=self._clear_depth_probes,
        )
        self.depth_display_frame = DepthDisplayPanel(
            mode=initial_depth_mode,
            callbacks=depth_display_callbacks,
            parent=self.left_sidebar_content,
        )
        self.depth_display_combo = self.depth_display_frame.mode_combo
        self.left_sidebar_layout.addWidget(self.depth_display_frame)

        self.depth_range_frame = DepthRangePanel(
            callbacks=depth_display_callbacks,
            parent=self.left_sidebar_content,
        )
        self.depth_range_label = self.depth_range_frame.range_label
        self.depth_probe_label = self.depth_range_frame.probe_label
        self.depth_clear_probes_btn = self.depth_range_frame.clear_btn
        self.left_sidebar_layout.addWidget(self.depth_range_frame)

        self.depth_assistant_frame = DepthModelPanel(
            DepthModelCallbacks(
                select_model=self._set_depth_model_path,
                choose_model=self._choose_depth_model_interactive,
            ),
            parent=self.left_sidebar_content,
        )
        self.depth_model_status_label = self.depth_assistant_frame.status_label
        self.depth_official_model_btn = self.depth_assistant_frame.official_model_btn
        self.depth_choose_model_btn = self.depth_assistant_frame.choose_model_btn
        self.depth_clear_model_btn = self.depth_assistant_frame.clear_model_btn
        self.left_sidebar_layout.addWidget(self.depth_assistant_frame)
        self._refresh_depth_assistant_controls()

        # -----------------------------
        # Top-right: video tools
        # -----------------------------
        operation_callbacks = OperationCallbacks(
            video_review=self.open_video_reviewer,
            analysis=self.open_analysis_dialog,
            validate_labels=self.normalize_labels_all,
            export_dataset=self.export_dataset,
            project_health=self.show_project_health,
            train=self.open_train_dialog,
            distill=self.open_distillation_dialog,
            project_models=self.load_model,
            inference=self.run_video_inference,
            apply_template=self.apply_template_for_current_class,
            save_template=self.save_template_for_current_class,
        )
        self.top_right_frame = VideoOperationsPanel(
            operation_callbacks.video_review,
            self.right_sidebar_content,
        )
        self.video_review_btn = self.top_right_frame.review_btn
        self.right_sidebar_layout.addWidget(self.top_right_frame)

        # -----------------------------
        # Right: layer-aware analysis
        # -----------------------------
        self.analysis_frame = AnalysisOperationsPanel(
            operation_callbacks.analysis,
            layer_id=self.active_layer,
            parent=self.right_sidebar_content,
        )
        self.analysis_title = self.analysis_frame.title_label
        self.analysis_btn = self.analysis_frame.analysis_btn

        # -----------------------------
        # Bottom-left: training tools
        # -----------------------------
        self.bottom_left_frame = DatasetOperationsPanel(
            operation_callbacks,
            layer_id=self.active_layer,
            parent=self.right_sidebar_content,
        )
        self.dataset_training_title = self.bottom_left_frame.title_label
        self.training_grid = self.bottom_left_frame.grid
        self.normalize_btn = self.bottom_left_frame.validate_btn
        self.export_dataset_btn = self.bottom_left_frame.export_btn
        self.project_health_btn = self.bottom_left_frame.health_btn
        self.train_btn = self.bottom_left_frame.train_btn
        self.distillation_btn = self.bottom_left_frame.distillation_btn

        # -----------------------------
        # Bottom-right: model + inference
        # -----------------------------
        self.bottom_right_frame = ModelOperationsPanel(
            operation_callbacks,
            layer_id=self.active_layer,
            parent=self.right_sidebar_content,
        )
        self.model_inference_title = self.bottom_right_frame.title_label
        self.model_status_label = self.bottom_right_frame.status_label
        self.inference_grid = self.bottom_right_frame.grid
        self.load_model_btn = self.bottom_right_frame.models_btn
        self.template_apply_btn = self.bottom_right_frame.apply_template_btn
        self.template_save_btn = self.bottom_right_frame.save_template_btn
        self.inference_btn = self.bottom_right_frame.inference_btn
        self.right_sidebar_layout.addWidget(self.bottom_right_frame)
        self.right_sidebar_layout.addWidget(self.bottom_left_frame)

        # -----------------------------
        # Bottom-left overlay: segmentation tools/help
        # -----------------------------
        self.seg_tools_frame = SegmentationToolsPanel(
            callbacks=SegmentationToolsCallbacks(
                load_model=self._load_sam_model_interactive,
                run=self._run_sam_segmentation,
                accept=self._accept_segmentation_preview,
                reset=self._clear_seg_prompt_state,
            ),
            brush_radius=self.seg_brush_radius,
            parent=self.left_sidebar_content,
        )
        self.seg_brush_size_label = self.seg_tools_frame.brush_size_label
        self.sam_load_btn = self.seg_tools_frame.load_btn
        self.sam_run_btn = self.seg_tools_frame.run_btn
        self.sam_accept_btn = self.seg_tools_frame.accept_btn
        self.sam_clear_btn = self.seg_tools_frame.reset_btn
        self.sam_helper_label = self.seg_tools_frame.helper_label
        self._refresh_seg_brush_size_badge()
        self.seg_tools_frame.hide()
        self.left_sidebar_layout.addWidget(self.seg_tools_frame)
        self.left_sidebar_layout.addStretch(1)
        self.right_sidebar_layout.addStretch(1)
        self.right_sidebar_layout.addWidget(self.analysis_frame)

        # reflect initial nav filter in the dropdown
        try:
            mapping = {"all": 0, "labeled": 1, "unlabeled": 2}
            self.filter_combo.setCurrentIndex(mapping.get(self.nav_filter, 0))
        except Exception:
            pass
        self.workflow_selector.setCurrentIndex(list(LAYER_DEFINITIONS).index(self.active_layer))

        self._layout_hot_corners()

        # Canvas presentation overlays. Compatibility aliases remain public while
        # construction, visibility, and hot-corner layout belong to the presenter.
        self.canvas_hud_presenter = CanvasHudPresenter(self.view)
        self.layer_context_frame = self.canvas_hud_presenter.layer_context
        self.layer_editing_label = self.layer_context_frame.editing_label
        self.layer_reference_label = self.layer_context_frame.reference_label
        self.legend_frame = self.canvas_hud_presenter.legend
        self.legend_title = self.legend_frame.title_label
        self.legend_label = self.legend_frame.legend_label
        self.zoom_frame = self.canvas_hud_presenter.zoom
        self.zoom_label = self.zoom_frame.value_label
        self.canvas_hud_presenter.show_context()
        self._refresh_layer_context_hud()

        # Status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # Shortcuts
        self._bind_shortcuts()

    # ---------- Class & annotation helpers ----------

    def _bind_pose_annotation_controller(self) -> Optional[PoseAnnotationController]:
        controller = getattr(self, "_pose_controller", None)
        if controller is None or not isinstance(self.annotation_cache, PoseAnnotationDocument):
            return None
        controller.bind_document(self.annotation_cache)
        controller.configure_schema(
            keypoint_order_for=self._kp_names_for_index,
            canonical_names=self.kp_names,
        )
        controller.state = self.pose_edit_state
        return controller

    def _bind_segmentation_annotation_controller(
        self,
    ) -> Optional[SegmentationAnnotationController]:
        controller = getattr(self, "_segmentation_controller", None)
        if controller is None or not isinstance(
            self.annotation_cache, SegmentationAnnotationDocument
        ):
            return None
        controller.bind_document(self.annotation_cache)
        controller.state = self.seg_edit_state
        return controller

    def _segmentation_box_transfer_plan(self):
        if not self._is_keypoints_layer():
            raise SegmentationBoxUnavailableError(
                "Switch to the Keypoints layer to use a segmentation box."
            )
        if not self.images or self.current_idx < 0 or self.current_idx >= len(self.images):
            raise SegmentationBoxUnavailableError("Load an image first.")
        class_id = self.class_selector.currentIndex()
        base = os.path.splitext(self.images[self.current_idx])[0]
        label_file = os.path.join(self.seg_label_dir, f"{base}.txt")
        segmentation_document = load_segmentation_document(
            label_file,
            classes_count=len(self.seg_classes),
            image_width=self.img_w,
            image_height=self.img_h,
        )
        return plan_segmentation_box_transfer(
            pose_class_id=class_id,
            pose_classes=self.pose_classes,
            segmentation_classes=self.seg_classes,
            segmentation_document=segmentation_document,
        )

    def _refresh_segmentation_box_action(self) -> None:
        panel = self.__dict__.get("annotation_panel")
        if panel is None:
            return
        try:
            plan = self._segmentation_box_transfer_plan()
        except SegmentationBoxUnavailableError as error:
            panel.set_segmentation_box_available(False, reason=str(error))
            return
        panel.set_segmentation_box_available(
            True,
            reason=f"Use saved segmentation bounds for '{plan.class_name}'.",
        )

    def _use_segmentation_box_for_pose(self) -> None:
        try:
            plan = self._segmentation_box_transfer_plan()
        except SegmentationBoxUnavailableError as error:
            QMessageBox.information(self, "Segmentation box unavailable", str(error))
            self._refresh_segmentation_box_action()
            return

        class_id = plan.pose_class_id
        state = self._sync_pose_state_from_scene(class_id)
        controller = self._bind_pose_annotation_controller()
        if controller is not None:
            controller.replace_box_preserving_keypoints(plan.box)
        else:
            state.push_undo_snapshot()
            state.replace_box(plan.box)
            self.annotation_cache.apply_edit_state(state, require_complete=True)
        self._render_pose_edit_state()
        self._update_status()
        self._update_item_editability()
        self.update_status_bar(
            f"Updated the {plan.class_name} keypoint box from segmentation; "
            "existing keypoints were preserved."
        )

    def _pose_state_for_class(self, class_id: int) -> PoseEditState:
        controller = self._bind_pose_annotation_controller()
        if controller is not None:
            controller.select_class(class_id)
            self.pose_edit_state = controller.state
            return self.pose_edit_state
        order = self._kp_names_for_index(class_id)
        entry = self.annotation_cache.get(class_id)
        state = self.pose_edit_state
        state.select_class(
            class_id,
            order,
            canonical_names=self.kp_names,
            entry=entry,
        )
        return state

    def _sync_pose_state_from_scene(self, class_id: int) -> PoseEditState:
        state = (
            self.pose_edit_state
            if class_id == self.class_selector.currentIndex()
            else PoseEditState()
        )
        state.active_class_id = class_id
        state.keypoint_order = list(self._kp_names_for_index(class_id))
        state.canonical_names = list(self.kp_names)
        box_item = self._class_box_item(class_id)
        if box_item is None:
            state.box = None
            state.keypoints = {}
            return state
        box_item.update_model()
        state.box = box_item.bbox
        items_by_name = {item.kp.name: item for item in self._class_keypoint_items(class_id)}
        state.keypoints = {
            name: KeypointEntry(
                name=name,
                display_name=name,
                kp=items_by_name[name].kp,
                visibility=int(getattr(items_by_name[name], "visibility", 2)),
            )
            for name in state.keypoint_order
            if name in items_by_name
        }
        return state

    def _sync_pose_legacy_mirrors(self) -> None:
        state = self.pose_edit_state
        self.current_class_id = state.active_class_id
        self.current_box = state.box
        self.bboxes = [state.box] if state.box is not None else []
        self.kps = [
            state.keypoints[name].kp for name in state.keypoint_order if name in state.keypoints
        ]
        self.current_kps = self.kps
        self.current_kp_idx = state.current_keypoint_index

    def _render_pose_edit_state(self) -> None:
        state = self.pose_edit_state
        cid = state.active_class_id
        if cid is None:
            self._sync_pose_legacy_mirrors()
            return
        self._clear_class_items(cid, drop_cache=False)
        if state.box is not None:
            item = BoxItem(
                state.box,
                self.classes[cid] if cid < len(self.classes) else str(cid),
            )
            self.scene.addItem(item)
            self._track_scene_item(item)
        for name in state.keypoint_order:
            entry = state.keypoints.get(name)
            if entry is None:
                continue
            kp_item = KeypointItem(entry.kp, self.kp_pixel_radius, self.kp_font_px)
            kp_item.visibility = int(entry.visibility)
            kp_item.update_appearance()
            self.scene.addItem(kp_item)
            self._track_scene_item(kp_item)
        self._sync_pose_legacy_mirrors()

    def _store_pose_edit_state(self, *, require_complete: bool = True) -> bool:
        controller = self._bind_pose_annotation_controller()
        if controller is not None:
            return controller.commit(require_complete=require_complete)
        annotation = self.annotation_cache.apply_edit_state(
            self.pose_edit_state,
            require_complete=require_complete,
        )
        return annotation is not None

    def _set_segmentation_cache_entry(self, class_id: int, entry: dict[str, object]) -> None:
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            points = entry.get("segments", [])
            score = float(entry.get("score", 0.0) or 0.0)
            controller.upsert_polygon(
                class_id,
                points,
                score=score,
                record_undo=False,
            )
            return
        stored = self.seg_edit_state.set_accepted_entry(class_id, entry)
        self.annotation_cache[int(class_id)] = stored

    def _drop_segmentation_cache_entry(self, class_id: int) -> None:
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            controller.remove_mask(class_id, record_undo=False)
            return
        self.seg_edit_state.clear_accepted_mask(class_id)
        self.annotation_cache.pop(int(class_id), None)

    def _sync_segmentation_state_from_cache(self) -> None:
        entries = {int(class_id): dict(entry) for class_id, entry in self.annotation_cache.items()}
        selected = self.class_selector.currentIndex() if hasattr(self, "class_selector") else -1
        controller = self._bind_segmentation_annotation_controller()
        if controller is not None:
            controller.replace_document(
                entries,
                selected_target=selected if selected >= 0 else None,
            )
            self.seg_edit_state = controller.state
            return
        self.seg_edit_state.reset(
            accepted_masks=entries,
            selected_target=selected if selected >= 0 else None,
        )

    def _on_class_changed(self, index: int):
        if index < 0 or index >= len(self.classes):
            session = getattr(self, "_project_session", None)
            if session is not None:
                session.select_class(-1)
            controller = self._bind_segmentation_annotation_controller()
            if controller is not None:
                controller.select_target(None)
            else:
                self.seg_edit_state.select_target(None)
            return
        prev = getattr(self, "_active_class_id", index)
        if prev != index and self._is_pose_workflow():
            self._cache_active_annotation(prev)
        self._active_class_id = index
        session = getattr(self, "_project_session", None)
        if session is not None:
            session.select_class(index)
        if self._is_seg_workflow():
            controller = self._bind_segmentation_annotation_controller()
            if controller is not None:
                controller.select_target(index, clear_prompts=False)
            else:
                self.seg_edit_state.select_target(index)
            self._clear_seg_prompt_state()
            has_item = self._class_seg_mask_item(index) is not None
        else:
            self._pose_state_for_class(index)
            has_item = self._class_box_item(index) is not None
        if self.annotation_cache.get(index) and not has_item:
            self._restore_annotation_for_class(index)
        else:
            self._sync_active_class_state()
            self._update_item_editability()
            self._update_status()
        self._clear_seg_edit_handles()
        self._refresh_sam_controls()
        self._refresh_segmentation_box_action()

    def _cache_active_annotation(self, class_id: Optional[int] = None) -> bool:
        if not self.images:
            return False
        cid = self.class_selector.currentIndex() if class_id is None else class_id

        if self._is_seg_workflow():
            seg_item = self._class_seg_mask_item(cid)
            if seg_item is not None:
                points = self._extract_seg_item_points(seg_item)
                if len(points) >= 3:
                    self._set_segmentation_cache_entry(
                        cid,
                        {
                            "class_id": cid,
                            "segments": [(float(x), float(y)) for x, y in points],
                        },
                    )
                    return True
                self._drop_segmentation_cache_entry(cid)
                return False
            entry = self.annotation_cache.get(cid, {})
            if len(entry.get("segments", [])) >= 3:
                self.seg_edit_state.set_accepted_entry(cid, dict(entry))
                return True
            return False

        state = self._sync_pose_state_from_scene(cid)
        if state.box is None:
            self.annotation_cache.delete_annotation(cid)
            if state is self.pose_edit_state:
                self._sync_pose_legacy_mirrors()
            return False
        annotation = self.annotation_cache.apply_edit_state(state, require_complete=True)
        if state is self.pose_edit_state:
            self._sync_pose_legacy_mirrors()
        return annotation is not None

    def _restore_annotation_for_class(self, cid: int):
        self._clear_class_items(cid)
        entry = self.annotation_cache.get(cid)
        if not entry:
            if cid == self.class_selector.currentIndex():
                self._sync_active_class_state()
                self._update_item_editability()
            return

        if self._is_seg_workflow():
            self.seg_edit_state.set_accepted_entry(cid, dict(entry))
            seg_points_raw = entry.get("segments", [])
            seg_points: list[tuple[float, float]] = []
            for pair in seg_points_raw:
                try:
                    seg_points.append((float(pair[0]), float(pair[1])))
                except Exception:
                    continue
            if len(seg_points) >= 3:
                self._add_seg_mask_item(cid, seg_points, preview=False)
            if cid == self.class_selector.currentIndex():
                self._sync_active_class_state()
                self._update_status()
                self._update_item_editability()
            return

        state = self.annotation_cache.to_edit_state(
            cid,
            self._kp_names_for_index(cid),
            canonical_names=self.kp_names,
        )
        if cid == self.class_selector.currentIndex():
            self.pose_edit_state.select_class(
                cid,
                state.keypoint_order,
                canonical_names=state.canonical_names,
                entry=entry,
            )
            state = self.pose_edit_state
        bbox = state.box
        if bbox is None:
            return
        item = BoxItem(bbox, self.classes[cid] if cid < len(self.classes) else str(cid))
        self.scene.addItem(item)
        self._track_scene_item(item)

        for name in state.keypoint_order:
            kp_entry = state.keypoints.get(name)
            if kp_entry is None:
                continue
            kp_item = KeypointItem(kp_entry.kp, self.kp_pixel_radius, self.kp_font_px)
            kp_item.visibility = int(kp_entry.visibility)
            kp_item.update_appearance()
            self.scene.addItem(kp_item)
            self._track_scene_item(kp_item)

        if cid == self.class_selector.currentIndex():
            self._sync_active_class_state()
            self._update_status()
            self._update_item_editability()

    def _load_annotations_from_file(self, label_file: str) -> dict[int, dict]:
        if self._is_seg_workflow():
            return self._load_seg_annotations_from_file(label_file)
        class_lookup = [self._kp_names_for_index(i) for i in range(len(self.classes))]
        result = load_pose_document(
            label_file,
            classes_count=len(self.classes),
            canonical_names=self.kp_names,
            class_keypoint_lookup=class_lookup,
            image_width=self.img_w,
            image_height=self.img_h,
        )
        if result.extra_keypoint_rows > 0:
            print(
                "⚠️ Ignored extra keypoint values in "
                f"{result.extra_keypoint_rows} row(s) while reading {label_file}",
                file=sys.stderr,
            )
        return result.document.snapshot()

    def _load_seg_annotations_from_file(self, label_file: str) -> dict[int, dict]:
        return load_segmentation_document(
            label_file,
            classes_count=len(self.classes),
            image_width=self.img_w,
            image_height=self.img_h,
        ).snapshot()

    def _parse_label_line(self, line: str) -> tuple[Optional[dict], bool]:
        class_lookup = [self._kp_names_for_index(i) for i in range(len(self.classes))]
        return parse_pose_label_line(
            line,
            classes_count=len(self.classes),
            canonical_names=self.kp_names,
            class_keypoint_lookup=class_lookup,
            img_w=self.img_w,
            img_h=self.img_h,
        )

    def _parse_segmentation_line(self, line: str) -> Optional[dict]:
        return parse_segmentation_label_line(
            line,
            classes_count=len(self.classes),
            img_w=self.img_w,
            img_h=self.img_h,
        )

    def _segmentation_entry_to_line(self, entry: dict) -> str:
        return segmentation_annotation_to_line(entry, img_w=self.img_w, img_h=self.img_h)

    def _annotation_entry_to_line(self, entry: dict) -> str:
        if self._is_seg_workflow():
            return self._segmentation_entry_to_line(entry)
        return pose_annotation_to_line(
            entry, kp_names=self.kp_names, img_w=self.img_w, img_h=self.img_h
        )

    def _render_overlay_from_cache(self, out_path: str) -> bool:
        if self.img_w <= 0 or self.img_h <= 0:
            return False
        rect = QRectF(0, 0, self.img_w, self.img_h)
        render_w = int(rect.width())
        render_h = int(rect.height())

        # Build a background from the currently loaded source image when possible.
        base_pix = QPixmap()
        if self.current_image_path:
            base_pix = QPixmap(self.current_image_path)
        if base_pix.isNull() and self.images:
            file_name = self.images[self.current_idx]
            for cand in (
                os.path.join(self.active_image_dir, file_name),
                os.path.join(self.image_dir_queue, file_name),
                os.path.join(self.image_dir_all, file_name),
            ):
                if not os.path.exists(cand):
                    continue
                probe = QPixmap(cand)
                if not probe.isNull():
                    base_pix = probe
                    break
        if base_pix.isNull():
            for item in self.scene.items():
                if isinstance(item, QGraphicsPixmapItem):
                    probe = item.pixmap()
                    if not probe.isNull():
                        base_pix = probe
                        break

        pm = QPixmap(render_w, render_h)
        pm.fill(Qt.GlobalColor.transparent)
        if not base_pix.isNull():
            bg = QPainter(pm)
            bg.drawPixmap(0, 0, render_w, render_h, base_pix)
            bg.end()

        painter = QPainter(pm)
        colors = [
            Qt.GlobalColor.cyan,
            Qt.GlobalColor.magenta,
            Qt.GlobalColor.yellow,
            Qt.GlobalColor.green,
            Qt.GlobalColor.red,
        ]
        try:
            for idx, (cid, entry) in enumerate(sorted(self.annotation_cache.items())):
                color = colors[idx % len(colors)]
                pen = QPen(color)
                pen.setWidth(2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                if self._is_seg_workflow():
                    seg = entry.get("segments", [])
                    if len(seg) < 3:
                        continue
                    path = self._polygon_path([(float(p[0]), float(p[1])) for p in seg])
                    if path is None:
                        continue
                    fill = QColor(color)
                    fill.setAlpha(85)
                    painter.setBrush(QBrush(fill))
                    painter.drawPath(path)
                    painter.setBrush(Qt.GlobalColor.transparent)
                    class_name = self._seg_class_name(cid)
                    label_pt = path.boundingRect().topLeft() + QPointF(4.0, 14.0)
                    painter.setPen(QPen(color))
                    painter.drawText(label_pt, class_name)
                else:
                    painter.setBrush(Qt.GlobalColor.transparent)
                    bbox = entry.get("bbox", {})
                    painter.drawRect(
                        QRectF(
                            bbox.get("x", 0.0),
                            bbox.get("y", 0.0),
                            bbox.get("w", 0.0),
                            bbox.get("h", 0.0),
                        )
                    )
                    for kp in entry.get("keypoints", []):
                        vis = int(kp.get("vis", 2))
                        if vis == 0:
                            painter.setBrush(QBrush(Qt.GlobalColor.transparent))
                            painter.setPen(QPen(Qt.GlobalColor.lightGray))
                        elif vis == 1:
                            painter.setBrush(QBrush(color))
                            pen = QPen(color)
                            pen.setStyle(Qt.PenStyle.DashLine)
                            painter.setPen(pen)
                        else:
                            painter.setBrush(QBrush(color))
                            painter.setPen(QPen(color))
                        painter.drawEllipse(QPointF(kp.get("x", 0.0), kp.get("y", 0.0)), 3, 3)
                    painter.setPen(pen)
        finally:
            painter.end()
        try:
            if not pm.save(out_path):
                raise OSError(f"Qt could not encode overlay image: {out_path}")
            print(f"✅ Saved annotated image to {out_path}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save annotated image: {e}")
            return False

    # ---------- Navigation helpers ----------

    def _find_next_unlabeled(self, start_from: int) -> int:
        """Return index of next frame without a label file. If none, returns current index."""
        return next_unlabeled_index(
            self.images_queue,
            start_from,
            self.label_dir,
            label_is_usable=self._label_file_is_usable,
        )

    # ---------- Navigation filtering ----------
    def _is_labeled_index(self, idx: int) -> bool:
        return self._label_file_is_usable(image_label_path(self.label_dir, self.images[idx]))

    def _filtered_indices(self) -> list[int]:
        navigator = self._sync_image_queue_navigator()
        return list(
            navigator.selection(
                self.label_dir,
                label_is_usable=self._label_file_is_usable,
            ).matching_indices
        )

    def _sync_image_queue_navigator(self) -> ImageQueueNavigator:
        navigator = getattr(self, "_image_queue_navigator", None)
        if navigator is None:
            navigator = ImageQueueNavigator()
            self._image_queue_navigator = navigator
        navigator.synchronize(
            self.images,
            current_index=self.current_idx,
            filter_mode=self.nav_filter,
        )
        return navigator

    def _apply_image_queue_selection(self, current_index: int) -> None:
        self.current_idx = current_index
        self._queue_current_idx = current_index

    def _set_nav_filter(self, mode: str):
        if mode not in ("all", "labeled", "unlabeled"):
            return
        self.nav_filter = mode
        self.images = self.images_queue[:]
        self.active_image_dir = self.image_dir_queue
        if not self.images:
            self.update_status_bar("No images available in the current batch.")
            return
        if self._queue_current_idx >= len(self.images):
            self._queue_current_idx = max(0, len(self.images) - 1)
        self.current_idx = self._queue_current_idx

        selection = self._sync_image_queue_navigator().set_filter(
            mode,
            self.label_dir,
            label_is_usable=self._label_file_is_usable,
        )
        fi = list(selection.matching_indices)
        if not selection.has_match:
            self.update_status_bar(f"No images match filter: {mode}.")
            return
        self._apply_image_queue_selection(selection.current_index)
        self.update_status_bar(f"Browsing: {mode} ({fi.index(self.current_idx) + 1}/{len(fi)})")
        self.load_image()

    def prev_index(self):
        selection = self._sync_image_queue_navigator().move(
            -1,
            self.label_dir,
            label_is_usable=self._label_file_is_usable,
        )
        if not selection.has_match:
            self.update_status_bar("No images found for current filter.")
            return
        self._apply_image_queue_selection(selection.current_index)
        self.mode = (
            "segment"
            if self._is_seg_workflow()
            else ("panzoom" if self._is_depth_layer() else "bbox")
        )
        self.load_image()

    def next_index(self):
        selection = self._sync_image_queue_navigator().move(
            1,
            self.label_dir,
            label_is_usable=self._label_file_is_usable,
        )
        if not selection.has_match:
            self.update_status_bar("No images found for current filter.")
            return
        self._apply_image_queue_selection(selection.current_index)
        self.mode = (
            "segment"
            if self._is_seg_workflow()
            else ("panzoom" if self._is_depth_layer() else "bbox")
        )
        self.load_image()

    def complete_and_next_unlabeled(self):
        if LabelingApp._is_depth_layer(self):
            if not self._is_fully_labeled():
                QMessageBox.information(
                    self,
                    "No Depth Map",
                    "Run Depth prediction before moving to the next image without a map.",
                )
                return
            self.skip_to_next_unlabeled()
            return
        if self._is_seg_workflow():
            self._cache_active_annotation()
            has_any_mask = any(
                len(entry.get("segments", [])) >= 3 for entry in self.annotation_cache.values()
            )
            if not has_any_mask:
                QMessageBox.information(
                    self,
                    "Incomplete",
                    "Add and accept at least one segmentation mask before completing this frame.",
                )
                return
            if self.seg_preview_points:
                QMessageBox.information(
                    self,
                    "Pending preview",
                    "Accept the current SAM preview mask before completing this frame.",
                )
                return
            if not self.save_labels():
                return
            next_idx = self._find_next_unlabeled(self.current_idx)
            if next_idx == self.current_idx:
                popup = CongratsPopup()
                popup.exec()
                return
            self.current_idx = next_idx
            self._queue_current_idx = self.current_idx
            self.mode = "segment"
            self.load_image()
            return
        if not self._is_fully_labeled():
            QMessageBox.information(
                self,
                "Incomplete",
                "Place one bounding box and all keypoints to complete this frame.",
            )
            return
        if not self.save_labels():
            return
        next_idx = self._find_next_unlabeled(self.current_idx)
        if next_idx == self.current_idx:
            popup = CongratsPopup()
            popup.exec()
            return
        self.current_idx = next_idx
        self._queue_current_idx = self.current_idx
        self.mode = (
            "segment"
            if self._is_seg_workflow()
            else ("panzoom" if self._is_depth_layer() else "bbox")
        )
        self.load_image()

    def skip_to_next_unlabeled(self):
        navigator = self._sync_image_queue_navigator()
        next_idx = navigator.move_to_next_unlabeled(
            self.label_dir,
            label_is_usable=self._label_file_is_usable,
        )
        if next_idx == self.current_idx:
            popup = CongratsPopup()
            popup.exec()
            return
        self._apply_image_queue_selection(next_idx)
        self.mode = (
            "segment"
            if self._is_seg_workflow()
            else ("panzoom" if self._is_depth_layer() else "bbox")
        )
        self.load_image()

    def _image_deletion_plan(self, image_name: str) -> ImageDeletionPlan:
        state = getattr(self, "__dict__", {})
        return plan_image_deletion(
            project_root=self.project_root,
            image_name=image_name,
            active_image_dir=getattr(self, "active_image_dir", ""),
            image_dir_queue=getattr(self, "image_dir_queue", ""),
            image_dir_all=getattr(self, "image_dir_all", ""),
            pose_label_dir=getattr(self, "pose_label_dir", ""),
            seg_label_dir=getattr(self, "seg_label_dir", ""),
            depth_image_dir=state.get("depth_image_dir", ""),
            depth_preview_dir=state.get("depth_preview_dir", ""),
        )

    def _delete_planned_image_files(
        self, paths: list[str] | tuple[str, ...]
    ) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        errors: list[str] = []
        for path in paths:
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                errors.append(f"{path}: expected a file but found a directory")
                continue
            try:
                os.remove(path)
                removed.append(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        return removed, errors

    def _clear_loaded_image_state(self):
        self.current_image_path = ""
        self.annotation_cache.clear()
        self.seg_edit_state.reset()
        self.bboxes.clear()
        self.kps.clear()
        self.current_kp_idx = 0
        depth_presenter = self.__dict__.get("_depth_preview_presenter")
        if depth_presenter is not None:
            depth_presenter.clear()
        self._scene_presenter.forget_scene_items()
        self.scene.clear()
        self._item_refs.clear()

    def _reload_after_current_image_delete(self):
        self._queue_current_idx = self.current_idx

        self._refresh_queue_images()
        self._update_progress_label()

        self.images = self.images_queue[:]
        self.active_image_dir = self.image_dir_queue
        if self._queue_current_idx >= len(self.images):
            self._queue_current_idx = max(0, len(self.images) - 1)
        self.current_idx = self._queue_current_idx

        fi = self._filtered_indices()
        if fi and self.current_idx not in fi:
            self.current_idx = fi[0]
            self._queue_current_idx = self.current_idx
        if self.images and fi:
            self.load_image()
        else:
            self._clear_loaded_image_state()
            if not self.images:
                self.update_status_bar("No images available in the current batch.")
            else:
                self.update_status_bar(f"No images match filter: {self.nav_filter}.")

    def delete_current_image(self):
        if not self.images:
            QMessageBox.information(self, "No image", "No current image is loaded.")
            return

        file_name = os.path.basename(self.images[self.current_idx])
        conflicting_names = image_stem_conflicts(
            file_name,
            (self.image_dir_queue, self.image_dir_all),
        )
        if conflicting_names:
            QMessageBox.warning(
                self,
                "Duplicate Image Name",
                f"Cannot safely delete '{file_name}' because another project image shares its label stem:\n\n"
                f"{', '.join(conflicting_names)}\n\n"
                "Rename the conflicting image first.",
            )
            return

        deletion_plan = LabelingApp._image_deletion_plan(self, file_name)
        existing_paths = [path for path in deletion_plan.paths if os.path.exists(path)]
        if not existing_paths:
            QMessageBox.information(
                self,
                "Image Not Found",
                f"'{file_name}' was not found in the current project.",
            )
            return

        decision = QMessageBox.question(
            self,
            "Delete Image",
            f"Are you sure you want to delete '{file_name}'?\n\n"
            "This permanently removes the current image from the project browser. "
            "If matching labels, annotated previews, or generated dataset train/val "
            "copies exist, those are removed too.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return

        removed, errors = self._delete_planned_image_files(deletion_plan.paths)
        self._reload_after_current_image_delete()

        if errors:
            QMessageBox.warning(
                self,
                "Delete Error",
                "Some files could not be deleted:\n\n" + "\n".join(errors[:8]),
            )
        elif removed:
            self.update_status_bar(f"Deleted '{file_name}'.")

    # ---------- Prediction ----------

    def _prediction_controller_busy_changed(self, busy: bool) -> None:
        self._predict_busy = bool(busy)
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(not busy)

    def _handle_prediction_controller_decision(self, decision) -> None:
        if decision.action == "background_error":
            return
        if decision.action == "cancel":
            self._cleanup_prediction_depth_staging()
            self.update_status_bar("Prediction canceled.")
            return
        if decision.action == "error":
            self._cleanup_prediction_depth_staging()
            self._on_predict_error(decision.error_message)
            return
        if decision.action == "discard":
            self._cleanup_prediction_depth_staging()
            self.update_status_bar("Prediction finished for a different image and was discarded.")
            return
        if decision.action == "apply" and decision.prediction is not None:
            self._apply_prediction_payload(decision.prediction)

    def _inference_controller_busy_changed(self, busy: bool) -> None:
        if busy:
            self._inference_previous_busy = self._predict_busy
            self._predict_busy = True
        else:
            self._predict_busy = getattr(self, "_inference_previous_busy", False)
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(not busy)
        if hasattr(self, "inference_btn"):
            self.inference_btn.setEnabled(not busy)

    def _inference_controller_job_started(self, job: InferenceJobPlan) -> None:
        progress = QProgressDialog(
            f"Pass {job.job_index}/{job.job_total}: running {job.display_name} inference…",
            "Cancel",
            0,
            0 if job.total_frames <= 0 else job.total_frames,
            self,
        )
        progress.setWindowTitle("Project Video Inference")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        if job.total_frames <= 0:
            progress.setRange(0, 0)
        progress.canceled.connect(self._cancel_inference_process)
        self._inference_progress = progress
        progress.show()

    def _inference_controller_progress(self, job: InferenceJobPlan, event: dict) -> None:
        progress = self._inference_progress
        if progress is None:
            return
        processed = int(event.get("processed_frames") or 0)
        total = int(event.get("total_frames") or 0)
        if total > 0:
            progress.setValue(min(processed, total))
        detail = str(event.get("message") or f"Inferencing frame {processed}")
        progress.setLabelText(
            f"Pass {job.job_index}/{job.job_total} · {job.display_name}\n{detail}"
        )
        QApplication.processEvents()

    def _inference_controller_pass_finished(self, result: InferencePassResult) -> None:
        if self._inference_progress is not None:
            self._inference_progress.close()
        self._inference_progress = None

    def _inference_controller_completed(self, summary: InferenceRunSummary) -> None:
        if not summary.results:
            return
        message = "\n\n".join(summary.details)
        if summary.failed_count:
            QMessageBox.warning(self, "Project Inference Finished", message)
        else:
            QMessageBox.information(self, "Project Inference Complete", message)

    def _refresh_depth_assistant_controls(self) -> None:
        label = getattr(self, "depth_model_status_label", None)
        if label is None:
            return
        path = str(getattr(self, "layer_model_paths", {}).get(LAYER_DEPTH) or "")
        if not path:
            text = "No depth model selected."
            tooltip = ""
        elif self._is_builtin_model_reference(path):
            text = f"{os.path.basename(path)} · official model; downloads on first use"
            tooltip = path
        else:
            text = f"Custom model · {os.path.basename(path)}"
            tooltip = path
        panel = getattr(self, "__dict__", {}).get("depth_assistant_frame")
        if isinstance(panel, DepthModelPanel):
            panel.set_model_status(text, tooltip=tooltip, can_clear=bool(path))
            return
        label.setText(text)
        label.setToolTip(tooltip)
        clear_button = getattr(self, "depth_clear_model_btn", None)
        if clear_button is not None:
            clear_button.setEnabled(bool(path))

    def _set_depth_model_path(self, path: str) -> None:
        normalized = str(path or "")
        if normalized and not self._is_builtin_model_reference(normalized):
            normalized = os.path.abspath(normalized)
        self.layer_model_paths[LAYER_DEPTH] = normalized
        if self._is_depth_layer():
            self.predict_model_path = normalized or None
        self._save_project_preferences()
        self._device = _auto_device()
        if self._is_depth_layer():
            self._restart_prediction_worker(warm=bool(normalized))
        self._refresh_depth_assistant_controls()
        if normalized:
            self.update_status_bar(
                f"Depth assistant model selected: {os.path.basename(normalized)}"
            )
        else:
            self.update_status_bar("Depth assistant model cleared.")

    def _choose_depth_model_interactive(self) -> None:
        current = str(self.layer_model_paths.get(LAYER_DEPTH) or "")
        start_dir = (
            os.path.dirname(current)
            if current and not self._is_builtin_model_reference(current)
            else self.project_root
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select depth assistant model",
            start_dir,
            "Depth Model Files (*.pt *.yaml *.onnx)",
        )
        if path:
            self._set_depth_model_path(path)

    def load_model(self):
        dialog = ProjectModelsDialog(
            self,
            self.layer_model_paths,
            active_layer=self.active_layer,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.layer_model_paths.update(dialog.model_paths)
            self.predict_model_path = self.layer_model_paths.get(self.active_layer) or None
            self._save_project_preferences()
            # Re-detect device in case hardware/availability changed
            self._device = _auto_device()
            print(f"🧠 Inference device: {self._device}")
            self._restart_prediction_worker(warm=bool(self.predict_model_path))
            self._update_layer_ui_state()
            configured = [
                layer_definition(layer_id).display_name
                for layer_id in (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
                if self.layer_model_paths.get(layer_id)
            ]
            summary = ", ".join(configured) if configured else "none"
            self.update_status_bar(f"Project prediction models configured: {summary}.")
        except Exception as e:
            QMessageBox.warning(self, "Model Load Error", f"Could not load model:\n{e}")

    def run_video_inference(self):
        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies."
            )
            return
        if getattr(self, "_inference_coordinator", None) is not None and (
            self._inference_coordinator.is_busy
        ):
            QMessageBox.information(
                self, "Inference Running", "An inference process is already running."
            )
            return
        configured_layers = configured_inference_layers(
            self.active_layer,
            self.layer_model_paths,
        )
        if not configured_layers:
            if self._is_depth_layer():
                QMessageBox.information(
                    self,
                    "No Depth Assistant Model",
                    "Select an official or custom model in the Depth Assistant panel first.",
                )
                self.update_status_bar("Select a model in the Depth Assistant panel first.")
                return
            QMessageBox.information(
                self,
                "No Project Models",
                "Configure a Keypoints or Segmentation prediction model first.",
            )
            self.load_model()
            configured_layers = configured_inference_layers(
                self.active_layer,
                self.layer_model_paths,
            )
            if not configured_layers:
                return

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video for inference",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.mpg *.mpeg);;All Files (*)",
        )
        if not video_path:
            return

        metadata = probe_video_metadata(video_path, _cv2)
        if not metadata.opened:
            QMessageBox.warning(self, "Video Error", f"Unable to open video:\n{video_path}")
            return

        device_name = str(getattr(self, "_device", "cpu")).lower()
        default_batch = 16 if device_name in {"cuda", "mps"} else 4
        batch_size, ok = QInputDialog.getInt(
            self,
            "Batch Size",
            "Frames per batch (larger uses more VRAM/RAM but speeds up inference):",
            value=max(1, default_batch),
            min=1,
            max=256,
        )
        if not ok:
            return

        try:
            plan = plan_inference_run(
                project_root=self.project_root,
                video_path=video_path,
                active_layer=self.active_layer,
                model_paths=self.layer_model_paths,
                pose_classes=self.pose_classes,
                segmentation_classes=self.seg_classes,
                keypoint_names=self.pose_kp_names,
                device=str(getattr(self, "_device", "cpu")),
                batch_size=batch_size,
                total_frames=metadata.total_frames,
                fps=metadata.fps,
            )
            prepare_inference_run(plan)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Output Error",
                f"Could not prepare project inference outputs.\n\n{e}",
            )
            return

        self._inference_coordinator.start(plan)

    def _cancel_inference_process(self) -> None:
        coordinator = getattr(self, "_inference_coordinator", None)
        if coordinator is not None and coordinator.is_busy:
            if coordinator.cancel() and self._inference_progress is not None:
                self._inference_progress.setLabelText("Canceling inference process…")

    @staticmethod
    def _discard_inference_outputs(result: InferencePassResult) -> None:
        for path in result.discard_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def set_mode(self, mode: str):
        if self._is_depth_layer() and mode not in {"panzoom", "predict"}:
            self.update_status_bar("The Depth layer supports Pan/Zoom and Predict modes.")
            return
        if self._is_seg_workflow() and mode in {"bbox", "keypoint"}:
            self.update_status_bar(
                "The Segmentation layer uses Segment Prompt (2), Edit Mask (E), and Predict (4) modes."
            )
            return
        if self._is_pose_workflow() and mode in {"segment", "segedit"}:
            self.update_status_bar(
                "Segment Prompt/Edit Mask modes are only available in the Segmentation layer."
            )
            return

        if mode == "predict":
            if not self.predict_model_path:
                if self._is_depth_layer():
                    self.update_status_bar("Select a model in the Depth Assistant panel first.")
                    return
                self.load_model()
                if not self.predict_model_path:
                    self.update_status_bar(
                        f"No {self._workflow_label()} prediction model configured."
                    )
                    return
            if not self.images:
                self.update_status_bar("No images to predict.")
                return
            if self._predict_busy:
                self.update_status_bar("Prediction already running...")
                return
            self.run_prediction_on_current_image()
            return

        if mode != "segedit":
            self.seg_brush_active = False
            self._clear_seg_edit_handles()

        self.mode = mode
        self._update_status()

        if hasattr(self.view, "_remove_crosshairs"):
            self.view._remove_crosshairs()

        if self.mode == "panzoom":
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.mode == "bbox":
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, "draw_crosshairs_at"):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == "keypoint":
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, "draw_crosshairs_at"):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == "segment":
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, "draw_crosshairs_at"):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == "segedit":
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
            self._clear_seg_edit_handles()
        if hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()

    def run_prediction_on_current_image(self):
        if not self.predict_model_path or not self.images:
            return
        img_path = os.path.join(self.active_image_dir, self.images[self.current_idx])

        if self._predict_busy:
            self.update_status_bar("Prediction already running...")
            return

        self._cleanup_prediction_depth_staging()
        depth_targets = None
        if self._is_depth_layer():
            base = os.path.splitext(os.path.basename(img_path))[0]
            final_targets = {
                "map": os.path.join(self.depth_image_dir, f"{base}.npy"),
                "preview": os.path.join(self.depth_preview_dir, f"{base}_depth.png"),
                "metadata": os.path.join(self.depth_image_dir, f"{base}_depth.json"),
            }
            staged_targets: dict[str, str] = {}
            try:
                for key, path in final_targets.items():
                    staged_targets[key] = staging_path_for(path)
            except Exception as exc:
                for path in staged_targets.values():
                    try:
                        remove_path(path)
                    except OSError:
                        pass
                self._on_predict_error(f"Could not prepare depth-map output files: {exc}")
                return
            self._prediction_depth_targets = {
                **{f"final_{key}": path for key, path in final_targets.items()},
                **{f"staged_{key}": path for key, path in staged_targets.items()},
            }
            depth_targets = DepthPredictionTargets.from_mapping(self._prediction_depth_targets)

        try:
            self._prediction_coordinator.submit_prediction(
                layer_id=self.active_layer,
                model_path=self.predict_model_path,
                image_path=img_path,
                device=self._device,
                depth_targets=depth_targets,
            )
        except Exception as exc:
            self._cleanup_prediction_depth_staging()
            self._on_predict_error(str(exc) or "Could not prepare prediction request")
            return

    def _cleanup_prediction_depth_staging(self) -> None:
        targets = getattr(self, "_prediction_depth_targets", None) or {}
        for key, path in targets.items():
            if key.startswith("staged_") and path:
                try:
                    remove_path(path)
                except OSError:
                    pass
        self._prediction_depth_targets = None

    def _displayed_image_path(self) -> str:
        images = getattr(self, "images", []) or []
        current_idx = int(getattr(self, "current_idx", 0) or 0)
        if current_idx < 0 or current_idx >= len(images):
            return ""
        return os.path.abspath(
            os.path.join(getattr(self, "active_image_dir", ""), images[current_idx])
        )

    def _restart_prediction_worker(self, *, warm: bool = False) -> None:
        self._cleanup_prediction_depth_staging()
        self._prediction_coordinator.restart_model(
            layer_id=self.active_layer,
            model_path=self.predict_model_path or "",
            device=self._device,
            warm=warm,
        )

    def _cancel_prediction_process(self) -> None:
        self._prediction_coordinator.cancel()

    def _apply_prediction_payload(self, prediction: dict):
        try:
            active_layer = validate_prediction_identity(
                prediction,
                expected_layer=self.active_layer,
            )
            if active_layer == LAYER_DEPTH:
                depth_targets = DepthPredictionTargets.from_mapping(
                    self._prediction_depth_targets or {}
                )
                application = plan_prediction_application(
                    prediction,
                    expected_layer=active_layer,
                    depth_targets=depth_targets,
                )
                commit_staged_paths(list(application.depth.replacements))
                self._prediction_depth_targets = None
                self._clear_depth_probes()
                self.load_image()
                self._update_progress_label()
                median = application.depth.median_depth
                suffix = f" Median estimated depth: {median:.3f} m." if median is not None else ""
                self.update_status_bar(
                    "Depth map saved and displayed (model-default scale)." + suffix
                )
                return
            self._cache_active_annotation()

            active_cid = self.class_selector.currentIndex()
            application = plan_prediction_application(
                prediction,
                expected_layer=active_layer,
                class_names=self.classes,
                canonical_keypoints=self.kp_names,
                class_keypoints=self.class_keypoints,
                active_class_id=active_cid,
            )
            if application.outcome == "no_detections":
                self.update_status_bar("Prediction returned no detections.")
                return
            if application.outcome == "no_usable_detections":
                self.update_status_bar("Prediction returned no usable detections.")
                return

            if active_layer == LAYER_SEGMENTATION:
                for planned_mask in application.segmentation:
                    cid = planned_mask.class_id
                    seg_points = list(planned_mask.points)
                    self._clear_class_items(cid, drop_cache=False)
                    self._set_segmentation_cache_entry(
                        cid,
                        {
                            "class_id": cid,
                            "segments": seg_points,
                            "score": planned_mask.confidence,
                        },
                    )
                    self._restore_annotation_for_class(cid)

                if application.segmentation:
                    self._clear_seg_prompt_state()
                    self._clear_seg_preview()
                    self._update_item_editability()
                    self._update_status()
                    self._jump_to_next_pending_class()
                    status_msg = "Segmentation prediction applied."
                    if application.missing_mask_count:
                        status_msg += (
                            f" Skipped {application.missing_mask_count} "
                            "detection(s) without usable masks."
                        )
                    self.update_status_bar(status_msg)
                else:
                    self.update_status_bar("Prediction returned no usable segmentation masks.")
                    QMessageBox.information(
                        self,
                        "No segmentation masks",
                        "The loaded model did not return any usable segmentation masks for this image.",
                    )
                return

            for planned_pose in application.pose:
                cid = planned_pose.class_id
                bb = BoundingBox(
                    planned_pose.x,
                    planned_pose.y,
                    planned_pose.width,
                    planned_pose.height,
                    cid,
                )
                self._clear_class_items(cid, drop_cache=True)
                item = BoxItem(bb, self.classes[cid] if cid < len(self.classes) else str(cid))
                self.scene.addItem(item)
                self._track_scene_item(item)

                kp_objs: list[Keypoint] = []
                class_kp_names = self._kp_names_for_index(cid)
                for planned_keypoint in planned_pose.keypoints:
                    kp_obj = Keypoint(
                        planned_keypoint.x,
                        planned_keypoint.y,
                        cid,
                        planned_keypoint.name,
                    )
                    kp_item = KeypointItem(kp_obj, self.kp_pixel_radius, self.kp_font_px)
                    setattr(kp_item, "pred_conf", planned_keypoint.confidence)
                    kp_item.update_appearance()
                    self.scene.addItem(kp_item)
                    self._track_scene_item(kp_item)
                    kp_objs.append(kp_obj)

                if cid == active_cid:
                    self.bboxes = [bb]
                    self.kps = kp_objs[:]
                    self.current_kp_idx = min(len(class_kp_names), len(self.kps))
                self._cache_active_annotation(cid)

            if not application.pose:
                self.update_status_bar("Prediction returned no usable boxes.")
                return

            self._update_item_editability()
            self._maybe_autoadvance()
            self._update_status()
            self.update_status_bar("Prediction applied.")
        except Exception as e:
            self._cleanup_prediction_depth_staging()
            image_name = self.images[self.current_idx] if self.images else ""
            logger.exception(
                "Could not apply prediction payload",
                extra={
                    "event": "prediction_payload_failed",
                    "operation": "apply_prediction",
                    "project_root": self.project_root,
                    "source_path": image_name,
                },
            )
            self._on_predict_error(str(e) or "Could not apply the prediction payload")

    def _on_predict_error(self, error_text: str):
        # Reset busy state and re-enable button
        self._predict_busy = False
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(True)
        # Surface the error to the user and point to the log
        try:
            QMessageBox.critical(
                self,
                "Prediction Error",
                f"An error occurred during prediction.\n\nDetails:\n{error_text[:1000]}\n\nA full traceback was written to:\n{self._log_path}",
            )
        except Exception:
            pass
        self.update_status_bar("Prediction failed. See log for details.")

    def _reset_zoom(self):
        self.view.resetTransform()
        self.update_zoom_label()

    def mark_current_kp_invisible(self):
        """Mark the next required keypoint as invisible (v=0) and advance."""
        if self.mode != "keypoint":
            self.update_status_bar("Switch to Keypoint mode to mark invisible (press 3).")
            return
        if not self.bboxes:
            self.update_status_bar("Place a bounding box first.")
            return
        state = self.pose_edit_state
        name = state.next_keypoint_name
        if name is None:
            self.update_status_bar("All keypoints already placed.")
            return

        # Use (0,0) for invisibles; YOLO ignores coords when v=0
        controller = self._bind_pose_annotation_controller()
        if controller is not None:
            controller.mark_next_invisible()
        else:
            state.push_undo_snapshot()
            state.mark_next_invisible()
        kp = state.keypoints[name].kp
        item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
        item.visibility = 0
        item.update_appearance()

        # Keep it in the scene so saving picks it up (subtle visual)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self._sync_pose_legacy_mirrors()

        self._update_status()
        self.update_status_bar(f"Marked '{name}' invisible (v=0).")
        self._maybe_autoadvance()

    def set_selected_invisible(self):
        """Convert selected keypoints to invisible (v=0) without moving them."""
        changed = False
        selected = [item for item in self.scene.selectedItems() if isinstance(item, KeypointItem)]
        controller = self._bind_pose_annotation_controller()
        if selected:
            self.pose_edit_state.push_undo_snapshot()
        for it in selected:
            it.visibility = 0
            it.update_appearance()
            if controller is not None:
                controller.set_visibility(it.kp.name, 0, record_undo=False)
            else:
                self.pose_edit_state.set_visibility(it.kp.name, 0)
            changed = True
        if changed:
            self._cache_active_annotation()
            self.update_status_bar("Selected keypoints set to invisible (v=0).")

    # ---------- Shortcuts & input ----------

    def _bind_shortcuts(self):
        # modes & core actions
        mapping = {
            "1": lambda: self.set_mode("panzoom"),
            "2": lambda: self.set_mode("segment" if self._is_seg_workflow() else "bbox"),
            "E": lambda: self.set_mode("segedit"),
            "3": lambda: self.set_mode("keypoint"),
            "4": lambda: self.set_mode("predict"),
            "S": self.save_labels,
            "Z": self.undo,
            "V": self.toggle_selected_visibility,
            "R": self._reset_zoom,  # <-- refresh zoom label too
            "G": self._run_sam_segmentation,
            "X": self._clear_seg_prompt_state,
            "A": self.apply_template_for_current_class,
            "C": lambda: self._cycle_class(+1),
            Qt.Key.Key_Delete: self.delete_selected,
            Qt.Key.Key_Backspace: self.delete_selected,  # optional: Mac-friendly
            Qt.Key.Key_P: self.prev_index,
            Qt.Key.Key_N: self.next_index,
        }
        for key, func in mapping.items():
            QShortcut(QKeySequence(key), self).activated.connect(func)

        # cancel drawing
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(self.view._cancel_draw)

        # Size controls + label toggle
        QShortcut(QKeySequence("="), self).activated.connect(lambda: self._bump_kp_size(+1))
        QShortcut(QKeySequence("-"), self).activated.connect(lambda: self._bump_kp_size(-1))
        QShortcut(QKeySequence("]"), self).activated.connect(lambda: self._bump_kp_font(+1))
        QShortcut(QKeySequence("["), self).activated.connect(lambda: self._bump_kp_font(-1))
        QShortcut(QKeySequence("L"), self).activated.connect(self._toggle_kp_labels)
        QShortcut(QKeySequence(","), self).activated.connect(
            lambda: self._adjust_seg_brush_radius(-2)
        )
        QShortcut(QKeySequence("."), self).activated.connect(
            lambda: self._adjust_seg_brush_radius(+2)
        )

        # Invisible keypoints
        QShortcut(QKeySequence("0"), self).activated.connect(self.mark_current_kp_invisible)
        QShortcut(QKeySequence("Shift+0"), self).activated.connect(self.set_selected_invisible)

        # Workflow jumps
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(
            self.complete_and_next_unlabeled
        )
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(
            self.complete_and_next_unlabeled
        )
        QShortcut(QKeySequence("K"), self).activated.connect(self.skip_to_next_unlabeled)
        QShortcut(QKeySequence("Meta+Return"), self).activated.connect(
            self.complete_and_next_unlabeled
        )  # optional: macOS
        QShortcut(QKeySequence("Shift+Return"), self).activated.connect(
            self._accept_segmentation_preview
        )

    def keyPressEvent(self, event):
        # Space = temporary pan
        if event.key() == Qt.Key.Key_Space:
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.setCursor(Qt.CursorShape.OpenHandCursor)
            return

        # Arrow keys:
        # - If we're in keypoint mode AND at least one KeypointItem is selected -> nudge
        # - Else -> browse frames prev/next
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            selected_kp = any(isinstance(it, KeypointItem) for it in self.scene.selectedItems())
            if self.mode == "keypoint" and selected_kp:
                step = 0.5
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    step = 3.0
                dx = dy = 0
                if event.key() == Qt.Key.Key_Left:
                    dx = -step
                elif event.key() == Qt.Key.Key_Right:
                    dx = step
                elif event.key() == Qt.Key.Key_Up:
                    dy = -step
                elif event.key() == Qt.Key.Key_Down:
                    dy = step
                for it in self.scene.selectedItems():
                    if isinstance(it, KeypointItem):
                        it.moveBy(dx, dy)
                        it.update_model()
                return
            else:
                # browse
                if event.key() == Qt.Key.Key_Left:
                    self.prev_index()
                elif event.key() == Qt.Key.Key_Right:
                    self.next_index()
                return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.set_mode(self.mode)  # restore cursor/drag per current tool
            return
        super().keyReleaseEvent(event)

    def _bump_kp_size(self, d):
        self.kp_pixel_radius = max(1, self.kp_pixel_radius + d)
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                it.refresh_display_sizes(self.kp_pixel_radius, self.kp_font_px)

    def _bump_kp_font(self, d):
        self.kp_font_px = max(6, self.kp_font_px + d)
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                it.refresh_display_sizes(self.kp_pixel_radius, self.kp_font_px)

    def _toggle_kp_labels(self):
        any_visible = any(
            isinstance(it, KeypointItem) and it.text_item.isVisible() for it in self.scene.items()
        )
        new_vis = not any_visible
        for it in self.scene.items():
            if isinstance(it, KeypointItem):
                it.text_item.setVisible(new_vis)
        self.update_status_bar("Keypoint labels " + ("shown" if new_vis else "hidden"))

    def _adjust_seg_brush_radius(self, delta: int):
        radius = int(getattr(self, "seg_brush_radius", 8))
        radius = max(2, min(96, radius + int(delta)))
        self.seg_brush_radius = radius
        self._refresh_seg_brush_size_badge()
        if hasattr(self, "view") and hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()
        if self._is_seg_workflow() and self.mode == "segedit":
            self._refresh_sam_controls()
            self.update_status_bar(f"Mask brush radius: {radius}px")

    def _apply_class_manager_results(
        self, classes: list[str], keypoints: list[str], kp_map: dict[str, list[str]]
    ) -> bool:
        if not classes or not keypoints:
            return False
        classes_clean = [name.strip() for name in classes if name and name.strip()]
        keypoints_clean = [name.strip() for name in keypoints if name and name.strip()]
        if not classes_clean or not keypoints_clean:
            return False

        class_dupes = find_duplicate_names(classes_clean)
        if class_dupes:
            QMessageBox.warning(
                self,
                "Duplicate classes",
                "Class names must be unique.\n\nDuplicates: " + ", ".join(class_dupes),
            )
            return False

        canonical: list[str] = []
        for name in keypoints_clean:
            if name not in canonical:
                canonical.append(name)

        normalized_map: dict[str, list[str]] = {}
        for class_name in classes_clean:
            raw_list = kp_map.get(class_name, [])
            cls_keypoints = [str(name).strip() for name in raw_list if str(name).strip()]
            dupes = find_duplicate_names(cls_keypoints)
            if dupes:
                QMessageBox.warning(
                    self,
                    "Duplicate keypoints",
                    f"Class '{class_name}' has duplicate keypoint names:\n{', '.join(dupes)}",
                )
                return False
            for name in cls_keypoints:
                if name not in canonical:
                    canonical.append(name)
            normalized_map[class_name] = cls_keypoints

        if self._schema_is_locked():
            allowed, reason = self._validate_locked_schema_changes(classes_clean, normalized_map)
            if not allowed:
                QMessageBox.warning(
                    self,
                    "Schema Locked",
                    reason + "\n\nLabeled data already exists for this project.",
                )
                return False

        try:
            atomic_write_text_files(
                {
                    self.class_file: "".join(f"{name}\n" for name in classes_clean),
                    self.keypoint_file: "".join(f"{name}\n" for name in canonical),
                    self.class_keypoints_path: json.dumps(normalized_map, indent=2),
                }
            )
        except Exception as e:
            logger.exception(
                "Project schema save failed",
                extra={
                    "event": "schema_save_failed",
                    "operation": "save_schema",
                    "project_root": getattr(self, "project_root", ""),
                    "target_path": self.class_keypoints_path,
                },
            )
            QMessageBox.warning(
                self,
                "Schema Save Error",
                "Could not update the class/keypoint schema. Existing schema files were restored.\n\n"
                f"{e}",
            )
            return False

        self.classes = classes_clean
        self.kp_names = canonical
        self.class_keypoints = normalized_map
        self._refresh_kp_index_lookup()
        current_name = self.class_selector.currentText()
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.addItems(self.classes)
        self._fit_class_selector_to_items()
        if current_name in self.classes:
            self.class_selector.setCurrentIndex(self.classes.index(current_name))
        else:
            self.class_selector.setCurrentIndex(0)
        self.class_selector.blockSignals(False)
        self.annotation_cache.clear()
        self.load_image()
        return True

    def open_class_manager(self):
        if self._is_seg_workflow():
            self._open_seg_class_manager()
            return
        dlg = ClassManagerDialog(
            self.classes,
            self.class_keypoints,
            self.kp_names,
            self,
            schema_locked=self._schema_is_locked(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        classes, keypoints, kp_map = dlg.get_results()
        if self._apply_class_manager_results(classes, keypoints, kp_map):
            self.update_status_bar("Classes and keypoints updated.")

    def _open_seg_class_manager(self):
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Segmentation Classes",
            "Enter one segmentation class per line:",
            "\n".join(self.seg_classes),
        )
        if not ok:
            return
        classes_clean = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not classes_clean:
            QMessageBox.warning(self, "No classes", "Add at least one segmentation class.")
            return
        dupes = find_duplicate_names(classes_clean)
        if dupes:
            QMessageBox.warning(
                self,
                "Duplicate classes",
                "Segmentation class names must be unique.\n\nDuplicates: " + ", ".join(dupes),
            )
            return
        if self._schema_is_locked():
            existing_classes = self.seg_classes[:]
            if len(classes_clean) < len(existing_classes):
                QMessageBox.warning(
                    self,
                    "Schema Locked",
                    "Cannot remove segmentation classes after labeled data exists.\n\n"
                    "Labeled data already exists for this segmentation project.",
                )
                return
            if classes_clean[: len(existing_classes)] != existing_classes:
                QMessageBox.warning(
                    self,
                    "Schema Locked",
                    "Existing segmentation class names/order are locked.\n"
                    "Only append new classes at the end.\n\n"
                    "Labeled data already exists for this segmentation project.",
                )
                return
        self.seg_classes = classes_clean[:]
        self._write_list_file(self.seg_class_file, self.seg_classes)
        self.classes = self.seg_classes[:]
        self._refresh_class_selector_for_workflow()
        self.annotation_cache.clear()
        self.seg_edit_state.reset(
            selected_target=(
                self.class_selector.currentIndex()
                if self.class_selector.currentIndex() >= 0
                else None
            )
        )
        self.load_image()
        self.update_status_bar("Segmentation classes updated.")

    def _template_path_for_class(self, class_name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", class_name)
        return os.path.join(self.template_dir, f"{safe}.json")

    def save_template_for_current_class(self):
        if self._is_seg_workflow():
            self.update_status_bar("Templates are only available in the Keypoints layer.")
            return
        if not self._cache_active_annotation():
            QMessageBox.warning(
                self,
                "Template error",
                "Complete the current class annotation before saving a template.",
            )
            return
        cid = self.class_selector.currentIndex()
        entry = self.annotation_cache.get(cid)
        if not entry:
            QMessageBox.warning(self, "Template error", "Nothing to save for this class.")
            return
        data = self.pose_edit_state.to_template(
            self.classes[cid],
            image_width=self.img_w,
            image_height=self.img_h,
        )
        if data is None:
            QMessageBox.warning(self, "Template error", "Nothing to save for this class.")
            return
        path = self._template_path_for_class(self.classes[cid])
        try:
            atomic_write_text(path, json.dumps(data, indent=2))
            QMessageBox.information(self, "Template saved", f"Template saved to {path}.")
        except Exception as e:
            QMessageBox.warning(self, "Template error", f"Failed to save template:\n{e}")

    def apply_template_for_current_class(self):
        if self._is_seg_workflow():
            self.update_status_bar("Templates are only available in the Keypoints layer.")
            return
        if not self.images:
            QMessageBox.warning(self, "Template error", "Load an image before applying templates.")
            return
        class_name = self.class_selector.currentText()
        path = self._template_path_for_class(class_name)
        if not os.path.exists(path):
            QMessageBox.warning(
                self, "Template missing", f"No template found for {class_name}.\nSave one first."
            )
            return
        try:
            data = read_json_file(path, max_bytes=4 * 1024 * 1024, require_object=True)
        except Exception as e:
            QMessageBox.warning(self, "Template error", f"Failed to load template:\n{e}")
            return

        cid = self.class_selector.currentIndex()
        class_name = self.classes[cid]
        self._sync_pose_state_from_scene(cid)
        controller = self._bind_pose_annotation_controller()
        if controller is not None:
            controller.apply_template(
                data,
                image_width=self.img_w,
                image_height=self.img_h,
            )
        else:
            self.pose_edit_state.push_undo_snapshot()
            self.pose_edit_state.apply_template(
                data,
                image_width=self.img_w,
                image_height=self.img_h,
            )
            self.annotation_cache.delete_annotation(cid)
        self._render_pose_edit_state()
        self._store_pose_edit_state()
        self._update_item_editability()
        self.update_status_bar(f"Applied template for {class_name}.")
        self._maybe_autoadvance()

    def update_status_bar(self, msg: str):
        self.status.showMessage(msg, 2500)

    def _kp_text(self) -> str:
        if self.mode == "keypoint":
            names = self._active_kp_names()
            if self.current_kp_idx < len(names):
                return f"Next: {names[self.current_kp_idx]}  ({self.current_kp_idx}/{len(names)})"
            return "All keypoints placed"
        return ""

    # ---------- Image load / navigation ----------

    def load_image(self):
        if hasattr(self.view, "_remove_crosshairs"):
            self.view._remove_crosshairs()
        if hasattr(self.view, "_reset_seg_brush_cursor"):
            self.view._reset_seg_brush_cursor()
        self._clear_seg_edit_handles()
        self._clear_seg_prompt_state()
        selected = self.class_selector.currentIndex() if self._is_seg_workflow() else -1
        self.seg_edit_state.reset(selected_target=selected if selected >= 0 else None)
        pose_controller = self._bind_pose_annotation_controller()
        if pose_controller is not None:
            pose_controller.select_class(None)
        else:
            self.pose_edit_state.select_class(None)
        self.current_box = None
        self.current_kps = []
        self.current_class_id = None

        depth_presenter = self.__dict__.get("_depth_preview_presenter")
        if depth_presenter is not None:
            depth_presenter.clear()
        self._scene_presenter.forget_scene_items()
        self.scene.clear()
        self._item_refs.clear()
        self._active_depth_map = None
        if not self.images:
            self._refresh_segmentation_box_action()
            return

        current_image_name = self.images[self.current_idx]
        if current_image_name != self._depth_probe_image_name:
            self._depth_probe_image_name = current_image_name
            self._depth_probes = []
            self._depth_probe_error = ""
        depth_controller = getattr(self, "_depth_controller", None)
        if depth_controller is not None:
            depth_controller.state.set_view_mode(self._depth_view_mode())
            depth_controller.load_image(current_image_name)
        self._refresh_depth_probe_label()

        img_path = os.path.join(self.active_image_dir, current_image_name)
        self.current_image_path = img_path
        pix = QPixmap(img_path)
        if pix.isNull():
            self.update_status_bar(f"Failed to load image: {self.images[self.current_idx]}")
            return
        self.img_w, self.img_h = pix.width(), pix.height()
        self.scene.setSceneRect(0, 0, self.img_w, self.img_h)
        self._scene_presenter.add_background(pix)

        self.bboxes.clear()
        self.kps.clear()
        self.current_kp_idx = 0
        base = os.path.splitext(self.images[self.current_idx])[0]
        if LabelingApp._is_depth_layer(self):
            self.annotation_cache.clear()
            artifact_plan = plan_depth_artifacts(
                depth_image_dir=self.depth_image_dir,
                depth_preview_dir=self.depth_preview_dir,
                image_name=current_image_name,
                image_width=self.img_w,
                image_height=self.img_h,
                project_root=self.project_root,
            )
            array_reader = (
                (lambda path: _np.load(path, mmap_mode="r", allow_pickle=False))
                if _np is not None
                else None
            )
            metadata_reader = lambda path: read_json_file(
                path,
                max_bytes=1024 * 1024,
                require_object=True,
            )
            if depth_controller is not None:
                loaded_artifacts = depth_controller.load_artifacts(
                    artifact_plan,
                    array_reader=array_reader,
                    metadata_reader=metadata_reader,
                    is_file=os.path.isfile,
                )
            else:
                loaded_artifacts = load_depth_artifacts(
                    artifact_plan,
                    array_reader=array_reader,
                    metadata_reader=metadata_reader,
                    is_file=os.path.isfile,
                )
            self._active_depth_map = loaded_artifacts.depth_map
            self._depth_probe_error = loaded_artifacts.map_error
            if depth_controller is not None:
                if loaded_artifacts.metadata_error:
                    depth_controller.state.set_metadata({})
                depth_range_label = self.__dict__.get("depth_range_label")
                if depth_range_label is not None:
                    depth_range_label.setText(depth_controller.state.range_text())
            else:
                self._update_depth_range_label(base)
            self._refresh_depth_probe_label()
            display_mode = self._depth_view_mode()
            presentation = self._depth_preview_presenter.present_preview(
                loaded_artifacts,
                mode=display_mode,
                image_width=self.img_w,
                image_height=self.img_h,
            )
            if presentation.status_message:
                self.update_status_bar(presentation.status_message)
            self._refresh_reference_layer_overlay()
            self._render_depth_probes()
            self._update_status()
            self.view.centerOn(self.scene.sceneRect().center())
            return
        label_file = os.path.join(self.label_dir, f"{base}.txt")

        if self._is_seg_workflow():
            entries = (
                self._load_seg_annotations_from_file(label_file)
                if os.path.exists(label_file)
                else {}
            )
            self.annotation_cache = SegmentationAnnotationDocument(entries)
            self._segmentation_document = self.annotation_cache
            self._sync_segmentation_state_from_cache()
            for cid in range(len(self.classes)):
                if cid in self.annotation_cache:
                    self._restore_annotation_for_class(cid)
            self._clear_seg_prompt_state()
            self._sync_active_class_state()
            self._update_item_editability()
            self._update_status()
            self._clear_seg_edit_handles()
            if hasattr(self.view, "refresh_seg_brush_cursor"):
                self.view.refresh_seg_brush_cursor()
            self._refresh_reference_layer_overlay()
            scene_center = self.scene.sceneRect().center()
            self.view.centerOn(scene_center)
            return

        entries = self._load_annotations_from_file(label_file) if os.path.exists(label_file) else {}
        self.annotation_cache = PoseAnnotationDocument()
        self.annotation_cache.load_annotations(entries)
        self._bind_pose_annotation_controller()
        for cid in range(len(self.classes)):
            if cid in self.annotation_cache:
                self._restore_annotation_for_class(cid)
        self._sync_active_class_state()
        self._update_item_editability()

        self._update_status()
        if hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()
        self._refresh_reference_layer_overlay()
        self._refresh_segmentation_box_action()
        scene_center = self.scene.sceneRect().center()
        self.view.centerOn(scene_center)

    def add_bbox(self, rect: QRectF):
        if not self.classes:
            QMessageBox.warning(
                self, "No classes", "Define at least one class before adding boxes."
            )
            return
        cid = self.class_selector.currentIndex()
        if cid < 0 or cid >= len(self.classes):
            QMessageBox.warning(self, "Invalid class", "Select a valid class before adding boxes.")
            return
        class_name = self.classes[cid]
        self._clear_class_items(cid, drop_cache=True)
        controller = self._bind_pose_annotation_controller()
        if controller is not None:
            controller.select_class(cid)
            controller.set_box(BoundingBox(rect.x(), rect.y(), rect.width(), rect.height(), cid))
        else:
            self.pose_edit_state.select_class(
                cid,
                self._kp_names_for_index(cid),
                canonical_names=self.kp_names,
            )
            self.pose_edit_state.push_undo_snapshot()
            self.pose_edit_state.set_box(
                BoundingBox(rect.x(), rect.y(), rect.width(), rect.height(), cid)
            )
        bbox = self.pose_edit_state.box
        assert bbox is not None
        item = BoxItem(bbox, class_name)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self._sync_pose_legacy_mirrors()
        self._update_item_editability()
        self.update_status_bar("Box added. Switch to Keypoint mode (3).")
        if not self._active_kp_names():
            if self._cache_active_annotation():
                self._update_item_editability()
                self._jump_to_next_pending_class()

    def add_keypoint(self, pos: QPointF):
        if not self.bboxes:
            self.update_status_bar("Place a bounding box first.")
            return
        state = self.pose_edit_state
        if state.active_class_id != self.class_selector.currentIndex():
            state = self._sync_pose_state_from_scene(self.class_selector.currentIndex())
        name = state.next_keypoint_name
        if name is None:
            self.update_status_bar("All keypoints placed for this frame.")
            return
        controller = self._bind_pose_annotation_controller()
        if controller is not None and state is self.pose_edit_state:
            controller.add_next_keypoint(pos.x(), pos.y())
        else:
            state.push_undo_snapshot()
            state.add_next_keypoint(pos.x(), pos.y())
        kp = state.keypoints[name].kp
        item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self._sync_pose_legacy_mirrors()
        self._update_status()
        self._maybe_autoadvance()

    def delete_selected(self):
        cid = self.class_selector.currentIndex()
        if self._is_pose_workflow():
            selected = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, (BoxItem, KeypointItem))
                and getattr(item, "bbox", getattr(item, "kp", None)).class_id == cid
            ]
            if not selected:
                return
            self._sync_pose_state_from_scene(cid)
            controller = self._bind_pose_annotation_controller()
            self.pose_edit_state.push_undo_snapshot()
            if any(isinstance(item, BoxItem) for item in selected):
                if controller is not None:
                    controller.delete_box(record_undo=False)
                else:
                    self.pose_edit_state.delete_box()
            else:
                for item in selected:
                    if isinstance(item, KeypointItem):
                        if controller is not None:
                            controller.delete_keypoint(item.kp.name, record_undo=False)
                        else:
                            self.pose_edit_state.delete_keypoint(item.kp.name)
            if controller is None:
                self.annotation_cache.delete_annotation(cid)
            self._render_pose_edit_state()
            self._store_pose_edit_state()
            self._update_status()
            self._update_item_editability()
            return
        changed = False
        drop_cache = False
        for item in list(self.scene.selectedItems()):
            if isinstance(item, BoxItem) and item.bbox.class_id == cid:
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                self.bboxes.clear()
                self.kps.clear()
                drop_cache = True
                changed = True
            elif isinstance(item, KeypointItem) and item.kp.class_id == cid:
                if item.kp in self.kps:
                    self.kps.remove(item.kp)
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                changed = True
            elif self._is_seg_mask_item(item) and int(getattr(item, "seg_class_id", -1)) == cid:
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                drop_cache = True
                changed = True
        names = self._active_kp_names()
        self.current_kp_idx = min(self.current_kp_idx, len(names), len(self.kps))
        if drop_cache:
            if self._is_seg_workflow():
                self._drop_segmentation_cache_entry(cid)
            else:
                self.annotation_cache.pop(cid, None)
        if changed:
            self._update_status()
            self._update_item_editability()

    def undo(self):
        cid = self.class_selector.currentIndex()
        if self._is_seg_workflow() and self.mode == "segment":
            if self.seg_preview_points:
                self._clear_seg_preview()
                self.update_status_bar("Cleared segmentation preview.")
                return
            if self.seg_prompt_points:
                controller = self._bind_segmentation_annotation_controller()
                if controller is not None:
                    controller.remove_last_prompt()
                else:
                    self.seg_edit_state.remove_last_prompt()
                self._refresh_seg_prompt_markers()
                self.update_status_bar("Removed last segmentation prompt.")
                return
            seg_item = self._class_seg_mask_item(cid)
            if seg_item is not None:
                self._safe_remove_scene_item(seg_item)
                self._untrack_scene_item(seg_item)
                self._drop_segmentation_cache_entry(cid)
                self.update_status_bar("Removed segmentation mask for current class.")
                self._update_item_editability()
                self._refresh_sam_controls()
                return
        pose_controller = self._bind_pose_annotation_controller()
        pose_undone = (
            pose_controller.undo()
            if self._is_pose_workflow() and pose_controller is not None
            else self._is_pose_workflow() and self.pose_edit_state.undo()
        )
        if pose_undone:
            if pose_controller is None:
                self.annotation_cache.delete_annotation(cid)
            self._render_pose_edit_state()
            self._store_pose_edit_state()
            self._update_status()
            self._update_item_editability()

    def _is_fully_labeled(self) -> bool:
        if self._is_depth_layer():
            if not self.images:
                return False
            base = os.path.splitext(self.images[self.current_idx])[0]
            return os.path.isfile(os.path.join(self.depth_image_dir, f"{base}.npy"))
        self._cache_active_annotation()
        if not self.classes:
            return False
        if self._is_seg_workflow():
            return any(
                len(entry.get("segments", [])) >= 3 for entry in self.annotation_cache.values()
            )
        for cid in range(len(self.classes)):
            if not self.annotation_cache.is_complete(
                cid,
                required_keypoints=self._kp_names_for_index(cid),
            ):
                return False
        return True

    def _update_status(self):
        panel = getattr(self, "__dict__", {}).get("annotation_panel")
        if panel is not None:
            panel.set_active_mode(self.mode)
        buttons = {
            "panzoom": self.panzoom_btn,
            "bbox": self.bbox_btn,
            "segment": self.segment_btn,
            "segedit": self.seg_edit_btn,
            "keypoint": self.keypoint_btn,
            "predict": self.predict_btn,
        }
        for mode_name, button in buttons.items():
            button.setProperty("activeMode", self.mode == mode_name)
            _refresh_qt_style(button)

        # Show filtered index / total in status bar
        fi = self._filtered_indices()
        if fi and self.current_idx in fi:
            idx_in_view = fi.index(self.current_idx) + 1
            self.status.showMessage(f"Viewing {self.nav_filter}: {idx_in_view}/{len(fi)}", 2000)

        presenter = getattr(self, "__dict__", {}).get("canvas_hud_presenter")
        if presenter is not None:
            presenter.set_mode(self.mode)
        if self.mode == "keypoint":
            if presenter is None:
                self.legend_frame.show()
                self.zoom_frame.hide()
            self._layout_overlays()
            self.update_status_bar(self._kp_text())
        elif self.mode == "panzoom":
            if presenter is None:
                self.legend_frame.hide()
                self.zoom_frame.show()
            self._layout_overlays()
            self.update_zoom_label()
        else:
            if presenter is None:
                self.legend_frame.hide()
                self.zoom_frame.hide()
        self._refresh_sam_controls()

    def toggle_selected_visibility(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, KeypointItem)]
        controller = self._bind_pose_annotation_controller()
        if selected and self._is_pose_workflow():
            self.pose_edit_state.push_undo_snapshot()
        for item in selected:
            item.toggle_visibility()
            if self._is_pose_workflow():
                if controller is not None:
                    controller.set_visibility(
                        item.kp.name,
                        item.visibility,
                        record_undo=False,
                    )
                else:
                    self.pose_edit_state.set_visibility(item.kp.name, item.visibility)
        if selected and self._is_pose_workflow():
            self._cache_active_annotation()

    def update_zoom_label(self):
        scale = self.view.transform().m11()
        presenter = getattr(self, "__dict__", {}).get("canvas_hud_presenter")
        if presenter is not None:
            presenter.set_zoom_scale(scale)
        else:
            self.zoom_label.setText(f"Zoom: {int(scale * 100)}%")

    def _layout_hot_corners(self):
        if not hasattr(self, "view"):
            return
        presenter = getattr(self, "__dict__", {}).get("canvas_hud_presenter")
        if presenter is not None:
            presenter.layout_context()
            return
        if hasattr(self, "layer_context_frame"):
            self.layer_context_frame.adjustSize()
            self.layer_context_frame.move(10, 10)
            self.layer_context_frame.raise_()

    def _layout_overlays(self):
        """Dynamically position and size legend / zoom overlays."""
        if not hasattr(self, "view"):
            return
        vw = self.view.viewport().width()
        vh = self.view.viewport().height()

        presenter = getattr(self, "__dict__", {}).get("canvas_hud_presenter")
        if presenter is not None:
            presenter.layout_overlays(viewport_width=vw, viewport_height=vh)
            return

        x = 10
        cursor_y = vh - 10

        # Keypoint legend (pose workflow / keypoint mode).
        if hasattr(self, "legend_frame") and self.legend_frame.isVisible():
            fm = self.legend_label.fontMetrics()
            ch = fm.horizontalAdvance("M")  # approx width of one character
            preferred = int(ch * 30 + 24)
            w = max(250, min(preferred, int(vw * 0.42), 420))
            self.legend_frame.setFixedWidth(w)
            self.legend_frame.adjustSize()
            lh = self.legend_frame.sizeHint().height()
            top = max(10, cursor_y - lh)
            self.legend_frame.move(x, top)
            cursor_y = top - 8

        # Keep zoom HUD stacked above whichever lower-left overlays are visible.
        if hasattr(self, "zoom_frame") and self.zoom_frame.isVisible():
            zh = self.zoom_frame.sizeHint().height()
            self.zoom_frame.move(x, max(10, cursor_y - zh))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_hot_corners()
        self._layout_overlays()

    def showEvent(self, event):
        super().showEvent(event)
        # Initial widget geometry can settle after the first paint; reflow once
        # now and once shortly after so corners land correctly on first open.
        QTimer.singleShot(0, self._relayout_after_show)
        QTimer.singleShot(40, self._relayout_after_show)

    def _relayout_after_show(self):
        if not self.isVisible() or not hasattr(self, "view"):
            return
        self._layout_hot_corners()
        self._layout_overlays()

    def _remove_all_boxes_and_keypoints(self, drop_cache: bool = False):
        self._clear_all_annotation_items()
        if drop_cache:
            self.annotation_cache.clear()
            if self._is_seg_workflow():
                self.seg_edit_state.replace_accepted_masks({})
        self._update_status()

    # ---------- Save ----------
    def _collect_keypoints_by_name(
        self, class_id: Optional[int] = None
    ) -> dict[str, tuple[Keypoint, int]]:
        """Return {kp_name: (Keypoint, visibility)} for all KeypointItems in the scene.
        If there are duplicates by name, the last one found wins."""
        if class_id is None:
            class_id = self.class_selector.currentIndex()
        out: dict[str, tuple[Keypoint, int]] = {}
        for it in self.scene.items():
            if isinstance(it, KeypointItem) and it.kp.class_id == class_id:
                out[it.kp.name] = (it.kp, getattr(it, "visibility", 2))
        return out

    def _sync_current_kp_idx(self):
        """Advance index to the first *missing* required name, counting from the start of kp_names."""
        name_to_entry = self._collect_keypoints_by_name()
        count = 0
        for name in self._active_kp_names():
            if name in name_to_entry:
                count += 1
            else:
                break
        self.current_kp_idx = min(count, len(self._active_kp_names()))

    def save_labels(self) -> bool:
        if not self.images:
            return False
        if LabelingApp._is_depth_layer(self):
            QMessageBox.information(
                self,
                "Depth Maps Save Automatically",
                "Depth predictions are saved automatically when inference completes.",
            )
            return False

        if self._is_seg_workflow() and self.seg_preview_points:
            QMessageBox.information(
                self,
                "Pending preview",
                "Accept the current SAM preview mask before saving.",
            )
            return False

        if self._is_pose_workflow() and not self._cache_active_annotation():
            QMessageBox.warning(
                self,
                "Save Error",
                "Place one bounding box and all keypoints for the selected class before saving.",
            )
            return False
        if self._is_seg_workflow():
            self._cache_active_annotation()

        file_name = os.path.basename(self.images[self.current_idx])
        base = os.path.splitext(file_name)[0]

        project_root = self.project_root
        images_all_dir = os.path.join(project_root, "images_all")
        labels_all_dir = self.label_dir
        annotations_dir = os.path.join(
            project_root,
            "annotations",
            normalize_layer_id(getattr(self, "active_layer", WORKFLOW_POSE)),
        )
        os.makedirs(images_all_dir, exist_ok=True)
        os.makedirs(labels_all_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        label_out_path = os.path.join(labels_all_dir, f"{base}.txt")
        annotated_out_path = os.path.join(annotations_dir, f"{base}_annotated.png")
        image_out_path = os.path.join(images_all_dir, file_name)

        conflicting_images = [
            name
            for name in list_image_files(images_all_dir)
            if os.path.splitext(name)[0].casefold() == base.casefold() and name != file_name
        ]
        if conflicting_images:
            QMessageBox.warning(
                self,
                "Duplicate Image Name",
                f"Cannot save '{file_name}' because images_all already contains another image "
                f"with the same stem:\n\n{', '.join(conflicting_images)}\n\n"
                "Rename one of the images before saving.",
            )
            return False

        if self._is_pose_workflow():
            document = (
                self.annotation_cache
                if isinstance(self.annotation_cache, PoseAnnotationDocument)
                else PoseAnnotationDocument(self.annotation_cache)
            )
            save_request = build_pose_save_request(
                document.typed_snapshot(),
                canonical_names=self.kp_names,
                image_width=self.img_w,
                image_height=self.img_h,
                project_root=self.project_root,
                source_image_path="",
                image_output_path=image_out_path,
                label_output_path=label_out_path,
                overlay_output_path=annotated_out_path,
            )
        else:
            document = (
                self.annotation_cache
                if isinstance(self.annotation_cache, SegmentationAnnotationDocument)
                else SegmentationAnnotationDocument(self.annotation_cache)
            )
            save_request = build_segmentation_save_request(
                document.typed_snapshot(),
                image_width=self.img_w,
                image_height=self.img_h,
                project_root=self.project_root,
                source_image_path="",
                image_output_path=image_out_path,
                label_output_path=label_out_path,
                overlay_output_path=annotated_out_path,
            )

        if not save_request.label_text.strip():
            QMessageBox.warning(self, "No annotations", "Nothing to save for this image.")
            return False

        src_candidates: list[str] = []
        if self.current_image_path:
            src_candidates.append(self.current_image_path)
        if os.path.isabs(file_name):
            src_candidates.append(file_name)
        src_candidates.extend(
            [
                os.path.join(self.active_image_dir, file_name),
                os.path.join(self.image_dir_queue, file_name),
                os.path.join(self.image_dir_all, file_name),
            ]
        )

        src_path = ""
        seen: set[str] = set()
        for cand in src_candidates:
            norm = os.path.abspath(cand)
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.exists(cand):
                src_path = cand
                break

        if not src_path and not os.path.exists(image_out_path):
            tried = "\n".join(sorted(seen))
            msg = f"Could not locate source image for '{file_name}'.\n\nTried:\n{tried}"
            QMessageBox.warning(self, "Save Error", msg)
            return False

        try:
            save_annotation_transaction(
                replace(save_request, source_image_path=src_path),
                render_overlay=self._render_overlay_from_cache,
                committer=commit_staged_paths,
            )
        except Exception as e:
            logger.exception(
                "Annotation save failed",
                extra={
                    "event": "annotation_save_failed",
                    "operation": "save_annotation",
                    "project_root": self.project_root,
                    "source_path": src_path,
                    "target_path": label_out_path,
                },
            )
            QMessageBox.warning(
                self,
                "Save Error",
                f"Could not save the annotation. Existing project files were restored.\n\n{e}",
            )
            return False

        logger.info(
            "Annotation saved",
            extra={
                "event": "annotation_saved",
                "operation": "save_annotation",
                "project_root": self.project_root,
                "source_path": src_path,
                "target_path": label_out_path,
            },
        )
        self._schema_locked = True
        self._update_progress_label()
        return True

    # ---------- Video ----------
    def export_dataset(self):
        """Split images_all/labels_all into train/val sets and regenerate dataset.yaml."""
        if self._is_depth_layer():
            QMessageBox.information(
                self, "Depth MVP", "Depth dataset export is not included in the inference-only MVP."
            )
            return
        seg_mode = self._is_seg_workflow()
        project_root = self.project_root
        images_all_dir = self.image_dir_all
        labels_all_dir = self.label_dir

        if not os.path.isdir(images_all_dir):
            QMessageBox.information(
                self, "No images_all directory", f"Expected {images_all_dir} to exist."
            )
            return
        if not os.path.isdir(labels_all_dir):
            QMessageBox.information(
                self, "No labels_all directory", f"Expected {labels_all_dir} to exist."
            )
            return

        images, image_collisions = filter_image_stem_collisions(list_image_files(images_all_dir))
        if image_collisions:
            groups = "\n".join(" / ".join(names) for names in image_collisions.values())
            QMessageBox.warning(
                self,
                "Duplicate Image Names",
                "Dataset export cannot continue because images_all contains files that share "
                "the same stem and therefore map to the same label:\n\n"
                f"{groups}\n\nRename the conflicting images first.",
            )
            return
        if not images:
            QMessageBox.information(
                self, "Nothing to export", "images_all does not contain any images."
            )
            return

        ratio, ok = QInputDialog.getDouble(
            self, "Train/Val Split", "Train split ratio (0.1 – 0.95):", 0.8, 0.1, 0.95, 2
        )
        if not ok:
            return

        split_seed, ok_seed = QInputDialog.getInt(
            self,
            "Split Seed",
            "Shuffle seed (same files and seed recreate the same split):",
            0,
            0,
            2147483647,
        )
        if not ok_seed:
            return

        if seg_mode:
            dataset_mode = DATASET_SEGMENT
        else:
            dataset_mode = DATASET_POSE

        images, skipped_images = partition_images_by_usable_labels(
            images,
            labels_dir=labels_all_dir,
            mode=dataset_mode,
            class_count=len(self.classes),
            keypoint_count=len(self.kp_names),
        )
        if not images:
            label_kind = "segmentation" if seg_mode else "keypoint"
            QMessageBox.information(
                self,
                "No labeled images",
                f"No images in images_all have usable {label_kind} labels for this export.\n\n"
                "Validate labels or select the layer containing the labels you want to export.",
            )
            return

        paths = dataset_export_paths(project_root, dataset_mode)
        os.makedirs(os.path.dirname(paths.base_dir), exist_ok=True)

        existing_dataset = (
            dataset_dirs_have_files(paths)
            or os.path.exists(paths.dataset_yaml_path)
            or os.path.exists(os.path.join(paths.base_dir, "images"))
            or os.path.exists(os.path.join(paths.base_dir, "labels"))
        )
        if existing_dataset:
            confirm = QMessageBox.question(
                self,
                "Replace dataset?",
                "An exported dataset already exists. Replace it after the new export completes successfully?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        images = sorted(images)
        random.Random(split_seed).shuffle(images)
        train_images, val_images = split_train_val_images(images, ratio)

        total = len(train_images) + len(val_images)
        prog = QProgressDialog("Copying dataset…", "Cancel", 0, total, self)
        prog.setWindowTitle("Export Dataset")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        def _progress(processed: int, img_file: str):
            prog.setValue(processed)
            prog.setLabelText(f"Copying {img_file}")
            QApplication.processEvents()

        export_result = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export_result = export_dataset_transaction(
                project_root=self.project_root,
                images_all_dir=images_all_dir,
                labels_all_dir=labels_all_dir,
                final_paths=paths,
                train_images=train_images,
                val_images=val_images,
                mode=dataset_mode,
                classes=self.classes,
                keypoint_names=self.kp_names,
                split_seed=split_seed,
                skipped_images=skipped_images,
                progress_callback=_progress,
                cancel_requested=prog.wasCanceled,
                committer=commit_staged_paths,
            )
        except Exception as e:
            logger.exception(
                "Dataset export failed",
                extra={
                    "event": "dataset_export_failed",
                    "operation": "export_dataset",
                    "project_root": self.project_root,
                    "source_path": images_all_dir,
                    "target_path": paths.base_dir,
                },
            )
            QMessageBox.critical(
                self,
                "Dataset Export Error",
                "Could not build and install the dataset. "
                "The previous dataset was restored.\n\n"
                f"{e}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()

        if export_result is None:
            return
        if export_result.canceled:
            QMessageBox.information(
                self,
                "Export canceled",
                "Dataset export was canceled. The previous dataset was left unchanged.",
            )
            return

        if export_result.errors:
            errors = "\n".join(export_result.errors[:10])
            if len(export_result.errors) > 10:
                errors += f"\n...{len(export_result.errors) - 10} more"
            QMessageBox.critical(
                self,
                "Dataset Export Error",
                "The new dataset could not be completed. The previous dataset was left unchanged.\n\n"
                + errors,
            )
            return

        QMessageBox.information(
            self, "Dataset exported", format_dataset_export_summary(export_result)
        )
        self.update_status_bar("Dataset export complete.")

    def show_project_health(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = scan_project_health(
                self.project_root,
                pose_class_count=len(self.pose_classes),
                pose_keypoint_count=len(self.pose_kp_names),
                segmentation_class_count=len(self.seg_classes),
            )
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self,
            "Project Health",
            format_project_health_summary(report),
        )
        if report.restorable_transaction_backups:
            answer = QMessageBox.question(
                self,
                "Restore Missing Transaction Targets?",
                (
                    f"Restore {len(report.restorable_transaction_backups)} missing "
                    "project target(s) from their sole transaction backup?\n\n"
                    "Ambiguous or conflicting backups will remain untouched."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                recovery = restore_missing_transaction_targets(self.project_root)
                if recovery.errors:
                    QMessageBox.warning(
                        self,
                        "Recovery Incomplete",
                        "Some targets could not be restored:\n\n" + "\n".join(recovery.errors[:8]),
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Recovery Complete",
                        f"Restored {len(recovery.restored_paths)} project target(s).",
                    )
        if not report.temporary_paths:
            self.update_status_bar("Project health scan complete.")
            return

        answer = QMessageBox.question(
            self,
            "Remove Temporary Files?",
            (
                f"Remove {len(report.temporary_paths)} stale transaction "
                "file(s) or staging folder(s)?\n\n"
                "Transaction backups, worker config files, and project data will not be removed."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.update_status_bar("Project health scan complete.")
            return

        errors = cleanup_project_temporary_paths(report)
        if errors:
            QMessageBox.warning(
                self,
                "Cleanup Incomplete",
                "Some temporary paths could not be removed:\n\n" + "\n".join(errors[:8]),
            )
        else:
            QMessageBox.information(
                self,
                "Cleanup Complete",
                f"Removed {len(report.temporary_paths)} temporary path(s).",
            )
        self.update_status_bar("Project health cleanup complete.")

    def normalize_labels_all(self):
        if self._is_depth_layer():
            QMessageBox.information(
                self, "Depth MVP", "Saved depth maps do not use editable YOLO label files."
            )
            return
        labels_dir = self.label_dir
        images_all_dir = self.image_dir_all
        images_to_label_dir = self.image_dir_queue

        label_files = list_label_files(labels_dir)
        if not label_files:
            folder_name = os.path.basename(labels_dir.rstrip(os.sep)) or labels_dir
            QMessageBox.information(
                self, "No labels", f"{folder_name} does not contain any .txt files."
            )
            return

        seg_mode = self._is_seg_workflow()
        dataset_mode = DATASET_SEGMENT if seg_mode else DATASET_POSE
        progress_label = "Validating segmentation labels…" if seg_mode else "Validating labels…"
        window_title = "Validate Segmentation Labels" if seg_mode else "Validate Labels"

        prog = QProgressDialog(progress_label, "Cancel", 0, len(label_files), self)
        prog.setWindowTitle(window_title)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        def _progress(idx: int, fname: str):
            prog.setValue(idx)
            prog.setLabelText(f"Normalizing {fname}")
            QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = normalize_label_directory(
                labels_dir=labels_dir,
                images_all_dir=images_all_dir,
                images_to_label_dir=images_to_label_dir,
                mode=dataset_mode,
                class_count=len(self.classes),
                keypoint_count=len(self.kp_names),
                label_files=label_files,
                progress_callback=_progress,
                cancel_requested=prog.wasCanceled,
            )
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()

        if result.canceled:
            self._update_progress_label()
            QMessageBox.information(
                self,
                "Normalization canceled",
                "Operation canceled. Some files may have been processed already.",
            )
            return

        QMessageBox.information(
            self, "Normalization complete", format_label_normalization_summary(result)
        )
        self._update_progress_label()
        status = (
            "Segmentation label normalization complete."
            if seg_mode
            else "Label normalization complete."
        )
        self.update_status_bar(status)

    def open_train_dialog(self):
        try:
            plan = plan_training_dialog(
                project_root=self.project_root,
                layer_id=self.active_layer,
            )
        except DialogUnavailableError as exc:
            QMessageBox.information(self, exc.title, exc.message)
            return
        dlg = TrainDialog(
            self,
            default_dataset=plan.default_dataset,
            default_task=plan.default_task,
            layer_id=plan.layer_id,
        )
        dlg.exec()

    def open_distillation_dialog(self):
        try:
            require_dialog_support("distillation", self.active_layer)
        except DialogUnavailableError as exc:
            QMessageBox.information(self, exc.title, exc.message)
            return
        dlg = DistillationDialog(self)
        dlg.exec()

    def open_analysis_dialog(self):
        try:
            plan = plan_analysis_dialog(
                project_root=self.project_root,
                app_base_dir=self.app_base_dir,
                layer_id=self.active_layer,
            )
        except DialogUnavailableError as exc:
            QMessageBox.information(self, exc.title, exc.message)
            return
        dlg = AnalysisDialog(
            self,
            project_root=plan.project_root,
            app_base_dir=plan.app_base_dir,
            layer_id=plan.layer_id,
        )
        dlg.exec()

    def open_video_reviewer(self):
        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies."
            )
            return
        plan = plan_video_review_dialog(
            active_layer=self.active_layer,
            layer_model_paths=self.layer_model_paths,
            pose_classes=self.pose_classes,
            pose_keypoints=self.pose_kp_names,
            pose_class_keypoints=self.pose_class_keypoints,
            segmentation_classes=self.seg_classes,
        )
        reviewer_schema = plan.active_schema
        dlg = VideoReviewDialog(
            self,
            self._device,
            reviewer_schema["kp_names"],
            reviewer_schema["classes"],
            class_keypoints=reviewer_schema["class_keypoints"],
            workflow=plan.workflow,
            layer_id=plan.layer_id,
            model_paths=plan.model_paths,
            layer_schemas=plan.layer_schemas,
        )
        dlg.exec()


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    from squeakpose.app import run

    raise SystemExit(run())
