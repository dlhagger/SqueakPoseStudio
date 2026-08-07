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
    QPainterPathStroker,
    QPen,
    QPixmap,
    QShortcut,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QMainWindow,
    QMenu,
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
from dataset_ops import (
    DATASET_DETECT,
    DATASET_POSE,
    DATASET_SEGMENT,
    backup_label_dir,
    cleanup_project_temporary_paths,
    dataset_dirs_have_files,
    dataset_export_paths,
    format_dataset_export_summary,
    format_label_normalization_summary,
    format_project_health_summary,
    label_file_has_usable_rows,
    list_image_files,
    list_label_files,
    normalize_label_directory,
    partition_images_by_usable_labels,
    scan_project_health,
    split_train_val_images,
)
from depth_ops import DepthMapError, keypoint_depth_label, sample_depth_map
from inference_ops import (
    probe_video_metadata,
    segmentation_rows_from_result,
)
from label_io import (
    load_pose_annotations_from_file,
    load_segmentation_annotations_from_file,
    parse_pose_label_line,
    parse_segmentation_label_line,
    pose_annotation_to_line,
    segmentation_annotation_to_line,
)
from prediction_ops import rank_prediction_frames
from squeakpose import __version__
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
from squeakpose.annotation.video_view import VideoView
from squeakpose.diagnostics import configure_project_logging, project_log_path
from squeakpose.project.distillation import (
    discover_distillation_exports,
    distillation_export_search_roots,
    distillation_sample_count,
    preferred_distillation_export,
)
from squeakpose.project.layers import (
    LAYER_DEFINITIONS,
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_definition,
    layer_model_paths,
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
from squeakpose.project.safety import (
    ProjectLock,
    ProjectLockedError,
    ProjectPathError,
    break_stale_project_lock,
    canonical_path,
)
from squeakpose.services.annotation_save import (
    AnnotationSaveRequest,
    save_annotation_transaction,
)
from squeakpose.services.dataset import export_dataset_transaction
from squeakpose.ui.class_manager import (
    AddClassDialog,
    ClassManagerDialog,
)
from squeakpose.ui.distillation_dialog import DistillationDialog
from squeakpose.ui.project_launcher import (
    ProjectLauncherDialog,
    choose_project_root,
    create_project_root,
)
from squeakpose.ui.project_models_dialog import ProjectModelsDialog
from squeakpose.ui.training_dialog import TrainDialog
from squeakpose.ui.video_reviewer import VideoReviewDialog
from squeakpose.workers.process import (
    remove_file_quietly,
    request_qprocess_stop,
    shutdown_qprocess,
)
from squeakpose.workers.protocol import WorkerProtocolError, parse_event_line
from squeakpose_core import (
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
from ui_style import (
    ThemedComboBox,
    apply_panel_shadow,
    hud_stylesheet,
    sidebar_stylesheet,
    style_combo_popup,
    train_dialog_stylesheet,
)

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

try:
    from ultralytics import SAM
except Exception:
    SAM = None

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
    def active_workflow(self) -> str:
        """Compatibility view of the active layer for existing workers/dialogs."""

        return layer_worker_mode(getattr(self, "active_layer", LAYER_KEYPOINTS))

    @active_workflow.setter
    def active_workflow(self, value: str) -> None:
        self.active_layer = normalize_layer_id(value)

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
            label.setText("No saved depth range · Near = bright")
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            low = float(metadata["p02_depth"])
            high = float(metadata["p98_depth"])
            median = float(metadata["median_depth"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            label.setText("Depth range unavailable · Near = bright")
            return
        label.setText(
            f"Range (2–98%): {low:.3f}–{high:.3f} m · median {median:.3f} m · Near = bright"
        )

    def _refresh_depth_probe_label(self) -> None:
        label = getattr(self, "depth_probe_label", None)
        if label is None:
            return
        probes = list(getattr(self, "_depth_probes", []))
        if not probes:
            error = str(getattr(self, "_depth_probe_error", "") or "")
            label.setText(error if error else "Right-click the image to sample raw depth.")
        else:
            lines = ["Pixel probes:"]
            for index, probe in enumerate(probes, start=1):
                value = probe.get("depth")
                value_text = f"{float(value):.3f} m" if value is not None else "invalid"
                lines.append(f"{index}. ({probe['x']}, {probe['y']}): {value_text}")
            valid = [
                float(probe["depth"]) for probe in probes[-2:] if probe.get("depth") is not None
            ]
            if len(valid) == 2:
                lines.append(f"Δ last two: {abs(valid[1] - valid[0]):.3f} m")
            label.setText("\n".join(lines))
        button = getattr(self, "depth_clear_probes_btn", None)
        if button is not None:
            button.setEnabled(bool(probes))

    def _clear_depth_probe_items(self) -> None:
        for item in list(getattr(self, "_depth_probe_items", [])):
            self._safe_remove_scene_item(item)
        self._depth_probe_items = []

    def _clear_depth_probes(self, _checked: bool = False) -> None:
        self._clear_depth_probe_items()
        self._depth_probes = []
        self._refresh_depth_probe_label()

    def _render_depth_probes(self) -> None:
        self._clear_depth_probe_items()
        if not self._is_depth_layer() or not hasattr(self, "scene"):
            return
        colors = (
            QColor("#73d7ff"),
            QColor("#ffd166"),
            QColor("#82e0aa"),
            QColor("#ff8fab"),
            QColor("#c7a0ff"),
            QColor("#f6bd60"),
        )
        for index, probe in enumerate(self._depth_probes, start=1):
            color = colors[(index - 1) % len(colors)]
            marker = QGraphicsEllipseItem(-5.0, -5.0, 10.0, 10.0)
            marker.setPos(float(probe["x"]) + 0.5, float(probe["y"]) + 0.5)
            marker.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            marker.setAcceptHoverEvents(False)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(2)
            marker.setPen(pen)
            marker.setBrush(QBrush(QColor(10, 15, 18, 190)))
            marker.setZValue(20.0)
            value = probe.get("depth")
            value_text = f"{float(value):.3f} m" if value is not None else "invalid"
            text_item = QGraphicsSimpleTextItem(f"{index} · {value_text}")
            text_item.setBrush(QBrush(color))
            text_item.setPos(
                float(probe["x"]) + 8.5,
                float(probe["y"]) - 10.5,
            )
            text_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            text_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            text_item.setAcceptHoverEvents(False)
            text_item.setZValue(20.0)
            self.scene.addItem(marker)
            self.scene.addItem(text_item)
            self._depth_probe_items.extend((marker, text_item))

    def _probe_depth_at(self, scene_pos: QPointF) -> bool:
        if not self._is_depth_layer():
            return False
        depth_map = getattr(self, "_active_depth_map", None)
        if depth_map is None or _np is None:
            self.update_status_bar("No aligned raw depth map is available for pixel sampling.")
            return True
        try:
            probe = sample_depth_map(
                depth_map,
                x=float(scene_pos.x()),
                y=float(scene_pos.y()),
                numpy_module=_np,
            )
        except DepthMapError as exc:
            self.update_status_bar(str(exc))
            return True
        self._depth_probes.append(probe)
        self._depth_probes = self._depth_probes[-6:]
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
        self.active_layer = normalize_layer_id(
            meta.get("active_layer") or meta.get("active_workflow")
        )
        self.layer_settings = normalize_layer_settings(meta.get("layers"))
        self.layer_model_paths = layer_model_paths(
            self.layer_settings,
            resolve_path=self._resolve_model_reference,
        )
        self.layer_model_paths = {
            layer_id: (
                path
                if path and (os.path.isfile(path) or self._is_builtin_model_reference(path))
                else ""
            )
            for layer_id, path in self.layer_model_paths.items()
        }
        raw_visibility = meta.get("layer_visibility")
        if isinstance(raw_visibility, dict):
            for layer_id in LAYER_DEFINITIONS:
                if layer_id in raw_visibility:
                    self.layer_visibility[layer_id] = bool(raw_visibility[layer_id])
        sam_path = self._meta_normalize_path(str(meta.get("sam_model_path", "") or ""))
        if not sam_path:
            sam_path = self._meta_normalize_path(
                str(self.layer_settings[LAYER_SEGMENTATION].get("assistant_model_path") or "")
            )
        if sam_path and os.path.isfile(sam_path):
            self.sam_model_path = sam_path

    def _save_project_preferences(self):
        active_layer = normalize_layer_id(getattr(self, "active_layer", LAYER_KEYPOINTS))
        if hasattr(self, "layer_model_paths"):
            self.layer_model_paths[active_layer] = str(
                getattr(self, "predict_model_path", "") or ""
            )
        layer_settings = normalize_layer_settings(getattr(self, "layer_settings", {}))
        for layer_id in LAYER_DEFINITIONS:
            model_path = str(getattr(self, "layer_model_paths", {}).get(layer_id) or "")
            if model_path:
                layer_settings[layer_id]["model_path"] = self._store_model_reference(model_path)
            else:
                layer_settings[layer_id].pop("model_path", None)
        payload = {
            "active_layer": active_layer,
            "active_workflow": layer_worker_mode(active_layer),
            "layers": layer_settings,
            "layer_visibility": dict(self.layer_visibility),
        }
        if self.sam_model_path and os.path.isfile(self.sam_model_path):
            payload["sam_model_path"] = self._meta_store_path(self.sam_model_path)
            layer_settings[LAYER_SEGMENTATION]["assistant_model_path"] = self._meta_store_path(
                self.sam_model_path
            )
        else:
            payload["sam_model_path"] = None
            layer_settings[LAYER_SEGMENTATION].pop("assistant_model_path", None)
        self.layer_settings = layer_settings
        self._write_project_meta(payload)

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

    def _persist_active_workflow_state(self):
        self._persist_active_layer_state()

    def _bind_layer_state(self, layer_id: str):
        layer_id = normalize_layer_id(layer_id)
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
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        self.class_selector.addItems(self.classes)
        self._fit_class_selector_to_items()
        if current in self.classes:
            self.class_selector.setCurrentIndex(self.classes.index(current))
        elif self.classes:
            self.class_selector.setCurrentIndex(0)
        self.class_selector.blockSignals(False)
        self._active_class_id = self.class_selector.currentIndex()

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
        checks = {
            LAYER_KEYPOINTS: getattr(self, "keypoints_visibility_check", None),
            LAYER_SEGMENTATION: getattr(self, "segmentation_visibility_check", None),
            LAYER_DEPTH: getattr(self, "depth_visibility_check", None),
        }
        active_layer = getattr(self, "active_layer", LAYER_KEYPOINTS)
        self.layer_visibility[active_layer] = True
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
        self.layer_editing_label.setText(
            f"{active_name.upper()} · {'VIEW' if active_layer == LAYER_DEPTH else 'EDITING'}"
        )
        self.layer_reference_label.setText(
            "● " + " + ".join(visible_references) + " references visible"
            if visible_references
            else "○ Reference layers hidden"
        )
        self.layer_context_frame.adjustSize()
        self._layout_hot_corners()

    def _on_layer_visibility_changed(self, layer_id: str, visible: bool) -> None:
        layer_id = normalize_layer_id(layer_id)
        if layer_id == getattr(self, "active_layer", LAYER_KEYPOINTS):
            self.layer_visibility[layer_id] = True
            self._sync_layer_visibility_controls()
            return
        self.layer_visibility[layer_id] = bool(visible)
        self._save_project_preferences()
        self._sync_layer_visibility_controls()
        if hasattr(self, "scene"):
            self._refresh_reference_layer_overlay()

    def _update_layer_ui_state(self):
        is_pose = self._is_keypoints_layer()
        is_segmentation = self._is_segmentation_layer()
        is_depth = self._is_depth_layer()
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
        if layer_id == getattr(self, "active_layer", LAYER_KEYPOINTS):
            return
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
        self._clear_seg_prompt_state()
        self._update_layer_ui_state()
        self.load_image()
        if self._is_segmentation_layer():
            loaded_now, loaded_path = self._try_autoload_sam_model_from_project_root()
            self._refresh_sam_controls()
            if loaded_now:
                self.update_status_bar(
                    f"Segmentation layer selected. Auto-loaded SAM model: {os.path.basename(loaded_path)}"
                )
            elif self.sam_model is not None:
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
                with open(self.class_keypoints_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if not isinstance(raw, dict):
                    raise ValueError("class_keypoints.json must contain a JSON object")
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
        total = len(images)
        labeled = 0
        for img in images:
            base = os.path.splitext(img)[0]
            label_file = os.path.join(label_dir, f"{base}.txt")
            if self._label_file_is_usable(label_file):
                labeled += 1
        return labeled, total

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
        self.progress_label.setText(f"Queue: {queue_labeled}/{queue_total} {noun}")

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
        for item in list(getattr(self, "_reference_layer_items", [])):
            self._safe_remove_scene_item(item)
        self._reference_layer_items = []

    def _add_reference_item(
        self,
        item: QGraphicsItem,
        *,
        layer_id: str,
        opacity: float,
        z_value: float = 1.0,
    ) -> None:
        item.reference_layer_id = normalize_layer_id(layer_id)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setAcceptHoverEvents(False)
        item.setOpacity(opacity)
        item.setZValue(z_value)
        self.scene.addItem(item)
        self._reference_layer_items.append(item)

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
        if depth_pixmap.width() != self.img_w or depth_pixmap.height() != self.img_h:
            depth_pixmap = depth_pixmap.scaled(
                self.img_w,
                self.img_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        depth_item = QGraphicsPixmapItem(depth_pixmap)
        self._add_reference_item(
            depth_item,
            layer_id=LAYER_DEPTH,
            opacity=0.42,
            z_value=0.5,
        )

    def _add_segmentation_reference_overlay(self, base: str) -> None:
        reference_color = QColor(104, 164, 207)
        label_file = os.path.join(self.seg_label_dir, f"{base}.txt")
        if not os.path.isfile(label_file):
            return
        entries = load_segmentation_annotations_from_file(
            label_file,
            classes_count=len(self.seg_classes),
            img_w=self.img_w,
            img_h=self.img_h,
        )
        for cid, entry in entries.items():
            points = []
            for pair in entry.get("segments", []):
                try:
                    points.append((float(pair[0]), float(pair[1])))
                except Exception:
                    continue
            path = self._polygon_path(points)
            if path is None:
                continue
            color = reference_color
            item = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(104, 164, 207, 48)))
            item.seg_class_id = int(cid)
            item.seg_points = points
            item.seg_preview = False
            label = self.seg_classes[cid] if 0 <= cid < len(self.seg_classes) else f"class_{cid}"
            label_item = QGraphicsSimpleTextItem(f"{label} · Segmentation", item)
            label_item.setBrush(QBrush(color))
            label_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            label_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            label_item.setPos(
                path.boundingRect().left() + 4.0,
                path.boundingRect().top() + 4.0,
            )
            label_item.setVisible(False)
            item.seg_label_item = label_item
            self._add_reference_item(
                item,
                layer_id=LAYER_SEGMENTATION,
                opacity=0.50,
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
        entries, _extra_rows = load_pose_annotations_from_file(
            label_file,
            classes_count=len(self.pose_classes),
            canonical_names=self.pose_kp_names,
            class_keypoint_lookup=class_lookup,
            img_w=self.img_w,
            img_h=self.img_h,
        )
        for cid, entry in entries.items():
            bbox_data = entry.get("bbox", {})
            bbox = BoundingBox(
                bbox_data.get("x", 0.0),
                bbox_data.get("y", 0.0),
                bbox_data.get("w", 0.0),
                bbox_data.get("h", 0.0),
                cid,
            )
            class_name = self.pose_classes[cid] if 0 <= cid < len(self.pose_classes) else str(cid)
            box_item = BoxItem(bbox, f"{class_name} · Keypoints")
            box_item.set_reference_style(reference_color, show_label=False)
            self._add_reference_item(
                box_item,
                layer_id=LAYER_KEYPOINTS,
                opacity=0.52,
            )
            for kp_info in entry.get("keypoints", []):
                name = str(kp_info.get("name") or "")
                kp = Keypoint(
                    kp_info.get("x", 0.0),
                    kp_info.get("y", 0.0),
                    cid,
                    name,
                )
                kp_item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
                kp_item.visibility = int(kp_info.get("vis", 2))
                kp_item.update_appearance()
                if show_depth_labels and kp_item.visibility > 0:
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
                    kp_item.text_item.setText(display_name)
                kp_item.set_reference_style(
                    reference_color,
                    show_label=show_depth_labels,
                )
                self._add_reference_item(
                    kp_item,
                    layer_id=LAYER_KEYPOINTS,
                    opacity=0.90 if show_depth_labels else 0.52,
                )

    # ---------- Annotation helpers ----------

    def _seg_class_color(self, class_id: int, alpha: int = 255) -> QColor:
        hue = int((class_id * 47) % 360)
        color = QColor.fromHsv(hue, 210, 245, alpha)
        return color

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
        if len(points) < 3:
            return None
        path = QPainterPath()
        first_x, first_y = points[0]
        path.moveTo(first_x, first_y)
        for x, y in points[1:]:
            path.lineTo(x, y)
        path.closeSubpath()
        return path

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
        max_w = max(1, int(round(float(self.img_w))) - 1)
        max_h = max(1, int(round(float(self.img_h))) - 1)
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0:
            xi = 0
        elif xi > max_w:
            xi = max_w
        if yi < 0:
            yi = 0
        elif yi > max_h:
            yi = max_h
        return xi, yi

    def _seg_mask_shape(self) -> tuple[int, int]:
        mask_h = max(1, int(round(float(self.img_h))))
        mask_w = max(1, int(round(float(self.img_w))))
        return mask_h, mask_w

    def _seg_mask_from_points(self, points: list[tuple[float, float]]) -> Optional[object]:
        if _np is None or _cv2 is None or len(points) < 3:
            return None
        mask_h, mask_w = self._seg_mask_shape()
        mask = _np.zeros((mask_h, mask_w), dtype=_np.uint8)
        poly = _np.array(points, dtype=_np.int32).reshape((-1, 1, 2))
        _cv2.fillPoly(mask, [poly], 255)
        return mask

    def _seg_points_from_mask(
        self,
        mask: object,
        anchor_points: Optional[list[tuple[float, float]]] = None,
    ) -> list[tuple[float, float]]:
        if _cv2 is None or mask is None:
            return []
        contours_info = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return []
        contour = None
        anchor = anchor_points or []
        if anchor:
            ax, ay = float(anchor[0][0]), float(anchor[0][1])
            anchored: list[object] = []
            for c in contours:
                try:
                    inside = _cv2.pointPolygonTest(c, (ax, ay), False)
                except Exception:
                    inside = -1
                if inside >= 0:
                    anchored.append(c)
            if anchored:
                contour = max(anchored, key=_cv2.contourArea)
        if contour is None:
            contour = max(contours, key=_cv2.contourArea)
        if contour is None or len(contour) < 3:
            return []

        points: list[tuple[float, float]] = []
        for node in contour:
            try:
                x = float(node[0][0])
                y = float(node[0][1])
            except Exception:
                continue
            points.append((x, y))
        if len(points) < 3:
            return []
        points = self._downsample_seg_points(points, max_points=1200)

        if len(anchor) >= 3 and len(points) >= 3:
            old_area = self._polygon_signed_area(anchor)
            new_area = self._polygon_signed_area(points)
            if old_area * new_area < 0:
                points.reverse()
            points = self._rotate_polygon_to_anchor(points, anchor[0])
        return points

    def _downsample_seg_points(
        self, points: list[tuple[float, float]], max_points: int = 1200
    ) -> list[tuple[float, float]]:
        if len(points) <= max_points:
            return points
        step = max(1, (len(points) + max_points - 1) // max_points)
        reduced = points[::step]
        if len(reduced) < 3:
            return points[:3]
        return reduced

    def _polygon_signed_area(self, points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        total = 0.0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            total += (float(x1) * float(y2)) - (float(x2) * float(y1))
        return 0.5 * total

    def _rotate_polygon_to_anchor(
        self,
        points: list[tuple[float, float]],
        anchor: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if not points:
            return points
        ax, ay = float(anchor[0]), float(anchor[1])
        best_idx = 0
        best_d2 = float("inf")
        for idx, (x, y) in enumerate(points):
            d2 = ((float(x) - ax) ** 2) + ((float(y) - ay) ** 2)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        if best_idx == 0:
            return points
        return points[best_idx:] + points[:best_idx]

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

    def _position_seg_visual_frame(self, item: QGraphicsPathItem, path: QPainterPath) -> None:
        """Keep the segmentation frame and class badge aligned to its mask."""
        frame_item = getattr(item, "seg_frame_item", None)
        label_bg = getattr(item, "seg_label_bg", None)
        label_item = getattr(item, "seg_label_item", None)
        if frame_item is None or label_bg is None or label_item is None:
            return

        bbox = path.boundingRect()
        frame_item.setRect(bbox)

        pad_x = 4.0
        pad_y = 1.0
        margin = 2.0
        text_rect = label_item.boundingRect()
        badge_w = text_rect.width() + (pad_x * 2.0)
        badge_h = text_rect.height() + (pad_y * 2.0)
        badge_x = bbox.left() + margin
        badge_y = bbox.top() - badge_h - margin
        if hasattr(self, "scene") and badge_y < self.scene.sceneRect().top():
            badge_y = bbox.bottom() + margin
        label_bg.setRect(badge_x, badge_y, badge_w, badge_h)
        label_item.setPos(
            badge_x + pad_x,
            badge_y + pad_y,
        )

    def _add_seg_visual_frame(
        self,
        item: QGraphicsPathItem,
        path: QPainterPath,
        label_text: str,
        *,
        preview: bool,
    ) -> None:
        """Add the same blue object frame/badge used by keypoint output."""
        frame_color = QColor(32, 78, 255)

        frame_item = QGraphicsRectItem(item)
        frame_pen = QPen(frame_color)
        frame_pen.setWidth(2)
        frame_pen.setCosmetic(True)
        if preview:
            frame_pen.setStyle(Qt.PenStyle.DashLine)
        frame_item.setPen(frame_pen)
        frame_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        frame_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        frame_item.setAcceptHoverEvents(False)
        frame_item.setZValue(0.3)

        label_bg = QGraphicsRectItem(item)
        label_bg.setBrush(QBrush(frame_color))
        label_bg.setPen(QPen(Qt.PenStyle.NoPen))
        label_bg.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        label_bg.setAcceptHoverEvents(False)
        label_bg.setZValue(0.4)

        label_item = QGraphicsSimpleTextItem(label_text, item)
        label_font = QFont()
        label_font.setPixelSize(12)
        label_item.setFont(label_font)
        label_item.setBrush(QBrush(Qt.GlobalColor.white))
        label_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        label_item.setZValue(0.5)

        item.seg_frame_item = frame_item
        item.seg_label_bg = label_bg
        item.seg_label_item = label_item
        self._position_seg_visual_frame(item, path)

    def _seg_update_item_geometry(
        self, item: Optional[QGraphicsPathItem], points: list[tuple[float, float]]
    ) -> bool:
        if item is None:
            return False
        normalized = [(float(x), float(y)) for x, y in points]
        path = self._polygon_path(normalized)
        if path is None:
            return False
        item.seg_points = normalized
        item.setPath(path)
        self._position_seg_visual_frame(item, path)
        return True

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

        mask = active_mask

        x2, y2 = self._clamp_scene_xy(scene_pos.x(), scene_pos.y())
        if prev_scene_pos is not None:
            x1, y1 = self._clamp_scene_xy(prev_scene_pos.x(), prev_scene_pos.y())
        else:
            x1, y1 = x2, y2

        radius = max(2, int(round(float(getattr(self, "seg_brush_radius", 8)))))
        value = 255 if add else 0
        thickness = max(2, radius * 2)
        _cv2.circle(mask, (x2, y2), radius, value, thickness=-1)
        if x1 != x2 or y1 != y2:
            _cv2.line(mask, (x1, y1), (x2, y2), value, thickness=thickness)

        if int(_cv2.countNonZero(mask)) == 0:
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

        anchor_points = getattr(self, "seg_brush_anchor_points", None) or points
        new_points = self._seg_points_from_mask(mask, anchor_points=anchor_points)
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
        path = self._polygon_path(points)
        if path is None:
            return None
        color = self._seg_class_color(class_id)
        item = QGraphicsPathItem(path)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(2 if preview else 3)
        if preview:
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        fill_alpha = 52 if preview else 76
        item.setBrush(QBrush(self._seg_class_color(class_id, alpha=fill_alpha)))
        item.setZValue(4.5 if preview else 4.0)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, not preview)
        item.seg_class_id = int(class_id)
        item.seg_points = [(float(x), float(y)) for x, y in points]
        item.seg_preview = bool(preview)

        label_text = self._seg_class_name(class_id)
        if preview:
            label_text += " (preview)"
        self._add_seg_visual_frame(
            item,
            path,
            label_text,
            preview=preview,
        )

        self.scene.addItem(item)
        self._track_scene_item(item)
        return item

    def _clear_seg_prompt_items(self):
        for item in list(self.seg_prompt_items):
            self._safe_remove_scene_item(item)
            self._untrack_scene_item(item)
        self.seg_prompt_items.clear()

    def _clear_seg_preview(self):
        self._clear_seg_edit_handles()
        if self.seg_preview_item is not None:
            self._safe_remove_scene_item(self.seg_preview_item)
            self._untrack_scene_item(self.seg_preview_item)
            self.seg_preview_item = None
        self.seg_preview_points = []
        self.seg_preview_score = 0.0
        self._refresh_sam_controls()

    def _clear_seg_prompt_state(self):
        self.seg_prompt_points.clear()
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
        model_loaded = self.sam_model is not None

        cid = self.class_selector.currentIndex() if hasattr(self, "class_selector") else -1
        class_name = self._seg_class_name(cid) if cid >= 0 else "class"
        has_mask = False
        if cid >= 0:
            entry = self.annotation_cache.get(cid, {})
            has_mask = len(entry.get("segments", [])) >= 3 or (
                self._class_seg_mask_item(cid) is not None
            )
        completed = sum(1 for idx in range(len(self.classes)) if self._class_is_complete(idx))
        run_enabled = has_image and in_segment_mode and total_prompts > 0 and model_loaded
        accept_enabled = has_preview
        clear_enabled = has_preview or total_prompts > 0
        load_enabled = SAM is not None and not model_loaded
        if hasattr(self, "sam_load_btn"):
            self.sam_load_btn.setEnabled(load_enabled)
            self.sam_load_btn.setText(
                "Load SAM" if load_enabled else ("SAM Ready" if model_loaded else "SAM N/A")
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
        if not model_loaded:
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
        model_text = "ready" if model_loaded else ("missing" if SAM is not None else "unavailable")
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
        radius = 5.0
        color = Qt.GlobalColor.green if positive else Qt.GlobalColor.red
        marker = QGraphicsEllipseItem(x - radius, y - radius, radius * 2.0, radius * 2.0)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(2)
        marker.setPen(pen)
        marker.setBrush(QBrush(Qt.GlobalColor.transparent))
        marker.setZValue(8.0)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.scene.addItem(marker)
        self._track_scene_item(marker)
        self.seg_prompt_items.append(marker)

        if not positive:
            cross_a = QGraphicsLineItem(
                x - radius + 1.0, y - radius + 1.0, x + radius - 1.0, y + radius - 1.0
            )
            cross_b = QGraphicsLineItem(
                x - radius + 1.0, y + radius - 1.0, x + radius - 1.0, y - radius + 1.0
            )
            for line in (cross_a, cross_b):
                lp = QPen(color)
                lp.setCosmetic(True)
                lp.setWidth(2)
                line.setPen(lp)
                line.setZValue(8.1)
                line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                self.scene.addItem(line)
                self._track_scene_item(line)
                self.seg_prompt_items.append(line)

    def _refresh_seg_prompt_markers(self):
        self._clear_seg_prompt_items()
        for x, y, label in self.seg_prompt_points:
            self._draw_seg_prompt_marker(float(x), float(y), positive=bool(label))
        self._refresh_sam_controls()

    def _add_seg_prompt(self, scene_pos: QPointF, positive: bool = True):
        if not self._is_seg_workflow() or self.mode != "segment":
            return
        if not self.images:
            return
        x = max(0.0, min(float(self.img_w - 1), float(scene_pos.x())))
        y = max(0.0, min(float(self.img_h - 1), float(scene_pos.y())))
        label = 1 if positive else 0
        self.seg_prompt_points.append((x, y, label))
        self._draw_seg_prompt_marker(x, y, positive=positive)
        self.update_status_bar(
            f"Added {'positive' if positive else 'negative'} prompt ({len(self.seg_prompt_points)} total)."
        )
        self._refresh_sam_controls()

    def _sam3_model_candidates_in_project_root(self) -> list[str]:
        root = os.path.abspath(getattr(self, "project_root", "") or "")
        if not root or not os.path.isdir(root):
            return []

        exact: list[str] = []
        prefix: list[str] = []
        other: list[str] = []
        try:
            names = sorted(os.listdir(root))
        except Exception:
            names = []
        for name in names:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            lower = name.lower()
            if not (lower.endswith(".pt") or lower.endswith(".pth")):
                continue
            if "sam3" not in lower:
                continue
            if lower == DEFAULT_SAM3_WEIGHTS.lower():
                exact.append(path)
            elif lower.startswith("sam3"):
                prefix.append(path)
            else:
                other.append(path)
        return exact + prefix + other

    def _try_autoload_sam_model_from_project_root(self) -> tuple[bool, str]:
        """Attempt non-interactive SAM autoload from project root.

        Returns (loaded_now, model_path). loaded_now is True only when this call
        actually loads a model instance.
        """
        if SAM is None:
            return False, ""
        if self.sam_model is not None:
            return False, str(getattr(self, "sam_model_path", "") or "")

        for path in self._sam3_model_candidates_in_project_root():
            try:
                self.sam_model = SAM(path)
                self.sam_model_path = path
                self._save_project_preferences()
                return True, path
            except Exception:
                self.sam_model = None
                continue
        return False, ""

    def _ensure_sam_model_loaded(self) -> bool:
        if SAM is None:
            QMessageBox.warning(
                self,
                "SAM unavailable",
                "Ultralytics SAM support is not available in this environment.",
            )
            return False

        if self.sam_model is not None:
            return True

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

        model_path = ""
        seen = set()
        for cand in candidate_paths:
            norm = os.path.abspath(cand)
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.isfile(norm):
                model_path = norm
                break

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

        try:
            self.sam_model = SAM(model_path)
            self.sam_model_path = model_path
            self._save_project_preferences()
            self.update_status_bar(f"Loaded SAM model: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            self.sam_model = None
            QMessageBox.warning(self, "SAM load error", f"Could not load SAM model:\n{e}")
            return False

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
        if not self._ensure_sam_model_loaded():
            self._refresh_sam_controls()
            return
        self._refresh_sam_controls()

        points = [[x, y] for x, y, _ in self.seg_prompt_points]
        labels = [int(lb) for _, _, lb in self.seg_prompt_points]
        img_source = self.current_image_path or os.path.join(
            self.active_image_dir, self.images[self.current_idx]
        )

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = self.sam_model.predict(
                source=img_source,
                points=points,
                labels=labels,
                verbose=False,
            )
        except Exception as e:
            QMessageBox.warning(self, "SAM inference error", f"SAM segmentation failed:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not results:
            QMessageBox.information(
                self, "No masks", "SAM did not return any segmentation mask for these prompts."
            )
            return

        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or len(masks) == 0:
            QMessageBox.information(
                self, "No masks", "SAM did not return any segmentation mask for these prompts."
            )
            return

        best_idx = 0
        best_score = 0.0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
            try:
                conf_values = boxes.conf.cpu().tolist()
                best_idx = max(range(len(conf_values)), key=lambda i: conf_values[i])
                best_score = float(conf_values[best_idx])
            except Exception:
                best_idx = 0
                best_score = 0.0

        points_xy = []
        try:
            contours = masks.xy
            if best_idx < len(contours):
                points_xy = [(float(p[0]), float(p[1])) for p in contours[best_idx] if len(p) >= 2]
        except Exception:
            points_xy = []

        if len(points_xy) < 3:
            QMessageBox.information(
                self, "No polygon", "SAM returned a mask without a usable contour polygon."
            )
            return

        self._clear_seg_preview()
        cid = self.class_selector.currentIndex()
        self.seg_preview_item = self._add_seg_mask_item(cid, points_xy, preview=True)
        self.seg_preview_points = points_xy
        self.seg_preview_score = best_score
        if self.seg_preview_item is None:
            QMessageBox.information(self, "No polygon", "Unable to render SAM mask preview.")
            return
        self.update_status_bar("SAM mask preview ready. Click Accept Mask to commit.")
        self._refresh_sam_controls()

    def _accept_segmentation_preview(self):
        if not self._is_seg_workflow():
            return
        if len(self.seg_preview_points) < 3:
            QMessageBox.information(
                self, "No preview", "Run SAM first to create a segmentation mask preview."
            )
            return

        accepted_points = [(float(x), float(y)) for x, y in self.seg_preview_points]
        accepted_score = float(self.seg_preview_score)
        cid = self.class_selector.currentIndex()
        self._clear_class_items(cid, drop_cache=False)
        self.annotation_cache[cid] = {
            "class_id": cid,
            "segments": accepted_points,
            "score": accepted_score,
        }
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
            self.annotation_cache.pop(class_id, None)
        if class_id == self.class_selector.currentIndex():
            self.bboxes.clear()
            self.kps.clear()
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
        self.current_kp_idx = 0
        self._clear_seg_prompt_state()

    def _sync_active_class_state(self):
        if self._is_seg_workflow():
            self.bboxes = []
            self.kps = []
            self.current_kp_idx = 0
            return
        cid = self.class_selector.currentIndex()
        box_item = self._class_box_item(cid)
        self.bboxes = [box_item.bbox] if box_item else []
        self.kps = [kp_item.kp for kp_item in self._class_keypoint_items(cid)]
        self.current_kp_idx = min(len(self._active_kp_names()), len(self.kps))

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
        entry = self.annotation_cache.get(class_id)
        if not entry:
            return False
        if self._is_seg_workflow():
            seg = entry.get("segments", [])
            return len(seg) >= 3
        needed = len(self._kp_names_for_index(class_id))
        if needed == 0:
            return bool(entry.get("bbox"))
        return len(entry.get("keypoints", [])) == needed

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
            candidates = sorted(list_image_files(self.image_dir_queue))
            self.images_queue, self._queue_stem_collisions = filter_image_stem_collisions(
                candidates
            )
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
        self._reference_layer_items: list[QGraphicsItem] = []
        self.layer_settings = normalize_layer_settings({})
        self.layer_model_paths: dict[str, str] = {layer_id: "" for layer_id in LAYER_DEFINITIONS}
        self.layer_visibility: dict[str, bool] = {layer_id: True for layer_id in LAYER_DEFINITIONS}
        self.predict_model_path: Optional[str] = None
        self.sam_model_path = os.path.join(self.project_root, DEFAULT_SAM3_WEIGHTS)
        self.sam_model: Optional[SAM] = None
        self.seg_prompt_points: list[tuple[float, float, int]] = []
        self.seg_prompt_items: list[QGraphicsItem] = []
        self.seg_preview_item: Optional[QGraphicsPathItem] = None
        self.seg_preview_points: list[tuple[float, float]] = []
        self.seg_preview_score: float = 0.0
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
        self._inference_process: Optional[QProcess] = None
        self._inference_progress: Optional[QProgressDialog] = None
        self._inference_stdout_buffer = ""
        self._inference_stderr = ""
        self._inference_result_event: Optional[dict] = None
        self._inference_config_path: Optional[str] = None
        self._inference_csv_path: Optional[str] = None
        self._inference_mode: str = WORKFLOW_POSE
        self._inference_cancel_requested = False
        self._inference_job_queue: list[dict] = []
        self._inference_active_job: Optional[dict] = None
        self._inference_run_results: list[dict] = []
        self._inference_run_total = 0
        self._inference_run_canceled = False
        self._inference_run_manifest_path = ""
        self._inference_run_video_path = ""
        self._prediction_process: Optional[QProcess] = None
        self._prediction_stdout_buffer = ""
        self._prediction_stderr = ""
        self._prediction_result_event: Optional[dict] = None
        self._prediction_config_path: Optional[str] = None
        self._prediction_image_path: Optional[str] = None
        self._prediction_cancel_requested = False
        self._prediction_worker_ready = False
        self._prediction_pending_request: Optional[dict] = None
        self._prediction_request_counter = 0
        self._prediction_current_request_id: Optional[int] = None
        self._prediction_expected_stop = False
        self._prediction_depth_targets: Optional[dict[str, str]] = None
        self._active_depth_map = None
        self._depth_probes: list[dict] = []
        self._depth_probe_items: list[QGraphicsItem] = []
        self._depth_probe_image_name = ""
        self._depth_probe_error = ""
        # Auto-select device once at startup
        self._device = _auto_device()
        print(f"🧠 Inference device: {self._device}")
        # Build UI and load first image
        self._setup_ui()
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
        self._inference_run_canceled = True
        self._inference_job_queue.clear()
        _shutdown_qprocess(self._inference_process)
        self._prediction_expected_stop = True
        _shutdown_qprocess(self._prediction_process)
        _remove_file_quietly(self._inference_config_path)
        _remove_file_quietly(self._prediction_config_path)
        self._cleanup_prediction_depth_staging()
        self._inference_process = None
        self._prediction_process = None
        self._inference_config_path = None
        self._prediction_config_path = None
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

        def prepare_panel_button(button: QPushButton, min_height: int = 30) -> QPushButton:
            button.setMinimumHeight(min_height)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return button

        # Shared widgets/state
        self.class_selector = ThemedComboBox()
        self.class_selector.setObjectName("classSelector")
        self.class_selector.addItems(self.classes)
        self.class_selector.setToolTip("Choose the active class to label")
        self.class_selector.setMinimumContentsLength(12)
        self.class_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.class_selector.setMinimumWidth(0)
        self.class_selector.setMinimumHeight(34)
        self.class_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.class_selector.setMaxVisibleItems(8)
        class_popup = QListView(self.class_selector)
        class_popup.setUniformItemSizes(True)
        class_popup.setSpacing(2)
        class_popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        class_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        style_combo_popup(class_popup)
        self.class_selector.setView(class_popup)
        self._fit_class_selector_to_items()
        self.class_selector.currentIndexChanged.connect(self._on_class_changed)
        self._active_class_id = self.class_selector.currentIndex()
        self.workflow_selector = ThemedComboBox()
        self.workflow_selector.setObjectName("workflowSelector")
        self.workflow_selector.addItem("Keypoints Layer", LAYER_KEYPOINTS)
        self.workflow_selector.addItem("Segmentation Layer", LAYER_SEGMENTATION)
        self.workflow_selector.addItem("Depth Layer", LAYER_DEPTH)
        self.workflow_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.workflow_selector.setMinimumContentsLength(18)
        self.workflow_selector.setMinimumWidth(0)
        self.workflow_selector.setMinimumHeight(34)
        self.workflow_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.workflow_selector.setMaxVisibleItems(6)
        workflow_popup = QListView(self.workflow_selector)
        workflow_popup.setUniformItemSizes(True)
        workflow_popup.setSpacing(2)
        workflow_popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        workflow_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        style_combo_popup(workflow_popup)
        self.workflow_selector.setView(workflow_popup)
        self.workflow_selector.setToolTip(
            "Choose the annotation layer to edit. Each layer keeps its own labels, model, dataset, and analysis context."
        )
        self.workflow_selector.currentIndexChanged.connect(self._on_layer_changed)

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

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        self.filter_combo = ThemedComboBox()
        self.filter_combo.setObjectName("browseSelector")
        self.filter_combo.addItems(["All", "Labeled", "Unlabeled"])
        self.filter_combo.setToolTip("Which images to browse with Prev/Next")
        self.filter_combo.setMinimumContentsLength(10)
        self.filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.filter_combo.setMinimumWidth(0)
        self.filter_combo.setMinimumHeight(34)
        self.filter_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.filter_combo.setMaxVisibleItems(8)
        filter_popup = QListView(self.filter_combo)
        filter_popup.setUniformItemSizes(True)
        filter_popup.setSpacing(2)
        filter_popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        filter_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        style_combo_popup(filter_popup)
        self.filter_combo.setView(filter_popup)
        self.filter_combo.currentTextChanged.connect(lambda t: self._set_nav_filter(t.lower()))
        browse_label = QLabel("Browse")
        browse_label.setObjectName("fieldLabel")
        filter_row.addWidget(browse_label)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch(1)
        top_left_layout.addLayout(filter_row)

        workflow_row = QHBoxLayout()
        workflow_row.setSpacing(6)
        layer_label = QLabel("Layer")
        layer_label.setObjectName("fieldLabel")
        workflow_row.addWidget(layer_label)
        workflow_row.addWidget(self.workflow_selector, 1)
        top_left_layout.addLayout(workflow_row)

        visibility_row = QHBoxLayout()
        visibility_row.setSpacing(8)
        visibility_label = QLabel("Layers")
        visibility_label.setObjectName("fieldLabel")
        visibility_row.addWidget(visibility_label)
        self.keypoints_visibility_check = QPushButton("● Keypoints")
        self.segmentation_visibility_check = QPushButton("● Segmentation")
        self.depth_visibility_check = QPushButton("● Depth")
        for layer_button in (
            self.keypoints_visibility_check,
            self.segmentation_visibility_check,
            self.depth_visibility_check,
        ):
            layer_button.setCheckable(True)
            layer_button.setProperty("layerVisibilityPill", True)
            layer_button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        self.keypoints_visibility_check.setChecked(True)
        self.segmentation_visibility_check.setChecked(True)
        self.depth_visibility_check.setChecked(True)
        self.keypoints_visibility_check.toggled.connect(
            lambda checked: self._on_layer_visibility_changed(LAYER_KEYPOINTS, checked)
        )
        self.segmentation_visibility_check.toggled.connect(
            lambda checked: self._on_layer_visibility_changed(LAYER_SEGMENTATION, checked)
        )
        self.depth_visibility_check.toggled.connect(
            lambda checked: self._on_layer_visibility_changed(LAYER_DEPTH, checked)
        )
        visibility_row.addWidget(self.keypoints_visibility_check)
        visibility_row.addWidget(self.segmentation_visibility_check)
        visibility_row.addWidget(self.depth_visibility_check)
        visibility_row.addStretch(1)
        top_left_layout.addLayout(visibility_row)

        nav_grid = QGridLayout()
        nav_grid.setHorizontalSpacing(6)
        nav_grid.setVerticalSpacing(6)
        btn_prev = QPushButton("◀ Prev")
        btn_prev.clicked.connect(self.prev_index)
        nav_grid.addWidget(btn_prev, 0, 0)

        btn_next = QPushButton("Next ▶")
        btn_next.clicked.connect(self.next_index)
        nav_grid.addWidget(btn_next, 0, 1)

        self.complete_btn = QPushButton("Complete")
        self.complete_btn.setToolTip("Save and jump to next unlabeled image")
        self.complete_btn.clicked.connect(self.complete_and_next_unlabeled)
        nav_grid.addWidget(self.complete_btn, 0, 2)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setToolTip("Jump to next unlabeled image")
        self.skip_btn.clicked.connect(self.skip_to_next_unlabeled)
        nav_grid.addWidget(self.skip_btn, 1, 0)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_labels)
        nav_grid.addWidget(self.save_btn, 1, 1)

        self.delete_image_btn = QPushButton("Delete Image")
        self.delete_image_btn.setToolTip("Delete the current image after confirmation")
        self.delete_image_btn.clicked.connect(self.delete_current_image)
        nav_grid.addWidget(self.delete_image_btn, 1, 2)
        for btn in (
            btn_prev,
            btn_next,
            self.complete_btn,
            self.skip_btn,
            self.save_btn,
            self.delete_image_btn,
        ):
            prepare_panel_button(btn, min_height=30)
        top_left_layout.addLayout(nav_grid)

        mode_section = QLabel("Mode")
        mode_section.setObjectName("sectionLabel")
        top_left_layout.addWidget(mode_section)

        mode_grid = QGridLayout()
        self.mode_grid = mode_grid
        mode_grid.setHorizontalSpacing(5)
        mode_grid.setVerticalSpacing(5)
        self.panzoom_btn = QPushButton("Pan/Zoom (1)")
        self.bbox_btn = QPushButton("BBox (2)")
        self.segment_btn = QPushButton("Segment (2)")
        self.segment_btn.setToolTip("Segmentation click prompts (left=positive, right=negative)")
        self.seg_edit_btn = QPushButton("Edit Mask (E)")
        self.seg_edit_btn.setToolTip("Manual mask edit mode using brush add/erase.")
        self.keypoint_btn = QPushButton("Keypoint (3)")
        self.predict_btn = QPushButton("Predict (4)")
        for btn, mode_name in [
            (self.panzoom_btn, "panzoom"),
            (self.bbox_btn, "bbox"),
            (self.keypoint_btn, "keypoint"),
        ]:
            btn.clicked.connect(lambda checked, m=mode_name: self.set_mode(m))
            btn.setMinimumWidth(116)
            btn.setMinimumHeight(28)
        self.segment_btn.clicked.connect(lambda checked: self.set_mode("segment"))
        self.seg_edit_btn.clicked.connect(lambda checked: self.set_mode("segedit"))
        self.predict_btn.clicked.connect(lambda checked: self.set_mode("predict"))
        self.panzoom_btn.setMinimumWidth(116)
        self.bbox_btn.setMinimumWidth(116)
        self.segment_btn.setMinimumWidth(116)
        self.seg_edit_btn.setMinimumWidth(116)
        self.keypoint_btn.setMinimumWidth(116)
        self.predict_btn.setMinimumWidth(116)
        self.segment_btn.setMinimumHeight(28)
        self.seg_edit_btn.setMinimumHeight(28)
        self.predict_btn.setMinimumHeight(28)
        mode_grid.addWidget(self.panzoom_btn, 0, 0)
        mode_grid.addWidget(self.bbox_btn, 0, 1)
        mode_grid.addWidget(self.keypoint_btn, 1, 0)
        mode_grid.addWidget(self.predict_btn, 1, 1)
        mode_grid.addWidget(self.segment_btn, 2, 0, 1, 2)
        mode_grid.addWidget(self.seg_edit_btn, 3, 0, 1, 2)
        top_left_layout.addLayout(mode_grid)
        self._reflow_mode_grid(is_pose=self._is_pose_workflow())

        self.class_controls_frame = QFrame()
        class_controls = QVBoxLayout(self.class_controls_frame)
        class_controls.setContentsMargins(0, 0, 0, 0)
        class_controls.setSpacing(5)
        self.class_label_widget = QLabel("Class")
        self.class_label_widget.setObjectName("fieldLabel")
        class_controls.addWidget(self.class_label_widget)
        class_controls.addWidget(self.class_selector)
        self.manage_classes_btn = QPushButton("Classes…")
        self.manage_classes_btn.setToolTip("Manage classes and per-class keypoints")
        self.manage_classes_btn.clicked.connect(self.open_class_manager)
        prepare_panel_button(self.manage_classes_btn, min_height=30)
        class_controls.addWidget(self.manage_classes_btn)
        top_left_layout.addWidget(self.class_controls_frame)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(6)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progressBadge")
        progress_row.addWidget(self.progress_label)
        progress_row.addStretch(1)
        top_left_layout.addLayout(progress_row)
        self.left_sidebar_layout.addWidget(self.top_left_frame)

        self.depth_display_frame = QFrame(self.left_sidebar_content)
        self.depth_display_frame.setObjectName("ToolPanel")
        self.depth_display_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.depth_display_frame)
        depth_display_layout = QVBoxLayout(self.depth_display_frame)
        depth_display_layout.setContentsMargins(10, 9, 10, 9)
        depth_display_layout.setSpacing(6)
        depth_display_title = QLabel("Depth Display")
        depth_display_title.setObjectName("panelTitle")
        depth_display_layout.addWidget(depth_display_title)
        self.depth_display_combo = ThemedComboBox()
        self.depth_display_combo.addItem("Original", "original")
        self.depth_display_combo.addItem("Depth", "depth")
        self.depth_display_combo.addItem("Overlay", "overlay")
        self.depth_display_combo.setToolTip(
            "Compare the source image, standalone depth map, or a blended overlay"
        )
        self.depth_display_combo.setMinimumHeight(32)
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
        self.depth_display_combo.setCurrentIndex(
            max(0, self.depth_display_combo.findData(initial_depth_mode))
        )
        self.depth_display_combo.currentIndexChanged.connect(self._on_depth_view_changed)
        depth_display_layout.addWidget(self.depth_display_combo)
        self.left_sidebar_layout.addWidget(self.depth_display_frame)

        self.depth_range_frame = QFrame(self.left_sidebar_content)
        self.depth_range_frame.setObjectName("ToolPanel")
        self.depth_range_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.depth_range_frame)
        depth_range_layout = QVBoxLayout(self.depth_range_frame)
        depth_range_layout.setContentsMargins(10, 9, 10, 9)
        depth_range_layout.setSpacing(6)
        depth_range_title = QLabel("Depth Range")
        depth_range_title.setObjectName("panelTitle")
        depth_range_layout.addWidget(depth_range_title)
        self.depth_range_label = QLabel("No saved depth range · Near = bright")
        self.depth_range_label.setWordWrap(True)
        self.depth_range_label.setStyleSheet("color: #9fb0bd; font-size: 9pt;")
        self.depth_range_label.setToolTip(
            "Depth values are estimated meters. The preview uses inverse depth, "
            "so brighter colors indicate surfaces closer to the camera."
        )
        depth_range_layout.addWidget(self.depth_range_label)
        self.depth_probe_label = QLabel("Right-click the image to sample raw depth.")
        self.depth_probe_label.setWordWrap(True)
        self.depth_probe_label.setStyleSheet("color: #c8d4dc; font-size: 9pt;")
        depth_range_layout.addWidget(self.depth_probe_label)
        self.depth_clear_probes_btn = QPushButton("Clear Probes")
        self.depth_clear_probes_btn.setToolTip("Remove depth sample markers from the current image")
        self.depth_clear_probes_btn.clicked.connect(self._clear_depth_probes)
        self.depth_clear_probes_btn.setEnabled(False)
        prepare_panel_button(self.depth_clear_probes_btn, min_height=28)
        depth_range_layout.addWidget(self.depth_clear_probes_btn)
        self.left_sidebar_layout.addWidget(self.depth_range_frame)

        self.depth_assistant_frame = QFrame(self.left_sidebar_content)
        self.depth_assistant_frame.setObjectName("ToolPanel")
        self.depth_assistant_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.depth_assistant_frame)
        depth_assistant_layout = QVBoxLayout(self.depth_assistant_frame)
        depth_assistant_layout.setContentsMargins(10, 9, 10, 9)
        depth_assistant_layout.setSpacing(6)
        depth_assistant_title = QLabel("Depth Assistant")
        depth_assistant_title.setObjectName("panelTitle")
        depth_assistant_layout.addWidget(depth_assistant_title)
        self.depth_model_status_label = QLabel("")
        self.depth_model_status_label.setWordWrap(True)
        self.depth_model_status_label.setObjectName("fieldLabel")
        depth_assistant_layout.addWidget(self.depth_model_status_label)
        depth_model_grid = QGridLayout()
        depth_model_grid.setHorizontalSpacing(6)
        depth_model_grid.setVerticalSpacing(6)
        self.depth_official_model_btn = QPushButton("YOLO26 Depth ▾")
        self.depth_official_model_btn.setToolTip(
            "Choose an official depth model; Ultralytics downloads it on first use"
        )
        depth_model_menu = QMenu(self.depth_official_model_btn)
        for size, description in (
            ("n", "Nano — fastest"),
            ("s", "Small"),
            ("m", "Medium"),
            ("l", "Large"),
            ("x", "Extra large — most accurate"),
        ):
            action = depth_model_menu.addAction(description)
            action.triggered.connect(
                lambda _checked=False, model_size=size: self._set_depth_model_path(
                    f"yolo26{model_size}-depth.pt"
                )
            )
        self.depth_official_model_btn.setMenu(depth_model_menu)
        depth_model_grid.addWidget(self.depth_official_model_btn, 0, 0)
        self.depth_choose_model_btn = QPushButton("Choose…")
        self.depth_choose_model_btn.setToolTip("Choose a custom Ultralytics depth checkpoint")
        self.depth_choose_model_btn.clicked.connect(self._choose_depth_model_interactive)
        depth_model_grid.addWidget(self.depth_choose_model_btn, 0, 1)
        self.depth_clear_model_btn = QPushButton("Clear Model")
        self.depth_clear_model_btn.clicked.connect(lambda: self._set_depth_model_path(""))
        depth_model_grid.addWidget(self.depth_clear_model_btn, 1, 0, 1, 2)
        for button in (
            self.depth_official_model_btn,
            self.depth_choose_model_btn,
            self.depth_clear_model_btn,
        ):
            prepare_panel_button(button, min_height=30)
        depth_assistant_layout.addLayout(depth_model_grid)
        self.left_sidebar_layout.addWidget(self.depth_assistant_frame)
        self._refresh_depth_assistant_controls()

        # -----------------------------
        # Top-right: video tools
        # -----------------------------
        self.top_right_frame = QFrame(self.right_sidebar_content)
        self.top_right_frame.setObjectName("ToolPanel")
        self.top_right_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.top_right_frame)
        top_right_layout = QVBoxLayout(self.top_right_frame)
        top_right_layout.setContentsMargins(12, 11, 12, 14)
        top_right_layout.setSpacing(8)
        top_right_title = QLabel("Video")
        top_right_title.setObjectName("panelTitle")
        top_right_layout.addWidget(top_right_title)
        btn_video = QPushButton("Video Reviewer")
        btn_video.setToolTip(
            "Run the configured project models over a video and review their overlays together"
        )
        btn_video.setMinimumHeight(34)
        btn_video.clicked.connect(self.open_video_reviewer)
        top_right_layout.addWidget(btn_video)
        top_right_layout.addSpacing(2)
        self.right_sidebar_layout.addWidget(self.top_right_frame)

        # -----------------------------
        # Right: layer-aware analysis
        # -----------------------------
        self.analysis_frame = QFrame(self.right_sidebar_content)
        self.analysis_frame.setObjectName("ToolPanel")
        self.analysis_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.analysis_frame)
        analysis_layout = QVBoxLayout(self.analysis_frame)
        analysis_layout.setContentsMargins(12, 11, 12, 14)
        analysis_layout.setSpacing(8)
        self.analysis_title = QLabel("Analysis")
        self.analysis_title.setObjectName("panelTitle")
        analysis_layout.addWidget(self.analysis_title)
        self.analysis_btn = QPushButton("Analysis")
        self.analysis_btn.setToolTip("Analyze inference results for the active layer")
        self.analysis_btn.clicked.connect(self.open_analysis_dialog)
        prepare_panel_button(self.analysis_btn, min_height=34)
        analysis_layout.addWidget(self.analysis_btn)

        # -----------------------------
        # Bottom-left: training tools
        # -----------------------------
        self.bottom_left_frame = QFrame(self.right_sidebar_content)
        self.bottom_left_frame.setObjectName("ToolPanel")
        self.bottom_left_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.bottom_left_frame)
        bottom_left_layout = QVBoxLayout(self.bottom_left_frame)
        bottom_left_layout.setContentsMargins(12, 11, 12, 11)
        bottom_left_layout.setSpacing(8)
        self.dataset_training_title = QLabel("Dataset & Training")
        self.dataset_training_title.setObjectName("panelTitle")
        bottom_left_layout.addWidget(self.dataset_training_title)
        training_grid = QGridLayout()
        self.training_grid = training_grid
        training_grid.setHorizontalSpacing(6)
        training_grid.setVerticalSpacing(6)
        self.normalize_btn = QPushButton("Validate Labels")
        self.normalize_btn.setToolTip(
            "Rewrite labels_all files and ensure matching images exist in images_all"
        )
        self.normalize_btn.clicked.connect(self.normalize_labels_all)
        training_grid.addWidget(self.normalize_btn, 0, 0)

        self.export_dataset_btn = QPushButton("Export Dataset")
        self.export_dataset_btn.setToolTip(
            "Split images_all/labels_all into train/val and regenerate dataset.yaml"
        )
        self.export_dataset_btn.clicked.connect(self.export_dataset)
        training_grid.addWidget(self.export_dataset_btn, 0, 1)

        self.project_health_btn = QPushButton("Project Health")
        self.project_health_btn.setToolTip(
            "Report orphan labels, duplicate stems, and stale transaction files"
        )
        self.project_health_btn.clicked.connect(self.show_project_health)
        training_grid.addWidget(self.project_health_btn, 1, 0)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.setToolTip("Launch a training run for a selected dataset")
        self.train_btn.clicked.connect(self.open_train_dialog)
        training_grid.addWidget(self.train_btn, 1, 1)

        self.distillation_btn = QPushButton("Distillation")
        self.distillation_btn.setToolTip(
            "Create an unlabeled image corpus and distill a DINO-backed pose model"
        )
        self.distillation_btn.clicked.connect(self.open_distillation_dialog)
        training_grid.addWidget(self.distillation_btn, 2, 0, 1, 2)
        for btn in (
            self.normalize_btn,
            self.export_dataset_btn,
            self.project_health_btn,
            self.train_btn,
            self.distillation_btn,
        ):
            prepare_panel_button(btn)

        bottom_left_layout.addLayout(training_grid)

        # -----------------------------
        # Bottom-right: model + inference
        # -----------------------------
        self.bottom_right_frame = QFrame(self.right_sidebar_content)
        self.bottom_right_frame.setObjectName("ToolPanel")
        self.bottom_right_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.bottom_right_frame)
        bottom_right_layout = QVBoxLayout(self.bottom_right_frame)
        bottom_right_layout.setContentsMargins(12, 11, 12, 11)
        bottom_right_layout.setSpacing(8)
        self.model_inference_title = QLabel("Project Models & Inference")
        self.model_inference_title.setObjectName("panelTitle")
        bottom_right_layout.addWidget(self.model_inference_title)
        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("fieldLabel")
        self.model_status_label.setWordWrap(True)
        bottom_right_layout.addWidget(self.model_status_label)
        inference_grid = QGridLayout()
        self.inference_grid = inference_grid
        inference_grid.setHorizontalSpacing(6)
        inference_grid.setVerticalSpacing(6)
        self.load_model_btn = QPushButton("Project Models…")
        self.load_model_btn.clicked.connect(self.load_model)
        inference_grid.addWidget(self.load_model_btn, 0, 0)

        self.template_apply_btn = QPushButton("Apply Template")
        self.template_apply_btn.setToolTip("Apply the saved template for the selected class")
        self.template_apply_btn.clicked.connect(self.apply_template_for_current_class)
        inference_grid.addWidget(self.template_apply_btn, 1, 0)

        self.template_save_btn = QPushButton("Save Template")
        self.template_save_btn.setToolTip("Capture the current annotation as the class template")
        self.template_save_btn.clicked.connect(self.save_template_for_current_class)
        inference_grid.addWidget(self.template_save_btn, 1, 1)

        self.inference_btn = QPushButton("Inference")
        self.inference_btn.setToolTip(
            "Select a video and run every configured project prediction model into layer-specific CSV outputs"
        )
        self.inference_btn.clicked.connect(self.run_video_inference)
        inference_grid.addWidget(self.inference_btn, 0, 1)
        for btn in (
            self.load_model_btn,
            self.template_apply_btn,
            self.template_save_btn,
            self.inference_btn,
        ):
            prepare_panel_button(btn)
        bottom_right_layout.addLayout(inference_grid)
        self.right_sidebar_layout.addWidget(self.bottom_right_frame)
        self.right_sidebar_layout.addWidget(self.bottom_left_frame)

        # -----------------------------
        # Bottom-left overlay: segmentation tools/help
        # -----------------------------
        self.seg_tools_frame = QFrame(self.left_sidebar_content)
        self.seg_tools_frame.setObjectName("ToolPanel")
        self.seg_tools_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.seg_tools_frame)
        seg_tools_layout = QVBoxLayout(self.seg_tools_frame)
        seg_tools_layout.setContentsMargins(10, 9, 10, 9)
        seg_tools_layout.setSpacing(6)

        seg_tools_title = QLabel("Segmentation Tools")
        seg_tools_title.setObjectName("panelTitle")
        seg_tools_layout.addWidget(seg_tools_title)

        seg_brush_row = QHBoxLayout()
        seg_brush_row.setSpacing(6)
        seg_brush_label = QLabel("Brush")
        seg_brush_label.setObjectName("fieldLabel")
        seg_brush_row.addWidget(seg_brush_label)
        self.seg_brush_size_label = QLabel("Brush: 8px")
        self.seg_brush_size_label.setObjectName("brushSizeBadge")
        seg_brush_row.addWidget(self.seg_brush_size_label)
        seg_brush_row.addStretch(1)
        seg_tools_layout.addLayout(seg_brush_row)

        sam_grid = QGridLayout()
        sam_grid.setHorizontalSpacing(6)
        sam_grid.setVerticalSpacing(6)
        self.sam_load_btn = QPushButton("Load SAM")
        self.sam_load_btn.setToolTip("Load a SAM model file for segmentation prompts")
        self.sam_load_btn.clicked.connect(self._load_sam_model_interactive)
        sam_grid.addWidget(self.sam_load_btn, 0, 0)

        self.sam_run_btn = QPushButton("Run (G)")
        self.sam_run_btn.setToolTip("Run SAM using current positive/negative prompts")
        self.sam_run_btn.clicked.connect(self._run_sam_segmentation)
        sam_grid.addWidget(self.sam_run_btn, 0, 1)

        self.sam_accept_btn = QPushButton("Accept")
        self.sam_accept_btn.setObjectName("samAcceptButton")
        self.sam_accept_btn.setToolTip("Commit the current SAM mask preview to this class")
        self.sam_accept_btn.clicked.connect(self._accept_segmentation_preview)
        sam_grid.addWidget(self.sam_accept_btn, 1, 0)

        self.sam_clear_btn = QPushButton("Reset")
        self.sam_clear_btn.setToolTip("Remove prompt points and the current SAM preview")
        self.sam_clear_btn.clicked.connect(self._clear_seg_prompt_state)
        sam_grid.addWidget(self.sam_clear_btn, 1, 1)
        for btn in (self.sam_load_btn, self.sam_run_btn, self.sam_accept_btn, self.sam_clear_btn):
            prepare_panel_button(btn, min_height=30)
        seg_tools_layout.addLayout(sam_grid)

        self.sam_helper_label = QLabel("")
        self.sam_helper_label.setWordWrap(True)
        self.sam_helper_label.setObjectName("samHelper")
        seg_tools_layout.addWidget(self.sam_helper_label)
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

        hud_style = hud_stylesheet()

        # Active/reference layer context (top-left).
        self.layer_context_frame = QFrame(self.view)
        self.layer_context_frame.setStyleSheet(hud_style)
        apply_panel_shadow(self.layer_context_frame, blur=14, y_offset=2, alpha=75)
        layer_context_layout = QVBoxLayout(self.layer_context_frame)
        layer_context_layout.setContentsMargins(10, 7, 10, 7)
        layer_context_layout.setSpacing(1)
        self.layer_editing_label = QLabel("")
        self.layer_editing_label.setObjectName("layerEditing")
        self.layer_reference_label = QLabel("")
        self.layer_reference_label.setObjectName("layerReference")
        layer_context_layout.addWidget(self.layer_editing_label)
        layer_context_layout.addWidget(self.layer_reference_label)
        self.layer_context_frame.show()
        self._refresh_layer_context_hud()

        # --- legend (bottom-left) ---
        self.legend_frame = QFrame(self.view)
        self.legend_frame.setStyleSheet(hud_style)
        apply_panel_shadow(self.legend_frame)
        # don't lock width; let it resize
        self.legend_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        legend_layout = QVBoxLayout(self.legend_frame)
        legend_layout.setContentsMargins(10, 9, 10, 9)
        legend_layout.setSpacing(6)

        self.legend_title = QLabel("Keypoint Visibility")
        self.legend_title.setObjectName("hudTitle")
        legend_layout.addWidget(self.legend_title)

        # multiline, can wrap, can expand
        self.legend_label = QLabel(
            "Keys:  🔴 Visible   🟡 Occluded   ⚪ Invisible (v=0)\n"
            "L: toggle labels   -/= point size   [/] text size\n"
            "0: mark next invisible   Shift+0: selected → invisible"
        )
        self.legend_label.setWordWrap(True)
        self.legend_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        legend_layout.addWidget(self.legend_label)

        self.legend_frame.hide()

        # Floating zoom HUD
        self.zoom_frame = QFrame(self.view)
        self.zoom_frame.setStyleSheet(hud_style)
        apply_panel_shadow(self.zoom_frame)
        self.zoom_frame.setFixedWidth(132)

        zoom_layout = QVBoxLayout(self.zoom_frame)
        zoom_layout.setContentsMargins(10, 8, 10, 8)
        zoom_layout.setSpacing(4)

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setObjectName("zoomValue")
        zoom_layout.addWidget(self.zoom_label)

        self.zoom_frame.move(10, 150)
        self.zoom_frame.hide()

        # Status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # Shortcuts
        self._bind_shortcuts()

    # ---------- Class & annotation helpers ----------

    def _on_class_changed(self, index: int):
        if index < 0 or index >= len(self.classes):
            return
        prev = getattr(self, "_active_class_id", index)
        if prev != index and self._is_pose_workflow():
            self._cache_active_annotation(prev)
        self._active_class_id = index
        if self._is_seg_workflow():
            self._clear_seg_prompt_state()
            has_item = self._class_seg_mask_item(index) is not None
        else:
            has_item = self._class_box_item(index) is not None
        if self.annotation_cache.get(index) and not has_item:
            self._restore_annotation_for_class(index)
        else:
            self._sync_active_class_state()
            self._update_item_editability()
            self._update_status()
        self._clear_seg_edit_handles()
        self._refresh_sam_controls()

    def _cache_active_annotation(self, class_id: Optional[int] = None) -> bool:
        if not self.images:
            return False
        cid = self.class_selector.currentIndex() if class_id is None else class_id

        if self._is_seg_workflow():
            seg_item = self._class_seg_mask_item(cid)
            if seg_item is not None:
                points = self._extract_seg_item_points(seg_item)
                if len(points) >= 3:
                    self.annotation_cache[cid] = {
                        "class_id": cid,
                        "segments": [(float(x), float(y)) for x, y in points],
                    }
                    return True
                self.annotation_cache.pop(cid, None)
                return False
            entry = self.annotation_cache.get(cid, {})
            return len(entry.get("segments", [])) >= 3

        bbox_item = self._class_box_item(cid)
        if bbox_item is None:
            self.annotation_cache.pop(cid, None)
            return False
        bbox_item.update_model()
        required_names = self._kp_names_for_index(cid)
        kp_items = self._class_keypoint_items(cid)
        if len(required_names) == 0:
            bbox = bbox_item.bbox
            self.annotation_cache[cid] = {
                "class_id": cid,
                "bbox": {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h},
                "keypoints": [],
            }
            return True
        if len(kp_items) != len(required_names):
            return False
        kp_map = {}
        for it in kp_items:
            kp_map[it.kp.name] = {
                "name": it.kp.name,
                "x": it.kp.x,
                "y": it.kp.y,
                "vis": int(getattr(it, "visibility", 2)),
            }
        ordered = []
        for idx, name in enumerate(required_names):
            entry = kp_map.get(name)
            if not entry:
                return False
            entry["idx"] = idx
            entry["canon_idx"] = self._ensure_canonical_name(name)
            ordered.append(entry)
        bbox = bbox_item.bbox
        self.annotation_cache[cid] = {
            "class_id": cid,
            "bbox": {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h},
            "keypoints": ordered,
        }
        return True

    def _restore_annotation_for_class(self, cid: int):
        self._clear_class_items(cid)
        entry = self.annotation_cache.get(cid)
        if not entry:
            if cid == self.class_selector.currentIndex():
                self._sync_active_class_state()
                self._update_item_editability()
            return

        if self._is_seg_workflow():
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

        bbox_data = entry.get("bbox", {})
        bbox = BoundingBox(
            bbox_data.get("x", 0.0),
            bbox_data.get("y", 0.0),
            bbox_data.get("w", 0.0),
            bbox_data.get("h", 0.0),
            cid,
        )
        item = BoxItem(bbox, self.classes[cid] if cid < len(self.classes) else str(cid))
        self.scene.addItem(item)
        self._track_scene_item(item)

        active_names = self._kp_names_for_index(cid)
        for kp_info in entry.get("keypoints", []):
            idx = int(kp_info.get("idx", -1))
            if 0 <= idx < len(active_names):
                name = active_names[idx]
            else:
                name = kp_info.get("name", f"kp_{idx + 1}")
            kp = Keypoint(kp_info.get("x", 0.0), kp_info.get("y", 0.0), cid, name)
            kp_item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
            kp_item.visibility = int(kp_info.get("vis", 2))
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
        cache, extra_rows = load_pose_annotations_from_file(
            label_file,
            classes_count=len(self.classes),
            canonical_names=self.kp_names,
            class_keypoint_lookup=class_lookup,
            img_w=self.img_w,
            img_h=self.img_h,
        )
        if extra_rows > 0:
            print(
                f"⚠️ Ignored extra keypoint values in {extra_rows} row(s) while reading {label_file}",
                file=sys.stderr,
            )
        return cache

    def _load_seg_annotations_from_file(self, label_file: str) -> dict[int, dict]:
        return load_segmentation_annotations_from_file(
            label_file,
            classes_count=len(self.classes),
            img_w=self.img_w,
            img_h=self.img_h,
        )

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
        total = len(self.images_queue)
        idx = start_from
        if total == 0:
            return 0
        for _ in range(total):
            idx = (idx + 1) % total
            base = os.path.splitext(self.images_queue[idx])[0]
            label_file = os.path.join(self.label_dir, f"{base}.txt")
            if not self._label_file_is_usable(label_file):
                return idx
        return start_from  # all labeled

    # ---------- Navigation filtering ----------
    def _is_labeled_index(self, idx: int) -> bool:
        base = os.path.splitext(self.images[idx])[0]
        label_file = os.path.join(self.label_dir, f"{base}.txt")
        return self._label_file_is_usable(label_file)

    def _filtered_indices(self) -> list[int]:
        if not self.images:
            return []
        if self.nav_filter == "all":
            return list(range(len(self.images)))
        elif self.nav_filter == "labeled":
            return [i for i in range(len(self.images)) if self._is_labeled_index(i)]
        else:  # 'unlabeled'
            return [i for i in range(len(self.images)) if not self._is_labeled_index(i)]

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

        fi = self._filtered_indices()
        if not fi:
            self.update_status_bar(f"No images match filter: {mode}.")
            return
        if self.current_idx not in fi:
            self.current_idx = fi[0]
        self._queue_current_idx = self.current_idx
        self.update_status_bar(f"Browsing: {mode} ({fi.index(self.current_idx) + 1}/{len(fi)})")
        self.load_image()

    def prev_index(self):
        fi = self._filtered_indices()
        if not fi:
            self.update_status_bar("No images found for current filter.")
            return
        if self.current_idx not in fi:
            self.current_idx = fi[0]
        else:
            pos = fi.index(self.current_idx)
            self.current_idx = fi[(pos - 1) % len(fi)]
        self.mode = (
            "segment"
            if self._is_seg_workflow()
            else ("panzoom" if self._is_depth_layer() else "bbox")
        )
        self.load_image()
        self._queue_current_idx = self.current_idx

    def next_index(self):
        fi = self._filtered_indices()
        if not fi:
            self.update_status_bar("No images found for current filter.")
            return
        if self.current_idx not in fi:
            self.current_idx = fi[0]
        else:
            pos = fi.index(self.current_idx)
            self.current_idx = fi[(pos + 1) % len(fi)]
        self.mode = (
            "segment"
            if self._is_seg_workflow()
            else ("panzoom" if self._is_depth_layer() else "bbox")
        )
        self.load_image()
        self._queue_current_idx = self.current_idx

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

    def _image_delete_paths(self, image_name: str) -> list[str]:
        file_name = os.path.basename(image_name)
        if not file_name:
            return []
        base = os.path.splitext(file_name)[0]
        label_name = f"{base}.txt"
        state = getattr(self, "__dict__", {})
        paths: list[str] = []
        for directory, target_name in (
            (getattr(self, "active_image_dir", ""), file_name),
            (getattr(self, "image_dir_queue", ""), file_name),
            (getattr(self, "image_dir_all", ""), file_name),
            (getattr(self, "pose_label_dir", ""), label_name),
            (getattr(self, "seg_label_dir", ""), label_name),
            (state.get("depth_image_dir", ""), f"{base}.npy"),
            (state.get("depth_image_dir", ""), f"{base}_depth.json"),
            (state.get("depth_preview_dir", ""), f"{base}_depth.png"),
            (os.path.join(self.project_root, "annotations"), f"{base}_annotated.png"),
            (
                os.path.join(self.project_root, "annotations", LAYER_KEYPOINTS),
                f"{base}_annotated.png",
            ),
            (
                os.path.join(self.project_root, "annotations", LAYER_SEGMENTATION),
                f"{base}_annotated.png",
            ),
        ):
            if directory:
                paths.append(os.path.join(directory, target_name))

        for mode in (DATASET_POSE, DATASET_SEGMENT, DATASET_DETECT):
            dataset_paths = dataset_export_paths(self.project_root, mode)
            paths.extend(
                [
                    os.path.join(dataset_paths.images_train_dir, file_name),
                    os.path.join(dataset_paths.images_val_dir, file_name),
                    os.path.join(dataset_paths.labels_train_dir, label_name),
                    os.path.join(dataset_paths.labels_val_dir, label_name),
                ]
            )

        unique_paths: list[str] = []
        seen: set[str] = set()
        for path in paths:
            norm = os.path.abspath(path)
            if norm in seen:
                continue
            seen.add(norm)
            unique_paths.append(path)
        return unique_paths

    def _delete_image_files(self, image_name: str) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        errors: list[str] = []
        for path in self._image_delete_paths(image_name):
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
        self.bboxes.clear()
        self.kps.clear()
        self.current_kp_idx = 0
        self.scene.clear()
        self._item_refs.clear()
        self._reference_layer_items.clear()

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
        base = os.path.splitext(file_name)[0].casefold()
        conflicting_names: set[str] = set()
        for directory in (self.image_dir_queue, self.image_dir_all):
            for candidate in list_image_files(directory):
                if os.path.splitext(candidate)[0].casefold() == base and candidate != file_name:
                    conflicting_names.add(candidate)
        if conflicting_names:
            QMessageBox.warning(
                self,
                "Duplicate Image Name",
                f"Cannot safely delete '{file_name}' because another project image shares its label stem:\n\n"
                f"{', '.join(sorted(conflicting_names, key=str.casefold))}\n\n"
                "Rename the conflicting image first.",
            )
            return

        existing_paths = [p for p in self._image_delete_paths(file_name) if os.path.exists(p)]
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

        removed, errors = self._delete_image_files(file_name)
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

    def _refresh_depth_assistant_controls(self) -> None:
        label = getattr(self, "depth_model_status_label", None)
        if label is None:
            return
        path = str(getattr(self, "layer_model_paths", {}).get(LAYER_DEPTH) or "")
        if not path:
            label.setText("No depth model selected.")
            label.setToolTip("")
        elif self._is_builtin_model_reference(path):
            label.setText(f"{os.path.basename(path)} · official model; downloads on first use")
            label.setToolTip(path)
        else:
            label.setText(f"Custom model · {os.path.basename(path)}")
            label.setToolTip(path)
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
        if (
            self._inference_process is not None
            and self._inference_process.state() != QProcess.ProcessState.NotRunning
        ):
            QMessageBox.information(
                self, "Inference Running", "An inference process is already running."
            )
            return

        configured_layers = [
            layer_id
            for layer_id in (self.active_layer, *LAYER_DEFINITIONS)
            if self.layer_model_paths.get(layer_id)
        ]
        configured_layers = list(dict.fromkeys(configured_layers))
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
            configured_layers = [
                layer_id
                for layer_id in (self.active_layer, *LAYER_DEFINITIONS)
                if self.layer_model_paths.get(layer_id)
            ]
            configured_layers = list(dict.fromkeys(configured_layers))
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

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        jobs: list[dict] = []
        for layer_id in configured_layers:
            layer = layer_definition(layer_id)
            output_root = os.path.join(self.project_root, "inference outputs", layer.id)
            try:
                os.makedirs(output_root, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Output Error",
                    f"Could not create output directory:\n{output_root}\n\n{e}",
                )
                return
            csv_name = f"{base_name}_{timestamp}{layer.inference_suffix}"
            if layer_id == LAYER_KEYPOINTS:
                classes = self.pose_classes[:]
                kp_names = self.pose_kp_names[:]
            elif layer_id == LAYER_SEGMENTATION:
                classes = self.seg_classes[:]
                kp_names = []
            else:
                classes = []
                kp_names = []
            csv_path = os.path.join(output_root, csv_name)
            jobs.append(
                {
                    "layer_id": layer_id,
                    "workflow": layer.worker_mode,
                    "model_path": self.layer_model_paths.get(layer_id) or "",
                    "video_path": video_path,
                    "csv_path": csv_path,
                    "preview_path": (
                        os.path.splitext(csv_path)[0] + "_preview.mp4"
                        if layer_id == LAYER_DEPTH
                        else ""
                    ),
                    "classes": classes,
                    "kp_names": kp_names,
                    "batch_size": batch_size,
                    "total_frames": metadata.total_frames,
                    "fps": metadata.fps,
                }
            )

        run_dir = os.path.join(self.project_root, "inference outputs", "runs")
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Output Error",
                f"Could not create inference run directory:\n{run_dir}\n\n{e}",
            )
            return
        self._inference_run_manifest_path = os.path.join(run_dir, f"{base_name}_{timestamp}.json")
        self._inference_run_video_path = video_path
        self._inference_job_queue = jobs
        self._inference_run_results = []
        self._inference_run_total = len(jobs)
        self._inference_run_canceled = False
        self._start_next_inference_job()

    def _start_next_inference_job(self) -> None:
        if not self._inference_job_queue:
            self._finish_project_inference_run()
            return
        job = self._inference_job_queue.pop(0)
        self._inference_active_job = job
        job_index = self._inference_run_total - len(self._inference_job_queue)
        self._start_inference_process(
            layer_id=job["layer_id"],
            workflow=job["workflow"],
            model_path=job["model_path"],
            video_path=job["video_path"],
            csv_path=job["csv_path"],
            classes=job["classes"],
            kp_names=job["kp_names"],
            batch_size=job["batch_size"],
            total_frames=job["total_frames"],
            fps=job["fps"],
            job_index=job_index,
            job_total=self._inference_run_total,
        )

    def _start_inference_process(
        self,
        *,
        layer_id: str,
        workflow: str,
        model_path: str,
        video_path: str,
        csv_path: str,
        classes: list[str],
        kp_names: list[str],
        batch_size: int,
        total_frames: int,
        fps: float,
        job_index: int = 1,
        job_total: int = 1,
    ) -> None:
        output_root = os.path.dirname(csv_path)
        config = {
            "layer_id": normalize_layer_id(layer_id),
            "mode": workflow,
            "model_path": model_path,
            "video_path": video_path,
            "csv_path": csv_path,
            "preview_path": str((self._inference_active_job or {}).get("preview_path") or ""),
            "classes": classes,
            "kp_names": kp_names,
            "device": self._device,
            "batch_size": int(batch_size),
            "total_frames": int(total_frames),
            "fps": float(fps),
        }
        config_path = os.path.join(
            output_root, f".{os.path.splitext(os.path.basename(csv_path))[0]}_config.json"
        )
        try:
            atomic_write_text(config_path, json.dumps(config, indent=2))
        except Exception as e:
            self._inference_run_results.append(
                {
                    "layer_id": normalize_layer_id(layer_id),
                    "workflow": workflow,
                    "model_path": model_path,
                    "csv_path": csv_path,
                    "rows_written": 0,
                    "processed_frames": 0,
                    "canceled": False,
                    "had_error": True,
                    "error_message": f"Could not write worker config: {e}",
                }
            )
            self._inference_active_job = None
            QTimer.singleShot(0, self._start_next_inference_job)
            return

        layer_name = layer_definition(layer_id).display_name
        title = "Project Video Inference"
        label = f"Pass {job_index}/{job_total}: running {layer_name} inference…"
        prog = QProgressDialog(label, "Cancel", 0, 0 if total_frames <= 0 else total_frames, self)
        prog.setWindowTitle(title)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        if total_frames <= 0:
            prog.setRange(0, 0)  # busy indicator for unknown length

        if job_index == 1:
            self._inference_previous_busy = getattr(self, "_predict_busy", False)
        self._predict_busy = True
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(False)
        if hasattr(self, "inference_btn"):
            self.inference_btn.setEnabled(False)

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "inference_worker", "--config", config_path])
        process.setWorkingDirectory(APP_BASE_DIR)
        process.readyReadStandardOutput.connect(self._read_inference_process_stdout)
        process.readyReadStandardError.connect(self._read_inference_process_stderr)
        process.finished.connect(self._finish_inference_process)
        process.errorOccurred.connect(self._handle_inference_process_error)
        prog.canceled.connect(self._cancel_inference_process)

        self._inference_process = process
        self._inference_progress = prog
        self._inference_stdout_buffer = ""
        self._inference_stderr = ""
        self._inference_result_event = None
        self._inference_config_path = config_path
        self._inference_csv_path = csv_path
        self._inference_mode = workflow
        self._inference_layer_id = normalize_layer_id(layer_id)
        self._inference_job_index = int(job_index)
        self._inference_job_total = int(job_total)
        self._inference_cancel_requested = False

        prog.show()
        process.start()
        if not process.waitForStarted(1000):
            self._inference_stderr = process.errorString()
            self._finish_inference_process(1, QProcess.ExitStatus.CrashExit)
            return

    def _read_inference_process_stdout(self) -> None:
        process = self._inference_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._inference_stdout_buffer += text
        lines = self._inference_stdout_buffer.splitlines(keepends=True)
        self._inference_stdout_buffer = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                self._handle_inference_event_line(line.strip())
            else:
                self._inference_stdout_buffer = line

    def _read_inference_process_stderr(self) -> None:
        process = self._inference_process
        if process is None:
            return
        self._inference_stderr += bytes(process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )

    def _handle_inference_event_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self._inference_stderr += line + "\n"
            return

        event_type = event.get("event")
        if event_type == "progress":
            progress = self._inference_progress
            if progress is not None:
                processed = int(event.get("processed_frames") or 0)
                total = int(event.get("total_frames") or 0)
                if total > 0:
                    progress.setValue(min(processed, total))
                layer_name = layer_definition(
                    getattr(self, "_inference_layer_id", LAYER_KEYPOINTS)
                ).display_name
                detail = str(event.get("message") or f"Inferencing frame {processed}")
                progress.setLabelText(
                    f"Pass {getattr(self, '_inference_job_index', 1)}/"
                    f"{getattr(self, '_inference_job_total', 1)} · "
                    f"{layer_name}\n{detail}"
                )
            QApplication.processEvents()
        elif event_type == "result":
            self._inference_result_event = event
        elif event_type == "error":
            self._inference_result_event = {
                "event": "result",
                "csv_path": self._inference_csv_path or "",
                "rows_written": 0,
                "processed_frames": 0,
                "canceled": False,
                "had_error": True,
                "error_message": str(event.get("error_message") or "Inference worker error"),
                "mode": self._inference_mode,
            }
        elif event_type == "started":
            progress = self._inference_progress
            if progress is not None:
                layer_name = layer_definition(
                    getattr(self, "_inference_layer_id", LAYER_KEYPOINTS)
                ).display_name
                progress.setLabelText(
                    f"Pass {getattr(self, '_inference_job_index', 1)}/"
                    f"{getattr(self, '_inference_job_total', 1)} · "
                    f"Loading {layer_name} model…"
                )

    def _cancel_inference_process(self) -> None:
        process = self._inference_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        self._inference_cancel_requested = True
        progress = self._inference_progress
        if progress is not None:
            progress.setLabelText("Canceling inference process…")
        request_qprocess_stop(
            process,
            schedule=QTimer.singleShot,
            force_kill=self._kill_inference_process_if_running,
            kill_after_ms=5000,
        )

    def _kill_inference_process_if_running(self) -> None:
        process = self._inference_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_inference_process_error(self, _error) -> None:
        process = self._inference_process
        if process is not None:
            self._inference_stderr += process.errorString() + "\n"

    def _finish_inference_process(self, exit_code: int, exit_status) -> None:
        if self._inference_process is None and self._inference_config_path is None:
            return
        if self._inference_stdout_buffer.strip():
            self._handle_inference_event_line(self._inference_stdout_buffer.strip())
            self._inference_stdout_buffer = ""

        progress = self._inference_progress
        if progress is not None:
            progress.close()

        event = self._inference_result_event
        csv_path = self._inference_csv_path or ""
        config_path = self._inference_config_path
        mode = self._inference_mode
        layer_id = getattr(self, "_inference_layer_id", normalize_layer_id(mode))
        cancel_requested = self._inference_cancel_requested
        stderr_text = self._inference_stderr.strip()

        _remove_file_quietly(config_path)

        self._inference_process = None
        self._inference_progress = None
        self._inference_config_path = None
        self._inference_csv_path = None
        self._inference_result_event = None
        self._inference_stdout_buffer = ""
        self._inference_stderr = ""
        self._inference_cancel_requested = False

        if event is None:
            event = {
                "rows_written": 0,
                "processed_frames": 0,
                "canceled": bool(cancel_requested),
                "had_error": not bool(cancel_requested),
                "error_message": stderr_text or f"Process exited with code {exit_code}.",
                "csv_path": csv_path,
            }

        rows_written = int(event.get("rows_written") or 0)
        canceled = bool(event.get("canceled")) or cancel_requested
        had_error = bool(event.get("had_error")) or (
            not canceled and (exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0)
        )
        error_message = str(
            event.get("error_message")
            or stderr_text
            or ("Unknown inference error" if had_error else "")
        )
        job = self._inference_active_job or {}
        csv_path = str(event.get("csv_path") or csv_path)
        preview_path = str(event.get("preview_path") or job.get("preview_path") or "")

        if rows_written == 0 and (had_error or canceled):
            try:
                if csv_path and os.path.exists(csv_path):
                    os.remove(csv_path)
            except Exception:
                pass
            try:
                if preview_path and os.path.exists(preview_path):
                    os.remove(preview_path)
            except Exception:
                pass

        self._inference_run_results.append(
            {
                "layer_id": layer_id,
                "workflow": mode,
                "model_path": str(job.get("model_path") or ""),
                "csv_path": csv_path,
                "preview_path": preview_path,
                "rows_written": rows_written,
                "processed_frames": int(event.get("processed_frames") or 0),
                "canceled": canceled,
                "had_error": had_error,
                "error_message": error_message,
            }
        )
        self._inference_active_job = None

        if canceled:
            self._inference_run_canceled = True
            self._inference_job_queue.clear()

        if self._inference_job_queue:
            QTimer.singleShot(0, self._start_next_inference_job)
            return
        self._finish_project_inference_run()

    def _finish_project_inference_run(self) -> None:
        if hasattr(self, "_inference_previous_busy"):
            self._predict_busy = self._inference_previous_busy
        else:
            self._predict_busy = False
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(True)
        if hasattr(self, "inference_btn"):
            self.inference_btn.setEnabled(True)

        results = list(self._inference_run_results)
        manifest_path = self._inference_run_manifest_path
        manifest = {
            "schema_version": 1,
            "created_at": datetime.datetime.now().isoformat(),
            "video_path": self._inference_run_video_path,
            "canceled": bool(self._inference_run_canceled),
            "passes": results,
        }
        if manifest_path:
            try:
                atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
            except Exception:
                manifest_path = ""

        self._inference_job_queue = []
        self._inference_active_job = None
        self._inference_run_results = []
        self._inference_run_total = 0
        self._inference_run_manifest_path = ""
        self._inference_run_video_path = ""

        if not results:
            return
        lines = []
        failures = 0
        for result in results:
            name = layer_definition(result["layer_id"]).display_name
            rows = int(result.get("rows_written") or 0)
            if result.get("had_error"):
                failures += 1
                detail = result.get("error_message") or "failed"
                lines.append(f"{name}: failed — {detail}")
            elif result.get("canceled"):
                lines.append(f"{name}: canceled ({rows} rows retained)")
            else:
                detail = f"{name}: {rows} rows → {result.get('csv_path', '')}"
                if result.get("preview_path"):
                    detail += f"\nPreview → {result.get('preview_path')}"
                lines.append(detail)
        if manifest_path:
            lines.append(f"Run manifest: {manifest_path}")
        message = "\n\n".join(lines)
        if failures:
            QMessageBox.warning(
                self,
                "Project Inference Finished",
                message,
            )
        else:
            QMessageBox.information(
                self,
                "Project Inference Complete",
                message,
            )

    def _segmentation_rows_from_result(
        self,
        result,
        frame_idx: int,
        *,
        include_binary_mask: bool = True,
    ) -> list[dict[str, object]]:
        return segmentation_rows_from_result(
            result,
            frame_idx,
            classes=self.classes,
            include_binary_mask=include_binary_mask,
            numpy_module=_np,
        )

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

        self._prediction_request_counter += 1
        request_id = self._prediction_request_counter
        request = {
            "command": "predict",
            "request_id": request_id,
            "layer_id": self.active_layer,
            "model_path": self.predict_model_path,
            "image_path": img_path,
            "workflow": self.active_workflow,
            "device": self._device,
        }
        self._cleanup_prediction_depth_staging()
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
            request.update(
                {
                    "depth_map_path": staged_targets["map"],
                    "depth_preview_path": staged_targets["preview"],
                    "depth_metadata_path": staged_targets["metadata"],
                }
            )

        self._predict_busy = True
        self._prediction_current_request_id = request_id
        self._prediction_image_path = img_path
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(False)
        self.update_status_bar("Running prediction...")
        self._send_prediction_request(request)

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

    def _start_prediction_worker(self) -> None:
        process = self._prediction_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            return
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "predict_worker", "--server"])
        process.setWorkingDirectory(APP_BASE_DIR)
        process.readyReadStandardOutput.connect(self._read_prediction_process_stdout)
        process.readyReadStandardError.connect(self._read_prediction_process_stderr)
        process.finished.connect(self._finish_prediction_process)
        process.errorOccurred.connect(self._handle_prediction_process_error)

        self._prediction_process = process
        self._prediction_stdout_buffer = ""
        self._prediction_stderr = ""
        self._prediction_result_event = None
        self._prediction_config_path = None
        self._prediction_cancel_requested = False
        self._prediction_worker_ready = False
        self._prediction_expected_stop = False
        process.start()
        if not process.waitForStarted(1000):
            self._prediction_stderr = process.errorString()
            self._finish_prediction_process(1, QProcess.ExitStatus.CrashExit)

    def _send_prediction_request(self, request: dict) -> None:
        self._start_prediction_worker()
        process = self._prediction_process
        if (
            process is None
            or process.state() == QProcess.ProcessState.NotRunning
            or not self._prediction_worker_ready
        ):
            self._prediction_pending_request = request
            return
        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        if process.write(payload) < 0:
            self._prediction_pending_request = request
            self._prediction_stderr += "Could not write prediction request to worker.\n"

    def _restart_prediction_worker(self, *, warm: bool = False) -> None:
        process = self._prediction_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._prediction_expected_stop = True
            _shutdown_qprocess(process)
        self._prediction_process = None
        self._prediction_worker_ready = False
        self._prediction_pending_request = None
        self._prediction_current_request_id = None
        self._cleanup_prediction_depth_staging()
        if not warm or not self.predict_model_path:
            return
        self._prediction_request_counter += 1
        self._send_prediction_request(
            {
                "command": "load",
                "request_id": self._prediction_request_counter,
                "layer_id": self.active_layer,
                "model_path": self.predict_model_path,
                "workflow": self.active_workflow,
                "device": self._device,
            }
        )

    def _read_prediction_process_stdout(self) -> None:
        process = self._prediction_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._prediction_stdout_buffer += text
        lines = self._prediction_stdout_buffer.splitlines(keepends=True)
        self._prediction_stdout_buffer = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                self._handle_prediction_event_line(line.strip())
            else:
                self._prediction_stdout_buffer = line

    def _read_prediction_process_stderr(self) -> None:
        process = self._prediction_process
        if process is None:
            return
        self._prediction_stderr += bytes(process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )

    def _handle_prediction_event_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = parse_event_line(line).as_dict()
        except WorkerProtocolError:
            self._prediction_stderr += line + "\n"
            return
        event_type = event.get("event")
        request_id = event.get("request_id")
        if event_type == "ready":
            self._prediction_worker_ready = True
            pending = self._prediction_pending_request
            self._prediction_pending_request = None
            if pending is not None:
                self._send_prediction_request(pending)
        elif event_type == "loading":
            self.update_status_bar("Loading prediction model...")
        elif event_type == "loaded":
            self.update_status_bar("Prediction model ready.")
        elif event_type == "started":
            self.update_status_bar("Prediction worker started...")
        elif event_type == "error":
            error_text = str(event.get("error_message") or "Prediction worker error")
            if request_id is None or request_id == self._prediction_current_request_id:
                self._prediction_current_request_id = None
                self._prediction_image_path = None
                self._predict_busy = False
                LabelingApp._cleanup_prediction_depth_staging(self)
                if hasattr(self, "predict_btn"):
                    self.predict_btn.setEnabled(True)
                self._on_predict_error(error_text)
            else:
                self.update_status_bar(f"Prediction model error: {error_text}")
        elif event_type == "result":
            if request_id != self._prediction_current_request_id:
                return
            requested_image_path = self._prediction_image_path
            displayed_image_path = self._displayed_image_path()
            self._prediction_current_request_id = None
            self._prediction_image_path = None
            self._predict_busy = False
            if hasattr(self, "predict_btn"):
                self.predict_btn.setEnabled(True)
            if bool(event.get("canceled")):
                LabelingApp._cleanup_prediction_depth_staging(self)
                self.update_status_bar("Prediction canceled.")
                return
            if bool(event.get("had_error")):
                LabelingApp._cleanup_prediction_depth_staging(self)
                self._on_predict_error(
                    str(event.get("error_message") or "Unknown prediction error")
                )
                return
            prediction = event.get("prediction")
            if not isinstance(prediction, dict):
                LabelingApp._cleanup_prediction_depth_staging(self)
                self._on_predict_error("Prediction worker returned no prediction payload.")
                return
            if not requested_image_path or os.path.normcase(
                os.path.abspath(requested_image_path)
            ) != os.path.normcase(displayed_image_path):
                LabelingApp._cleanup_prediction_depth_staging(self)
                self.update_status_bar(
                    "Prediction finished for a different image and was discarded."
                )
                return
            self._apply_prediction_payload(prediction)

    def _cancel_prediction_process(self) -> None:
        process = self._prediction_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._prediction_cancel_requested = True
            self._prediction_expected_stop = True
            request_qprocess_stop(
                process,
                schedule=QTimer.singleShot,
                force_kill=self._kill_prediction_process_if_running,
                kill_after_ms=3000,
            )

    def _kill_prediction_process_if_running(self) -> None:
        process = self._prediction_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_prediction_process_error(self, _error) -> None:
        process = self._prediction_process
        if process is not None:
            self._prediction_stderr += process.errorString() + "\n"

    def _finish_prediction_process(self, exit_code: int, exit_status) -> None:
        if self._prediction_process is None:
            return
        if self._prediction_stdout_buffer.strip():
            self._handle_prediction_event_line(self._prediction_stdout_buffer.strip())
            self._prediction_stdout_buffer = ""

        stderr_text = self._prediction_stderr.strip()
        cancel_requested = self._prediction_cancel_requested
        expected_stop = self._prediction_expected_stop
        self._prediction_process = None
        self._prediction_config_path = None
        self._prediction_result_event = None
        self._prediction_stdout_buffer = ""
        self._prediction_stderr = ""
        self._prediction_cancel_requested = False
        self._prediction_worker_ready = False
        self._prediction_expected_stop = False

        if cancel_requested:
            self._prediction_pending_request = None
            self._prediction_current_request_id = None
            self._prediction_image_path = None
            self._predict_busy = False
            self._cleanup_prediction_depth_staging()
            if hasattr(self, "predict_btn"):
                self.predict_btn.setEnabled(True)
            self.update_status_bar("Prediction canceled.")
            return
        if expected_stop:
            return
        if self._predict_busy:
            self._prediction_pending_request = None
            self._prediction_current_request_id = None
            self._prediction_image_path = None
            self._predict_busy = False
            self._cleanup_prediction_depth_staging()
            if hasattr(self, "predict_btn"):
                self.predict_btn.setEnabled(True)
            self._on_predict_error(
                stderr_text or f"Prediction worker exited with code {exit_code}."
            )
        else:
            self.update_status_bar(
                "Prediction worker stopped; it will restart on the next prediction."
            )

    def _apply_prediction_payload(self, prediction: dict):
        try:
            if str(prediction.get("workflow") or "") == WORKFLOW_DEPTH:
                targets = self._prediction_depth_targets or {}
                replacements = [
                    (targets.get("staged_map", ""), targets.get("final_map", "")),
                    (
                        targets.get("staged_preview", ""),
                        targets.get("final_preview", ""),
                    ),
                    (
                        targets.get("staged_metadata", ""),
                        targets.get("final_metadata", ""),
                    ),
                ]
                if not all(stage and target for stage, target in replacements):
                    raise RuntimeError("Depth prediction output transaction is incomplete.")
                commit_staged_paths(replacements)
                self._prediction_depth_targets = None
                self._clear_depth_probes()
                self.load_image()
                self._update_progress_label()
                metadata = prediction.get("depth_metadata") or {}
                median = metadata.get("median_depth")
                suffix = (
                    f" Median estimated depth: {float(median):.3f} m." if median is not None else ""
                )
                self.update_status_bar(
                    "Depth map saved and displayed (model-default scale)." + suffix
                )
                return
            self._cache_active_annotation()

            active_cid = self.class_selector.currentIndex()
            raw_detections = prediction.get("detections") or []
            if not isinstance(raw_detections, list) or not raw_detections:
                self.update_status_bar("Prediction returned no detections.")
                return

            detections = [det for det in raw_detections if isinstance(det, dict)]
            if not detections:
                self.update_status_bar("Prediction returned no usable detections.")
                return

            best_by_class: dict[int, int] = {}
            conf_list: list[float] = []
            for det_idx, det in enumerate(detections):
                try:
                    conf = float(det.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf = 0.0
                conf_list.append(conf)
                try:
                    cid = int(det.get("class_id", active_cid))
                except Exception:
                    cid = active_cid
                if cid < 0 or cid >= len(self.classes):
                    continue
                prev_idx = best_by_class.get(cid)
                if prev_idx is None or conf >= conf_list[prev_idx]:
                    best_by_class[cid] = det_idx

            if not best_by_class:
                best_idx = max(range(len(detections)), key=lambda i: conf_list[i])
                best_by_class[active_cid] = best_idx

            if self._is_seg_workflow():
                applied_count = 0
                missing_mask_count = 0
                for cid, det_idx in best_by_class.items():
                    det = detections[det_idx]
                    seg_points: list[tuple[float, float]] = []
                    for raw_pair in det.get("segments") or []:
                        try:
                            if len(raw_pair) < 2:
                                continue
                            seg_points.append((float(raw_pair[0]), float(raw_pair[1])))
                        except Exception:
                            continue
                    if len(seg_points) < 3:
                        missing_mask_count += 1
                        continue
                    self._clear_class_items(cid, drop_cache=False)
                    self.annotation_cache[cid] = {
                        "class_id": cid,
                        "segments": seg_points,
                        "score": conf_list[det_idx] if det_idx < len(conf_list) else 0.0,
                    }
                    self._restore_annotation_for_class(cid)
                    applied_count += 1

                if applied_count > 0:
                    self._clear_seg_prompt_state()
                    self._clear_seg_preview()
                    self._update_item_editability()
                    self._update_status()
                    self._jump_to_next_pending_class()
                    status_msg = "Segmentation prediction applied."
                    if missing_mask_count:
                        status_msg += (
                            f" Skipped {missing_mask_count} detection(s) without usable masks."
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

            applied_count = 0
            for cid, det_idx in best_by_class.items():
                det = detections[det_idx]
                xyxy = det.get("xyxy") or []
                try:
                    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
                except Exception:
                    continue
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                bb = BoundingBox(x1, y1, w, h, cid)
                self._clear_class_items(cid, drop_cache=True)
                item = BoxItem(bb, self.classes[cid] if cid < len(self.classes) else str(cid))
                self.scene.addItem(item)
                self._track_scene_item(item)

                kp_objs: list[Keypoint] = []
                class_kp_names = self._kp_names_for_index(cid)
                for idx_pt, raw_kp in enumerate(det.get("keypoints") or []):
                    try:
                        if len(raw_kp) < 3:
                            continue
                        x, y, kp_conf = float(raw_kp[0]), float(raw_kp[1]), float(raw_kp[2])
                    except Exception:
                        continue
                    if idx_pt >= len(self.kp_names):
                        break
                    canonical_name = self.kp_names[idx_pt]
                    if canonical_name not in class_kp_names:
                        continue
                    kp_obj = Keypoint(x, y, cid, canonical_name)
                    kp_item = KeypointItem(kp_obj, self.kp_pixel_radius, self.kp_font_px)
                    setattr(kp_item, "pred_conf", kp_conf)
                    kp_item.update_appearance()
                    self.scene.addItem(kp_item)
                    self._track_scene_item(kp_item)
                    kp_objs.append(kp_obj)

                if cid == active_cid:
                    self.bboxes = [bb]
                    self.kps = kp_objs[:]
                    self.current_kp_idx = min(len(class_kp_names), len(self.kps))
                self._cache_active_annotation(cid)
                applied_count += 1

            if applied_count == 0:
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
        names = self._active_kp_names()
        if self.current_kp_idx >= len(names):
            self.update_status_bar("All keypoints already placed.")
            return

        name = names[self.current_kp_idx]
        cid = self.class_selector.currentIndex()

        # Use (0,0) for invisibles; YOLO ignores coords when v=0
        kp = Keypoint(0.0, 0.0, cid, name)
        item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
        item.visibility = 0
        item.update_appearance()

        # Keep it in the scene so saving picks it up (subtle visual)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self.kps.append(kp)

        # Advance to next missing name
        if hasattr(self, "_sync_current_kp_idx"):
            self._sync_current_kp_idx()
        else:
            self.current_kp_idx = min(self.current_kp_idx + 1, len(names))

        self._update_status()
        self.update_status_bar(f"Marked '{name}' invisible (v=0).")
        self._maybe_autoadvance()

    def set_selected_invisible(self):
        """Convert selected keypoints to invisible (v=0) without moving them."""
        changed = False
        for it in self.scene.selectedItems():
            if isinstance(it, KeypointItem):
                it.visibility = 0
                it.update_appearance()
                changed = True
        if changed:
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
        bbox = entry.get("bbox", {})
        data = {
            "class": self.classes[cid],
            "bbox": {
                "xc": (bbox.get("x", 0.0) + bbox.get("w", 0.0) / 2.0) / max(1.0, float(self.img_w)),
                "yc": (bbox.get("y", 0.0) + bbox.get("h", 0.0) / 2.0) / max(1.0, float(self.img_h)),
                "w": bbox.get("w", 0.0) / max(1.0, float(self.img_w)),
                "h": bbox.get("h", 0.0) / max(1.0, float(self.img_h)),
            },
            "keypoints": [],
        }
        for kp in entry.get("keypoints", []):
            vis = int(kp.get("vis", 2))
            data["keypoints"].append(
                {
                    "name": kp.get("name", ""),
                    "idx": int(kp.get("idx", len(data["keypoints"]))),
                    "canon_idx": int(kp.get("canon_idx", -1)),
                    "x": 0.0 if vis == 0 else kp.get("x", 0.0) / max(1.0, float(self.img_w)),
                    "y": 0.0 if vis == 0 else kp.get("y", 0.0) / max(1.0, float(self.img_h)),
                    "vis": vis,
                }
            )
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
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Template error", f"Failed to load template:\n{e}")
            return

        bbox = data.get("bbox", {})
        xc = bbox.get("xc", 0.5)
        yc = bbox.get("yc", 0.5)
        w = bbox.get("w", 1.0)
        h = bbox.get("h", 1.0)
        x = (xc - w / 2.0) * self.img_w
        y = (yc - h / 2.0) * self.img_h
        rect = QRectF(x, y, w * self.img_w, h * self.img_h)

        cid = self.class_selector.currentIndex()
        class_name = self.classes[cid]
        self._clear_class_items(cid, drop_cache=True)
        bbox_obj = BoundingBox(rect.x(), rect.y(), rect.width(), rect.height(), cid)
        item = BoxItem(bbox_obj, class_name)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self.bboxes = [bbox_obj]

        kp_lookup = {}
        for kp in data.get("keypoints", []):
            idx = int(kp.get("idx", -1))
            if idx >= 0:
                kp_lookup[idx] = kp
        self.kps.clear()
        active_names = self._active_kp_names()
        for idx_name, name in enumerate(active_names):
            kp_info = kp_lookup.get(idx_name)
            if kp_info:
                vis = int(kp_info.get("vis", 2))
                if vis == 0:
                    x_pix = 0.0
                    y_pix = 0.0
                else:
                    x_pix = kp_info.get("x", 0.0) * self.img_w
                    y_pix = kp_info.get("y", 0.0) * self.img_h
            else:
                vis = 0
                x_pix = 0.0
                y_pix = 0.0
            kp = Keypoint(x_pix, y_pix, cid, name)
            kp_item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
            kp_item.visibility = vis
            kp_item.update_appearance()
            self.scene.addItem(kp_item)
            self._track_scene_item(kp_item)
            self.kps.append(kp)
        self.current_kp_idx = len(active_names)
        self._cache_active_annotation()
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

        self.scene.clear()
        self._item_refs.clear()
        self._depth_probe_items = []
        self._active_depth_map = None
        if not self.images:
            return

        current_image_name = self.images[self.current_idx]
        if current_image_name != self._depth_probe_image_name:
            self._depth_probe_image_name = current_image_name
            self._depth_probes = []
            self._depth_probe_error = ""
        self._refresh_depth_probe_label()

        img_path = os.path.join(self.active_image_dir, current_image_name)
        self.current_image_path = img_path
        pix = QPixmap(img_path)
        if pix.isNull():
            self.update_status_bar(f"Failed to load image: {self.images[self.current_idx]}")
            return
        self.img_w, self.img_h = pix.width(), pix.height()
        self.scene.setSceneRect(0, 0, self.img_w, self.img_h)
        bg_item = QGraphicsPixmapItem(pix)
        bg_item.setZValue(0)
        self.scene.addItem(bg_item)

        self.bboxes.clear()
        self.kps.clear()
        self.current_kp_idx = 0
        base = os.path.splitext(self.images[self.current_idx])[0]
        if LabelingApp._is_depth_layer(self):
            self.annotation_cache.clear()
            map_path = os.path.join(self.depth_image_dir, f"{base}.npy")
            if _np is None:
                self._depth_probe_error = "NumPy is unavailable; pixel sampling is disabled."
            elif not os.path.isfile(map_path):
                self._depth_probe_error = "No raw depth map is available for pixel sampling."
            else:
                try:
                    depth_map = _np.load(map_path, mmap_mode="r", allow_pickle=False)
                    if depth_map.ndim != 2:
                        raise ValueError(f"expected 2 dimensions, received {depth_map.ndim}")
                    if tuple(depth_map.shape) != (self.img_h, self.img_w):
                        raise ValueError(
                            f"map {tuple(depth_map.shape)} does not match "
                            f"image {(self.img_h, self.img_w)}"
                        )
                    self._active_depth_map = depth_map
                    self._depth_probe_error = ""
                except (OSError, ValueError) as exc:
                    self._depth_probe_error = f"Pixel sampling unavailable: {exc}"
            self._refresh_depth_probe_label()
            self._update_depth_range_label(base)
            display_mode = self._depth_view_mode()
            preview_path = os.path.join(self.depth_preview_dir, f"{base}_depth.png")
            if display_mode != "original" and os.path.isfile(preview_path):
                depth_pixmap = QPixmap(preview_path)
                if not depth_pixmap.isNull():
                    if depth_pixmap.width() != self.img_w or depth_pixmap.height() != self.img_h:
                        depth_pixmap = depth_pixmap.scaled(
                            self.img_w,
                            self.img_h,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    depth_item = QGraphicsPixmapItem(depth_pixmap)
                    depth_item.setZValue(1.0)
                    depth_item.setOpacity(0.55 if display_mode == "overlay" else 1.0)
                    self.scene.addItem(depth_item)
                    self._track_scene_item(depth_item)
                    self.update_status_bar(
                        "Saved depth overlay displayed."
                        if display_mode == "overlay"
                        else "Saved depth map displayed (near = bright)."
                    )
            elif display_mode == "original":
                self.update_status_bar(
                    "Original image displayed; a saved depth map remains available."
                )
            else:
                self.update_status_bar(
                    "No saved depth map for this image. Select Predict to create one."
                )
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
        self.annotation_cache = PoseAnnotationDocument(entries)
        for cid in range(len(self.classes)):
            if cid in self.annotation_cache:
                self._restore_annotation_for_class(cid)
        self._sync_active_class_state()
        self._update_item_editability()

        self._update_status()
        if hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()
        self._refresh_reference_layer_overlay()
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
        bbox = BoundingBox(rect.x(), rect.y(), rect.width(), rect.height(), cid)
        item = BoxItem(bbox, class_name)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self.bboxes = [bbox]
        self.kps.clear()
        self.current_kp_idx = 0
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
        names = self._active_kp_names()
        if self.current_kp_idx >= len(names):
            self.update_status_bar("All keypoints placed for this frame.")
            return
        cid = self.class_selector.currentIndex()
        name = names[self.current_kp_idx]
        kp = Keypoint(pos.x(), pos.y(), cid, name)
        item = KeypointItem(kp, self.kp_pixel_radius, self.kp_font_px)
        self.scene.addItem(item)
        self._track_scene_item(item)
        self.kps.append(kp)
        self.current_kp_idx = min(self.current_kp_idx + 1, len(names))
        self._update_status()
        self._maybe_autoadvance()

    def delete_selected(self):
        cid = self.class_selector.currentIndex()
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
                self.seg_prompt_points.pop()
                self._refresh_seg_prompt_markers()
                self.update_status_bar("Removed last segmentation prompt.")
                return
            seg_item = self._class_seg_mask_item(cid)
            if seg_item is not None:
                self._safe_remove_scene_item(seg_item)
                self._untrack_scene_item(seg_item)
                self.annotation_cache.pop(cid, None)
                self.update_status_bar("Removed segmentation mask for current class.")
                self._update_item_editability()
                self._refresh_sam_controls()
                return
        if self.mode == "keypoint" and self.kps:
            kp = self.kps.pop()
            for it in list(self.scene.items()):
                if isinstance(it, KeypointItem) and it.kp is kp and it.kp.class_id == cid:
                    self._safe_remove_scene_item(it)
                    self._untrack_scene_item(it)
                    break
            self.current_kp_idx = max(0, self.current_kp_idx - 1)
            self._update_status()
        elif self.mode == "bbox" and self.bboxes:
            bb = self.bboxes.pop()
            for it in list(self.scene.items()):
                if isinstance(it, BoxItem) and it.bbox is bb and it.bbox.class_id == cid:
                    self._safe_remove_scene_item(it)
                    self._untrack_scene_item(it)
                    break
            for it in list(self.scene.items()):
                if isinstance(it, KeypointItem) and it.kp.class_id == cid:
                    self._safe_remove_scene_item(it)
                    self._untrack_scene_item(it)
            self.kps.clear()
            self.current_kp_idx = 0
            self.annotation_cache.pop(cid, None)
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
            entry = self.annotation_cache.get(cid)
            if not entry:
                return False
            required = len(self._kp_names_for_index(cid))
            if len(entry.get("keypoints", [])) != required:
                return False
        return True

    def _update_status(self):
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

        if self.mode == "keypoint":
            self.legend_frame.show()
            self.zoom_frame.hide()
            self._layout_overlays()
            self.update_status_bar(self._kp_text())
        elif self.mode == "panzoom":
            self.legend_frame.hide()
            self.zoom_frame.show()
            self._layout_overlays()
            self.update_zoom_label()
        else:
            self.legend_frame.hide()
            self.zoom_frame.hide()
        self._refresh_sam_controls()

    def toggle_selected_visibility(self):
        for item in self.scene.selectedItems():
            if isinstance(item, KeypointItem):
                item.toggle_visibility()

    def update_zoom_label(self):
        zoom = int(self.view.transform().m11() * 100)
        self.zoom_label.setText(f"Zoom: {zoom}%")

    def _layout_hot_corners(self):
        if not hasattr(self, "view"):
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

        lines = []
        for class_idx in range(len(self.classes)):
            entry = self.annotation_cache.get(class_idx)
            if not entry:
                continue
            line = self._annotation_entry_to_line(entry)
            if line:
                lines.append(line)

        if not lines:
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
                AnnotationSaveRequest(
                    project_root=self.project_root,
                    source_image_path=src_path,
                    image_output_path=image_out_path,
                    label_output_path=label_out_path,
                    overlay_output_path=annotated_out_path,
                    label_text="\n".join(lines) + "\n",
                ),
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
        if not report.temporary_paths:
            self.update_status_bar("Project health scan complete.")
            return

        answer = QMessageBox.question(
            self,
            "Remove Temporary Files?",
            (
                f"Remove {len(report.temporary_paths)} stale transaction "
                "file(s) or staging folder(s)?\n\n"
                "Worker config files and project data will not be removed."
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
        layer = self._active_layer_definition()
        if self._is_depth_layer():
            QMessageBox.information(
                self, "Depth MVP", "Depth training is not included in the inference-only MVP."
            )
            return
        if self._is_segmentation_layer():
            default_dataset = os.path.join(self.project_root, "datasets", "segment")
            if not os.path.isdir(default_dataset):
                default_dataset = os.path.join(self.project_root, "datasets")
            dlg = TrainDialog(
                self,
                default_dataset=default_dataset,
                default_task="segment",
                layer_id=layer.id,
            )
        else:
            default_dataset = os.path.join(self.project_root, "datasets", "pose")
            if not os.path.isdir(default_dataset):
                default_dataset = os.path.join(self.project_root, "datasets")
            dlg = TrainDialog(
                self,
                default_dataset=default_dataset,
                default_task="pose",
                layer_id=layer.id,
            )
        dlg.exec()

    def open_distillation_dialog(self):
        if self._is_depth_layer():
            QMessageBox.information(
                self, "Depth MVP", "Depth distillation is not included in the inference-only MVP."
            )
            return
        dlg = DistillationDialog(self)
        dlg.exec()

    def open_analysis_dialog(self):
        if self._is_depth_layer():
            QMessageBox.information(
                self, "Depth MVP", "Depth analysis tools are not included in the MVP yet."
            )
            return
        dlg = AnalysisDialog(
            self,
            project_root=self.project_root,
            app_base_dir=self.app_base_dir,
            layer_id=self.active_layer,
        )
        dlg.exec()

    def open_video_reviewer(self):
        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run `uv sync --locked` to restore project dependencies."
            )
            return
        review_model_paths = {
            layer_id: self.layer_model_paths.get(layer_id) or ""
            for layer_id in (LAYER_KEYPOINTS, LAYER_SEGMENTATION)
        }
        reviewer_layer = self.active_layer
        if reviewer_layer == LAYER_DEPTH:
            reviewer_layer = (
                LAYER_KEYPOINTS
                if review_model_paths[LAYER_KEYPOINTS] or not review_model_paths[LAYER_SEGMENTATION]
                else LAYER_SEGMENTATION
            )
        layer_schemas = {
            LAYER_KEYPOINTS: {
                "classes": self.pose_classes[:],
                "kp_names": self.pose_kp_names[:],
                "class_keypoints": {
                    name: self.pose_class_keypoints.get(name, [])[:] for name in self.pose_classes
                },
            },
            LAYER_SEGMENTATION: {
                "classes": self.seg_classes[:],
                "kp_names": [],
                "class_keypoints": {},
            },
        }
        reviewer_schema = layer_schemas[reviewer_layer]
        dlg = VideoReviewDialog(
            self,
            self._device,
            reviewer_schema["kp_names"],
            reviewer_schema["classes"],
            class_keypoints=reviewer_schema["class_keypoints"],
            workflow=layer_worker_mode(reviewer_layer),
            layer_id=reviewer_layer,
            model_paths=review_model_paths,
            layer_schemas=layer_schemas,
        )
        dlg.exec()


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    from squeakpose.app import run

    raise SystemExit(run())
