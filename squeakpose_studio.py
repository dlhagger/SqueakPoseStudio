#!/usr/bin/env python3
import sys, os, shutil, json, random, yaml, re, shlex, platform, datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem,
    QGraphicsSimpleTextItem, QGraphicsLineItem, QGraphicsPathItem, QVBoxLayout, QHBoxLayout,
    QGridLayout,
    QComboBox, QPushButton, QLabel, QSplashScreen, QMessageBox,
    QDialog, QFrame, QStatusBar, QGraphicsDropShadowEffect, QSizePolicy,
    QProgressDialog, QDialogButtonBox, QTabWidget, QSlider, QSpinBox, QDoubleSpinBox, QProgressBar,
    QInputDialog, QFileDialog, QFormLayout, QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QListView
)
from PyQt6.QtGui import (
    QPixmap, QPen, QBrush, QKeySequence, QFont, QPainter, QShortcut,
    QFontDatabase, QIcon, QCursor, QPainterPath, QPainterPathStroker,
    QFontInfo, QColor, QTextCursor
)
from PyQt6.QtCore import (
    Qt, QRectF, QPointF, QTimer, QPoint, QProcess, QLibraryInfo
)
from squeakpose_core import (
    atomic_write_text,
    effective_prediction_batch,
    find_duplicate_names,
    resolve_default_training_dataset_path,
)
from dataset_ops import (
    DATASET_DETECT,
    DATASET_POSE,
    DATASET_SEGMENT,
    backup_label_dir,
    dataset_dirs_have_files,
    dataset_export_paths,
    export_dataset_files,
    format_dataset_export_summary,
    format_label_normalization_summary,
    list_image_files,
    list_label_files,
    normalize_label_directory,
    remove_dataset_split_dirs,
    split_train_val_images,
    write_dataset_yaml_for_mode,
)
from label_io import (
    load_pose_annotations_from_file,
    load_segmentation_annotations_from_file,
    parse_pose_label_line,
    parse_segmentation_label_line,
    pose_annotation_to_line,
    segmentation_annotation_to_line,
)
from inference_ops import (
    probe_video_metadata,
    segmentation_rows_from_result,
)

DEFAULT_CLASS_NAMES = ["mouse"]
DEFAULT_KEYPOINT_NAMES = ["nose", "head", "left_ear", "right_ear", "back", "tail_base"]
DEFAULT_SAM3_WEIGHTS = "sam3.pt"
PROJECT_META_FILE = "squeakpose_project.json"
LAST_PROJECT_STATE_FILE = os.path.join(os.path.expanduser("~"), ".squeakpose_studio_last_project.json")
WORKFLOW_POSE = "pose"
WORKFLOW_SEG = "segmentation"


def _qt_app_instance():
    return QApplication.instance()


def _retain_main_window(window) -> None:
    app = _qt_app_instance()
    if app is not None:
        app._squeakpose_main_window = window


def _project_paths(project_root: str) -> dict[str, str]:
    root = os.path.abspath(project_root)
    return {
        "root": root,
        "images_to_label": os.path.join(root, "images_to_label"),
        "images_all": os.path.join(root, "images_all"),
        "labels_all": os.path.join(root, "labels_all"),
        "labels_seg_all": os.path.join(root, "labels_seg_all"),
        "annotations": os.path.join(root, "annotations"),
        "datasets": os.path.join(root, "datasets"),
        "runs": os.path.join(root, "runs"),
        "templates": os.path.join(root, "templates"),
        "inference_outputs": os.path.join(root, "inference outputs"),
        "logs": os.path.join(root, "logs"),
        "classes_file": os.path.join(root, "classes.txt"),
        "keypoints_file": os.path.join(root, "keypoints.txt"),
        "class_keypoints_file": os.path.join(root, "class_keypoints.json"),
        "classes_seg_file": os.path.join(root, "classes_seg.txt"),
    }


def _project_window_title(project_root: str) -> str:
    root = os.path.abspath(project_root)
    name = os.path.basename(root.rstrip(os.sep)) or root
    return f"SqueakPose Studio — {name}"


def _ensure_project_structure(project_root: str) -> dict[str, str]:
    paths = _project_paths(project_root)
    for key in (
        "images_to_label",
        "images_all",
        "labels_all",
        "labels_seg_all",
        "annotations",
        "datasets",
        "runs",
        "templates",
        "inference_outputs",
        "logs",
    ):
        os.makedirs(paths[key], exist_ok=True)

    if not os.path.exists(paths["classes_seg_file"]):
        try:
            atomic_write_text(paths["classes_seg_file"], "".join(f"{name}\n" for name in DEFAULT_CLASS_NAMES))
        except Exception:
            pass

    meta_path = os.path.join(paths["root"], PROJECT_META_FILE)
    if not os.path.exists(meta_path):
        payload = {
            "schema_version": 1,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            atomic_write_text(meta_path, json.dumps(payload, indent=2))
        except Exception:
            pass
    return paths


def _load_last_project() -> Optional[str]:
    if not os.path.exists(LAST_PROJECT_STATE_FILE):
        return None
    try:
        with open(LAST_PROJECT_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = str(data.get("last_project", "")).strip()
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    except Exception:
        return None
    return None


def _save_last_project(project_root: str):
    payload = {"last_project": os.path.abspath(project_root)}
    try:
        atomic_write_text(LAST_PROJECT_STATE_FILE, json.dumps(payload, indent=2))
    except Exception:
        pass


def _default_projects_root() -> str:
    """Return the default parent directory for SqueakPose projects."""
    # Linux users may customize their user dirs; honor it when present.
    if sys.platform.startswith("linux"):
        xdg_docs = os.environ.get("XDG_DOCUMENTS_DIR", "").strip()
        if xdg_docs:
            return os.path.join(os.path.expanduser(xdg_docs), "SqueakPose Studio Projects")
    documents = os.path.join(os.path.expanduser("~"), "Documents")
    return os.path.join(documents, "SqueakPose Studio Projects")


def _choose_project_root(default_dir: str, parent: Optional[QWidget] = None) -> Optional[str]:
    start_dir = default_dir if os.path.isdir(default_dir) else os.path.expanduser("~")
    selected = QFileDialog.getExistingDirectory(
        parent,
        "Select Project Folder",
        start_dir,
    )
    if not selected:
        return None
    return os.path.abspath(selected)


def _create_project_root(default_dir: str, parent: Optional[QWidget] = None) -> Optional[str]:
    start_dir = default_dir if os.path.isdir(default_dir) else os.path.expanduser("~")
    parent_dir = QFileDialog.getExistingDirectory(
        parent,
        "Select Parent Folder for New Project",
        start_dir,
    )
    if not parent_dir:
        return None

    project_name, ok = QInputDialog.getText(parent, "New Project", "Project name:")
    if not ok:
        return None
    project_name = project_name.strip()
    if not project_name:
        QMessageBox.warning(parent, "Invalid name", "Project name cannot be empty.")
        return None

    project_root = os.path.abspath(os.path.join(parent_dir, project_name))
    if os.path.exists(project_root):
        if not os.path.isdir(project_root):
            QMessageBox.warning(parent, "Invalid path", "A file exists with that project name.")
            return None
        if os.listdir(project_root):
            confirm = QMessageBox.question(
                parent,
                "Use Existing Folder?",
                "The selected project folder already contains files.\nUse it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return None
    else:
        try:
            os.makedirs(project_root, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(parent, "Create project failed", f"Could not create project folder:\n{e}")
            return None
    return project_root


class ProjectLauncherDialog(QDialog):
    """Startup dialog for opening or creating a project."""

    def __init__(self, default_dir: str, logo_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.default_dir = default_dir
        self.project_root: Optional[str] = None
        self.selection_mode: Optional[str] = None  # "open" | "create"

        self.setWindowTitle("SqueakPose Studio")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        if logo_path and os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                logo_label = QLabel()
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                logo_label.setPixmap(
                    pix.scaled(
                        220,
                        220,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                layout.addWidget(logo_label)

        title = QLabel("Open a project or create a new one")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Project folders contain classes, keypoints, images, labels, datasets, runs, and analysis outputs."
        )
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open Project")
        open_btn.clicked.connect(self._open_project)
        btn_row.addWidget(open_btn)

        create_btn = QPushButton("Create Project")
        create_btn.clicked.connect(self._create_project)
        btn_row.addWidget(create_btn)
        layout.addLayout(btn_row)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

    def _open_project(self):
        chosen = _choose_project_root(self.default_dir, parent=self)
        if not chosen:
            return
        self.project_root = chosen
        self.selection_mode = "open"
        self.accept()

    def _create_project(self):
        chosen = _create_project_root(self.default_dir, parent=self)
        if not chosen:
            return
        self.project_root = chosen
        self.selection_mode = "create"
        self.accept()


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
            print(f"[Qt bootstrap] Warning: unable to resolve Qt plugins path: {primary_error}", file=sys.stderr)
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
    seen = set()
    for family in ordered:
        if not family or family in seen:
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
            if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                return 'cuda'
            # On macOS, MPS can be present but not fully usable; check both built and available
            if hasattr(_torch, 'backends') and hasattr(_torch.backends, 'mps'):
                mps = _torch.backends.mps
                if getattr(mps, 'is_built', lambda: False)() and getattr(mps, 'is_available', lambda: False)():
                    return 'mps'
        return 'cpu'
    except Exception:
        return 'cpu'

# =========================
# Data Classes
# =========================

@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float
    class_id: int

    def to_yolo(self, img_w: float, img_h: float) -> Tuple[int, float, float, float, float]:
        xc = (self.x + self.w / 2) / img_w
        yc = (self.y + self.h / 2) / img_h
        return (self.class_id, xc, yc, self.w / img_w, self.h / img_h)


@dataclass(slots=True)
class Keypoint:
    x: float
    y: float
    class_id: int
    name: str

    def to_yolo(self, img_w: float, img_h: float) -> Tuple[int, float, float, str]:
        return (self.class_id, self.x / img_w, self.y / img_h, self.name)


@dataclass(slots=True)
class KeypointEntry:
    name: str
    display_name: str
    kp: Keypoint
    visibility: int


@dataclass(slots=True)
class Annotation:
    ann_id: int
    bbox: BoundingBox
    keypoints: Dict[str, KeypointEntry]
    order: List[str]


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


class AddClassDialog(QDialog):
    def __init__(self, existing_keypoints: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Class")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        form.addRow("Class name:", self.name_edit)

        default_count = max(0, len(existing_keypoints))
        if default_count == 0:
            default_count = 6

        self.keypoints_edit = QTextEdit()
        if existing_keypoints:
            initial_lines = existing_keypoints[:]
        else:
            initial_lines = [f"kp_{idx+1}" for idx in range(default_count)]
        self.keypoints_edit.setPlainText("\n".join(initial_lines))
        self.count_label = QLabel("")
        self.keypoints_edit.textChanged.connect(self._update_count_label)
        self._update_count_label()

        info = QLabel("Keypoint names apply to all classes. Enter one per line.")
        info.setWordWrap(True)

        layout.addLayout(form)
        layout.addWidget(info)
        layout.addWidget(self.keypoints_edit, 1)
        layout.addWidget(self.count_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_count_label(self):
        lines = [ln for ln in self.keypoints_edit.toPlainText().splitlines() if ln.strip()]
        self.count_label.setText(f"Keypoint count: {len(lines)}")

    def get_data(self) -> tuple[str, list[str]]:
        name = self.name_edit.text().strip()
        keys = [ln.strip() for ln in self.keypoints_edit.toPlainText().splitlines() if ln.strip()]
        return name, keys


class ClassManagerDialog(QDialog):
    def __init__(
        self,
        classes: list[str],
        keypoint_map: dict[str, list[str]],
        canonical: list[str],
        parent=None,
        schema_locked: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Manage Classes & Keypoints")
        self.resize(420, 480)

        self._classes = classes[:]
        self._kp_map: dict[str, list[str]] = {}
        for name in self._classes:
            lst = keypoint_map.get(name, canonical[:])
            self._kp_map[name] = lst[:]
        self._canonical_default = canonical[:]
        self._current_row = -1
        self._schema_locked = bool(schema_locked)

        layout = QVBoxLayout(self)

        if self._schema_locked:
            lock_info = QLabel(
                "Schema is locked because labeled data exists.\n"
                "Allowed: add class, append keypoints.\n"
                "Blocked: remove/reorder/rename existing classes/keypoints."
            )
            lock_info.setWordWrap(True)
            layout.addWidget(lock_info)

        self.class_list = QListWidget()
        for name in self._classes:
            self.class_list.addItem(name)
        layout.addWidget(QLabel("Classes"))
        layout.addWidget(self.class_list, 1)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Class")
        self.add_btn.clicked.connect(self._add_class)
        btn_row.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.remove_btn)
        if self._schema_locked:
            self.remove_btn.setEnabled(False)
            self.remove_btn.setToolTip("Schema locked after labels exist.")
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Keypoint Names (per class, one per line)"))
        self.keypoints_edit = QTextEdit()
        layout.addWidget(self.keypoints_edit, 2)

        self.status_label = QLabel("Keypoint count: 0")
        layout.addWidget(self.status_label)
        self.keypoints_edit.textChanged.connect(self._update_count_label)
        self.class_list.currentRowChanged.connect(self._load_selected_class)
        if self._classes:
            self.class_list.setCurrentRow(0)
        else:
            self._load_selected_class(-1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.result_classes: Optional[list[str]] = None
        self.result_keypoints: Optional[list[str]] = None
        self.result_map: Optional[dict[str, list[str]]] = None

    def _add_class(self):
        seed = []
        current = self.class_list.currentRow()
        if current >= 0 and current < len(self._classes):
            seed = self._kp_map.get(self._classes[current], [])
        if not seed:
            if self._canonical_default:
                seed = self._canonical_default[:]
        dlg = AddClassDialog(seed, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, keypoints = dlg.get_data()
        if not name:
            QMessageBox.warning(self, "Class name required", "Enter a class name.")
            return
        if name in self._classes:
            QMessageBox.warning(self, "Duplicate class", "That class already exists.")
            return
        if not keypoints:
            keypoints = []
        self._classes.append(name)
        self.class_list.addItem(name)
        self._kp_map[name] = keypoints[:]
        self.class_list.setCurrentRow(len(self._classes) - 1)

    def _remove_selected(self):
        if self._schema_locked:
            QMessageBox.information(
                self,
                "Schema locked",
                "Cannot remove classes after labeled data exists.",
            )
            return
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self._classes):
            return
        name = self._classes.pop(row)
        if name in self._kp_map:
            del self._kp_map[name]
        item = self.class_list.takeItem(row)
        del item
        QMessageBox.information(self, "Class removed", f"Removed '{name}'.")
        next_row = min(row, len(self._classes) - 1)
        self.class_list.setCurrentRow(next_row)

    def _update_count_label(self):
        kp = [ln.strip() for ln in self.keypoints_edit.toPlainText().splitlines() if ln.strip()]
        self.status_label.setText(f"Keypoint count: {len(kp)}")

    def _load_selected_class(self, row: int):
        self._save_current_keypoints()
        self._current_row = row
        if row < 0 or row >= len(self._classes):
            self.keypoints_edit.clear()
            self.status_label.setText("Keypoint count: 0")
            return
        name = self._classes[row]
        kp = self._kp_map.get(name, [])
        self.keypoints_edit.blockSignals(True)
        self.keypoints_edit.setPlainText("\n".join(kp))
        self.keypoints_edit.blockSignals(False)
        self._update_count_label()

    def _save_current_keypoints(self):
        if self._current_row < 0 or self._current_row >= len(self._classes):
            return
        name = self._classes[self._current_row]
        kp = [ln.strip() for ln in self.keypoints_edit.toPlainText().splitlines() if ln.strip()]
        self._kp_map[name] = kp[:]

    def _on_accept(self):
        self._save_current_keypoints()
        if not self._classes:
            QMessageBox.warning(self, "No classes", "Add at least one class.")
            return
        defined_any = any(self._kp_map.get(name) for name in self._classes)
        if not defined_any:
            QMessageBox.warning(self, "Keypoints required", "Enter at least one keypoint for any class.")
            return
        for class_name in self._classes:
            dupes = find_duplicate_names(self._kp_map.get(class_name, []))
            if dupes:
                joined = ", ".join(dupes)
                QMessageBox.warning(
                    self,
                    "Duplicate keypoints",
                    f"Class '{class_name}' has duplicate keypoint names:\n{joined}\n\n"
                    "Each keypoint name must be unique within a class."
                )
                return
        canonical = []
        seen = set()
        for name in self._canonical_default:
            if name not in seen:
                canonical.append(name)
                seen.add(name)
        for cls in self._classes:
            for kp_name in self._kp_map.get(cls, []):
                if kp_name not in seen:
                    canonical.append(kp_name)
                    seen.add(kp_name)
        if not canonical:
            QMessageBox.warning(self, "Keypoints required", "No keypoint names defined.")
            return
        self._canonical_default = canonical[:]
        self.result_classes = self._classes[:]
        self.result_keypoints = canonical[:]
        self.result_map = {name: self._kp_map.get(name, [])[:] for name in self._classes}
        self.accept()

    def get_results(self) -> tuple[list[str], list[str], dict[str, list[str]]]:
        return (
            self.result_classes or [],
            self.result_keypoints or [],
            self.result_map or {},
        )

class BoxItem(QGraphicsRectItem):
    HANDLE = 8
    MIN_W = 6
    MIN_H = 6
    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 4, 8

    def __init__(self, bbox: BoundingBox, class_name: str):
        super().__init__(0, 0, max(self.MIN_W, bbox.w), max(self.MIN_H, bbox.h))
        self.setPos(bbox.x, bbox.y)
        self.bbox = bbox
        self.class_name = class_name

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(2)

        pen = QPen(Qt.GlobalColor.blue)
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.setPen(pen)

        self._label_bg = QGraphicsRectItem(self)
        self._label_bg.setBrush(QBrush(Qt.GlobalColor.blue))
        self._label_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self._label_bg.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label_bg.setZValue(0.0)

        self._label = QGraphicsSimpleTextItem(class_name, self)
        self._label.setFont(_ui_font(12))
        self._label.setBrush(QBrush(Qt.GlobalColor.white))
        self._label.setZValue(0.1)
        self._label_pad_x = 4.0
        self._label_pad_y = 1.0
        self._reposition_label()

        self._drag_edges = 0
        self._press_rect = QRectF()
        self._press_pos_local = QPointF()
        self._press_item_pos = QPointF()

        self.update_model()  # ✅ now exists

    def _reposition_label(self):
        """Keep class label outside the bbox to avoid keypoint overlap.

        Preferred placement: above top-left corner. If the box is near the top
        edge of the scene, place the label just below the box instead.
        """
        margin = 2.0
        text_rect = self._label.boundingRect()
        bg_w = text_rect.width() + (self._label_pad_x * 2.0)
        bg_h = text_rect.height() + (self._label_pad_y * 2.0)
        x = margin
        y = -(bg_h + margin)

        if self.scene():
            sr = self.scene().sceneRect()
            # If above would clip off-screen, move label below the box.
            if self.pos().y() + y < sr.top():
                y = self.rect().height() + margin

        self._label_bg.setRect(x, y, bg_w, bg_h)
        self._label.setPos(x + self._label_pad_x, y + self._label_pad_y)

    # --- only outline is clickable/selectable ---
    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        stroker = QPainterPathStroker()
        stroker.setWidth(max(6.0, float(self.HANDLE) * 2.0))  # clickable band
        return stroker.createStroke(path)

    def contains(self, point: QPointF) -> bool:
        return self.shape().contains(point)

    # --- edge hit/cursor ---
    def _hit_edges(self, p_local: QPointF) -> int:
        r = self.rect()
        tol = max(6.0, float(self.HANDLE))
        edges = 0
        if abs(p_local.x() - r.left())   <= tol: edges |= self.LEFT
        if abs(p_local.x() - r.right())  <= tol: edges |= self.RIGHT
        if abs(p_local.y() - r.top())    <= tol: edges |= self.TOP
        if abs(p_local.y() - r.bottom()) <= tol: edges |= self.BOTTOM
        return edges

    def _cursor_for_edges(self, edges: int):
        # diagonals first, then single-axis
        if edges in (self.TOP | self.LEFT, self.BOTTOM | self.RIGHT):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (self.TOP | self.RIGHT, self.BOTTOM | self.LEFT):
            return Qt.CursorShape.SizeBDiagCursor
        if edges & (self.LEFT | self.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edges & (self.TOP | self.BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeAllCursor

    # --- events ---
    def hoverMoveEvent(self, event):
        edges = self._hit_edges(event.pos())
        self.setCursor(self._cursor_for_edges(edges) if edges else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._hit_edges(event.pos())
            if not edges:
                # clicked inside -> let keypoints receive it
                event.ignore()
                return
            self._drag_edges = edges
            self._press_rect = QRectF(self.rect())
            self._press_pos_local = QPointF(event.pos())
            self._press_item_pos = QPointF(self.pos())
            self.setZValue(2.5)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_edges:
            delta = event.pos() - self._press_pos_local
            self._apply_resize(self._drag_edges, delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_edges:
            self._drag_edges = 0
            self.setZValue(2)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # --- resize & clamp ---
    def _apply_resize(self, edges: int, delta_local: QPointF):
        new_rect = QRectF(self._press_rect)
        new_pos = QPointF(self._press_item_pos)

        if edges & self.LEFT:
            new_pos.setX(self._press_item_pos.x() + delta_local.x())
            new_rect.setWidth(max(self.MIN_W, self._press_rect.width() - delta_local.x()))
        if edges & self.RIGHT:
            new_rect.setWidth(max(self.MIN_W, self._press_rect.width() + delta_local.x()))
        if edges & self.TOP:
            new_pos.setY(self._press_item_pos.y() + delta_local.y())
            new_rect.setHeight(max(self.MIN_H, self._press_rect.height() - delta_local.y()))
        if edges & self.BOTTOM:
            new_rect.setHeight(max(self.MIN_H, self._press_rect.height() + delta_local.y()))

        if self.scene():
            sr = self.scene().sceneRect()
            new_pos.setX(max(sr.left(), new_pos.x()))
            new_pos.setY(max(sr.top(),  new_pos.y()))
            new_pos.setX(min(new_pos.x(), sr.right() - new_rect.width()))
            new_pos.setY(min(new_pos.y(), sr.bottom() - new_rect.height()))

        self.setPos(new_pos)
        self.setRect(0, 0, new_rect.width(), new_rect.height())
        self.update_model()

    # --- sync bbox dataclass ---
    def update_model(self):
        self.bbox.x = self.pos().x()
        self.bbox.y = self.pos().y()
        self.bbox.w = self.rect().width()
        self.bbox.h = self.rect().height()
        self._reposition_label()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            sr = self.scene().sceneRect()
            r = self.rect()
            new_pos = value
            nx = min(max(new_pos.x(), sr.left()), sr.right() - r.width())
            ny = min(max(new_pos.y(), sr.top()),  sr.bottom() - r.height())
            return QPointF(nx, ny)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            self._reposition_label()
        elif change in (QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
                        QGraphicsItem.GraphicsItemChange.ItemTransformChange):
            self.update_model()
        return super().itemChange(change, value)

class KeypointItem(QGraphicsEllipseItem):
    """
    Precise keypoint:
      - position in scene coords (setPos)
      - local rect centered at origin
      - ignores transformations (constant on-screen size)
      - cosmetic pen; clamped to image bounds
    """
    def __init__(self, kp: Keypoint, pixel_radius: int = 4, font_px: int = 10):
        super().__init__(-pixel_radius, -pixel_radius, pixel_radius*2, pixel_radius*2)
        self.kp = kp
        self.visibility = 2
        self._pixel_radius = max(1, pixel_radius)
        self._font_px = max(6, font_px)

        self.setPos(kp.x, kp.y)  # scene-space anchor
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(3)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

        color = Qt.GlobalColor.red
        pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
        self.setPen(pen); self.setBrush(QBrush(color))

        self.text_item = QGraphicsSimpleTextItem(kp.name, self)
        self.text_item.setFont(_ui_font(self._font_px))
        self.text_item.setBrush(QBrush(color))

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(0); shadow.setOffset(1, 1); shadow.setColor(Qt.GlobalColor.black)
        self.text_item.setGraphicsEffect(shadow)

        self._reposition_label()

    def _reposition_label(self):
        self.text_item.setPos(self._pixel_radius + 2, -self._pixel_radius - 2)

    def refresh_display_sizes(self, pixel_radius: int, font_px: int):
        self._pixel_radius = max(1, pixel_radius)
        self._font_px = max(6, font_px)
        self.prepareGeometryChange()
        self.setRect(-self._pixel_radius, -self._pixel_radius, self._pixel_radius*2, self._pixel_radius*2)
        self.text_item.setFont(_ui_font(self._font_px))
        self._reposition_label()

    def update_appearance(self):
        # 2 = visible (red), 1 = occluded (yellow), 0 = invisible/not present (gray dashed)
        if self.visibility == 2:
            color = Qt.GlobalColor.red
            pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
            self.setPen(pen); self.setBrush(QBrush(color))
            self.text_item.setBrush(QBrush(color))
            self.text_item.setVisible(True)
        elif self.visibility == 1:
            color = Qt.GlobalColor.yellow
            pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
            self.setPen(pen); self.setBrush(QBrush(color))
            self.text_item.setBrush(QBrush(color))
            self.text_item.setVisible(True)
        else:  # self.visibility == 0
            color = Qt.GlobalColor.lightGray
            pen = QPen(color); pen.setStyle(Qt.PenStyle.DashLine); pen.setWidth(1); pen.setCosmetic(True)
            self.setPen(pen)
            self.setBrush(QBrush(Qt.GlobalColor.transparent))
            self.text_item.setBrush(QBrush(color))
            # optional: keep labels hidden for invisible to reduce clutter
            self.text_item.setVisible(False)

    def toggle_visibility(self):
        # Cycle: 2 (visible) -> 1 (occluded) -> 0 (invisible) -> 2 ...
        if self.visibility == 2:
            self.visibility = 1
        elif self.visibility == 1:
            self.visibility = 0
        else:
            self.visibility = 2
        self.update_appearance()

    def update_model(self):
        p = self.pos()
        self.kp.x = p.x()
        self.kp.y = p.y()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            sr = self.scene().sceneRect()
            p = value
            x = min(max(p.x(), sr.left()), sr.right())
            y = min(max(p.y(), sr.top()),  sr.bottom())
            return QPointF(x, y)
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.update_model()
        return super().itemChange(change, value)


# =========================
# View
# =========================

class LabelView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, app_ref):
        super().__init__(scene)
        self.app = app_ref
        self._start_pos: Optional[QPointF] = None
        self._crosshair_v: Optional[QGraphicsLineItem] = None
        self._crosshair_h: Optional[QGraphicsLineItem] = None
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._drawing_cancelled = False
        self._seg_brush_active = False
        self._seg_brush_add = True
        self._seg_brush_last_pos: Optional[QPointF] = None
        self._seg_brush_cursor_ring: Optional[QGraphicsEllipseItem] = None

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        if self.app.mode == 'panzoom':
            old_pos = self.mapToScene(event.position().toPoint())
            zoom_in_factor = 1.05
            zoom_out_factor = 1 / zoom_in_factor
            zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
            new_scale = self.transform().m11() * zoom_factor
            if new_scale < 1.0:
                zoom_factor = 1.0 / self.transform().m11()
            elif new_scale > 8.0:
                zoom_factor = 8.0 / self.transform().m11()
            self.scale(zoom_factor, zoom_factor)
            self.app.update_zoom_label()
            new_pos = self.mapToScene(event.position().toPoint())
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        else:
            super().wheelEvent(event)

    def _remove_crosshairs(self):
        if self._crosshair_v:
            owner_scene = self._crosshair_v.scene()
            if owner_scene is not None:
                owner_scene.removeItem(self._crosshair_v)
            self._crosshair_v = None
        if self._crosshair_h:
            owner_scene = self._crosshair_h.scene()
            if owner_scene is not None:
                owner_scene.removeItem(self._crosshair_h)
            self._crosshair_h = None

    def _ensure_crosshairs(self):
        """Create crosshair items with consistent styling if they do not exist."""
        if self._crosshair_v is None:
            self._crosshair_v = QGraphicsLineItem()
            self._crosshair_v.setZValue(10)
            pen = QPen(Qt.GlobalColor.cyan)
            pen.setCosmetic(True)
            self._crosshair_v.setPen(pen)
            self.scene().addItem(self._crosshair_v)
        if self._crosshair_h is None:
            self._crosshair_h = QGraphicsLineItem()
            self._crosshair_h.setZValue(10)
            pen = QPen(Qt.GlobalColor.cyan)
            pen.setCosmetic(True)
            self._crosshair_h.setPen(pen)
            self.scene().addItem(self._crosshair_h)

    def _update_crosshairs(self, scene_pos: QPointF):
        """Ensure crosshairs exist and update them to intersect at scene_pos."""
        if not self.scene():
            return
        self._ensure_crosshairs()
        img_bounds = self.scene().sceneRect()
        self._crosshair_v.setLine(scene_pos.x(), img_bounds.top(), scene_pos.x(), img_bounds.bottom())
        self._crosshair_h.setLine(img_bounds.left(), scene_pos.y(), img_bounds.right(), scene_pos.y())

    def draw_crosshairs_at(self, global_pos: QPoint):
        scene_pos = self.mapToScene(self.mapFromGlobal(global_pos))
        self._update_crosshairs(scene_pos)

    def _should_show_seg_brush_cursor(self) -> bool:
        try:
            return (
                self.app._is_seg_workflow()
                and self.app.mode == "segedit"
                and self.app._is_seg_edit_tool_brush()
                and self.scene() is not None
            )
        except Exception:
            return False

    def _ensure_seg_brush_cursor_ring(self):
        if self._seg_brush_cursor_ring is not None and self._seg_brush_cursor_ring.scene() is self.scene():
            return
        self._seg_brush_cursor_ring = QGraphicsEllipseItem()
        pen = QPen(QColor(118, 188, 255, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self._seg_brush_cursor_ring.setPen(pen)
        self._seg_brush_cursor_ring.setBrush(QBrush(Qt.GlobalColor.transparent))
        self._seg_brush_cursor_ring.setZValue(9.2)
        self._seg_brush_cursor_ring.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._seg_brush_cursor_ring.setAcceptHoverEvents(False)
        self._seg_brush_cursor_ring.setVisible(False)
        self.scene().addItem(self._seg_brush_cursor_ring)

    def _hide_seg_brush_cursor(self):
        ring = self._seg_brush_cursor_ring
        if ring is not None:
            ring.setVisible(False)

    def _reset_seg_brush_cursor(self):
        ring = self._seg_brush_cursor_ring
        if ring is None:
            return
        try:
            if ring.scene() is not None:
                ring.scene().removeItem(ring)
        except Exception:
            pass
        self._seg_brush_cursor_ring = None

    def _update_seg_brush_cursor(self, scene_pos: QPointF):
        if not self._should_show_seg_brush_cursor():
            self._hide_seg_brush_cursor()
            return
        self._ensure_seg_brush_cursor_ring()
        ring = self._seg_brush_cursor_ring
        if ring is None:
            return
        radius = max(2.0, float(getattr(self.app, "seg_brush_radius", 8)))
        x = float(scene_pos.x())
        y = float(scene_pos.y())
        ring.setRect(x - radius, y - radius, radius * 2.0, radius * 2.0)
        ring.setVisible(True)

    def refresh_seg_brush_cursor(self):
        if not self._should_show_seg_brush_cursor():
            self._hide_seg_brush_cursor()
            return
        vp = self.viewport().mapFromGlobal(QCursor.pos())
        if not self.viewport().rect().contains(vp):
            self._hide_seg_brush_cursor()
            return
        self._update_seg_brush_cursor(self.mapToScene(vp))

    def _cancel_draw(self):
        self._drawing_cancelled = True
        if self._temp_rect:
            owner_scene = self._temp_rect.scene()
            if owner_scene is not None:
                owner_scene.removeItem(self._temp_rect)
            self._temp_rect = None
        self._start_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self.app.mode == "segment":
            self.app._clear_seg_prompt_state()
            self.app.update_status_bar("Segmentation prompts cleared.")
        else:
            self.app.update_status_bar("Box drawing cancelled.")

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if self.app.mode == "segedit":
            if event.button() == Qt.MouseButton.LeftButton and self.app._start_seg_brush(scene_pos, add=True):
                self._seg_brush_active = True
                self._seg_brush_add = True
                self._seg_brush_last_pos = scene_pos
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton and self.app._start_seg_brush(scene_pos, add=False):
                self._seg_brush_active = True
                self._seg_brush_add = False
                self._seg_brush_last_pos = scene_pos
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.app.mode == 'panzoom':
                super().mousePressEvent(event)
            elif self.app.mode == 'bbox':
                # Do NOT clear here — clear only when committing a valid rect
                self._start_pos = scene_pos
                self._drawing_cancelled = False
                self._remove_crosshairs()
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif self.app.mode == 'keypoint':
                self.app.add_keypoint(scene_pos)
            elif self.app.mode == 'segment':
                self.app._add_seg_prompt(scene_pos, positive=True)
                event.accept()
                return
        elif event.button() == Qt.MouseButton.RightButton and self.app.mode == "segment":
            self.app._add_seg_prompt(scene_pos, positive=False)
            event.accept()
            return
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self._update_seg_brush_cursor(scene_pos)
        if self.app.mode == "segedit" and self._seg_brush_active:
            if self.app._apply_seg_brush(scene_pos, add=self._seg_brush_add, prev_scene_pos=self._seg_brush_last_pos):
                self._seg_brush_last_pos = scene_pos
            event.accept()
            return
        if (self.app.mode == 'bbox' and self._start_pos is None) or (self.app.mode in {'keypoint', 'segment'}):
            self._update_crosshairs(scene_pos)
        elif self._start_pos and self.app.mode == 'bbox':
            self._remove_crosshairs()
            end_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._start_pos, end_pos).normalized()
            if not self._temp_rect:
                self._temp_rect = QGraphicsRectItem(rect)
                pen = QPen(Qt.GlobalColor.yellow); pen.setWidth(2); pen.setCosmetic(True)
                self._temp_rect.setPen(pen); self._temp_rect.setZValue(1.5)
                self.scene().addItem(self._temp_rect)
            else:
                self._temp_rect.setRect(rect)
        elif self.app.mode == 'panzoom':
            self._remove_crosshairs()
            super().mouseMoveEvent(event)

    def enterEvent(self, event):
        self.refresh_seg_brush_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hide_seg_brush_cursor()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.app.mode == "segedit" and self._seg_brush_active:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.app._apply_seg_brush(scene_pos, add=self._seg_brush_add, prev_scene_pos=self._seg_brush_last_pos)
            self._seg_brush_active = False
            self._seg_brush_last_pos = None
            self.app._finish_seg_brush()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos and self.app.mode == 'bbox':
            if not self._drawing_cancelled:
                end_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._start_pos, end_pos).normalized()
                if rect.width() >= 2 and rect.height() >= 2:
                    self.app.add_bbox(rect)
            if self._temp_rect:
                owner_scene = self._temp_rect.scene()
                if owner_scene is not None:
                    owner_scene.removeItem(self._temp_rect)
                self._temp_rect = None
            self._start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.app.mode == 'panzoom':
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetTransform()
            self.app.update_zoom_label()


# =========================
# Video Review Pan/Zoom View
# =========================

class VideoView(QGraphicsView):
    """Lightweight pan/zoom view for the video reviewer.
    - Mouse wheel: zoom in/out centered on cursor
    - Left-drag: pan (ScrollHandDrag)
    - Double-click: reset zoom
    - Shortcuts (+/-) handled by the dialog via QShortcut
    """
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_in = 1.05
        zoom_out = 1.0 / zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        # clamp zoom between 10% and 800%
        cur = self.transform().m11()
        new_scale = cur * factor
        if new_scale < 0.10:
            factor = 0.10 / cur
        elif new_scale > 8.0:
            factor = 8.0 / cur
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.resetTransform()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def reset_view(self):
        self.resetTransform()

# =========================
# Main Application
# =========================

class LabelingApp(QMainWindow):

    def _is_pose_workflow(self) -> bool:
        return getattr(self, "active_workflow", WORKFLOW_POSE) == WORKFLOW_POSE

    def _is_seg_workflow(self) -> bool:
        return getattr(self, "active_workflow", WORKFLOW_POSE) == WORKFLOW_SEG

    def _workflow_label(self) -> str:
        return "Pose" if self._is_pose_workflow() else "Segmentation"

    def _ensure_classes_file(self, class_file: str, defaults: list[str]) -> tuple[list[str], bool]:
        created_any = False
        project_root = os.path.dirname(class_file) if class_file else os.getcwd()
        if not class_file:
            class_file = os.path.join(project_root, "classes.txt")

        try:
            cf_dir = os.path.dirname(class_file)
            if cf_dir:
                os.makedirs(cf_dir, exist_ok=True)
        except Exception:
            pass

        if not os.path.exists(class_file):
            try:
                with open(class_file, "a", encoding="utf-8"):
                    pass
                created_any = True
            except Exception:
                pass

        classes: list[str] = []
        try:
            with open(class_file, "r", encoding="utf-8") as f:
                classes = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            classes = []

        if not classes:
            classes = defaults[:] or DEFAULT_CLASS_NAMES[:]
            try:
                atomic_write_text(class_file, "".join(f"{name}\n" for name in classes))
                created_any = True
            except Exception:
                pass

        return classes, created_any

    def _project_meta_path(self) -> str:
        return os.path.join(self.project_root, PROJECT_META_FILE)

    def _read_project_meta(self) -> dict:
        path = self._project_meta_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _write_project_meta(self, updates: dict):
        if not isinstance(updates, dict):
            return
        path = self._project_meta_path()
        payload = self._read_project_meta()
        if not payload:
            payload = {
                "schema_version": 1,
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        for key, value in updates.items():
            if value is None:
                payload.pop(str(key), None)
            else:
                payload[str(key)] = value
        try:
            atomic_write_text(path, json.dumps(payload, indent=2))
        except Exception:
            pass

    def _meta_normalize_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        return os.path.abspath(os.path.join(self.project_root, raw))

    def _meta_store_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        abs_path = os.path.abspath(raw)
        try:
            rel = os.path.relpath(abs_path, self.project_root)
            if rel == ".":
                return os.path.basename(abs_path)
            if not rel.startswith(".."):
                return rel
        except Exception:
            pass
        return abs_path

    def _load_project_preferences(self):
        meta = self._read_project_meta()
        workflow = str(meta.get("active_workflow", "")).strip().lower()
        if workflow in {WORKFLOW_POSE, WORKFLOW_SEG}:
            self.active_workflow = workflow
        sam_path = self._meta_normalize_path(str(meta.get("sam_model_path", "") or ""))
        if sam_path and os.path.isfile(sam_path):
            self.sam_model_path = sam_path

    def _save_project_preferences(self):
        workflow = WORKFLOW_POSE if self._is_pose_workflow() else WORKFLOW_SEG
        payload = {"active_workflow": workflow}
        if self.sam_model_path and os.path.isfile(self.sam_model_path):
            payload["sam_model_path"] = self._meta_store_path(self.sam_model_path)
        else:
            payload["sam_model_path"] = None
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

    def _persist_active_workflow_state(self):
        if self._is_pose_workflow():
            self.pose_classes = self.classes[:]
            self.pose_kp_names = self.kp_names[:]
            self.pose_class_keypoints = {name: self.class_keypoints.get(name, [])[:] for name in self.classes}
        else:
            self.seg_classes = self.classes[:]

    def _bind_workflow_state(self, workflow: str):
        if workflow == WORKFLOW_POSE:
            self.active_workflow = WORKFLOW_POSE
            self.label_dir = self.pose_label_dir
            self.class_file = self.pose_class_file
            self.keypoint_file = self.pose_keypoint_file
            self.class_keypoints_path = self.pose_class_keypoints_path
            self.classes = self.pose_classes[:]
            self.kp_names = self.pose_kp_names[:]
            self.class_keypoints = {name: self.pose_class_keypoints.get(name, [])[:] for name in self.classes}
            if self._sync_canonical_keypoints_from_class_map():
                self.pose_kp_names = self.kp_names[:]
            self._schema_locked = self._detect_schema_locked()
        else:
            self.active_workflow = WORKFLOW_SEG
            self.label_dir = self.seg_label_dir
            self.class_file = self.seg_class_file
            self.keypoint_file = ""
            self.class_keypoints_path = ""
            self.classes = self.seg_classes[:]
            self.kp_names = []
            self.class_keypoints = {}
            self._schema_locked = self._detect_schema_locked()

        self._refresh_kp_index_lookup()

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

    def _ensure_workflow_selector_items(self):
        if not hasattr(self, "workflow_selector"):
            return
        expected = [
            ("Pose Workflow (BBox + Keypoints)", WORKFLOW_POSE),
            ("Segmentation Workflow (SAM)", WORKFLOW_SEG),
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

        current_workflow = getattr(self, "active_workflow", WORKFLOW_POSE)
        self.workflow_selector.blockSignals(True)
        self.workflow_selector.clear()
        for text, data in expected:
            self.workflow_selector.addItem(text, data)
        self.workflow_selector.setCurrentIndex(0 if current_workflow == WORKFLOW_POSE else 1)
        self.workflow_selector.blockSignals(False)

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
            "Before labeling in Segmentation workflow, define what objects/classes you want to segment.\n\n"
            "Open Segmentation Classes now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if decision == QMessageBox.StandardButton.Yes:
            self._open_seg_class_manager()
        else:
            self.update_status_bar("Using default segmentation class ('mouse'). Edit via Seg Classes… anytime.")

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

    def _update_workflow_ui_state(self):
        is_pose = self._is_pose_workflow()
        self._ensure_workflow_selector_items()

        self.save_btn.setEnabled(True)
        self.complete_btn.setEnabled(True)
        self.bbox_btn.setEnabled(is_pose)
        self.segment_btn.setEnabled(not is_pose)
        self.keypoint_btn.setEnabled(is_pose)
        self.predict_btn.setEnabled(True)
        self.seg_edit_btn.setEnabled(not is_pose)
        self.sam_load_btn.setEnabled(not is_pose)
        self.sam_run_btn.setEnabled(not is_pose)
        self.sam_accept_btn.setEnabled(not is_pose)
        self.sam_clear_btn.setEnabled(not is_pose)
        self.template_apply_btn.setEnabled(is_pose)
        self.template_save_btn.setEnabled(is_pose)
        self.inference_btn.setEnabled(True)
        self.normalize_btn.setEnabled(True)
        self.export_dataset_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        if hasattr(self, "delete_image_btn"):
            self.delete_image_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)

        self.manage_classes_btn.setToolTip(
            "Manage classes and per-class keypoints" if is_pose else "Manage segmentation classes"
        )
        self.manage_classes_btn.setText("Classes…")
        if hasattr(self, "class_label_widget"):
            self.class_label_widget.setText("Class")
        if hasattr(self, "bbox_btn"):
            self.bbox_btn.setVisible(is_pose)
        if hasattr(self, "keypoint_btn"):
            self.keypoint_btn.setVisible(is_pose)
        if hasattr(self, "predict_btn"):
            self.predict_btn.setVisible(True)
            self.predict_btn.setToolTip(
                "Run YOLO pose prediction on the current image"
                if is_pose else
                "Run YOLO segmentation prediction on the current image"
            )
        if hasattr(self, "segment_btn"):
            self.segment_btn.setVisible(not is_pose)
        if hasattr(self, "seg_edit_btn"):
            self.seg_edit_btn.setVisible(not is_pose)
        if hasattr(self, "seg_tools_frame"):
            self.seg_tools_frame.setVisible(not is_pose)
        self._reflow_mode_grid(is_pose=is_pose)
        self.save_btn.setText("Save")
        self.save_btn.setToolTip(
            "Save labels for current frame" if is_pose else "Save segmentation masks for current frame"
        )

        if is_pose and self.mode in {"segment", "segedit"}:
            self.mode = "panzoom"
        if not is_pose and self.mode not in {"panzoom", "segment", "segedit"}:
            self.mode = "segment"

        self._clear_seg_edit_handles()
        self._refresh_seg_brush_size_badge()
        if hasattr(self, "view") and hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()

        self._update_status()
        self._update_progress_label()
        self._refresh_sam_controls()
        self._layout_hot_corners()
        self._layout_overlays()

    def _switch_workflow(self, workflow: str):
        workflow = WORKFLOW_POSE if workflow == WORKFLOW_POSE else WORKFLOW_SEG
        if workflow == getattr(self, "active_workflow", WORKFLOW_POSE):
            return
        self._ensure_workflow_selector_items()
        if hasattr(self, "workflow_selector"):
            idx = 0 if workflow == WORKFLOW_POSE else 1
            if self.workflow_selector.currentIndex() != idx:
                self.workflow_selector.blockSignals(True)
                self.workflow_selector.setCurrentIndex(idx)
                self.workflow_selector.blockSignals(False)
        self._persist_active_workflow_state()
        self._bind_workflow_state(workflow)
        self._save_project_preferences()
        if workflow == WORKFLOW_SEG:
            self.mode = "segment"
        elif self.mode == "segment":
            self.mode = "panzoom"
        self._refresh_class_selector_for_workflow()
        self.annotation_cache.clear()
        self._clear_seg_prompt_state()
        self._update_workflow_ui_state()
        self.load_image()
        if self._is_seg_workflow():
            loaded_now, loaded_path = self._try_autoload_sam_model_from_project_root()
            self._refresh_sam_controls()
            if loaded_now:
                self.update_status_bar(
                    f"Segmentation workflow enabled. Auto-loaded SAM model: {os.path.basename(loaded_path)}"
                )
            elif self.sam_model is not None:
                self.update_status_bar("Segmentation workflow enabled. SAM model ready.")
            else:
                self.update_status_bar("Segmentation workflow enabled. Use Segment mode and SAM prompts.")
            QTimer.singleShot(0, self._maybe_prompt_seg_class_manager_initial)
        else:
            self.update_status_bar("Pose workflow enabled.")

    def _on_workflow_changed(self, _index: int):
        self._ensure_workflow_selector_items()
        workflow = self.workflow_selector.currentData()
        self._switch_workflow(str(workflow))

    def _ensure_label_files(self, class_file: str, keypoint_file: str) -> tuple[list[str], list[str], bool]:
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

        # Ensure parent dirs exist
        try:
            cf_dir = os.path.dirname(class_file)
            kf_dir = os.path.dirname(keypoint_file)
            if cf_dir:
                os.makedirs(cf_dir, exist_ok=True)
            if kf_dir and kf_dir != cf_dir:
                os.makedirs(kf_dir, exist_ok=True)
        except Exception:
            pass

        # Helper to write a list to file
        def _write_lines(path: str, items: list[str]):
            try:
                atomic_write_text(path, "".join(f"{s}\n" for s in items))
            except Exception:
                pass

        # Create files if missing
        def _touch(path: str) -> bool:
            try:
                with open(path, "a", encoding="utf-8"):
                    return True
            except Exception:
                return False

        if not os.path.exists(class_file):
            if _touch(class_file):
                created_any = True
        if not os.path.exists(keypoint_file):
            if _touch(keypoint_file):
                created_any = True

        # Read current values.
        def _read_nonempty_lines(path: str) -> list[str]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if not lines:
                    return []
                return lines
            except Exception:
                return []

        classes = _read_nonempty_lines(class_file)
        kp_names = _read_nonempty_lines(keypoint_file)

        # Backfill defaults so the app is always usable even if initial setup is skipped.
        if not classes:
            classes = DEFAULT_CLASS_NAMES[:]
            _write_lines(class_file, classes)
            created_any = True
        if not kp_names:
            kp_names = DEFAULT_KEYPOINT_NAMES[:]
            _write_lines(keypoint_file, kp_names)
            created_any = True

        return classes, kp_names, created_any

    def _load_class_keypoints(self) -> dict[str, list[str]]:
        data: dict[str, list[str]] = {}
        if os.path.exists(self.class_keypoints_path):
            try:
                with open(self.class_keypoints_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    for name, lst in raw.items():
                        if isinstance(name, str) and isinstance(lst, list):
                            cleaned = [str(item) for item in lst if isinstance(item, str)]
                            if cleaned:
                                data[name] = cleaned
            except Exception:
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
        keypoint_file = getattr(self, "keypoint_file", "") or getattr(self, "pose_keypoint_file", "")
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

    def _count_labeled_images(self, images: list[str], label_dir: str) -> tuple[int, int]:
        total = len(images)
        labeled = 0
        for img in images:
            base = os.path.splitext(img)[0]
            label_file = os.path.join(label_dir, f"{base}.txt")
            if os.path.exists(label_file):
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
        if classes_clean[:len(existing_classes)] != existing_classes:
            return (
                False,
                "Existing class names/order are locked.\n"
                "Only append new classes at the end.",
            )
        for class_name in existing_classes:
            old_kp = self.class_keypoints.get(class_name, [])[:]
            new_kp = normalized_map.get(class_name, [])[:]
            if len(new_kp) < len(old_kp):
                return False, f"Cannot remove keypoints from class '{class_name}'."
            if new_kp[:len(old_kp)] != old_kp:
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
        self.progress_label.setText(f"Queue: {queue_labeled}/{queue_total} labeled")

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

    def _extract_seg_item_points(self, item: Optional[QGraphicsPathItem]) -> list[tuple[float, float]]:
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

    def _downsample_seg_points(self, points: list[tuple[float, float]], max_points: int = 1200) -> list[tuple[float, float]]:
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

    def _seg_update_item_geometry(self, item: Optional[QGraphicsPathItem], points: list[tuple[float, float]]) -> bool:
        if item is None:
            return False
        normalized = [(float(x), float(y)) for x, y in points]
        path = self._polygon_path(normalized)
        if path is None:
            return False
        item.seg_points = normalized
        item.setPath(path)
        label_item = getattr(item, "seg_label_item", None)
        if label_item is not None:
            bbox = path.boundingRect()
            label_item.setPos(bbox.left() + 4.0, bbox.top() + 4.0)
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
            fg = "#ecf5ff"
            bg = "rgba(69, 101, 132, 190)"
            border = "rgba(164, 195, 226, 180)"
            self.seg_brush_size_label.setStyleSheet(
                "font-size: 9pt; font-weight: 700; padding: 2px 9px; border-radius: 8px; "
                f"color: {fg}; background-color: {bg}; border: 1px solid {border};"
            )
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

    def _apply_seg_brush(self, scene_pos: QPointF, add: bool, prev_scene_pos: Optional[QPointF] = None) -> bool:
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

    def _add_seg_mask_item(self, class_id: int, points: list[tuple[float, float]], preview: bool = False):
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
        fill_alpha = 60 if preview else 105
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
        label_item = QGraphicsSimpleTextItem(label_text, item)
        label_item.setBrush(QBrush(color))
        label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        bbox = path.boundingRect()
        label_item.setPos(bbox.left() + 4.0, bbox.top() + 4.0)
        label_item.setZValue(0.2)
        item.seg_label_item = label_item

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
            has_mask = len(entry.get("segments", [])) >= 3 or (self._class_seg_mask_item(cid) is not None)
        completed = sum(1 for idx in range(len(self.classes)) if self._class_is_complete(idx))
        run_enabled = has_image and in_segment_mode and total_prompts > 0 and model_loaded
        accept_enabled = has_preview
        clear_enabled = has_preview or total_prompts > 0
        load_enabled = SAM is not None and not model_loaded
        if hasattr(self, "sam_load_btn"):
            self.sam_load_btn.setEnabled(load_enabled)
            self.sam_load_btn.setText("Load SAM" if load_enabled else ("SAM Ready" if model_loaded else "SAM N/A"))
            self.sam_load_btn.setStyleSheet(
                "background-color: rgba(84, 98, 112, 225); border-color: rgba(164, 180, 195, 165);"
                if load_enabled else ""
            )
        self.sam_run_btn.setEnabled(run_enabled)
        self.sam_accept_btn.setEnabled(accept_enabled)
        self.sam_clear_btn.setEnabled(clear_enabled)
        self.sam_run_btn.setStyleSheet(
            "background-color: rgba(67, 104, 149, 230); border-color: rgba(150, 178, 207, 170);"
            if (run_enabled and not has_preview) else ""
        )
        self.sam_accept_btn.setStyleSheet(
            "background-color: rgba(62, 124, 92, 235); border-color: rgba(137, 201, 169, 180);"
            if accept_enabled else ""
        )
        self.sam_clear_btn.setStyleSheet(
            "background-color: rgba(108, 79, 79, 220); border-color: rgba(183, 147, 147, 165);"
            if clear_enabled else ""
        )
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
            edit_text = f"left-drag add, right-drag erase (brush {brush_px}px, ,/. resize)."
            if has_preview or has_mask:
                action = f"Mask edit ({tool_text}): {edit_text}"
            else:
                action = "No mask yet. Run SAM and accept, or edit a current preview."
        elif not model_loaded:
            action = "Next: Load SAM, add prompts, then Run (G)."
        elif not has_image:
            action = "Open an image to segment."
        elif not in_segment_mode:
            action = "Press 2 for Segment mode. Left click = positive prompt, right click = negative prompt."
        elif not total_prompts and not has_preview:
            action = "Left click = positive prompt, right click = negative prompt."
        elif total_prompts and not has_preview:
            action = "Run SAM (G) to generate a preview mask."
        else:
            action = "Accept (Shift+Enter) to commit this mask."

        mask_text = "saved" if has_mask else "none"
        preview_text = "ready" if has_preview else "none"
        model_text = "ready" if model_loaded else ("missing" if SAM is not None else "unavailable")
        self.sam_helper_label.setText(
            f"Class {class_name} | Done {completed}/{len(self.classes)} | Model {model_text} | Mask {mask_text}\n"
            f"Prompts +{pos_prompts}/-{neg_prompts} | Preview {preview_text} | {action}"
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
            cross_a = QGraphicsLineItem(x - radius + 1.0, y - radius + 1.0, x + radius - 1.0, y + radius - 1.0)
            cross_b = QGraphicsLineItem(x - radius + 1.0, y + radius - 1.0, x + radius - 1.0, y - radius + 1.0)
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
            QMessageBox.information(self, "No prompts", "Add at least one prompt point before running SAM.")
            return
        if not self._ensure_sam_model_loaded():
            self._refresh_sam_controls()
            return
        self._refresh_sam_controls()

        points = [[x, y] for x, y, _ in self.seg_prompt_points]
        labels = [int(lb) for _, _, lb in self.seg_prompt_points]
        img_source = self.current_image_path or os.path.join(self.active_image_dir, self.images[self.current_idx])

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
            QMessageBox.information(self, "No masks", "SAM did not return any segmentation mask for these prompts.")
            return

        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or len(masks) == 0:
            QMessageBox.information(self, "No masks", "SAM did not return any segmentation mask for these prompts.")
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
            QMessageBox.information(self, "No polygon", "SAM returned a mask without a usable contour polygon.")
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
            QMessageBox.information(self, "No preview", "Run SAM first to create a segmentation mask preview.")
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
            if isinstance(item, BoxItem) and item.bbox.class_id == class_id:
                return item
        return None

    def _class_keypoint_items(self, class_id: int) -> list[KeypointItem]:
        return [
            item for item in self.scene.items()
            if isinstance(item, KeypointItem) and item.kp.class_id == class_id
        ]

    def _clear_class_items(self, class_id: int, drop_cache: bool = False):
        removed = False
        for item in list(self.scene.items()):
            if isinstance(item, BoxItem) and item.bbox.class_id == class_id:
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                removed = True
            elif isinstance(item, KeypointItem) and item.kp.class_id == class_id:
                self._safe_remove_scene_item(item)
                self._untrack_scene_item(item)
                removed = True
            elif self._is_seg_mask_item(item) and int(getattr(item, "seg_class_id", -1)) == class_id:
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
            if isinstance(item, BoxItem):
                editable = (item.bbox.class_id == active_cid)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
                item.setOpacity(1.0 if editable else 0.4)
            elif isinstance(item, KeypointItem):
                editable = (item.kp.class_id == active_cid)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, editable)
                item.setOpacity(1.0 if editable else 0.4)
            elif self._is_seg_mask_item(item):
                editable = (int(getattr(item, "seg_class_id", -1)) == active_cid)
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

    def _refresh_queue_images(self):
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        try:
            self.images_queue = sorted(
                f for f in os.listdir(self.image_dir_queue) if f.lower().endswith(exts)
            )
        except Exception:
            self.images_queue = []

    def __init__(
        self,
        image_dir: Optional[str],
        label_dir: Optional[str],
        class_file: Optional[str],
        keypoint_file: Optional[str],
        project_root: Optional[str] = None,
        force_initial_setup: bool = False,
    ):
        super().__init__()
        self.app_base_dir = os.path.dirname(__file__)
        inferred_root = project_root or os.path.dirname(image_dir or "") or os.getcwd()
        self.project_root = os.path.abspath(inferred_root)
        self._force_initial_setup = bool(force_initial_setup)

        self.image_dir_queue = image_dir or os.path.join(self.project_root, "images_to_label")
        # Backward-compatible alias used by some dialogs/tools.
        self.image_dir = self.image_dir_queue
        self.image_dir_all = os.path.join(self.project_root, "images_all")
        self.pose_label_dir = label_dir or os.path.join(self.project_root, "labels_all")
        self.seg_label_dir = os.path.join(self.project_root, "labels_seg_all")
        os.makedirs(self.image_dir_queue, exist_ok=True)
        os.makedirs(self.pose_label_dir, exist_ok=True)
        os.makedirs(self.seg_label_dir, exist_ok=True)
        os.makedirs(self.image_dir_all, exist_ok=True)
        self.pose_class_file = class_file or os.path.join(self.project_root, "classes.txt")
        self.pose_keypoint_file = keypoint_file or os.path.join(self.project_root, "keypoints.txt")
        self.pose_class_keypoints_path = os.path.join(self.project_root, "class_keypoints.json")
        self.seg_class_file = os.path.join(self.project_root, "classes_seg.txt")
        self.base_dir = self.project_root

        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        self.images_queue = sorted(f for f in os.listdir(self.image_dir_queue) if f.lower().endswith(exts))
        self.images = self.images_queue[:]
        self.active_image_dir = self.image_dir_queue
        self.current_image_path = ""
        self.current_idx = 0
        self._queue_current_idx = 0

        # Pose workflow resources
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
        self.pose_class_keypoints = {name: self.class_keypoints.get(name, [])[:] for name in self.pose_classes}

        # Segmentation workflow resources
        self.seg_classes, self._created_seg_class_file = self._ensure_classes_file(
            self.seg_class_file, DEFAULT_CLASS_NAMES
        )

        self.active_workflow = WORKFLOW_POSE
        self.classes = self.pose_classes[:]
        self.kp_names = self.pose_kp_names[:]
        self.class_keypoints = {name: self.pose_class_keypoints.get(name, [])[:] for name in self.pose_classes}
        self.label_dir = self.pose_label_dir
        self.class_file = self.pose_class_file
        self.keypoint_file = self.pose_keypoint_file
        self._schema_locked = self._detect_schema_locked()
        self._kp_index_lookup: dict[str, int] = {}
        self._refresh_kp_index_lookup()
        self.annotation_cache: dict[int, dict] = {}
        self.template_dir = os.path.join(self.project_root, "templates")
        os.makedirs(self.template_dir, exist_ok=True)

        self.mode = 'panzoom'
        self.bboxes: List[BoundingBox] = []
        self.kps: List[Keypoint] = []
        self.current_kp_idx = 0
        self._item_refs: list[QGraphicsItem] = []
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
        self.nav_filter = 'all'  # 'all' | 'labeled' | 'unlabeled'

        self._load_project_preferences()
        self._bind_workflow_state(self.active_workflow)
        self._save_project_preferences()

        # keypoint display (screen-space)
        self.kp_pixel_radius = 4
        self.kp_font_px = 10
        self._precision_active = False

        self._log_path = os.path.join(self.project_root, "logs", "squeakpose_debug.log")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
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
        self._prediction_process: Optional[QProcess] = None
        self._prediction_stdout_buffer = ""
        self._prediction_stderr = ""
        self._prediction_result_event: Optional[dict] = None
        self._prediction_config_path: Optional[str] = None
        self._prediction_image_path: Optional[str] = None
        self._prediction_cancel_requested = False
        # Auto-select device once at startup
        self._device = _auto_device()
        print(f"🧠 Inference device: {self._device}")
        # Build UI and load first image
        self._setup_ui()
        self._update_workflow_ui_state()
        self.load_image()
        self._update_progress_label()
        if self._is_seg_workflow():
            QTimer.singleShot(0, self._maybe_prompt_seg_class_manager_initial)
        else:
            QTimer.singleShot(0, self._maybe_prompt_class_manager)
    def closeEvent(self, event):
        if self._inference_process is not None and self._inference_process.state() != QProcess.ProcessState.NotRunning:
            self._cancel_inference_process()
        if self._prediction_process is not None and self._prediction_process.state() != QProcess.ProcessState.NotRunning:
            self._cancel_prediction_process()
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

        paths = _ensure_project_structure(target_root)
        _save_last_project(target_root)

        new_window = LabelingApp(
            paths["images_to_label"],
            paths["labels_all"],
            paths["classes_file"],
            paths["keypoints_file"],
            project_root=paths["root"],
            force_initial_setup=force_initial_setup,
        )
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
        default_dir = os.path.dirname(self.project_root) if self.project_root else _default_projects_root()
        project_root = _choose_project_root(default_dir, parent=self)
        if not project_root:
            return
        self._switch_to_project_root(project_root, force_initial_setup=False)

    def close_project_command(self):
        if not self._confirm_project_change("Close this project and return to the project launcher?"):
            return
        default_dir = os.path.dirname(self.project_root) if self.project_root else _default_projects_root()
        launcher = ProjectLauncherDialog(default_dir, os.path.join(self.app_base_dir, "squeakpose_studio_logo.png"), self)
        if launcher.exec() != QDialog.DialogCode.Accepted:
            self.update_status_bar("Close project canceled.")
            return
        project_root = launcher.project_root
        if not project_root:
            self.update_status_bar("No project selected.")
            return
        self._switch_to_project_root(project_root, force_initial_setup=(launcher.selection_mode == "create"))

    # ---------- UI Setup ----------

    def _setup_ui(self):
        self.setWindowTitle('SqueakPose Studio')
        self._setup_menu()
        central = QWidget()
        self.setCentralWidget(central)

        self.scene = QGraphicsScene()
        self.view = LabelView(self.scene, self)

        panel_style = """
            QFrame {
                background-color: rgba(34, 38, 42, 200);
                border: 1px solid rgba(128, 141, 152, 130);
                border-radius: 13px;
            }
            QLabel {
                background: transparent;
                border: none;
                padding: 0px;
                color: #e2e8ee;
            }
            QLabel#panelTitle {
                font-weight: 700;
                font-size: 10pt;
                color: #f5f8fb;
                padding-bottom: 4px;
                border-bottom: 1px solid rgba(130, 144, 156, 110);
            }
            QLabel#fieldLabel {
                font-size: 9pt;
                color: #c8d0d8;
            }
            QLabel#brushSizeBadge {
                font-size: 9pt;
                font-weight: 700;
                color: #dce8f4;
                background-color: rgba(65, 76, 87, 170);
                border: 1px solid rgba(130, 145, 160, 130);
                border-radius: 8px;
                padding: 2px 9px;
                min-width: 84px;
            }
            QLabel#sectionLabel {
                font-size: 8pt;
                font-weight: 700;
                color: #aebac8;
                letter-spacing: 0.8px;
            }
            QLabel#samHelper {
                font-size: 9pt;
                color: #e3ebf3;
                background-color: rgba(56, 64, 72, 150);
                border: 1px solid rgba(120, 135, 149, 120);
                border-radius: 8px;
                padding: 5px 7px;
                line-height: 1.15;
            }
            QLabel#progressBadge {
                font-weight: 700;
                color: #f3f7fb;
                background-color: rgba(69, 82, 93, 165);
                border: 1px solid rgba(130, 144, 156, 130);
                border-radius: 8px;
                padding: 3px 8px;
            }
            QPushButton {
                background-color: rgba(52, 58, 64, 220);
                border: 1px solid rgba(133, 146, 158, 120);
                border-radius: 8px;
                padding: 4px 10px;
                color: #eff3f7;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(67, 75, 83, 230);
                border-color: rgba(154, 169, 183, 160);
            }
            QPushButton:pressed {
                background-color: rgba(42, 48, 53, 235);
            }
            QPushButton:disabled {
                color: rgba(224, 232, 239, 155);
                background-color: rgba(51, 58, 64, 196);
                border-color: rgba(134, 147, 159, 125);
            }
            QPushButton#samAcceptButton:disabled {
                color: rgba(229, 236, 241, 170);
                background-color: rgba(54, 68, 62, 198);
                border-color: rgba(135, 158, 147, 148);
            }
            QComboBox {
                background-color: rgba(43, 49, 54, 218);
                border: 1px solid rgba(129, 142, 154, 120);
                border-radius: 8px;
                padding: 3px 8px;
                min-height: 24px;
                color: #eef3f8;
            }
            QComboBox::drop-down {
                border-left: 1px solid rgba(129, 142, 154, 100);
                width: 18px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(35, 40, 45, 242);
                border: 1px solid rgba(120, 133, 145, 140);
                selection-background-color: rgba(94, 129, 161, 210);
                selection-color: #ffffff;
            }
            QComboBox#workflowSelector {
                combobox-popup: 0;
                font-size: 10pt;
                font-weight: 600;
                padding: 4px 12px;
                min-height: 31px;
            }
            QComboBox#workflowSelector::drop-down {
                width: 24px;
            }
            QComboBox#workflowSelector QAbstractItemView {
                font-size: 10pt;
                background-color: rgba(36, 42, 48, 246);
                color: #edf4fc;
                border: 1px solid rgba(129, 146, 162, 176);
                border-radius: 9px;
                outline: 0px;
                padding: 4px;
                selection-background-color: rgba(97, 136, 171, 230);
                selection-color: #ffffff;
            }
            QComboBox#workflowSelector QAbstractItemView::item {
                min-height: 30px;
                padding: 4px 10px;
            }
            QComboBox#browseSelector {
                combobox-popup: 0;
                font-size: 10pt;
                font-weight: 600;
                padding: 4px 10px;
                min-height: 31px;
            }
            QComboBox#browseSelector::drop-down {
                width: 22px;
            }
            QComboBox#browseSelector QAbstractItemView {
                font-size: 10pt;
                background-color: rgba(36, 42, 48, 246);
                color: #edf4fc;
                border: 1px solid rgba(129, 146, 162, 176);
                border-radius: 9px;
                outline: 0px;
                padding: 4px;
                selection-background-color: rgba(97, 136, 171, 230);
                selection-color: #ffffff;
            }
            QComboBox#browseSelector QAbstractItemView::item {
                min-height: 28px;
                padding: 4px 10px;
            }
            QComboBox#classSelector {
                combobox-popup: 0;
                font-size: 10pt;
                font-weight: 600;
                padding: 4px 10px;
                min-height: 31px;
            }
            QComboBox#classSelector::drop-down {
                width: 22px;
            }
            QComboBox#classSelector QAbstractItemView {
                font-size: 10pt;
                background-color: rgba(36, 42, 48, 246);
                color: #edf4fc;
                border: 1px solid rgba(129, 146, 162, 176);
                border-radius: 9px;
                outline: 0px;
                padding: 4px;
                selection-background-color: rgba(97, 136, 171, 230);
                selection-color: #ffffff;
            }
            QComboBox#classSelector QAbstractItemView::item {
                min-height: 28px;
                padding: 4px 10px;
            }
        """

        # Main layout: keep canvas clean and place controls as hot-corner overlays.
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view)
        central.setLayout(layout)

        def apply_panel_shadow(frame: QFrame):
            shadow = QGraphicsDropShadowEffect(frame)
            shadow.setBlurRadius(24)
            shadow.setOffset(0, 3)
            shadow.setColor(QColor(0, 0, 0, 120))
            frame.setGraphicsEffect(shadow)

        # Shared widgets/state
        self.class_selector = QComboBox()
        self.class_selector.setObjectName("classSelector")
        self.class_selector.addItems(self.classes)
        self.class_selector.setToolTip("Choose the active class to label")
        self.class_selector.setMinimumContentsLength(14)
        self.class_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.class_selector.setMinimumWidth(156)
        self.class_selector.setMinimumHeight(34)
        self.class_selector.setMaxVisibleItems(8)
        class_popup = QListView(self.class_selector)
        class_popup.setUniformItemSizes(True)
        class_popup.setSpacing(2)
        class_popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        class_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.class_selector.setView(class_popup)
        self._fit_class_selector_to_items()
        self.class_selector.currentIndexChanged.connect(self._on_class_changed)
        self._active_class_id = self.class_selector.currentIndex()
        self.workflow_selector = QComboBox()
        self.workflow_selector.setObjectName("workflowSelector")
        self.workflow_selector.addItem("Pose Workflow (BBox + Keypoints)", WORKFLOW_POSE)
        self.workflow_selector.addItem("Segmentation Workflow (SAM)", WORKFLOW_SEG)
        self.workflow_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.workflow_selector.setMinimumContentsLength(30)
        self.workflow_selector.setMinimumWidth(286)
        self.workflow_selector.setMinimumHeight(34)
        self.workflow_selector.setMaxVisibleItems(6)
        workflow_popup = QListView(self.workflow_selector)
        workflow_popup.setUniformItemSizes(True)
        workflow_popup.setSpacing(2)
        workflow_popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        workflow_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.workflow_selector.setView(workflow_popup)
        self.workflow_selector.setToolTip(
            "Choose labeling workflow: Pose for boxes/keypoints, or Segmentation for SAM masks."
        )
        self.workflow_selector.currentIndexChanged.connect(self._on_workflow_changed)

        # -----------------------------
        # Top-left: navigation + labeling
        # -----------------------------
        self.top_left_frame = QFrame(self.view)
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
        self.filter_combo = QComboBox()
        self.filter_combo.setObjectName("browseSelector")
        self.filter_combo.addItems(["All", "Labeled", "Unlabeled"])
        self.filter_combo.setToolTip("Which images to browse with Prev/Next")
        self.filter_combo.setMinimumContentsLength(10)
        self.filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.filter_combo.setMinimumWidth(132)
        self.filter_combo.setMinimumHeight(34)
        self.filter_combo.setMaxVisibleItems(8)
        filter_popup = QListView(self.filter_combo)
        filter_popup.setUniformItemSizes(True)
        filter_popup.setSpacing(2)
        filter_popup.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        filter_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
        workflow_label = QLabel("Workflow")
        workflow_label.setObjectName("fieldLabel")
        workflow_row.addWidget(workflow_label)
        workflow_row.addWidget(self.workflow_selector, 1)
        top_left_layout.addLayout(workflow_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        btn_prev = QPushButton('◀ Prev')
        btn_prev.clicked.connect(self.prev_index)
        nav_row.addWidget(btn_prev)

        btn_next = QPushButton('Next ▶')
        btn_next.clicked.connect(self.next_index)
        nav_row.addWidget(btn_next)

        self.complete_btn = QPushButton('Complete')
        self.complete_btn.setToolTip("Save and jump to next unlabeled image")
        self.complete_btn.clicked.connect(self.complete_and_next_unlabeled)
        nav_row.addWidget(self.complete_btn)

        self.skip_btn = QPushButton('Skip')
        self.skip_btn.setToolTip("Jump to next unlabeled image")
        self.skip_btn.clicked.connect(self.skip_to_next_unlabeled)
        nav_row.addWidget(self.skip_btn)

        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self.save_labels)
        nav_row.addWidget(self.save_btn)
        for btn in (btn_prev, btn_next, self.complete_btn, self.skip_btn, self.save_btn):
            btn.setMinimumHeight(28)
        top_left_layout.addLayout(nav_row)

        delete_row = QHBoxLayout()
        delete_row.setSpacing(6)
        self.delete_image_btn = QPushButton("Delete Image")
        self.delete_image_btn.setToolTip("Delete the current image after confirmation")
        self.delete_image_btn.setMinimumHeight(28)
        self.delete_image_btn.clicked.connect(self.delete_current_image)
        delete_row.addWidget(self.delete_image_btn)
        delete_row.addStretch(1)
        top_left_layout.addLayout(delete_row)

        mode_section = QLabel("Mode")
        mode_section.setObjectName("sectionLabel")
        top_left_layout.addWidget(mode_section)

        mode_grid = QGridLayout()
        self.mode_grid = mode_grid
        mode_grid.setHorizontalSpacing(5)
        mode_grid.setVerticalSpacing(5)
        self.panzoom_btn = QPushButton('Pan/Zoom (1)')
        self.bbox_btn = QPushButton('BBox (2)')
        self.segment_btn = QPushButton('Segment (2)')
        self.segment_btn.setToolTip("Segmentation click prompts (left=positive, right=negative)")
        self.seg_edit_btn = QPushButton('Edit Mask (E)')
        self.seg_edit_btn.setToolTip(
            "Manual mask edit mode using brush add/erase."
        )
        self.keypoint_btn = QPushButton('Keypoint (3)')
        self.predict_btn = QPushButton('Predict (4)')
        for btn, mode_name in [(self.panzoom_btn, 'panzoom'),
                               (self.bbox_btn, 'bbox'),
                               (self.keypoint_btn, 'keypoint')]:
            btn.clicked.connect(lambda checked, m=mode_name: self.set_mode(m))
            btn.setMinimumWidth(116)
            btn.setMinimumHeight(28)
        self.segment_btn.clicked.connect(lambda checked: self.set_mode('segment'))
        self.seg_edit_btn.clicked.connect(lambda checked: self.set_mode('segedit'))
        self.predict_btn.clicked.connect(lambda checked: self.set_mode('predict'))
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

        class_row = QHBoxLayout()
        class_row.setSpacing(6)
        self.class_label_widget = QLabel("Class")
        self.class_label_widget.setObjectName("fieldLabel")
        class_row.addWidget(self.class_label_widget)
        class_row.addWidget(self.class_selector, 1)
        self.manage_classes_btn = QPushButton("Classes…")
        self.manage_classes_btn.setToolTip("Manage classes and per-class keypoints")
        self.manage_classes_btn.clicked.connect(self.open_class_manager)
        class_row.addWidget(self.manage_classes_btn)
        top_left_layout.addLayout(class_row)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(6)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progressBadge")
        progress_row.addWidget(self.progress_label)
        progress_row.addStretch(1)
        top_left_layout.addLayout(progress_row)

        # -----------------------------
        # Top-right: video tools
        # -----------------------------
        self.top_right_frame = QFrame(self.view)
        self.top_right_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.top_right_frame)
        top_right_layout = QVBoxLayout(self.top_right_frame)
        top_right_layout.setContentsMargins(12, 11, 12, 14)
        top_right_layout.setSpacing(8)
        top_right_title = QLabel("Video")
        top_right_title.setObjectName("panelTitle")
        top_right_layout.addWidget(top_right_title)
        btn_video = QPushButton("Video Reviewer")
        btn_video.setToolTip("Predict an entire video, then review frames with overlays")
        btn_video.setMinimumHeight(34)
        btn_video.clicked.connect(self.open_video_reviewer)
        top_right_layout.addWidget(btn_video)
        top_right_layout.addSpacing(2)

        # -----------------------------
        # Bottom-left: training tools
        # -----------------------------
        self.bottom_left_frame = QFrame(self.view)
        self.bottom_left_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.bottom_left_frame)
        bottom_left_layout = QVBoxLayout(self.bottom_left_frame)
        bottom_left_layout.setContentsMargins(12, 11, 12, 11)
        bottom_left_layout.setSpacing(8)
        bottom_left_title = QLabel("Dataset & Training")
        bottom_left_title.setObjectName("panelTitle")
        bottom_left_layout.addWidget(bottom_left_title)
        training_row = QHBoxLayout()
        training_row.setSpacing(6)
        self.normalize_btn = QPushButton("Validate Labels")
        self.normalize_btn.setToolTip("Rewrite labels_all files and ensure matching images exist in images_all")
        self.normalize_btn.setMinimumHeight(30)
        self.normalize_btn.clicked.connect(self.normalize_labels_all)
        training_row.addWidget(self.normalize_btn)

        self.export_dataset_btn = QPushButton("Export Dataset")
        self.export_dataset_btn.setToolTip("Split images_all/labels_all into train/val and regenerate dataset.yaml")
        self.export_dataset_btn.setMinimumHeight(30)
        self.export_dataset_btn.clicked.connect(self.export_dataset)
        training_row.addWidget(self.export_dataset_btn)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.setToolTip("Launch a training run for a selected dataset")
        self.train_btn.setMinimumHeight(30)
        self.train_btn.clicked.connect(self.open_train_dialog)
        training_row.addWidget(self.train_btn)

        bottom_left_layout.addLayout(training_row)

        # -----------------------------
        # Bottom-right: model + inference
        # -----------------------------
        self.bottom_right_frame = QFrame(self.view)
        self.bottom_right_frame.setStyleSheet(panel_style)
        apply_panel_shadow(self.bottom_right_frame)
        bottom_right_layout = QVBoxLayout(self.bottom_right_frame)
        bottom_right_layout.setContentsMargins(12, 11, 12, 11)
        bottom_right_layout.setSpacing(8)
        bottom_right_title = QLabel("Model & Inference")
        bottom_right_title.setObjectName("panelTitle")
        bottom_right_layout.addWidget(bottom_right_title)
        inference_row = QHBoxLayout()
        inference_row.setSpacing(6)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setMinimumHeight(30)
        self.load_model_btn.clicked.connect(self.load_model)
        inference_row.addWidget(self.load_model_btn)

        self.template_apply_btn = QPushButton("Apply Tmpl")
        self.template_apply_btn.setToolTip("Apply the saved template for the selected class")
        self.template_apply_btn.setMinimumHeight(30)
        self.template_apply_btn.clicked.connect(self.apply_template_for_current_class)
        inference_row.addWidget(self.template_apply_btn)

        self.template_save_btn = QPushButton("Save Tmpl")
        self.template_save_btn.setToolTip("Capture the current annotation as the class template")
        self.template_save_btn.setMinimumHeight(30)
        self.template_save_btn.clicked.connect(self.save_template_for_current_class)
        inference_row.addWidget(self.template_save_btn)

        self.inference_btn = QPushButton("Inference")
        self.inference_btn.setToolTip("Select a video, run YOLO, and export per-frame metrics to CSV")
        self.inference_btn.setMinimumHeight(30)
        self.inference_btn.clicked.connect(self.run_video_inference)
        inference_row.addWidget(self.inference_btn)
        bottom_right_layout.addLayout(inference_row)

        # -----------------------------
        # Bottom-left overlay: segmentation tools/help
        # -----------------------------
        self.seg_tools_frame = QFrame(self.view)
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

        sam_row = QHBoxLayout()
        sam_row.setSpacing(6)
        self.sam_load_btn = QPushButton("Load SAM")
        self.sam_load_btn.setToolTip("Load a SAM model file for segmentation prompts")
        self.sam_load_btn.setMinimumHeight(28)
        self.sam_load_btn.clicked.connect(self._load_sam_model_interactive)
        sam_row.addWidget(self.sam_load_btn)

        self.sam_run_btn = QPushButton("Run (G)")
        self.sam_run_btn.setToolTip("Run SAM using current positive/negative prompts")
        self.sam_run_btn.setMinimumHeight(28)
        self.sam_run_btn.clicked.connect(self._run_sam_segmentation)
        sam_row.addWidget(self.sam_run_btn)

        self.sam_accept_btn = QPushButton("Accept")
        self.sam_accept_btn.setObjectName("samAcceptButton")
        self.sam_accept_btn.setToolTip("Commit the current SAM mask preview to this class")
        self.sam_accept_btn.setMinimumHeight(28)
        self.sam_accept_btn.clicked.connect(self._accept_segmentation_preview)
        sam_row.addWidget(self.sam_accept_btn)

        self.sam_clear_btn = QPushButton("Reset")
        self.sam_clear_btn.setToolTip("Remove prompt points and the current SAM preview")
        self.sam_clear_btn.setMinimumHeight(28)
        self.sam_clear_btn.clicked.connect(self._clear_seg_prompt_state)
        sam_row.addWidget(self.sam_clear_btn)
        seg_tools_layout.addLayout(sam_row)

        self.sam_helper_label = QLabel("")
        self.sam_helper_label.setWordWrap(True)
        self.sam_helper_label.setObjectName("samHelper")
        seg_tools_layout.addWidget(self.sam_helper_label)
        self._refresh_seg_brush_size_badge()
        self.seg_tools_frame.hide()

        # reflect initial nav filter in the dropdown
        try:
            mapping = {"all": 0, "labeled": 1, "unlabeled": 2}
            self.filter_combo.setCurrentIndex(mapping.get(self.nav_filter, 0))
        except Exception:
            pass
        self.workflow_selector.setCurrentIndex(0 if self._is_pose_workflow() else 1)

        self._layout_hot_corners()

        hud_style = """
            QFrame {
                background-color: rgba(34, 38, 42, 200);
                border: 1px solid rgba(128, 141, 152, 130);
                border-radius: 13px;
            }
            QLabel {
                background: transparent;
                border: none;
                padding: 0px;
                color: #e2e8ee;
            }
            QLabel#hudTitle {
                font-weight: 700;
                font-size: 10pt;
                color: #f5f8fb;
                padding-bottom: 2px;
            }
            QLabel#zoomValue {
                font-weight: 700;
                font-size: 11pt;
                color: #f3f7fb;
            }
        """

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
        self.legend_title.setStyleSheet(
            "background: transparent; border: none; font-weight: 700; font-size: 10pt; color: #f5f8fb;"
        )
        legend_layout.addWidget(self.legend_title)

        # multiline, can wrap, can expand
        self.legend_label = QLabel(
            "Keys:  🔴 Visible   🟡 Occluded   ⚪ Invisible (v=0)\n"
            "L: toggle labels   -/= point size   [/] text size\n"
            "0: mark next invisible   Shift+0: selected → invisible"
        )
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet("background: transparent; border: none; font-size: 10pt; color: #e2e8ee;")
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
        self.zoom_label.setStyleSheet(
            "background: transparent; border: none; font-weight: 700; font-size: 11pt; color: #f3f7fb;"
        )
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
                "vis": int(getattr(it, "visibility", 2))
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
            cid
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
                name = kp_info.get("name", f"kp_{idx+1}")
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
        return pose_annotation_to_line(entry, kp_names=self.kp_names, img_w=self.img_w, img_h=self.img_h)

    def _render_overlay_from_cache(self, out_path: str):
        if self.img_w <= 0 or self.img_h <= 0:
            return
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
                pen = QPen(color); pen.setWidth(2); pen.setCosmetic(True)
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
                    painter.drawRect(QRectF(bbox.get("x", 0.0), bbox.get("y", 0.0),
                                            bbox.get("w", 0.0), bbox.get("h", 0.0)))
                    for kp in entry.get("keypoints", []):
                        vis = int(kp.get("vis", 2))
                        if vis == 0:
                            painter.setBrush(QBrush(Qt.GlobalColor.transparent))
                            painter.setPen(QPen(Qt.GlobalColor.lightGray))
                        elif vis == 1:
                            painter.setBrush(QBrush(color))
                            pen = QPen(color); pen.setStyle(Qt.PenStyle.DashLine); painter.setPen(pen)
                        else:
                            painter.setBrush(QBrush(color))
                            painter.setPen(QPen(color))
                        painter.drawEllipse(QPointF(kp.get("x", 0.0), kp.get("y", 0.0)), 3, 3)
                    painter.setPen(pen)
        finally:
            painter.end()
        try:
            pm.save(out_path)
            print(f"✅ Saved annotated image to {out_path}")
        except Exception as e:
            print(f"⚠️ Failed to save annotated image: {e}")

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
            if not os.path.exists(label_file):
                return idx
        return start_from  # all labeled

    # ---------- Navigation filtering ----------
    def _is_labeled_index(self, idx: int) -> bool:
        base = os.path.splitext(self.images[idx])[0]
        label_file = os.path.join(self.label_dir, f"{base}.txt")
        return os.path.exists(label_file)

    def _filtered_indices(self) -> list[int]:
        if not self.images:
            return []
        if self.nav_filter == 'all':
            return list(range(len(self.images)))
        elif self.nav_filter == 'labeled':
            return [i for i in range(len(self.images)) if self._is_labeled_index(i)]
        else:  # 'unlabeled'
            return [i for i in range(len(self.images)) if not self._is_labeled_index(i)]

    def _set_nav_filter(self, mode: str):
        if mode not in ('all', 'labeled', 'unlabeled'):
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
        self.update_status_bar(f"Browsing: {mode} ({fi.index(self.current_idx)+1}/{len(fi)})")
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
        self.mode = 'segment' if self._is_seg_workflow() else 'bbox'
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
        self.mode = 'segment' if self._is_seg_workflow() else 'bbox'
        self.load_image()
        self._queue_current_idx = self.current_idx

    def complete_and_next_unlabeled(self):
        if self._is_seg_workflow():
            self._cache_active_annotation()
            has_any_mask = any(len(entry.get("segments", [])) >= 3 for entry in self.annotation_cache.values())
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
            self.save_labels()
            base = os.path.splitext(self.images[self.current_idx])[0]
            if not os.path.exists(os.path.join(self.label_dir, f"{base}.txt")):
                return
            next_idx = self._find_next_unlabeled(self.current_idx)
            if next_idx == self.current_idx:
                popup = CongratsPopup(); popup.exec()
                return
            self.current_idx = next_idx
            self._queue_current_idx = self.current_idx
            self.mode = 'segment'
            self.load_image()
            return
        if not self._is_fully_labeled():
            QMessageBox.information(self, "Incomplete",
                                    "Place one bounding box and all keypoints to complete this frame.")
            return
        self.save_labels()
        next_idx = self._find_next_unlabeled(self.current_idx)
        if next_idx == self.current_idx:
            popup = CongratsPopup(); popup.exec()
            return
        self.current_idx = next_idx
        self._queue_current_idx = self.current_idx
        self.mode = 'segment' if self._is_seg_workflow() else 'bbox'
        self.load_image()

    def skip_to_next_unlabeled(self):
        next_idx = self._find_next_unlabeled(self.current_idx)
        if next_idx == self.current_idx:
            popup = CongratsPopup(); popup.exec()
            return
        self.current_idx = next_idx
        self._queue_current_idx = self.current_idx
        self.mode = 'segment' if self._is_seg_workflow() else 'bbox'
        self.load_image()

    def _image_delete_paths(self, image_name: str) -> list[str]:
        file_name = os.path.basename(image_name)
        if not file_name:
            return []
        base = os.path.splitext(file_name)[0]
        label_name = f"{base}.txt"
        paths: list[str] = []
        for directory, target_name in (
            (getattr(self, "active_image_dir", ""), file_name),
            (getattr(self, "image_dir_queue", ""), file_name),
            (getattr(self, "image_dir_all", ""), file_name),
            (getattr(self, "pose_label_dir", ""), label_name),
            (getattr(self, "seg_label_dir", ""), label_name),
            (os.path.join(self.project_root, "annotations"), f"{base}_annotated.png"),
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

    def load_model(self):
        model_file, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model file", "", "Model Files (*.pt *.yaml *.onnx)"
        )
        if not model_file:
            return
        try:
            self.predict_model_path = model_file
            # Re-detect device in case hardware/availability changed
            self._device = _auto_device()
            print(f"🧠 Inference device: {self._device}")
            QMessageBox.information(self, "Model Selected", f"Selected model:\n{os.path.basename(model_file)}")
        except Exception as e:
            QMessageBox.warning(self, "Model Load Error", f"Could not load model:\n{e}")

    def run_video_inference(self):
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV:\n\n  pip install opencv-python")
            return
        if not getattr(self, "predict_model_path", None):
            QMessageBox.information(self, "No Model", "Load a model before running inference.")
            return
        if self._inference_process is not None and self._inference_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Inference Running", "An inference process is already running.")
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

        output_root = os.path.join(self.project_root, "inference outputs")
        try:
            os.makedirs(output_root, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Output Error", f"Could not create output directory:\n{output_root}\n\n{e}")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        workflow = WORKFLOW_SEG if self._is_seg_workflow() else WORKFLOW_POSE
        suffix = "_segmentation.csv" if workflow == WORKFLOW_SEG else ".csv"
        csv_name = f"{base_name}_{timestamp}{suffix}"
        csv_path = os.path.join(output_root, csv_name)
        self._start_inference_process(
            workflow=workflow,
            video_path=video_path,
            csv_path=csv_path,
            batch_size=batch_size,
            total_frames=metadata.total_frames,
            fps=metadata.fps,
        )

    def _start_inference_process(
        self,
        *,
        workflow: str,
        video_path: str,
        csv_path: str,
        batch_size: int,
        total_frames: int,
        fps: float,
    ) -> None:
        output_root = os.path.dirname(csv_path)
        config = {
            "mode": workflow,
            "model_path": self.predict_model_path,
            "video_path": video_path,
            "csv_path": csv_path,
            "classes": self.classes,
            "kp_names": self.kp_names,
            "device": self._device,
            "batch_size": int(batch_size),
            "total_frames": int(total_frames),
            "fps": float(fps),
        }
        config_path = os.path.join(output_root, f".{os.path.splitext(os.path.basename(csv_path))[0]}_config.json")
        try:
            atomic_write_text(config_path, json.dumps(config, indent=2))
        except Exception as e:
            QMessageBox.warning(self, "Output Error", f"Could not write inference config:\n{config_path}\n\n{e}")
            return

        title = "Segmentation Video Inference" if workflow == WORKFLOW_SEG else "Video Inference"
        label = "Running segmentation inference…" if workflow == WORKFLOW_SEG else "Running inference…"
        prog = QProgressDialog(label, "Cancel", 0, 0 if total_frames <= 0 else total_frames, self)
        prog.setWindowTitle(title)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        if total_frames <= 0:
            prog.setRange(0, 0)  # busy indicator for unknown length

        was_busy = getattr(self, "_predict_busy", False)
        self._inference_previous_busy = was_busy
        self._predict_busy = True
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(False)
        if hasattr(self, "inference_btn"):
            self.inference_btn.setEnabled(False)

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "inference_worker", "--config", config_path])
        process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
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
        self._inference_stderr += bytes(process.readAllStandardError()).decode("utf-8", errors="replace")

    def _handle_inference_event_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = json.loads(line)
        except Exception:
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
                progress.setLabelText(str(event.get("message") or f"Inferencing frame {processed}"))
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
                progress.setLabelText("Loading model in inference process…")

    def _cancel_inference_process(self) -> None:
        process = self._inference_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        self._inference_cancel_requested = True
        progress = self._inference_progress
        if progress is not None:
            progress.setLabelText("Canceling inference process…")
        process.terminate()
        QTimer.singleShot(5000, self._kill_inference_process_if_running)

    def _kill_inference_process_if_running(self) -> None:
        process = self._inference_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_inference_process_error(self, _error) -> None:
        process = self._inference_process
        if process is not None:
            self._inference_stderr += process.errorString() + "\n"

    def _finish_inference_process(self, exit_code: int, exit_status) -> None:
        if self._inference_stdout_buffer.strip():
            self._handle_inference_event_line(self._inference_stdout_buffer.strip())
            self._inference_stdout_buffer = ""

        progress = self._inference_progress
        if progress is not None:
            progress.close()

        if hasattr(self, "_inference_previous_busy"):
            self._predict_busy = self._inference_previous_busy
        else:
            self._predict_busy = False
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(True)
        if hasattr(self, "inference_btn"):
            self.inference_btn.setEnabled(True)

        event = self._inference_result_event
        csv_path = self._inference_csv_path or ""
        config_path = self._inference_config_path
        mode = self._inference_mode
        cancel_requested = self._inference_cancel_requested
        stderr_text = self._inference_stderr.strip()

        if config_path:
            try:
                if os.path.exists(config_path):
                    os.remove(config_path)
            except Exception:
                pass

        self._inference_process = None
        self._inference_progress = None
        self._inference_config_path = None
        self._inference_csv_path = None
        self._inference_result_event = None
        self._inference_stdout_buffer = ""
        self._inference_stderr = ""
        self._inference_cancel_requested = False

        if cancel_requested and event is None:
            QMessageBox.information(
                self,
                "Inference Canceled",
                f"Inference process was canceled.\n\nPartial CSV may remain at:\n{csv_path}",
            )
            return

        if event is None:
            detail = stderr_text or f"Process exited with code {exit_code}."
            QMessageBox.critical(self, "Inference Error", f"Inference process failed:\n{detail}")
            return

        rows_written = int(event.get("rows_written") or 0)
        had_error = bool(event.get("had_error"))
        canceled = bool(event.get("canceled"))
        error_message = str(event.get("error_message") or stderr_text or "Unknown inference error")
        csv_path = str(event.get("csv_path") or csv_path)
        label = "segmentation row(s)" if mode == WORKFLOW_SEG else "row(s)"

        if had_error or exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0:
            if rows_written == 0:
                try:
                    if csv_path and os.path.exists(csv_path):
                        os.remove(csv_path)
                except Exception:
                    pass
                QMessageBox.critical(self, "Inference Error", f"An error occurred during inference:\n{error_message}")
            else:
                QMessageBox.critical(
                    self,
                    "Inference Error",
                    "An error occurred during inference.\n\n"
                    f"{error_message}\n\n"
                    f"Partial CSV saved ({rows_written} {label}):\n{csv_path}",
                )
            return

        if canceled and rows_written == 0:
            try:
                if csv_path and os.path.exists(csv_path):
                    os.remove(csv_path)
            except Exception:
                pass
            QMessageBox.information(self, "Inference Canceled", "Inference canceled before any results were generated.")
            return

        if rows_written == 0:
            QMessageBox.information(self, "Inference Complete", "Inference completed without writing result rows.")
            return

        message = f"Saved {rows_written} {label} to:\n{csv_path}"
        if canceled:
            message = "Inference canceled early.\n" + message
        QMessageBox.information(self, "Inference Complete", message)

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
        if self._is_seg_workflow() and mode in {"bbox", "keypoint"}:
            self.update_status_bar("Segmentation workflow uses Segment Prompt (2), Edit Mask (E), and Predict (4) modes.")
            return
        if self._is_pose_workflow() and mode in {"segment", "segedit"}:
            self.update_status_bar("Segment Prompt/Edit Mask modes are only available in Segmentation workflow.")
            return

        if mode == 'predict':
            if not self.predict_model_path:
                QMessageBox.information(self, "No Model", "Please click 'Load Model' first.")
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

        if hasattr(self.view, '_remove_crosshairs'):
            self.view._remove_crosshairs()

        if self.mode == 'panzoom':
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
        elif self.mode == 'bbox':
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, 'draw_crosshairs_at'):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == 'keypoint':
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, 'draw_crosshairs_at'):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == 'segment':
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self.view, 'draw_crosshairs_at'):
                self.view.draw_crosshairs_at(QCursor.pos())
        elif self.mode == 'segedit':
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

        if self._prediction_process is not None and self._prediction_process.state() != QProcess.ProcessState.NotRunning:
            self.update_status_bar("Prediction already running...")
            return

        config = {
            "model_path": self.predict_model_path,
            "image_path": img_path,
            "workflow": self.active_workflow,
            "device": self._device,
        }
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            config_path = os.path.join(os.path.dirname(self._log_path), ".single_image_predict_config.json")
            atomic_write_text(config_path, json.dumps(config, indent=2))
        except Exception as e:
            QMessageBox.warning(self, "Prediction Error", f"Could not write prediction config:\n{e}")
            return

        self._predict_busy = True
        if hasattr(self, 'predict_btn'):
            self.predict_btn.setEnabled(False)
        self.update_status_bar("Running prediction...")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "predict_worker", "--config", config_path])
        process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        process.readyReadStandardOutput.connect(self._read_prediction_process_stdout)
        process.readyReadStandardError.connect(self._read_prediction_process_stderr)
        process.finished.connect(self._finish_prediction_process)
        process.errorOccurred.connect(self._handle_prediction_process_error)

        self._prediction_process = process
        self._prediction_stdout_buffer = ""
        self._prediction_stderr = ""
        self._prediction_result_event = None
        self._prediction_config_path = config_path
        self._prediction_image_path = img_path
        self._prediction_cancel_requested = False
        process.start()
        if not process.waitForStarted(1000):
            self._prediction_stderr = process.errorString()
            self._finish_prediction_process(1, QProcess.ExitStatus.CrashExit)
            return

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
        self._prediction_stderr += bytes(process.readAllStandardError()).decode("utf-8", errors="replace")

    def _handle_prediction_event_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = json.loads(line)
        except Exception:
            self._prediction_stderr += line + "\n"
            return
        if event.get("event") == "result":
            self._prediction_result_event = event
        elif event.get("event") == "error":
            self._prediction_result_event = {
                "event": "result",
                "canceled": False,
                "had_error": True,
                "error_message": str(event.get("error_message") or "Prediction worker error"),
                "prediction": None,
            }
        elif event.get("event") == "started":
            self.update_status_bar("Prediction worker started...")

    def _cancel_prediction_process(self) -> None:
        process = self._prediction_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._prediction_cancel_requested = True
            process.terminate()
            QTimer.singleShot(3000, self._kill_prediction_process_if_running)

    def _kill_prediction_process_if_running(self) -> None:
        process = self._prediction_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_prediction_process_error(self, _error) -> None:
        process = self._prediction_process
        if process is not None:
            self._prediction_stderr += process.errorString() + "\n"

    def _finish_prediction_process(self, exit_code: int, exit_status) -> None:
        if self._prediction_stdout_buffer.strip():
            self._handle_prediction_event_line(self._prediction_stdout_buffer.strip())
            self._prediction_stdout_buffer = ""

        config_path = self._prediction_config_path
        if config_path:
            try:
                if os.path.exists(config_path):
                    os.remove(config_path)
            except Exception:
                pass

        event = self._prediction_result_event
        stderr_text = self._prediction_stderr.strip()
        cancel_requested = self._prediction_cancel_requested
        self._prediction_process = None
        self._prediction_config_path = None
        self._prediction_image_path = None
        self._prediction_result_event = None
        self._prediction_stdout_buffer = ""
        self._prediction_stderr = ""
        self._prediction_cancel_requested = False
        self._predict_busy = False
        if hasattr(self, 'predict_btn'):
            self.predict_btn.setEnabled(True)

        if cancel_requested and event is None:
            self.update_status_bar("Prediction canceled.")
            return
        if event is None:
            self._on_predict_error(stderr_text or f"Process exited with code {exit_code}.")
            return
        if bool(event.get("canceled")):
            self.update_status_bar("Prediction canceled.")
            return
        if bool(event.get("had_error")) or exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0:
            self._on_predict_error(str(event.get("error_message") or stderr_text or "Unknown prediction error"))
            return
        prediction = event.get("prediction")
        if not isinstance(prediction, dict):
            self._on_predict_error("Prediction worker returned no prediction payload.")
            return
        self._apply_prediction_payload(prediction)

    def _apply_prediction_payload(self, prediction: dict):
        try:
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
                        status_msg += f" Skipped {missing_mask_count} detection(s) without usable masks."
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
            import traceback, datetime
            tb = traceback.format_exc()
            try:
                with open(self._log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f"\n[{datetime.datetime.now().isoformat()}] Apply-prediction payload error on {self.images[self.current_idx] if self.images else 'N/A'}\n{tb}\n")
            except Exception:
                pass
            self._on_predict_error(str(e) or tb)

    def _on_predict_error(self, error_text: str):
        # Reset busy state and re-enable button
        self._predict_busy = False
        if hasattr(self, 'predict_btn'):
            self.predict_btn.setEnabled(True)
        # Surface the error to the user and point to the log
        try:
            QMessageBox.critical(self, "Prediction Error",
                                 f"An error occurred during prediction.\n\nDetails:\n{error_text[:1000]}\n\nA full traceback was written to:\n{self._log_path}")
        except Exception:
            pass
        self.update_status_bar("Prediction failed. See log for details.")

    def _reset_zoom(self):
        self.view.resetTransform()
        self.update_zoom_label()

    def mark_current_kp_invisible(self):
        """Mark the next required keypoint as invisible (v=0) and advance."""
        if self.mode != 'keypoint':
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
            '1': lambda: self.set_mode('panzoom'),
            '2': lambda: self.set_mode('segment' if self._is_seg_workflow() else 'bbox'),
            'E': lambda: self.set_mode('segedit'),
            '3': lambda: self.set_mode('keypoint'),
            '4': lambda: self.set_mode('predict'),
            'S': self.save_labels,
            'Z': self.undo,
            'V': self.toggle_selected_visibility,
            'R': self._reset_zoom,  # <-- refresh zoom label too
            'G': self._run_sam_segmentation,
            'X': self._clear_seg_prompt_state,
            'A': self.apply_template_for_current_class,
            'C': lambda: self._cycle_class(+1),
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
        QShortcut(QKeySequence('='), self).activated.connect(lambda: self._bump_kp_size(+1))
        QShortcut(QKeySequence('-'), self).activated.connect(lambda: self._bump_kp_size(-1))
        QShortcut(QKeySequence(']'), self).activated.connect(lambda: self._bump_kp_font(+1))
        QShortcut(QKeySequence('['), self).activated.connect(lambda: self._bump_kp_font(-1))
        QShortcut(QKeySequence('L'), self).activated.connect(self._toggle_kp_labels)
        QShortcut(QKeySequence(','), self).activated.connect(lambda: self._adjust_seg_brush_radius(-2))
        QShortcut(QKeySequence('.'), self).activated.connect(lambda: self._adjust_seg_brush_radius(+2))

        # Invisible keypoints
        QShortcut(QKeySequence('0'), self).activated.connect(self.mark_current_kp_invisible)
        QShortcut(QKeySequence('Shift+0'), self).activated.connect(self.set_selected_invisible)

        # Workflow jumps
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.complete_and_next_unlabeled)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self.complete_and_next_unlabeled)
        QShortcut(QKeySequence('K'), self).activated.connect(self.skip_to_next_unlabeled)
        QShortcut(QKeySequence('Meta+Return'), self).activated.connect(self.complete_and_next_unlabeled)  # optional: macOS
        QShortcut(QKeySequence("Shift+Return"), self).activated.connect(self._accept_segmentation_preview)

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
            if self.mode == 'keypoint' and selected_kp:
                step = 0.5
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    step = 3.0
                dx = dy = 0
                if event.key() == Qt.Key.Key_Left:  dx = -step
                elif event.key() == Qt.Key.Key_Right: dx = step
                elif event.key() == Qt.Key.Key_Up:   dy = -step
                elif event.key() == Qt.Key.Key_Down: dy = step
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
        any_visible = any(isinstance(it, KeypointItem) and it.text_item.isVisible() for it in self.scene.items())
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

    def _apply_class_manager_results(self, classes: list[str], keypoints: list[str], kp_map: dict[str, list[str]]) -> bool:
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

        self._write_list_file(self.class_file, classes_clean)
        self._write_list_file(self.keypoint_file, canonical)
        self.classes = classes_clean
        self.kp_names = canonical
        self.class_keypoints = normalized_map
        self._save_class_keypoints()
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
            if classes_clean[:len(existing_classes)] != existing_classes:
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
            self.update_status_bar("Templates are only available in Pose workflow.")
            return
        if not self._cache_active_annotation():
            QMessageBox.warning(self, "Template error", "Complete the current class annotation before saving a template.")
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
            data["keypoints"].append({
                "name": kp.get("name", ""),
                "idx": int(kp.get("idx", len(data["keypoints"]))),
                "canon_idx": int(kp.get("canon_idx", -1)),
                "x": 0.0 if vis == 0 else kp.get("x", 0.0) / max(1.0, float(self.img_w)),
                "y": 0.0 if vis == 0 else kp.get("y", 0.0) / max(1.0, float(self.img_h)),
                "vis": vis,
            })
        path = self._template_path_for_class(self.classes[cid])
        try:
            atomic_write_text(path, json.dumps(data, indent=2))
            QMessageBox.information(self, "Template saved", f"Template saved to {path}.")
        except Exception as e:
            QMessageBox.warning(self, "Template error", f"Failed to save template:\n{e}")

    def apply_template_for_current_class(self):
        if self._is_seg_workflow():
            self.update_status_bar("Templates are only available in Pose workflow.")
            return
        if not self.images:
            QMessageBox.warning(self, "Template error", "Load an image before applying templates.")
            return
        class_name = self.class_selector.currentText()
        path = self._template_path_for_class(class_name)
        if not os.path.exists(path):
            QMessageBox.warning(self, "Template missing", f"No template found for {class_name}.\nSave one first.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Template error", f"Failed to load template:\n{e}")
            return

        bbox = data.get("bbox", {})
        xc = bbox.get("xc", 0.5); yc = bbox.get("yc", 0.5)
        w = bbox.get("w", 1.0); h = bbox.get("h", 1.0)
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
        if self.mode == 'keypoint':
            names = self._active_kp_names()
            if self.current_kp_idx < len(names):
                return f"Next: {names[self.current_kp_idx]}  ({self.current_kp_idx}/{len(names)})"
            return "All keypoints placed"
        return ""

    # ---------- Image load / navigation ----------

    def load_image(self):
        if hasattr(self.view, '_remove_crosshairs'):
            self.view._remove_crosshairs()
        if hasattr(self.view, "_reset_seg_brush_cursor"):
            self.view._reset_seg_brush_cursor()
        self._clear_seg_edit_handles()

        self.scene.clear()
        self._item_refs.clear()
        if not self.images:
            return

        img_path = os.path.join(self.active_image_dir, self.images[self.current_idx])
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
        label_file = os.path.join(self.label_dir, f"{base}.txt")

        if self._is_seg_workflow():
            self.annotation_cache = {}
            if os.path.exists(label_file):
                self.annotation_cache = self._load_seg_annotations_from_file(label_file)
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
            scene_center = self.scene.sceneRect().center()
            self.view.centerOn(scene_center)
            return

        self.annotation_cache = {}
        if os.path.exists(label_file):
            self.annotation_cache = self._load_annotations_from_file(label_file)
        for cid in range(len(self.classes)):
            if cid in self.annotation_cache:
                self._restore_annotation_for_class(cid)
        self._sync_active_class_state()
        self._update_item_editability()

        self._update_status()
        if hasattr(self.view, "refresh_seg_brush_cursor"):
            self.view.refresh_seg_brush_cursor()
        scene_center = self.scene.sceneRect().center()
        self.view.centerOn(scene_center)

    def add_bbox(self, rect: QRectF):
        if not self.classes:
            QMessageBox.warning(self, "No classes", "Define at least one class before adding boxes.")
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
        if self.mode == 'keypoint' and self.kps:
            kp = self.kps.pop()
            for it in list(self.scene.items()):
                if isinstance(it, KeypointItem) and it.kp is kp and it.kp.class_id == cid:
                    self._safe_remove_scene_item(it)
                    self._untrack_scene_item(it)
                    break
            self.current_kp_idx = max(0, self.current_kp_idx - 1)
            self._update_status()
        elif self.mode == 'bbox' and self.bboxes:
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
        self._cache_active_annotation()
        if not self.classes:
            return False
        if self._is_seg_workflow():
            return any(len(entry.get("segments", [])) >= 3 for entry in self.annotation_cache.values())
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
            'panzoom': self.panzoom_btn,
            'bbox': self.bbox_btn,
            'segment': self.segment_btn,
            'segedit': self.seg_edit_btn,
            'keypoint': self.keypoint_btn,
            'predict': self.predict_btn,
        }
        for mode_name, button in buttons.items():
            if self.mode == mode_name:
                button.setStyleSheet("background-color: #505357; font-weight: bold;")
            else:
                button.setStyleSheet("")

        # Show filtered index / total in status bar
        fi = self._filtered_indices()
        if fi and self.current_idx in fi:
            idx_in_view = fi.index(self.current_idx) + 1
            self.status.showMessage(f"Viewing {self.nav_filter}: {idx_in_view}/{len(fi)}", 2000)

        if self.mode == 'keypoint':
            self.legend_frame.show()
            self.zoom_frame.hide()
            self._layout_overlays()
            self.update_status_bar(self._kp_text())
        elif self.mode == 'panzoom':
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
        vw = self.view.viewport().width()
        vh = self.view.viewport().height()
        margin = 10
        gap = 8

        if hasattr(self, "top_left_frame"):
            self.top_left_frame.adjustSize()
            self.top_left_frame.move(margin, margin)

        if hasattr(self, "top_right_frame"):
            self.top_right_frame.adjustSize()
            tr_w = self.top_right_frame.sizeHint().width()
            self.top_right_frame.move(max(margin, vw - tr_w - margin), margin)

        if hasattr(self, "bottom_left_frame"):
            self.bottom_left_frame.adjustSize()
            bl_h = self.bottom_left_frame.sizeHint().height()
            self.bottom_left_frame.move(margin, max(margin, vh - bl_h - margin))

        if hasattr(self, "bottom_right_frame"):
            self.bottom_right_frame.adjustSize()
            br_w = self.bottom_right_frame.sizeHint().width()
            br_h = self.bottom_right_frame.sizeHint().height()
            self.bottom_right_frame.move(max(margin, vw - br_w - margin), max(margin, vh - br_h - margin))

    def _layout_overlays(self):
        """Dynamically position and size legend / zoom overlays."""
        vw = self.view.viewport().width()
        vh = self.view.viewport().height()

        # Keep overlays above the bottom-left tool panel.
        bottom_offset = 10
        if hasattr(self, "bottom_left_frame") and self.bottom_left_frame.isVisible():
            bottom_offset += self.bottom_left_frame.sizeHint().height() + 8

        x = 10
        cursor_y = vh - bottom_offset

        # Segmentation tools box (seg workflow).
        if hasattr(self, "seg_tools_frame") and self.seg_tools_frame.isVisible():
            self.seg_tools_frame.adjustSize()
            sh = self.seg_tools_frame.sizeHint().height()
            self.seg_tools_frame.move(x, cursor_y - sh)
            cursor_y -= sh + 8

        # Keypoint legend (pose workflow / keypoint mode).
        if hasattr(self, "legend_frame") and self.legend_frame.isVisible():
            fm = self.legend_label.fontMetrics()
            ch = fm.horizontalAdvance('M')  # approx width of one character
            preferred = int(ch * 30 + 24)
            w = max(250, min(preferred, int(vw * 0.36), 400))
            self.legend_frame.setFixedWidth(w)
            lh = self.legend_frame.sizeHint().height()
            self.legend_frame.move(x, cursor_y - lh)
            cursor_y -= lh + 8

        # Keep zoom HUD stacked above whichever lower-left overlays are visible.
        if hasattr(self, "zoom_frame") and self.zoom_frame.isVisible():
            zh = self.zoom_frame.sizeHint().height()
            self.zoom_frame.move(x, cursor_y - zh)

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
    def _collect_keypoints_by_name(self, class_id: Optional[int] = None) -> dict[str, tuple[Keypoint, int]]:
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

    def save_labels(self):
        if not self.images:
            return

        if self._is_seg_workflow() and self.seg_preview_points:
            QMessageBox.information(
                self,
                "Pending preview",
                "Accept the current SAM preview mask before saving.",
            )
            return

        if self._is_pose_workflow() and not self._cache_active_annotation():
            QMessageBox.warning(self, "Save Error", "Place one bounding box and all keypoints for the selected class before saving.")
            return
        if self._is_seg_workflow():
            self._cache_active_annotation()

        base = os.path.splitext(self.images[self.current_idx])[0]

        project_root = self.project_root
        images_all_dir = os.path.join(project_root, "images_all")
        labels_all_dir = self.label_dir
        annotations_dir = os.path.join(project_root, "annotations")
        os.makedirs(images_all_dir, exist_ok=True)
        os.makedirs(labels_all_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        label_out_path = os.path.join(labels_all_dir, f"{base}.txt")
        annotated_out_path = os.path.join(annotations_dir, f"{base}_annotated.png")
        image_out_path = os.path.join(images_all_dir, self.images[self.current_idx])

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
            return

        try:
            atomic_write_text(label_out_path, "\n".join(lines) + "\n")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not write label file:\n{label_out_path}\n\n{e}")
            return
        print(f"✅ Saved label to {label_out_path}")
        self._schema_locked = True

        self._render_overlay_from_cache(annotated_out_path)

        file_name = self.images[self.current_idx]
        src_candidates: list[str] = []
        if self.current_image_path:
            src_candidates.append(self.current_image_path)
        if os.path.isabs(file_name):
            src_candidates.append(file_name)
        src_candidates.extend([
            os.path.join(self.active_image_dir, file_name),
            os.path.join(self.image_dir_queue, file_name),
            os.path.join(self.image_dir_all, file_name),
        ])

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

        copied_ok = False
        if src_path:
            try:
                if os.path.abspath(src_path) != os.path.abspath(image_out_path):
                    shutil.copy2(src_path, image_out_path)
                    print(f"✅ Copied original image to {image_out_path}")
                else:
                    print(f"ℹ️ Image already stored at {image_out_path}")
                copied_ok = True
            except Exception as e:
                print(f"⚠️ Warning: Failed to copy image {src_path}: {e}")

        if not copied_ok and not os.path.exists(image_out_path):
            tried = "\n".join(sorted(seen))
            msg = (
                f"Could not locate source image for '{file_name}'.\n\n"
                f"Tried:\n{tried}"
            )
            print(f"⚠️ Warning: {msg}")
            try:
                QMessageBox.warning(self, "Image copy warning", msg)
            except Exception:
                pass

        # Update the labeled frame counter immediately after saving.
        self._update_progress_label()

    # ---------- Video ----------
    def export_dataset(self):
        """Split images_all/labels_all into train/val sets and regenerate dataset.yaml."""
        seg_mode = self._is_seg_workflow()
        project_root = self.project_root
        images_all_dir = self.image_dir_all
        labels_all_dir = self.label_dir

        if not os.path.isdir(images_all_dir):
            QMessageBox.information(self, "No images_all directory",
                                    f"Expected {images_all_dir} to exist.")
            return
        if not os.path.isdir(labels_all_dir):
            QMessageBox.information(self, "No labels_all directory",
                                    f"Expected {labels_all_dir} to exist.")
            return

        images = list_image_files(images_all_dir)
        if not images:
            QMessageBox.information(self, "Nothing to export",
                                    "images_all does not contain any images.")
            return

        ratio, ok = QInputDialog.getDouble(
            self,
            "Train/Val Split",
            "Train split ratio (0.1 – 0.95):",
            0.8,
            0.1,
            0.95,
            2
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

        pose_mode = False
        if not seg_mode:
            dataset_choice, ok_choice = QInputDialog.getItem(
                self,
                "Dataset Type",
                "Choose dataset format:",
                ["Pose (keypoints)", "Detection (bbox only)"],
                0,
                False
            )
            if not ok_choice:
                return
            pose_mode = dataset_choice.startswith("Pose")

        if seg_mode:
            dataset_mode = DATASET_SEGMENT
        else:
            dataset_mode = DATASET_POSE if pose_mode else DATASET_DETECT

        paths = dataset_export_paths(project_root, dataset_mode)
        os.makedirs(paths.base_dir, exist_ok=True)

        if dataset_dirs_have_files(paths):
            confirm = QMessageBox.question(
                self,
                "Overwrite dataset?",
                "Existing train/val folders contain files. Overwrite them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            remove_dataset_split_dirs(paths)

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

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export_result = export_dataset_files(
                images_all_dir=images_all_dir,
                labels_all_dir=labels_all_dir,
                paths=paths,
                train_images=train_images,
                val_images=val_images,
                mode=dataset_mode,
                progress_callback=_progress,
                cancel_requested=prog.wasCanceled,
            )
            export_result.split_seed = split_seed
        finally:
            QApplication.restoreOverrideCursor()
            prog.close()

        if export_result.canceled:
            QMessageBox.information(self, "Export canceled",
                                    "Dataset export was canceled. Partially copied files may remain.")
            return

        try:
            export_result.dataset_yaml_path = write_dataset_yaml_for_mode(
                paths.base_dir,
                dataset_mode,
                self.classes,
                self.kp_names,
            )
        except Exception as e:
            QMessageBox.warning(self, "dataset.yaml error",
                                f"Failed to create dataset.yaml:\n{e}")
            return

        QMessageBox.information(self, "Dataset exported", format_dataset_export_summary(export_result))
        self.update_status_bar("Dataset export complete.")

    def normalize_labels_all(self):
        labels_dir = self.label_dir
        images_all_dir = self.image_dir_all
        images_to_label_dir = self.image_dir_queue

        label_files = list_label_files(labels_dir)
        if not label_files:
            folder_name = os.path.basename(labels_dir.rstrip(os.sep)) or labels_dir
            QMessageBox.information(self, "No labels", f"{folder_name} does not contain any .txt files.")
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
            QMessageBox.information(self, "Normalization canceled",
                                    "Operation canceled. Some files may have been processed already.")
            return

        QMessageBox.information(self, "Normalization complete", format_label_normalization_summary(result))
        status = "Segmentation label normalization complete." if seg_mode else "Label normalization complete."
        self.update_status_bar(status)

    def open_train_dialog(self):
        if self._is_seg_workflow():
            default_dataset = os.path.join(self.project_root, "datasets", "segment")
            if not os.path.isdir(default_dataset):
                default_dataset = os.path.join(self.project_root, "datasets")
            dlg = TrainDialog(self, default_dataset=default_dataset, default_task="segment")
        else:
            default_dataset = resolve_default_training_dataset_path(self.project_root)
            dlg = TrainDialog(self, default_dataset=default_dataset, default_task=None)
        dlg.exec()

    def open_video_reviewer(self):
        if _cv2 is None:
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV:\n\n  pip install opencv-python")
            return
        # It’s okay if no model is loaded yet; dialog will warn before predicting
        dlg = VideoReviewDialog(
            self,
            self._device,
            self.kp_names,
            self.classes,
            class_keypoints=self.class_keypoints,
            workflow=self.active_workflow,
        )
        dlg.exec()

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
        self.spin_batch.setRange(-1, 256)
        self.spin_batch.setSpecialValueText("Auto")
        self.spin_batch.setValue(8)  # default batch size
        self.spin_batch.setToolTip("Auto uses a safe chunk size (8 on CUDA/MPS, 1 on CPU).")
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
        self.btn_send_low.setToolTip("Export N lowest-confidence predicted frames to the labeler")
        self.btn_send_low.setEnabled(False)
        self.btn_send_low.clicked.connect(self._export_low_confidence_frames)
        buttons.addButton(self.btn_send_low, QDialogButtonBox.ButtonRole.ActionRole)

        self.btn_send_high = QPushButton("Send High…")
        self.btn_send_high.setToolTip("Export N highest-confidence predicted frames to the labeler")
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
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
            return

        cap = _cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            QMessageBox.warning(self, "Video Error", "Failed to open video.")
            return

        self.cap = cap
        self.path = path
        self.base = os.path.splitext(os.path.basename(path))[0]
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
        requested_batch = int(self.spin_batch.value()) if hasattr(self, "spin_batch") else 1
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
        config_dir = os.path.dirname(parent_log_path) if parent_log_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
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
        process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        process.readyReadStandardOutput.connect(self._read_review_prediction_stdout)
        process.readyReadStandardError.connect(self._read_review_prediction_stderr)
        process.finished.connect(self._finish_review_prediction_process)
        process.errorOccurred.connect(self._handle_review_prediction_error)

        self._review_process = process
        self._review_progress = prog
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_result_event = None
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
            event = json.loads(line)
        except Exception:
            self._review_stderr += line + "\n"
            return

        event_type = event.get("event")
        if event_type == "started":
            progress = self._review_progress
            if progress is not None:
                progress.setLabelText("Loading model in video prediction process…")
        elif event_type == "progress":
            progress = self._review_progress
            if progress is not None:
                processed = int(event.get("processed") or 0)
                total = int(event.get("total") or progress.maximum())
                progress.setMaximum(max(1, total))
                progress.setValue(min(processed, max(1, total)))
                progress.setLabelText(str(event.get("message") or f"Predicting {processed}/{total}"))
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
        process.terminate()
        QTimer.singleShot(5000, self._kill_review_prediction_if_running)

    def _kill_review_prediction_if_running(self):
        process = self._review_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _handle_review_prediction_error(self, _error):
        process = self._review_process
        if process is not None:
            self._review_stderr += process.errorString() + "\n"

    def _finish_review_prediction_process(self, exit_code: int, exit_status):
        if self._review_stdout_buffer.strip():
            self._handle_review_prediction_event_line(self._review_stdout_buffer.strip())
            self._review_stdout_buffer = ""

        progress = self._review_progress
        if progress is not None:
            progress.close()

        config_path = self._review_config_path
        if config_path:
            try:
                if os.path.exists(config_path):
                    os.remove(config_path)
            except Exception:
                pass

        event = self._review_result_event
        stderr_text = self._review_stderr.strip()
        cancel_requested = self._review_cancel_requested
        run_meta = self._review_run_meta or {}

        self._review_process = None
        self._review_progress = None
        self._review_config_path = None
        self._review_result_event = None
        self._review_stdout_buffer = ""
        self._review_stderr = ""
        self._review_cancel_requested = False
        self._review_run_meta = None
        self.btn_predict.setEnabled(self.cap is not None)

        if cancel_requested and event is None:
            QMessageBox.information(self, "Prediction canceled", "Video prediction was canceled.")
            return

        if event is None:
            detail = stderr_text or f"Process exited with code {exit_code}."
            QMessageBox.critical(self, "Prediction Error", f"Video prediction failed:\n{detail}")
            return

        raw_preds = event.get("preds") or {}
        self.preds = {}
        if isinstance(raw_preds, dict):
            for key, value in raw_preds.items():
                try:
                    self.preds[int(key)] = value if isinstance(value, dict) else {"ok": False}
                except Exception:
                    continue

        had_error = bool(event.get("had_error")) or exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0
        canceled = bool(event.get("canceled")) or cancel_requested
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
            base_name = f"{self.base}_f{self.cur:06d}.png"
            out_path = os.path.join(dest_dir, base_name)

            if _cv2 is None:
                QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
                return

            ok = _cv2.imwrite(out_path, self._last_frame_bgr)
            if not ok:
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
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
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
                base_name = f"{self.base}_f{fi:06d}.png"
                dest_path = os.path.join(dest_dir, base_name)
                suffix = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{self.base}_f{fi:06d}_{suffix}.png")
                    suffix += 1
                if _cv2.imwrite(dest_path, frame):
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
            # Matches: {base}_f000123.png (optionally with suffixes, any common image ext)
            pat = re.compile(rf"^{re.escape(self.base)}_f(\d{{6}})(?:_.*)?\.(?:png|jpg|jpeg|bmp|webp)$", re.IGNORECASE)
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

        candidates = [(fi, float(p.get("conf", 0.0))) for fi, p in self.preds.items() if p.get("ok")]
        if not candidates:
            QMessageBox.information(self, "No predictions", "No successful predictions available to export.")
            return

        if order_key == "low":
            candidates.sort(key=lambda t: t[1])
            order_label = "lowest"
            dialog_title = "Export Lowest Confidence"
        else:
            candidates.sort(key=lambda t: (-t[1], t[0]))
            order_label = "highest"
            dialog_title = "Export Highest Confidence"
        conf_map = {fi: conf for fi, conf in candidates}

        already = self._existing_export_indices()
        pending = [fi for fi, _ in candidates if fi not in already]
        if not pending:
            QMessageBox.information(self, "Nothing to export", f"All {order_label}-confidence frames are already exported.")
            return

        max_n = len(pending)
        default_n = min(25, max_n)
        n, ok = QInputDialog.getInt(
            self,
            dialog_title,
            "How many frames should I send to the labeler?",
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
            QMessageBox.warning(self, "OpenCV missing", "Install OpenCV: pip install opencv-python")
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
        saved_confs: list[float] = []
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
                base_name = f"{self.base}_f{fi:06d}.png"
                dest_path = os.path.join(dest_dir, base_name)
                suffix = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{self.base}_f{fi:06d}_{suffix}.png")
                    suffix += 1
                if _cv2.imwrite(dest_path, frame):
                    saved += 1
                    saved_confs.append(conf_map.get(fi, 0.0))
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
            if saved_confs:
                lo = min(saved_confs)
                hi = max(saved_confs)
                msg += f"\nConfidence range of exported set: {lo:.2f}–{hi:.2f}"
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
            if i < len(class_kp_names):
                name = class_kp_names[i]
            elif i < len(self.kp_names):
                name = self.kp_names[i]
            else:
                name = f"kp{i}"
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
            self._cancel_review_prediction_process()
            QMessageBox.information(self, "Prediction running", "Canceling video prediction. Close the reviewer after it stops.")
            return
        # cleanup
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().reject()

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

    def __init__(self, parent, default_dataset: str, default_task: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.resize(1100, 720)
        self.setMinimumSize(760, 520)

        self.default_dataset = default_dataset
        self.default_task = (default_task or "").strip().lower() or None
        self.app_base_dir = os.path.abspath(getattr(parent, "app_base_dir", os.path.dirname(os.path.abspath(__file__))))
        self.project_root = os.path.abspath(getattr(parent, "project_root", self.app_base_dir))
        self.project_runs_dir = os.path.join(self.project_root, "runs")
        project_distillations_root = os.path.join(
            self.project_root,
            "dino_distillation",
            "DINOv3_Distillation_YOLO-pose",
            "dino_distillations",
        )
        fallback_distillations_root = os.path.join(
            self.app_base_dir,
            "dino_distillation",
            "DINOv3_Distillation_YOLO-pose",
            "dino_distillations",
        )
        self.distillations_root = (
            project_distillations_root if os.path.isdir(project_distillations_root) else fallback_distillations_root
        )
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
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "Standard YOLO backbone",
            "DINO distillation export",
            "Continue from YOLO checkpoint",
            "Resume YOLO run (exact)",
        ])
        self.source_combo.currentIndexChanged.connect(self._update_source_controls)
        form.addRow("Backbone source:", self.source_combo)

        # Model choice
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.MODEL_OPTIONS.keys())
        self.model_row = QWidget()
        model_layout = QHBoxLayout(self.model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_combo)
        form.addRow("YOLO model:", self.model_row)

        # DINO export selection
        self.dino_row = QWidget()
        dino_layout = QVBoxLayout(self.dino_row)
        dino_layout.setContentsMargins(0, 0, 0, 0)
        dino_top = QHBoxLayout()
        self.dino_combo = QComboBox()
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
        form.addRow("Distilled export:", self.dino_row)
        self.dino_row.hide()
        self._refresh_dino_list()

        # Resume YOLO selection
        self.resume_row = QWidget()
        resume_layout = QVBoxLayout(self.resume_row)
        resume_layout.setContentsMargins(0, 0, 0, 0)
        resume_top = QHBoxLayout()
        self.resume_combo = QComboBox()
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
        form.addRow("Checkpoint:", self.resume_row)
        self.resume_row.hide()
        self._refresh_resume_list()

        # Device info
        self.device_label = QLabel(self.device.upper())
        form.addRow("Device:", self.device_label)

        # Task selection
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "Auto (from dataset)",
            "Detection",
            "Pose",
            "Segmentation",
        ])
        if self.default_task == "segment":
            self.task_combo.setCurrentText("Segmentation")
        elif self.default_task == "pose":
            self.task_combo.setCurrentText("Pose")
        elif self.default_task == "detect":
            self.task_combo.setCurrentText("Detection")
        form.addRow("Training task:", self.task_combo)

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
        self.batch_hint.setStyleSheet("color: #bbbbbb; font-size: 9pt;")
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

        self.setStyleSheet(
            self.styleSheet()
            + """
            QFrame#TrainSettingsPanel, QFrame#TrainOutputPanel {
                background-color: #24272a;
                border: 1px solid #3e4449;
                border-radius: 8px;
            }
            QLabel#TrainPanelTitle {
                background: transparent;
                border: none;
                color: #f0f3f5;
                font-size: 12pt;
                font-weight: 700;
                padding: 0;
            }
            QLabel#TrainStatusLabel {
                background-color: #343a40;
                border: 1px solid #515a61;
                border-radius: 10px;
                color: #dce3e8;
                font-size: 9pt;
                padding: 3px 10px;
            }
            QPlainTextEdit#TrainLogView {
                background-color: #0e1113;
                color: #d8dee4;
                border: 1px solid #2f363d;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #315f8f;
            }
            """
        )

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
        use_dino = idx == 1
        use_checkpoint_continue = idx == 2
        use_exact_resume = idx == 3
        use_resume = use_checkpoint_continue or use_exact_resume
        self.model_row.setVisible(idx == 0)
        self.dino_row.setVisible(use_dino)
        self.resume_row.setVisible(use_resume)
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
        exports: list[tuple[str, str]] = []
        root = getattr(self, "distillations_root", "")
        if root and os.path.isdir(root):
            try:
                for entry in sorted(os.listdir(root)):
                    run_dir = os.path.join(root, entry)
                    if not os.path.isdir(run_dir):
                        continue
                    exported_dir = os.path.join(run_dir, "exported_models")
                    candidates: list[str] = []
                    if os.path.isdir(exported_dir):
                        preferred = [
                            os.path.join(exported_dir, "exported_last.pt"),
                            os.path.join(exported_dir, f"{entry}_last.pt"),
                        ]
                        for cand in preferred:
                            if os.path.isfile(cand):
                                candidates.append(cand)
                                break
                        if not candidates:
                            for file in sorted(os.listdir(exported_dir)):
                                if file.endswith(".pt"):
                                    candidates.append(os.path.join(exported_dir, file))
                                    break
                    if not candidates:
                        continue
                    exports.append((entry, candidates[0]))
            except Exception:
                exports = []
        self.dino_exports = exports
        if not exports:
            self.dino_combo.addItem("No exports found", "")
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
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select distillation checkpoint (.pt)",
            self.distillations_root if os.path.isdir(self.distillations_root) else os.getcwd(),
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
        colors = {
            "idle": ("#343a40", "#515a61", "#dce3e8"),
            "running": ("#214f63", "#3f879c", "#dff8ff"),
            "complete": ("#214f3a", "#3d8b61", "#e4fff1"),
            "failed": ("#5a2528", "#94434a", "#ffe4e8"),
            "canceled": ("#5a4a25", "#93763c", "#fff5d8"),
        }
        bg, border, fg = colors.get(tone, colors["idle"])
        self.train_status_label.setStyleSheet(
            f"background-color: {bg}; border: 1px solid {border}; border-radius: 10px; "
            f"color: {fg}; font-size: 9pt; padding: 3px 10px;"
        )

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
            QMessageBox.information(self, "Training running", "Cancel training before closing this dialog.")
            event.ignore()
            return
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
        if isinstance(data, dict):
            task_raw = str(data.get("task", "")).strip().lower()
            if task_raw in {"segment", "seg"}:
                return "segment"
            if "kpt_shape" in data or "kp_names" in data:
                return "pose"
        return "detect"

    def _start_training(self):
        if self.training_running:
            QMessageBox.information(self, "Training running", "A training session is already in progress.")
            return

        source_idx = self.source_combo.currentIndex()
        use_dino = source_idx == 1
        use_checkpoint_continue = source_idx == 2
        use_exact_resume = source_idx == 3

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
            if not task_selection.lower().startswith("pose"):
                QMessageBox.information(
                    self,
                    "Pose task enforced",
                    "DINO distillation exports are pose heads. Training task set to Pose."
                )
            task_value = "pose"
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
            event = json.loads(line)
        except Exception:
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
        process.terminate()
        QTimer.singleShot(5000, self._kill_training_process_if_running)

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
        self._flush_training_terminal_output()
        if self.train_stdout_buffer.strip():
            self._handle_training_event_line(self.train_stdout_buffer.strip())
            self.train_stdout_buffer = ""

        config_path = self.train_config_path
        if config_path:
            try:
                if os.path.exists(config_path):
                    os.remove(config_path)
            except Exception:
                pass

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

# =========================
# Entrypoint
# =========================

if __name__ == '__main__':
    _ensure_qt_plugin_paths()
    app = QApplication(sys.argv)

    app_base = os.path.dirname(__file__)

    app.setApplicationName("SqueakPose Studio")
    app.setApplicationDisplayName("SqueakPose Studio")

    icon_path = os.path.join(app_base, "squeakpose_studio_logo.png")
    app.setWindowIcon(QIcon(icon_path))

    default_project = _load_last_project() or _default_projects_root()
    launcher = ProjectLauncherDialog(default_project, icon_path)
    if launcher.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    project_root = launcher.project_root
    if not project_root:
        sys.exit(0)
    project_paths = _ensure_project_structure(project_root)
    _save_last_project(project_root)
    force_initial_setup = launcher.selection_mode == "create"

    splash_pix = QPixmap(os.path.join(app_base, "squeakpose_studio_logo.png"))
    splash = QSplashScreen(splash_pix, Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
    splash.show(); app.processEvents(); splash.raise_(); splash.activateWindow()
    screen = app.primaryScreen(); screen_geometry = screen.availableGeometry()
    x = (screen_geometry.width() - splash_pix.width()) // 2
    y = (screen_geometry.height() - splash_pix.height()) // 2
    splash.move(x, y)
    font_path = os.path.join(app_base, 'fonts', 'FiraSans-Regular.ttf')
    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    preferred_family = system_family
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            loaded_families = QFontDatabase.applicationFontFamilies(font_id)
            if loaded_families:
                preferred_family = loaded_families[0]
        else:
            print("⚠️ Failed to load bundled Fira Sans font; using system font.")

    dark_stylesheet = f"""
    QWidget {{
        background-color: #2b2b2b;
        color: #e0e0e0;
        font-family: '{preferred_family}', '{system_family}', 'Arial', 'Helvetica';
        font-size: 11pt;
    }}
    QPushButton {{
        background-color: #3c3f41;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 6px;
    }}
    QPushButton:hover {{ background-color: #505357; }}
    QComboBox, QLabel {{
        background-color: #3c3f41;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 4px;
    }}
    QComboBox QAbstractItemView {{
        background-color: #2b2b2b;
        selection-background-color: #606366;
    }}
    QGraphicsView {{ background-color: #1e1e1e; }}
    """
    app.setStyleSheet(dark_stylesheet)

    def start_main_window():
        window = LabelingApp(
            project_paths["images_to_label"],
            project_paths["labels_all"],
            project_paths["classes_file"],
            project_paths["keypoints_file"],
            project_root=project_paths["root"],
            force_initial_setup=force_initial_setup,
        )
        _retain_main_window(window)
        window.setWindowTitle(_project_window_title(project_paths["root"]))
        splash.finish(window)
        window.show()
        screen = app.primaryScreen(); screen_geometry = screen.availableGeometry()
        window_width = window.frameGeometry().width(); window_height = window.frameGeometry().height()
        x = (screen_geometry.width() - window_width) // 2
        y = (screen_geometry.height() - window_height) // 2
        window.move(x, y)
        window.raise_()
        window.activateWindow()
        window._update_status()
        window.update_status_bar(f"Project loaded: {project_paths['root']}")

    QTimer.singleShot(1000, start_main_window)
    sys.exit(app.exec())
