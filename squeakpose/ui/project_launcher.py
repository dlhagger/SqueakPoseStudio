"""Project selection and creation dialog."""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from squeakpose.ui.style import launcher_stylesheet


def choose_project_root(
    default_dir: str,
    parent: Optional[QWidget] = None,
) -> Optional[str]:
    start_dir = default_dir if os.path.isdir(default_dir) else os.path.expanduser("~")
    selected = QFileDialog.getExistingDirectory(
        parent,
        "Select Project Folder",
        start_dir,
    )
    return os.path.abspath(selected) if selected else None


def create_project_root(
    default_dir: str,
    parent: Optional[QWidget] = None,
) -> Optional[str]:
    start_dir = default_dir if os.path.isdir(default_dir) else os.path.expanduser("~")
    parent_dir = QFileDialog.getExistingDirectory(
        parent,
        "Select Parent Folder for New Project",
        start_dir,
    )
    if not parent_dir:
        return None

    project_name, accepted = QInputDialog.getText(
        parent,
        "New Project",
        "Project name:",
    )
    if not accepted:
        return None
    project_name = project_name.strip()
    if not project_name:
        QMessageBox.warning(parent, "Invalid name", "Project name cannot be empty.")
        return None

    project_root = os.path.abspath(os.path.join(parent_dir, project_name))
    if os.path.exists(project_root):
        if not os.path.isdir(project_root):
            QMessageBox.warning(
                parent,
                "Invalid path",
                "A file exists with that project name.",
            )
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
        except OSError as exc:
            QMessageBox.warning(
                parent,
                "Create project failed",
                f"Could not create project folder:\n{exc}",
            )
            return None
    return project_root


class ProjectLauncherDialog(QDialog):
    """Startup dialog for opening or creating a project."""

    def __init__(
        self,
        default_dir: str,
        logo_path: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.default_dir = default_dir
        self.project_root: Optional[str] = None
        self.selection_mode: Optional[str] = None

        self.setWindowTitle("SqueakPose Studio")
        self.setModal(True)
        self.setMinimumSize(640, 500)
        self.setStyleSheet(launcher_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(18)
        layout.addStretch(1)
        if logo_path and os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                logo_label = QLabel()
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                logo_label.setPixmap(
                    pixmap.scaled(
                        220,
                        220,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                layout.addWidget(logo_label)

        title = QLabel("Open a project or create a new one")
        title.setObjectName("LauncherTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(
            "Project folders contain classes, keypoints, images, labels, "
            "datasets, runs, and analysis outputs."
        )
        subtitle.setObjectName("LauncherSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        open_button = QPushButton("Open Project")
        open_button.setObjectName("PrimaryLauncherButton")
        open_button.clicked.connect(self._open_project)
        button_row.addWidget(open_button)
        create_button = QPushButton("Create Project")
        create_button.clicked.connect(self._create_project)
        button_row.addWidget(create_button)
        layout.addLayout(button_row)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)
        layout.addStretch(1)

    def _open_project(self) -> None:
        chosen = choose_project_root(self.default_dir, parent=self)
        if chosen:
            self.project_root = chosen
            self.selection_mode = "open"
            self.accept()

    def _create_project(self) -> None:
        chosen = create_project_root(self.default_dir, parent=self)
        if chosen:
            self.project_root = chosen
            self.selection_mode = "create"
            self.accept()
