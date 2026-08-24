"""QApplication setup and SqueakPose Studio startup."""

from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QSplashScreen

from squeakpose.project.paths import (
    default_projects_root,
    ensure_project_structure,
    load_last_project,
    project_window_title,
    save_last_project,
)
from squeakpose.project.safety import ProjectPathError
from squeakpose.ui.style import app_stylesheet


def run(argv: list[str] | None = None) -> int:
    """Launch the desktop application and return its Qt exit code."""
    # Imported lazily so ``squeakpose_studio.py`` remains a compatible module
    # for integrations and tests while this module owns application startup.
    from squeakpose.ui.main_window import (
        DEFAULT_CLASS_NAMES,
        LabelingApp,
        ProjectLauncherDialog,
        _acquire_project_lock_for_ui,
        _ensure_qt_plugin_paths,
        _recover_project_transactions_for_ui,
        _retain_main_window,
    )

    _ensure_qt_plugin_paths()
    app_argv = list(sys.argv if argv is None else argv)
    app = QApplication(app_argv)
    app_base = os.path.dirname(os.path.dirname(__file__))

    app.setApplicationName("SqueakPose Studio")
    app.setApplicationDisplayName("SqueakPose Studio")
    if sys.platform.startswith("linux"):
        app.setDesktopFileName("squeakpose-studio")
    icon_path = os.path.join(app_base, "squeakpose_studio_logo.png")
    app.setWindowIcon(QIcon(icon_path))

    default_project = load_last_project() or default_projects_root()
    launcher = ProjectLauncherDialog(default_project, icon_path)
    if launcher.exec() != QDialog.DialogCode.Accepted:
        return 0
    project_root = launcher.project_root
    if not project_root:
        return 0

    project_lock = _acquire_project_lock_for_ui(project_root, parent=launcher)
    if project_lock is None:
        return 1
    try:
        _recover_project_transactions_for_ui(project_root, parent=launcher)
        project_paths = ensure_project_structure(
            project_root,
            default_segmentation_classes=tuple(DEFAULT_CLASS_NAMES),
        )
    except (OSError, ProjectPathError) as exc:
        project_lock.release()
        QMessageBox.critical(
            launcher,
            "Invalid Project Structure",
            f"The selected project contains an unsafe or unavailable managed path.\n\n{exc}",
        )
        return 1
    save_last_project(project_root)
    force_initial_setup = launcher.selection_mode == "create"

    splash_pix = QPixmap(os.path.join(app_base, "squeakpose_studio_logo.png"))
    splash = QSplashScreen(
        splash_pix,
        Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint,
    )
    splash.show()
    app.processEvents()
    splash.raise_()
    splash.activateWindow()
    screen = app.primaryScreen()
    screen_geometry = screen.availableGeometry()
    splash.move(
        (screen_geometry.width() - splash_pix.width()) // 2,
        (screen_geometry.height() - splash_pix.height()) // 2,
    )

    font_path = os.path.join(app_base, "fonts", "FiraSans-Regular.ttf")
    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    preferred_family = system_family
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            loaded_families = QFontDatabase.applicationFontFamilies(font_id)
            if loaded_families:
                preferred_family = loaded_families[0]
        else:
            print("Failed to load bundled Fira Sans font; using system font.")

    app.setStyleSheet(app_stylesheet(preferred_family, system_family))

    def start_main_window() -> None:
        try:
            window = LabelingApp(
                project_paths.images_to_label,
                project_paths.labels_all,
                project_paths.classes_file,
                project_paths.keypoints_file,
                project_root=project_paths.root,
                force_initial_setup=force_initial_setup,
                project_lock=project_lock,
            )
        except Exception as exc:
            project_lock.release()
            splash.close()
            QMessageBox.critical(
                launcher,
                "Open Project Failed",
                f"Could not initialize the selected project.\n\n{exc}",
            )
            app.quit()
            return
        _retain_main_window(window)
        window.setWindowTitle(project_window_title(project_paths.root))
        splash.finish(window)
        window.show()
        current_screen = app.primaryScreen()
        available = current_screen.availableGeometry()
        window.move(
            (available.width() - window.frameGeometry().width()) // 2,
            (available.height() - window.frameGeometry().height()) // 2,
        )
        window.raise_()
        window.activateWindow()
        window._update_status()
        window.update_status_bar(f"Project loaded: {project_paths.root}")

    QTimer.singleShot(1000, start_main_window)
    return int(app.exec())
