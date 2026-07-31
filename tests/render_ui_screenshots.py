#!/usr/bin/env python3
"""Render offscreen SqueakPose Studio UI screenshots for visual regression checks.

Example:
    QT_QPA_PLATFORM=offscreen uv run python tests/render_ui_screenshots.py --output-dir /tmp/squeakpose-ui
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "squeakpose-mpl"))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PyQt6

QT_PLUGINS = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
QT_PLATFORMS = QT_PLUGINS / "platforms"
if QT_PLUGINS.is_dir():
    os.environ.setdefault("QT_PLUGIN_PATH", str(QT_PLUGINS))
if QT_PLATFORMS.is_dir():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(QT_PLATFORMS))

from PyQt6.QtGui import QColor, QFontDatabase, QImage, QPainter, QPen
from PyQt6.QtWidgets import QApplication

import squeakpose_studio as studio
from ui_style import app_stylesheet
from squeakpose.ui.project_models_dialog import ProjectModelsDialog


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(["squeakpose-ui-screenshots"])
    system_family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    app.setStyleSheet(app_stylesheet(system_family, system_family))
    return app


def _write_demo_image(path: Path) -> None:
    image = QImage(960, 540, QImage.Format.Format_RGB32)
    image.fill(QColor("#d8dee4"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.fillRect(0, 0, 960, 540, QColor("#eef2f5"))
    painter.setPen(QPen(QColor("#c7d1da"), 1))
    for x in range(0, 960, 48):
        painter.drawLine(x, 0, x, 540)
    for y in range(0, 540, 48):
        painter.drawLine(0, y, 960, y)
    painter.setBrush(QColor("#8bb6d6"))
    painter.setPen(QPen(QColor("#315f8f"), 4))
    painter.drawEllipse(355, 170, 250, 150)
    painter.setBrush(QColor("#e7a45e"))
    painter.setPen(QPen(QColor("#8b5a2b"), 3))
    painter.drawEllipse(570, 215, 70, 52)
    painter.drawLine(350, 245, 285, 255)
    painter.end()
    if not image.save(str(path)):
        raise RuntimeError(f"Could not write demo image: {path}")


def _build_demo_project(root: Path) -> dict[str, str]:
    paths = studio._ensure_project_structure(str(root))
    Path(paths["classes_file"]).write_text("mouse\n", encoding="utf-8")
    Path(paths["keypoints_file"]).write_text(
        "nose\nhead\nleft_ear\nright_ear\nback\ntail_base\n",
        encoding="utf-8",
    )
    Path(paths["classes_seg_file"]).write_text("mouse\n", encoding="utf-8")
    _write_demo_image(Path(paths["images_to_label"]) / "frame001.png")
    Path(paths["labels_all"], "frame001.txt").write_text(
        "0 0.505 0.505 0.350 0.310 "
        "0.635 0.510 2 0.590 0.475 2 0.565 0.440 2 "
        "0.580 0.525 2 0.480 0.485 2 0.345 0.515 2\n",
        encoding="utf-8",
    )
    Path(paths["labels_seg_all"], "frame001.txt").write_text(
        "0 0.340 0.405 0.485 0.365 0.610 0.395 0.655 0.485 "
        "0.625 0.590 0.445 0.625 0.335 0.535\n",
        encoding="utf-8",
    )
    return paths


def _flush_events(app: QApplication, rounds: int = 4) -> None:
    for _ in range(rounds):
        app.processEvents()


def _save_widget(widget, output_path: Path, app: QApplication) -> None:
    widget.show()
    _flush_events(app)
    pixmap = widget.grab()
    if pixmap.isNull() or pixmap.width() <= 1 or pixmap.height() <= 1:
        raise RuntimeError(f"Empty screenshot for {output_path.name}")
    if not pixmap.save(str(output_path)):
        raise RuntimeError(f"Could not save screenshot: {output_path}")


def render_screenshots(output_dir: Path) -> list[Path]:
    app = _ensure_app()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(tempfile.mkdtemp(prefix="squeakpose-ui-project-"))
    paths = _build_demo_project(project_dir)
    logo_path = Path(__file__).resolve().parents[1] / "squeakpose_studio_logo.png"

    screenshots: list[Path] = []

    launcher = studio.ProjectLauncherDialog(str(project_dir), str(logo_path))
    launcher.resize(680, 540)
    launcher_path = output_dir / "launcher.png"
    _save_widget(launcher, launcher_path, app)
    screenshots.append(launcher_path)
    launcher.close()

    window = studio.LabelingApp(
        paths["images_to_label"],
        paths["labels_all"],
        paths["classes_file"],
        paths["keypoints_file"],
        project_root=paths["root"],
        force_initial_setup=False,
    )
    window._prompted_class_manager = True
    window._seg_setup_prompted = True
    window.resize(1500, 900)
    window.show()
    _flush_events(app, rounds=8)

    pose_path = output_dir / "main_pose.png"
    _save_widget(window, pose_path, app)
    screenshots.append(pose_path)

    window.workflow_selector.setCurrentIndex(1)
    window._seg_setup_prompted = True
    window.set_mode("segment")
    _flush_events(app, rounds=8)
    seg_path = output_dir / "main_segmentation.png"
    _save_widget(window, seg_path, app)
    screenshots.append(seg_path)

    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    pose_model = models_dir / "mouse-keypoints.pt"
    seg_model = models_dir / "mouse-segmentation.pt"
    pose_model.write_bytes(b"demo")
    seg_model.write_bytes(b"demo")
    project_models = ProjectModelsDialog(
        window,
        {
            "keypoints": str(pose_model),
            "segmentation": str(seg_model),
        },
        active_layer="segmentation",
    )
    project_models.resize(900, 330)
    project_models_path = output_dir / "project_models_dialog.png"
    _save_widget(project_models, project_models_path, app)
    screenshots.append(project_models_path)
    project_models.close()

    reviewer = studio.VideoReviewDialog(
        window,
        "cpu",
        ["nose", "head", "left_ear", "right_ear", "back", "tail_base"],
        ["mouse"],
        class_keypoints={
            "mouse": [
                "nose",
                "head",
                "left_ear",
                "right_ear",
                "back",
                "tail_base",
            ]
        },
        workflow="segmentation",
        layer_id="segmentation",
        model_paths={
            "keypoints": str(pose_model),
            "segmentation": str(seg_model),
        },
        layer_schemas={
            "keypoints": {
                "classes": ["mouse"],
                "kp_names": [
                    "nose",
                    "head",
                    "left_ear",
                    "right_ear",
                    "back",
                    "tail_base",
                ],
                "class_keypoints": {
                    "mouse": [
                        "nose",
                        "head",
                        "left_ear",
                        "right_ear",
                        "back",
                        "tail_base",
                    ]
                },
            },
            "segmentation": {
                "classes": ["mouse"],
                "kp_names": [],
                "class_keypoints": {},
            },
        },
    )
    reviewer.resize(1180, 780)
    reviewer_path = output_dir / "video_reviewer_project_models.png"
    _save_widget(reviewer, reviewer_path, app)
    screenshots.append(reviewer_path)
    reviewer.close()

    train = studio.TrainDialog(window, str(Path(paths["datasets"]) / "demo_dataset"), default_task="pose")
    train.resize(1100, 720)
    train_path = output_dir / "train_dialog.png"
    _save_widget(train, train_path, app)
    screenshots.append(train_path)
    train.close()

    analysis = studio.AnalysisDialog(window, project_root=paths["root"], app_base_dir=str(REPO_ROOT))
    analysis.resize(1240, 900)
    analysis_path = output_dir / "analysis_dialog.png"
    _save_widget(analysis, analysis_path, app)
    screenshots.append(analysis_path)
    analysis.close()

    window.close()
    _flush_events(app)
    return screenshots


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join(tempfile.gettempdir(), "squeakpose-ui-screenshots"),
        help="Directory for rendered PNG screenshots.",
    )
    args = parser.parse_args(argv)
    paths = render_screenshots(Path(args.output_dir))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
