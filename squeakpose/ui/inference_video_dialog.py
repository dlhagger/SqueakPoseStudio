"""Project-aware multi-video picker for inference runs."""

from __future__ import annotations

import datetime
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from squeakpose.project.layers import layer_definition
from squeakpose.services.inference import (
    project_video_inference_statuses,
    video_identity,
)
from squeakpose.services.video_library import VIDEO_EXTENSIONS, list_project_videos
from squeakpose.ui.style import inference_dialog_stylesheet

VIDEO_FILTER = "Videos (" + " ".join(f"*{extension}" for extension in VIDEO_EXTENSIONS) + ")"


class InferenceVideoDialog(QDialog):
    """Select any combination of project or manually browsed videos."""

    def __init__(
        self,
        project_root: str,
        *,
        configured_layers: tuple[str, ...] = (),
        default_batch_size: int = 4,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.project_root = os.path.abspath(project_root)
        self.configured_layers = tuple(configured_layers)
        self._paths_by_key: dict[str, str] = {}
        self._statuses = project_video_inference_statuses(self.project_root)
        self.setWindowTitle("Choose Videos for Inference")
        self.setMinimumSize(860, 540)
        self.resize(980, 620)
        self.setStyleSheet(inference_dialog_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)

        title = QLabel("Run inference on project videos", self)
        title.setObjectName("InferencePickerTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Choose one video, a combination, or the entire project. Previous project "
            "inference is shown so you can quickly spot videos that are already processed.",
            self,
        )
        subtitle.setObjectName("InferencePickerSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        toolbar = QHBoxLayout()
        self.select_all_button = QPushButton("Select All", self)
        self.clear_button = QPushButton("Clear", self)
        self.browse_button = QPushButton("Browse for Video…", self)
        self.browse_button.setObjectName("InferenceBrowseButton")
        self.select_all_button.clicked.connect(self.select_all)
        self.clear_button.clicked.connect(self.clear_selection)
        self.browse_button.clicked.connect(self.browse_for_videos)
        toolbar.addWidget(self.select_all_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.browse_button)
        layout.addLayout(toolbar)

        self.video_table = QTableWidget(0, 3, self)
        self.video_table.setObjectName("InferenceVideoTable")
        self.video_table.setHorizontalHeaderLabels(("Video", "Location", "Inference status"))
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.setAlternatingRowColors(True)
        self.video_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.video_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.video_table.setShowGrid(False)
        header = self.video_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)
        header.setSectionResizeMode(1, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, header.ResizeMode.ResizeToContents)
        self.video_table.itemChanged.connect(self._selection_changed)
        layout.addWidget(self.video_table, 1)

        footer_panel = QFrame(self)
        footer_panel.setObjectName("InferencePickerFooter")
        footer = QHBoxLayout(footer_panel)
        footer.setContentsMargins(12, 9, 12, 9)
        self.selection_label = QLabel("No videos selected", footer_panel)
        self.selection_label.setObjectName("InferenceSelectionSummary")
        footer.addWidget(self.selection_label)
        footer.addStretch(1)
        footer.addWidget(QLabel("Frames per batch", footer_panel))
        self.batch_spin = QSpinBox(footer_panel)
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(max(1, int(default_batch_size)))
        self.batch_spin.setToolTip("Larger batches are faster but use more VRAM or RAM.")
        footer.addWidget(self.batch_spin)
        layout.addWidget(footer_panel)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=self)
        self.run_button = self.buttons.addButton(
            "Run Inference", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.run_button.setObjectName("InferenceRunButton")
        self.run_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._load_project_videos()

    @property
    def selected_video_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                paths.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return tuple(paths)

    @property
    def batch_size(self) -> int:
        return self.batch_spin.value()

    def _load_project_videos(self) -> None:
        videos_dir = os.path.join(self.project_root, "videos")
        entries = list_project_videos(videos_dir)
        for entry in entries:
            location = "Project link" if entry.is_link else "In project"
            if not entry.target_exists:
                location = "Missing source"
            self._add_video(entry.path, entry.name, location, available=entry.target_exists)
        if not entries:
            self.selection_label.setText(
                "No project videos yet — use Browse for Video to choose one manually."
            )

    def _add_video(self, path: str, name: str, location: str, *, available: bool = True) -> None:
        key = video_identity(path)
        if key in self._paths_by_key:
            if available:
                self._set_checked_for_key(key, True)
            return
        self._paths_by_key[key] = path
        row = self.video_table.rowCount()
        self.video_table.insertRow(row)

        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, path)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
        if not available:
            flags = Qt.ItemFlag.NoItemFlags
            name_item.setToolTip("This project's video link no longer has a readable source.")
        name_item.setFlags(flags)
        name_item.setCheckState(Qt.CheckState.Unchecked)
        self.video_table.setItem(row, 0, name_item)
        self.video_table.setItem(row, 1, QTableWidgetItem(location))
        status_item = QTableWidgetItem(self._status_text(key))
        status_item.setToolTip(self._status_tooltip(key))
        self.video_table.setItem(row, 2, status_item)

    def _status_text(self, key: str) -> str:
        status = self._statuses.get(key)
        if status is None or not status.successful_layers:
            return "Not run"
        completed = set(status.successful_layers)
        configured = set(self.configured_layers)
        prefix = "Complete" if configured and configured.issubset(completed) else "Available"
        names = ", ".join(
            layer_definition(layer_id).display_name for layer_id in status.successful_layers
        )
        return f"{prefix} · {names}"

    def _status_tooltip(self, key: str) -> str:
        status = self._statuses.get(key)
        if status is None:
            return "No completed project inference run was found for this video."
        latest = status.latest_created_at
        try:
            latest = datetime.datetime.fromisoformat(latest).strftime("%b %d, %Y at %I:%M %p")
        except ValueError:
            pass
        return f"{status.run_count} saved run(s). Latest: {latest or 'unknown date'}"

    def browse_for_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose videos for inference",
            os.path.join(self.project_root, "videos"),
            f"{VIDEO_FILTER};;All Files (*)",
        )
        for path in paths:
            if not os.path.isfile(path):
                continue
            key = video_identity(path)
            self._add_video(path, os.path.basename(path), "Browsed file")
            self._set_checked_for_key(key, True)
        self._selection_changed()

    def _set_checked_for_key(self, key: str, checked: bool) -> None:
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is not None and video_identity(str(item.data(Qt.ItemDataRole.UserRole))) == key:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                return

    def select_all(self) -> None:
        self.video_table.blockSignals(True)
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked)
        self.video_table.blockSignals(False)
        self._selection_changed()

    def clear_selection(self) -> None:
        self.video_table.blockSignals(True)
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.video_table.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, _item=None) -> None:
        count = len(self.selected_video_paths)
        self.run_button.setEnabled(count > 0)
        self.selection_label.setText(
            "No videos selected"
            if count == 0
            else f"{count} video{'s' if count != 1 else ''} selected"
        )


__all__ = ["InferenceVideoDialog"]
