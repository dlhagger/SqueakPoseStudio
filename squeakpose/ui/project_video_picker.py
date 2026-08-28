"""Reusable multi-select picker for videos already linked to a project."""

from __future__ import annotations

import os
from collections.abc import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from squeakpose.services.video_library import VideoLibraryEntry, list_project_videos


class ProjectVideoPickerDialog(QDialog):
    """Choose one or more readable entries from the project video library."""

    def __init__(
        self,
        videos_dir: str,
        *,
        selected_names: Iterable[str] = (),
        title: str = "Select Project Videos",
        description: str = "Choose videos already linked to this project.",
        accept_label: str = "Use Selected Videos",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.videos_dir = os.path.abspath(videos_dir)
        self.entries = list_project_videos(self.videos_dir)
        self._selected_names = {str(name) for name in selected_names}

        self.setWindowTitle(title)
        self.setMinimumSize(760, 440)
        self.resize(920, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 16)
        layout.setSpacing(10)

        intro = QLabel(description, self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        toolbar = QHBoxLayout()
        self.select_all_button = QPushButton("Select All", self)
        self.clear_button = QPushButton("Clear", self)
        self.select_all_button.clicked.connect(self.select_all)
        self.clear_button.clicked.connect(self.clear_selection)
        toolbar.addWidget(self.select_all_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.video_table = QTableWidget(0, 3, self)
        self.video_table.setHorizontalHeaderLabels(("Video", "Status", "Source"))
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.setAlternatingRowColors(True)
        self.video_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.video_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.video_table.setShowGrid(False)
        header = self.video_table.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, header.ResizeMode.Stretch)
        layout.addWidget(self.video_table, 1)

        self.summary_label = QLabel(self)
        layout.addWidget(self.summary_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=self)
        self.accept_button = self.buttons.addButton(
            accept_label, QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.accept_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.video_table.itemChanged.connect(self._selection_changed)
        self._populate()

    @property
    def selected_entries(self) -> tuple[VideoLibraryEntry, ...]:
        selected: list[VideoLibraryEntry] = []
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                entry = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(entry, VideoLibraryEntry):
                    selected.append(entry)
        return tuple(selected)

    def _populate(self) -> None:
        self.video_table.blockSignals(True)
        self.video_table.setRowCount(0)
        for entry in self.entries:
            row = self.video_table.rowCount()
            self.video_table.insertRow(row)

            name_item = QTableWidgetItem(entry.name)
            name_item.setData(Qt.ItemDataRole.UserRole, entry)
            if entry.target_exists:
                name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                name_item.setCheckState(
                    Qt.CheckState.Checked
                    if entry.name in self._selected_names
                    else Qt.CheckState.Unchecked
                )
            else:
                name_item.setFlags(Qt.ItemFlag.NoItemFlags)
                name_item.setCheckState(Qt.CheckState.Unchecked)
                name_item.setToolTip("The linked source is missing or unreadable.")

            status = (
                "Missing source"
                if not entry.target_exists
                else "Linked"
                if entry.is_link
                else "In project"
            )
            self.video_table.setItem(row, 0, name_item)
            self.video_table.setItem(row, 1, QTableWidgetItem(status))
            self.video_table.setItem(row, 2, QTableWidgetItem(entry.target))
        self.video_table.blockSignals(False)
        self._selection_changed()

    def select_all(self) -> None:
        self.video_table.blockSignals(True)
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
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
        selected = len(self.selected_entries)
        available = sum(entry.target_exists for entry in self.entries)
        missing = len(self.entries) - available
        self.accept_button.setEnabled(selected > 0)
        summary = f"{selected} selected • {available} available"
        if missing:
            summary += f" • {missing} missing"
        self.summary_label.setText(summary)


__all__ = ["ProjectVideoPickerDialog"]
