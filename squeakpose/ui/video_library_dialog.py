"""Project video-link manager dialog."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from squeakpose.services.video_library import (
    VIDEO_EXTENSIONS,
    add_video_links,
    list_project_videos,
    remove_video_link,
    rename_video_link,
    retarget_video_link,
)

VIDEO_FILTER = "Videos (" + " ".join(f"*{extension}" for extension in VIDEO_EXTENSIONS) + ")"


class VideoLibraryDialog(QDialog):
    """Manage lightweight links in one project's videos directory."""

    def __init__(self, videos_dir: str, parent=None):
        super().__init__(parent)
        self.videos_dir = os.path.abspath(videos_dir)
        self.setWindowTitle("Project Videos")
        self.setMinimumSize(720, 420)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Add links to videos stored elsewhere. Links use almost no disk space, and removing "
            "a link never deletes its source video."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.summary_label = QLabel(self)
        self.summary_label.setObjectName("VideoLibrarySummary")
        layout.addWidget(self.summary_label)

        self.video_list = QListWidget(self)
        self.video_list.setObjectName("VideoLibraryList")
        self.video_list.currentItemChanged.connect(self._selection_changed)
        self.video_list.itemDoubleClicked.connect(lambda _item: self._rename_selected())
        layout.addWidget(self.video_list, 1)

        self.detail_label = QLabel(self)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.detail_label)

        actions = QHBoxLayout()
        self.add_button = QPushButton("Add Video Links…", self)
        self.rename_button = QPushButton("Rename Link…", self)
        self.source_button = QPushButton("Change Source…", self)
        self.remove_button = QPushButton("Remove Link", self)
        self.add_button.clicked.connect(self.add_video_links)
        self.rename_button.clicked.connect(self._rename_selected)
        self.source_button.clicked.connect(self._change_source)
        self.remove_button.clicked.connect(self._remove_selected)
        actions.addWidget(self.add_button)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.source_button)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self, select_name: str = "") -> None:
        self.video_list.clear()
        selected_item = None
        entries = list_project_videos(self.videos_dir)
        linked_count = sum(entry.is_link and entry.target_exists for entry in entries)
        missing_count = sum(entry.is_link and not entry.target_exists for entry in entries)
        local_count = sum(not entry.is_link for entry in entries)
        self.summary_label.setText(
            f"{len(entries)} video(s)  •  {linked_count} linked  •  "
            f"{missing_count} missing  •  {local_count} local"
        )
        for entry in entries:
            if entry.is_link:
                state = "Linked" if entry.target_exists else "Missing source"
            else:
                state = "Local file"
            item = QListWidgetItem(f"{entry.name}    —    {state}")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            if not entry.target_exists:
                item.setToolTip("The source video was moved, renamed, or disconnected.")
            self.video_list.addItem(item)
            if entry.name == select_name:
                selected_item = item
        if selected_item is not None:
            self.video_list.setCurrentItem(selected_item)
        elif self.video_list.count():
            self.video_list.setCurrentRow(0)
        else:
            self.detail_label.setText("No project videos yet.")
        self._selection_changed(self.video_list.currentItem())

    def _selected_entry(self):
        item = self.video_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _selection_changed(self, _current, _previous=None) -> None:
        entry = self._selected_entry()
        editable = bool(entry is not None and entry.is_link)
        self.rename_button.setEnabled(editable)
        self.source_button.setEnabled(editable)
        self.remove_button.setEnabled(editable)
        if entry is None:
            self.detail_label.setText("No project videos yet.")
        elif entry.is_link:
            self.detail_label.setText(f"Source: {entry.target}")
        else:
            self.detail_label.setText(
                f"Stored in project: {entry.path}\nThis is a real file, so link-management actions are disabled."
            )

    def add_video_links(self) -> None:
        sources, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose videos to link",
            os.path.dirname(self.videos_dir),
            f"{VIDEO_FILTER};;All Files (*)",
        )
        if not sources:
            return
        try:
            created = add_video_links(self.videos_dir, sources)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Add Video Link", str(exc))
            return
        self.refresh(created[-1].name if created else "")
        if not created:
            QMessageBox.information(self, "Videos Already Linked", "Those videos are already linked.")

    def _rename_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None or not entry.is_link:
            return
        new_name, accepted = QInputDialog.getText(
            self, "Rename Video Link", "Link name:", text=entry.name
        )
        if not accepted or new_name.strip() == entry.name:
            return
        try:
            destination = rename_video_link(self.videos_dir, entry.name, new_name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Rename Link", str(exc))
            return
        self.refresh(os.path.basename(destination))

    def _change_source(self) -> None:
        entry = self._selected_entry()
        if entry is None or not entry.is_link:
            return
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Choose replacement video source",
            os.path.dirname(entry.target),
            f"{VIDEO_FILTER};;All Files (*)",
        )
        if not source:
            return
        try:
            retarget_video_link(self.videos_dir, entry.name, source)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Change Source", str(exc))
            return
        self.refresh(entry.name)

    def _remove_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None or not entry.is_link:
            return
        decision = QMessageBox.question(
            self,
            "Remove Video Link",
            f"Remove '{entry.name}' from this project?\n\nThe source video will not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return
        try:
            remove_video_link(self.videos_dir, entry.name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could Not Remove Link", str(exc))
            return
        self.refresh()
