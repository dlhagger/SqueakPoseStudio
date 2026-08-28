"""Project-aware multi-video picker for inference runs."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from squeakpose.project.layers import layer_definition
from squeakpose.services.inference import (
    project_video_inference_statuses,
    video_identity,
)
from squeakpose.services.tracking import (
    MAX_EXPECTED_ANIMALS,
    TRACKER_AUTO,
    TRACKER_BOTSORT,
    TRACKER_BYTETRACK,
    resolve_tracking_config,
)
from squeakpose.services.video_library import VIDEO_EXTENSIONS, list_project_videos
from squeakpose.ui.style import ThemedComboBox, inference_dialog_stylesheet

VIDEO_FILTER = "Videos (" + " ".join(f"*{extension}" for extension in VIDEO_EXTENSIONS) + ")"


@dataclass(frozen=True, slots=True)
class InferenceVideoSelection:
    """One selected video and its user-controlled tracking settings."""

    video_path: str
    expected_animal_count: int
    requested_tracker: str


class AnimalCountControl(QWidget):
    """Compact, platform-consistent animal counter with explicit controls."""

    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None, *, maximum: int = MAX_EXPECTED_ANIMALS) -> None:
        super().__init__(parent)
        self._minimum = 1
        self._maximum = max(self._minimum, int(maximum))
        self._value = self._minimum
        counter_layout = QHBoxLayout(self)
        counter_layout.setContentsMargins(6, 3, 6, 3)
        counter_layout.setSpacing(3)
        self.decrement_button = QToolButton(self)
        self.decrement_button.setObjectName("AnimalCountButton")
        self.decrement_button.setText("−")
        self.decrement_button.setToolTip("One fewer animal")
        self.value_label = QLabel(str(self._value), self)
        self.value_label.setObjectName("AnimalCountValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.increment_button = QToolButton(self)
        self.increment_button.setObjectName("AnimalCountButton")
        self.increment_button.setText("+")
        self.increment_button.setToolTip("One more animal")
        counter_layout.addWidget(self.decrement_button)
        counter_layout.addWidget(self.value_label, 1)
        counter_layout.addWidget(self.increment_button)
        self.decrement_button.clicked.connect(lambda: self.setValue(self._value - 1))
        self.increment_button.clicked.connect(lambda: self.setValue(self._value + 1))
        self._refresh()

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        normalized = max(self._minimum, min(self._maximum, int(value)))
        if normalized == self._value:
            return
        self._value = normalized
        self._refresh()
        self.valueChanged.emit(self._value)

    def _refresh(self) -> None:
        self.value_label.setText(str(self._value))
        self.decrement_button.setEnabled(self._value > self._minimum)
        self.increment_button.setEnabled(self._value < self._maximum)


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
        self.tracking_available = any(
            layer_id in {"keypoints", "segmentation"} for layer_id in self.configured_layers
        )
        self._paths_by_key: dict[str, str] = {}
        self._tracking_widgets_by_key: dict[str, tuple[AnimalCountControl, QComboBox]] = {}
        self._statuses = project_video_inference_statuses(self.project_root)
        self.setWindowTitle("Choose Videos for Inference")
        self.setMinimumSize(1000, 620)
        self.resize(1180, 700)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(inference_dialog_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)

        title = QLabel("Run inference on project videos", self)
        title.setObjectName("InferencePickerTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Select the videos to process, then confirm the animal count and tracker for "
            "each one. Previous inference is shown for reference.",
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

        self.tracking_defaults_panel = QFrame(self)
        self.tracking_defaults_panel.setObjectName("InferenceTrackingDefaults")
        tracking_defaults = QHBoxLayout(self.tracking_defaults_panel)
        tracking_defaults.setContentsMargins(14, 10, 14, 10)
        tracking_defaults.setSpacing(10)
        tracking_copy = QVBoxLayout()
        tracking_copy.setSpacing(2)
        tracking_title = QLabel("Tracking defaults", self.tracking_defaults_panel)
        tracking_title.setObjectName("InferenceTrackingTitle")
        tracking_hint = QLabel(
            "Auto uses ByteTrack for one animal and BoT-SORT for multiple animals.",
            self.tracking_defaults_panel,
        )
        tracking_hint.setObjectName("InferenceTrackingHint")
        tracking_copy.addWidget(tracking_title)
        tracking_copy.addWidget(tracking_hint)
        tracking_defaults.addLayout(tracking_copy, 1)
        tracking_defaults.addWidget(QLabel("Animals", self.tracking_defaults_panel))
        self.default_animals_spin = AnimalCountControl(self.tracking_defaults_panel)
        self.default_animals_spin.setObjectName("InferenceDefaultAnimals")
        self.default_animals_spin.setValue(1)
        self.default_animals_spin.setMinimumWidth(112)
        tracking_defaults.addWidget(self.default_animals_spin)
        tracking_defaults.addWidget(QLabel("Tracker", self.tracking_defaults_panel))
        self.default_tracker_combo = ThemedComboBox(self.tracking_defaults_panel)
        self.default_tracker_combo.setObjectName("InferenceDefaultTracker")
        self._populate_tracker_combo(self.default_tracker_combo)
        self.default_tracker_combo.setMinimumWidth(180)
        tracking_defaults.addWidget(self.default_tracker_combo)
        self.apply_defaults_button = QPushButton("Apply to Selected", self.tracking_defaults_panel)
        self.apply_defaults_button.setObjectName("InferenceApplyDefaults")
        self.apply_defaults_button.setEnabled(False)
        self.apply_defaults_button.clicked.connect(self.apply_tracking_defaults)
        tracking_defaults.addWidget(self.apply_defaults_button)
        self.default_animals_spin.valueChanged.connect(
            lambda count: self._update_auto_tracker_label(self.default_tracker_combo, count)
        )
        self._update_auto_tracker_label(
            self.default_tracker_combo, self.default_animals_spin.value()
        )
        self.tracking_defaults_panel.setVisible(self.tracking_available)
        layout.addWidget(self.tracking_defaults_panel)

        self.video_table = QTableWidget(0, 5, self)
        self.video_table.setObjectName("InferenceVideoTable")
        self.video_table.setHorizontalHeaderLabels(
            ("Video", "Source", "Previous inference", "Animals", "Tracker")
        )
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.setAlternatingRowColors(True)
        self.video_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.video_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.video_table.setShowGrid(False)
        self.video_table.verticalHeader().setDefaultSectionSize(44)
        self.video_table.verticalHeader().setMinimumSectionSize(44)
        header = self.video_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)
        for column, width in ((1, 95), (2, 255), (3, 105), (4, 225)):
            header.setSectionResizeMode(column, header.ResizeMode.Fixed)
            self.video_table.setColumnWidth(column, width)
        if not self.tracking_available:
            self.video_table.setColumnHidden(3, True)
            self.video_table.setColumnHidden(4, True)
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
        self.sequential_tracking_label = QLabel(
            "Identity tracking runs sequentially by frame.", footer_panel
        )
        self.sequential_tracking_label.setObjectName("InferenceSequentialHint")
        self.sequential_tracking_label.setVisible(self.tracking_available)
        footer.addWidget(self.sequential_tracking_label)
        self.batch_label = QLabel("Frames per batch", footer_panel)
        self.batch_label.setVisible(not self.tracking_available)
        footer.addWidget(self.batch_label)
        self.batch_spin = QSpinBox(footer_panel)
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(max(1, int(default_batch_size)))
        self.batch_spin.setToolTip("Larger batches are faster but use more VRAM or RAM.")
        if self.tracking_available:
            self.batch_spin.setEnabled(False)
            self.batch_spin.setVisible(False)
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
    def selected_video_settings(self) -> tuple[InferenceVideoSelection, ...]:
        """Return selected videos with the settings shown in their table rows."""
        selections: list[InferenceVideoSelection] = []
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole))
            widgets = self._tracking_widgets_by_key.get(video_identity(path))
            if widgets is None:
                selections.append(InferenceVideoSelection(path, 1, TRACKER_AUTO))
                continue
            animal_spin, tracker_combo = widgets
            selections.append(
                InferenceVideoSelection(
                    video_path=path,
                    expected_animal_count=animal_spin.value(),
                    requested_tracker=str(tracker_combo.currentData() or TRACKER_AUTO),
                )
            )
        return tuple(selections)

    @property
    def batch_size(self) -> int:
        return self.batch_spin.value()

    def _load_project_videos(self) -> None:
        videos_dir = os.path.join(self.project_root, "videos")
        entries = list_project_videos(videos_dir)
        for entry in entries:
            location = "Linked" if entry.is_link else "Project"
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

        status = self._statuses.get(key)
        animal_spin = AnimalCountControl(self.video_table)
        animal_spin.setObjectName("InferenceRowAnimals")
        animal_spin.setValue(status.expected_animal_count if status is not None else 1)
        animal_spin.setToolTip("Expected number of animals visible in this video.")
        animal_spin.setMinimumWidth(110)
        tracker_combo = ThemedComboBox(self.video_table)
        tracker_combo.setObjectName("InferenceRowTracker")
        self._populate_tracker_combo(tracker_combo)
        tracker_combo.setMinimumWidth(205)
        requested = status.requested_tracker if status is not None else TRACKER_AUTO
        selected_index = tracker_combo.findData(requested)
        tracker_combo.setCurrentIndex(max(0, selected_index))
        tracker_combo.setToolTip(
            "Auto uses ByteTrack for one animal and BoT-SORT for multiple animals."
        )
        self._tracking_widgets_by_key[key] = (animal_spin, tracker_combo)
        animal_spin.valueChanged.connect(
            lambda count, combo=tracker_combo: self._update_auto_tracker_label(combo, count)
        )
        self._update_auto_tracker_label(tracker_combo, animal_spin.value())
        if self.configured_layers and not self.tracking_available:
            animal_spin.setEnabled(False)
            tracker_combo.setEnabled(False)
            disabled_tip = "Tracking is not used for depth-only inference."
            animal_spin.setToolTip(disabled_tip)
            tracker_combo.setToolTip(disabled_tip)
        self.video_table.setCellWidget(row, 3, animal_spin)
        self.video_table.setCellWidget(row, 4, tracker_combo)

    @staticmethod
    def _populate_tracker_combo(combo: QComboBox) -> None:
        combo.addItem("Auto", TRACKER_AUTO)
        combo.addItem("ByteTrack", TRACKER_BYTETRACK)
        combo.addItem("BoT-SORT", TRACKER_BOTSORT)
        combo.setToolTip("Auto uses ByteTrack for one animal and BoT-SORT for multiple animals.")

    @staticmethod
    def _update_auto_tracker_label(combo: QComboBox, expected_animals: int) -> None:
        resolved = resolve_tracking_config(expected_animals, TRACKER_AUTO).resolved_tracker
        name = "ByteTrack" if resolved == TRACKER_BYTETRACK else "BoT-SORT"
        combo.setItemText(0, f"Auto → {name}")

    def _status_text(self, key: str) -> str:
        status = self._statuses.get(key)
        if status is None or not status.successful_layers:
            return "Not run"
        completed = set(status.successful_layers)
        configured = set(self.configured_layers)
        complete = bool(configured and configured.issubset(completed))
        names = " + ".join(
            layer_definition(layer_id).display_name for layer_id in status.successful_layers
        )
        return f"✓ {names}" if complete else f"{names} available"

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

    def apply_tracking_defaults(self) -> None:
        """Apply the bulk tracking controls without changing unselected videos."""
        count = self.default_animals_spin.value()
        tracker = str(self.default_tracker_combo.currentData() or TRACKER_AUTO)
        for row in range(self.video_table.rowCount()):
            item = self.video_table.item(row, 0)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            key = video_identity(str(item.data(Qt.ItemDataRole.UserRole)))
            widgets = self._tracking_widgets_by_key.get(key)
            if widgets is None:
                continue
            animal_spin, tracker_combo = widgets
            animal_spin.setValue(count)
            tracker_index = tracker_combo.findData(tracker)
            tracker_combo.setCurrentIndex(max(0, tracker_index))

    def _selection_changed(self, _item=None) -> None:
        count = len(self.selected_video_paths)
        self.run_button.setEnabled(count > 0)
        self.apply_defaults_button.setEnabled(self.tracking_available and count > 0)
        self.selection_label.setText(
            "No videos selected"
            if count == 0
            else f"{count} video{'s' if count != 1 else ''} selected"
        )


__all__ = ["InferenceVideoDialog", "InferenceVideoSelection"]
