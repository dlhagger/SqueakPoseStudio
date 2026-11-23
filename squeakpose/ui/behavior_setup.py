"""Project behavior-definition editor shared by first-open setup and later edits."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from squeakpose.annotation.behavior import (
    COLORS,
    Behavior,
    new_id,
    validate_behavior_definitions,
)


class BehaviorDefinitionEditor(QWidget):
    submitted = pyqtSignal(object)
    canceled = pyqtSignal()

    def __init__(self, behaviors=(), *, first_open=False, protected_ids=(), parent=None):
        super().__init__(parent)
        self.protected_ids = set(protected_ids)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        title = QLabel("Define behaviors" if first_open else "Behavior definitions")
        title.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(title)
        self.description = QLabel(
            "Choose what you want to label before opening a video. Start with one behavior or add several. "
            "These definitions will be saved for this project and used across its videos."
            if first_open
            else "Review the names and definitions you use while labeling."
        )
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        help_text = QLabel(
            "Give each behavior a name. Use the definition to describe what counts, "
            "when it starts, and when it stops. Definitions are optional."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(
            ["Behavior name", "Definition / start and stop criteria"]
        )
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 220)
        self.table.setMinimumHeight(200)
        self.table.setWordWrap(True)
        self.table.setToolTip("Double-click a name or definition to edit it.")
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.add_button = QPushButton("Add behavior")
        self.remove_button = QPushButton("Remove selected")
        for button in (self.add_button, self.remove_button):
            button.setAutoDefault(False)
            controls.addWidget(button)
        controls.addStretch()
        self.add_button.clicked.connect(lambda: self.add_row(edit=True))
        self.remove_button.clicked.connect(self.remove_selected)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemChanged.connect(lambda _: self.error.clear())
        layout.addLayout(controls)
        reserved = QLabel(
            "Mark only the behaviors you want to label. All other frames stay unlabeled; "
            "they are not automatically treated as examples of behavior absence."
        )
        reserved.setWordWrap(True)
        layout.addWidget(reserved)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #fca5a5;")
        layout.addWidget(self.error)
        footer = QHBoxLayout()
        footer.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.canceled.emit)
        footer.addWidget(cancel)
        self.save_button = QPushButton(
            "Save behaviors & continue" if first_open else "Save definitions"
        )
        self.save_button.setAutoDefault(False)
        self.save_button.clicked.connect(self._submit)
        footer.addWidget(self.save_button)
        layout.addLayout(footer)
        for behavior in behaviors:
            if behavior.name.strip().casefold() not in {"other", "unscorable"}:
                self.add_row(behavior)
        if not self.table.rowCount():
            self.add_row()
        self._selection_changed()

    def add_row(self, behavior=None, *, edit=False):
        row = self.table.rowCount()
        behavior = behavior or Behavior(new_id(), "", COLORS[(row + 1) % len(COLORS)])
        self.table.insertRow(row)
        name = QTableWidgetItem(behavior.name)
        name.setData(Qt.ItemDataRole.UserRole, behavior)
        name.setForeground(QColor(behavior.color))
        self.table.setItem(row, 0, name)
        self.table.setItem(row, 1, QTableWidgetItem(behavior.definition))
        self.table.setRowHeight(row, 58)
        self.table.setCurrentCell(row, 0)
        if edit:
            self.table.editItem(name)

    def _selection_changed(self):
        item = self.table.item(self.table.currentRow(), 0)
        protected = item and item.data(Qt.ItemDataRole.UserRole).id in self.protected_ids
        self.remove_button.setEnabled(item is not None and not protected)
        self.remove_button.setToolTip(
            "This behavior is used by saved intervals and cannot be removed." if protected else ""
        )

    def remove_selected(self):
        item = self.table.item(self.table.currentRow(), 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole).id not in self.protected_ids:
            self.table.removeRow(item.row())
        self._selection_changed()

    def behaviors(self):
        records = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            original = item.data(Qt.ItemDataRole.UserRole)
            name = item.text().strip()
            definition = self.table.item(row, 1).text().strip()
            if not name and not definition and original.id not in self.protected_ids:
                continue
            if not name:
                raise ValueError(f"Enter a name for behavior {row + 1}.")
            if name.casefold() in {"other", "unscorable", "unreviewed"}:
                raise ValueError(
                    f"'{name}' is a reserved review category. Enter a target behavior name."
                )
            records.append(Behavior(original.id, name, original.color, definition))
        if not records:
            raise ValueError("Add at least one behavior you want to label.")
        validate_behavior_definitions(records)
        return records

    def _submit(self):
        self.table.clearFocus()  # Commit an active cell edit before reading values.
        self.save_button.setFocus()
        try:
            records = self.behaviors()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.error.clear()
        self.submitted.emit(records)
