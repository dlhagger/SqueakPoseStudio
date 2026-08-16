"""Dialogs for editing pose classes and their keypoint schemas."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from squeakpose.core import find_duplicate_names


class AddClassDialog(QDialog):
    def __init__(self, existing_keypoints: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Class")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("Class name:", self.name_edit)

        default_count = max(0, len(existing_keypoints)) or 6
        self.keypoints_edit = QTextEdit()
        initial_lines = (
            existing_keypoints[:]
            if existing_keypoints
            else [f"kp_{idx + 1}" for idx in range(default_count)]
        )
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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_count_label(self) -> None:
        lines = [line for line in self.keypoints_edit.toPlainText().splitlines() if line.strip()]
        self.count_label.setText(f"Keypoint count: {len(lines)}")

    def get_data(self) -> tuple[str, list[str]]:
        name = self.name_edit.text().strip()
        keypoints = [
            line.strip() for line in self.keypoints_edit.toPlainText().splitlines() if line.strip()
        ]
        return name, keypoints


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
        self._kp_map = {name: keypoint_map.get(name, canonical[:])[:] for name in self._classes}
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

        button_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Class")
        self.add_btn.clicked.connect(self._add_class)
        button_row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        button_row.addWidget(self.remove_btn)
        if self._schema_locked:
            self.remove_btn.setEnabled(False)
            self.remove_btn.setToolTip("Schema locked after labels exist.")
        button_row.addStretch()
        layout.addLayout(button_row)

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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.result_classes: Optional[list[str]] = None
        self.result_keypoints: Optional[list[str]] = None
        self.result_map: Optional[dict[str, list[str]]] = None

    def _add_class(self) -> None:
        seed: list[str] = []
        current = self.class_list.currentRow()
        if 0 <= current < len(self._classes):
            seed = self._kp_map.get(self._classes[current], [])
        if not seed:
            seed = self._canonical_default[:]
        dialog = AddClassDialog(seed, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, keypoints = dialog.get_data()
        if not name:
            QMessageBox.warning(self, "Class name required", "Enter a class name.")
            return
        if name in self._classes:
            QMessageBox.warning(self, "Duplicate class", "That class already exists.")
            return
        self._classes.append(name)
        self.class_list.addItem(name)
        self._kp_map[name] = keypoints[:]
        self.class_list.setCurrentRow(len(self._classes) - 1)

    def _remove_selected(self) -> None:
        if self._schema_locked:
            QMessageBox.information(
                self,
                "Schema locked",
                "Cannot remove classes after labeled data exists.",
            )
            return
        row = self.class_list.currentRow()
        if not 0 <= row < len(self._classes):
            return
        name = self._classes.pop(row)
        self._kp_map.pop(name, None)
        item = self.class_list.takeItem(row)
        del item
        QMessageBox.information(self, "Class removed", f"Removed '{name}'.")
        self.class_list.setCurrentRow(min(row, len(self._classes) - 1))

    def _update_count_label(self) -> None:
        keypoints = [
            line.strip() for line in self.keypoints_edit.toPlainText().splitlines() if line.strip()
        ]
        self.status_label.setText(f"Keypoint count: {len(keypoints)}")

    def _load_selected_class(self, row: int) -> None:
        self._save_current_keypoints()
        self._current_row = row
        if not 0 <= row < len(self._classes):
            self.keypoints_edit.clear()
            self.status_label.setText("Keypoint count: 0")
            return
        keypoints = self._kp_map.get(self._classes[row], [])
        self.keypoints_edit.blockSignals(True)
        self.keypoints_edit.setPlainText("\n".join(keypoints))
        self.keypoints_edit.blockSignals(False)
        self._update_count_label()

    def _save_current_keypoints(self) -> None:
        if not 0 <= self._current_row < len(self._classes):
            return
        name = self._classes[self._current_row]
        self._kp_map[name] = [
            line.strip() for line in self.keypoints_edit.toPlainText().splitlines() if line.strip()
        ]

    def _on_accept(self) -> None:
        self._save_current_keypoints()
        if not self._classes:
            QMessageBox.warning(self, "No classes", "Add at least one class.")
            return
        if not any(self._kp_map.get(name) for name in self._classes):
            QMessageBox.warning(
                self,
                "Keypoints required",
                "Enter at least one keypoint for any class.",
            )
            return
        for class_name in self._classes:
            duplicates = find_duplicate_names(self._kp_map.get(class_name, []))
            if duplicates:
                QMessageBox.warning(
                    self,
                    "Duplicate keypoints",
                    f"Class '{class_name}' has duplicate keypoint names:\n"
                    f"{', '.join(duplicates)}\n\n"
                    "Each keypoint name must be unique within a class.",
                )
                return

        canonical: list[str] = []
        seen: set[str] = set()
        for name in self._canonical_default:
            if name not in seen:
                canonical.append(name)
                seen.add(name)
        for class_name in self._classes:
            for keypoint_name in self._kp_map.get(class_name, []):
                if keypoint_name not in seen:
                    canonical.append(keypoint_name)
                    seen.add(keypoint_name)
        if not canonical:
            QMessageBox.warning(
                self,
                "Keypoints required",
                "No keypoint names defined.",
            )
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
