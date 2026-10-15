"""Review the box assigned to each pose before replacing geometry."""

from collections.abc import Mapping, Sequence
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from squeakpose.services.frame_annotations import SegmentationBoxTransferPlan


class SegmentationBoxReviewDialog(QDialog):
    """Show current/proposed boxes in shared pixels and let the user correct pairings."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        plans: Sequence[SegmentationBoxTransferPlan],
        assignments: Sequence[int],
        image_path: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Replace boxes from segmentation")
        self.resize(760, 650)
        self._entries = entries
        self._plans = plans
        self._pixmap = QPixmap(image_path)
        layout = QVBoxLayout(self)
        description = QLabel(
            "The geometry does not clearly distinguish these objects. Select a row to compare its current box "
            "(blue) with the proposed segmentation box (gold). Keypoints stay in their original image coordinates."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(220)
        layout.addWidget(self.view, 1)
        self.table = QTableWidget(len(entries), 2)
        self.table.setHorizontalHeaderLabels(["Keypoint instance", "Use saved segmentation mask"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 230)
        self.table.setMaximumHeight(180)
        self.selectors: list[QComboBox] = []
        for row, source in enumerate(assignments):
            self.table.setItem(row, 0, QTableWidgetItem(f"{plans[0].class_name} {row + 1}"))
            combo = QComboBox()
            for index, plan in enumerate(plans):
                box = plan.box
                combo.addItem(
                    f"Mask {index + 1} · x {box.x:.0f}, y {box.y:.0f}, {box.w:.0f} × {box.h:.0f}"
                )
            combo.setCurrentIndex(source)
            combo.currentIndexChanged.connect(
                lambda _index, selected_row=row: self._selection_changed(selected_row)
            )
            self.table.setCellWidget(row, 1, combo)
            self.selectors.append(combo)
        layout.addWidget(self.table)
        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Replace boxes")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.table.currentCellChanged.connect(lambda row, *_args: self._preview(row))
        self.table.selectRow(0)
        self._validate()

    @property
    def assignments(self) -> tuple[int, ...]:
        return tuple(combo.currentIndex() for combo in self.selectors)

    def _validate(self) -> None:
        valid = len(set(self.assignments)) == len(self.selectors)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        self.validation_label.setText(
            ""
            if valid
            else "Each mask must be assigned once. Choose a different mask for duplicate selections."
        )

    def _selection_changed(self, row: int) -> None:
        self.table.selectRow(row)
        self._validate()
        self._preview(row)

    def _preview(self, row: int) -> None:
        if row < 0 or row >= len(self.selectors):
            return
        self.scene.clear()
        self.scene.addPixmap(self._pixmap)
        entry = self._entries[row]
        box = entry["bbox"]
        plan = self._plans[self.selectors[row].currentIndex()]
        for x, y, width, height, color in (
            (box["x"], box["y"], box["w"], box["h"], "#68bfff"),
            (plan.box.x, plan.box.y, plan.box.w, plan.box.h, "#ffd166"),
        ):
            pen = QPen(QColor(color), 2)
            pen.setCosmetic(True)
            self.scene.addRect(x, y, width, height, pen)
        for point in entry.get("keypoints", []):
            if point.get("vis", 2):
                self.scene.addEllipse(
                    point["x"] - 2, point["y"] - 2, 4, 4, QPen(QColor("#68bfff")), QColor("#68bfff")
                )
        self.scene.setSceneRect(0, 0, self._pixmap.width(), self._pixmap.height())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._preview(self.table.currentRow())
