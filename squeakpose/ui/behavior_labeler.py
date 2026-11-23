"""Human behavior labeling workspace. No trained behavior model is required."""

from __future__ import annotations

import bisect
import os
from dataclasses import replace

from PyQt6.QtCore import QElapsedTimer, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from squeakpose.annotation.behavior import (
    BehaviorDocument,
    Interval,
    Subject,
    new_id,
    timestamp,
)
from squeakpose.annotation.video_view import VideoView
from squeakpose.services.behavior_labels import (
    export_document,
    load_document,
    load_labeling_catalog,
    new_document,
    reconcile_behavior_catalog,
    resolve_frame_annotations,
    save_behavior_catalog,
    save_document,
)
from squeakpose.services.behavior_video import BehaviorVideoReader, VideoIndex, open_frame_video
from squeakpose.services.video_library import VIDEO_EXTENSIONS, list_project_videos
from squeakpose.ui.behavior_setup import BehaviorDefinitionEditor


class VideoIndexJob(QThread):
    progress = pyqtSignal(str)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, project_root: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.project_root = project_root

    def run(self):
        try:
            index = open_frame_video(
                self.path,
                self.progress.emit,
                self.isInterruptionRequested,
                project_root=self.project_root,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(index)
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class SourcePositionSpinBox(QDoubleSpinBox):
    """Keep source precision internally without displaying redundant zeros."""

    def textFromValue(self, value):
        text = super().textFromValue(value)
        if self.decimals():
            whole, separator, fraction = text.partition(self.locale().decimalPoint())
            if separator:
                zero = self.locale().zeroDigit()
                return (
                    whole + separator + fraction.rstrip(zero).ljust(min(3, self.decimals()), zero)
                )
        return text


class BehaviorFrameView(VideoView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class BehaviorTimeline(QWidget):
    seek_requested = pyqtSignal(object)
    selected = pyqtSignal(str)
    resized = pyqtSignal(str, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(125)
        self.setMouseTracking(True)
        self.document = None
        self.index = None
        self.subject_id = ""
        self.current = 0
        self.selected_id = ""
        self.zoom = 1
        self.drag = None
        self.hit_boxes = []
        self.setToolTip(
            "Click to seek or select a label. Drag either edge of a label to adjust it. Use zoom for short bouts."
        )

    def span(self):
        if self.drag:
            return self.drag[3]
        total = self.index.source.seconds(self.index.source.end_pts) if self.index else 1
        width = total / self.zoom
        current = (
            self.index.source.seconds(self.index.position_pts(self.current)) if self.index else 0
        )
        start = max(0, min(total - width, current - width / 2))
        return start, start + width

    def x_for(self, seconds):
        start, end = self.span()
        return 12 + (seconds - start) / max(1e-9, end - start) * max(1, self.width() - 24)

    def frame_for(self, x, *, boundary=False):
        start, end = self.span()
        seconds = start + (x - 12) / max(1, self.width() - 24) * (end - start)
        v = self.index.source
        pts = v.first_pts + seconds * v.time_base_den / v.time_base_num
        if not self.index.pts:
            return max(v.first_pts, min(v.end_pts if boundary else v.end_pts - 1, round(pts)))
        if boundary:
            points = (*self.index.pts, v.end_pts)
            pos = bisect.bisect_left(points, pts)
            pos = max(0, min(len(points) - 1, pos))
            if pos and abs(points[pos - 1] - pts) < abs(points[pos] - pts):
                pos -= 1
            return pos
        return max(0, min(v.frame_count - 1, bisect.bisect_right(self.index.pts, pts) - 1))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161c24"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.hit_boxes = []
        if not self.document or not self.index:
            painter.setPen(QColor("#b9c5d4"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to begin labeling"
            )
            return
        v = self.index.source
        start, end = self.span()
        painter.fillRect(QRectF(12, 35, self.width() - 24, 45), QColor("#303844"))
        behaviors = {b.id: b for b in self.document.behaviors}
        for interval in self.document.intervals:
            if interval.subject_id != self.subject_id:
                continue
            left, right = v.seconds(interval.start_pts), v.seconds(interval.end_pts)
            if right < start or left > end:
                continue
            x1, x2 = self.x_for(max(start, left)), self.x_for(min(end, right))
            rect = QRectF(x1, 35, max(2, x2 - x1), 45)
            behavior = behaviors.get(interval.behavior_id)
            painter.fillRect(rect, QColor(behavior.color if behavior else "#7c8795"))
            painter.setPen(QColor("#101820"))
            if rect.width() > 50:
                painter.save()
                painter.setClipRect(rect)
                painter.drawText(
                    rect.adjusted(6, 0, -3, 0),
                    Qt.AlignmentFlag.AlignVCenter,
                    behavior.name if behavior else "Unscorable",
                )
                painter.restore()
            if interval.id == self.selected_id:
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawRect(rect.adjusted(1, 1, -1, -1))
            self.hit_boxes.append((rect, interval))
        painter.setPen(QColor("#b9c5d4"))
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            secs = start + fraction * (end - start)
            x = self.x_for(secs)
            painter.drawText(
                QRectF(max(0, min(self.width() - 100, x - 40)), 7, 100, 22),
                Qt.AlignmentFlag.AlignCenter,
                timestamp(secs),
            )
        painter.drawText(
            12,
            106,
            "Gray gaps: unlabeled  •  Labels are per animal",
        )
        painter.setPen(QPen(QColor("#f8fafc"), 2))
        x = self.x_for(v.seconds(self.index.position_pts(self.current)))
        painter.drawLine(int(x), 30, int(x), 86)

    def mousePressEvent(self, event):
        if not self.index or event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        for rect, interval in reversed(self.hit_boxes):
            if rect.adjusted(-4, 0, 4, 0).contains(event.position()):
                span = self.span()
                self.selected.emit(interval.id)
                if abs(x - rect.left()) < 7:
                    self.drag = (interval, "start", x, span)
                elif abs(x - rect.right()) < 7:
                    self.drag = (interval, "end", x, span)
                return
        self.seek_requested.emit(self.frame_for(x))

    def mouseMoveEvent(self, event):
        if self.drag:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            on_edge = any(
                rect.top() <= event.position().y() <= rect.bottom()
                and min(
                    abs(event.position().x() - rect.left()),
                    abs(event.position().x() - rect.right()),
                )
                < 7
                for rect, _ in self.hit_boxes
            )
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if on_edge else Qt.CursorShape.PointingHandCursor
            )

    def mouseReleaseEvent(self, event):
        if self.drag:
            interval, edge, x, _ = self.drag
            boundary = self.frame_for(event.position().x(), boundary=True)
            self.drag = None
            if abs(event.position().x() - x) > 2:
                start = (
                    boundary
                    if edge == "start"
                    else (interval.start_frame if self.index.pts else interval.start_pts)
                )
                end = (
                    boundary
                    if edge == "end"
                    else (interval.end_frame_exclusive if self.index.pts else interval.end_pts)
                )
                self.resized.emit(interval.id, start, end)
            self.update()


class BehaviorLabelerDialog(QDialog):
    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self.document: BehaviorDocument | None = None
        self.index: VideoIndex | None = None
        self.reader = None
        self.job = None
        self.closing = False
        self.current = 0
        self.edit_id = None
        self.draft = False
        self.history = []
        self.future = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._advance)
        self.playing = False
        self.play_clock = QElapsedTimer()
        self.play_origin_s = 0.0
        self.position_timer = QTimer(self)
        self.position_timer.setSingleShot(True)
        self.position_timer.timeout.connect(self._save_position)
        self.setWindowTitle("Label Behaviors — SqueakPose Studio")
        self.resize(1260, 850)
        self.setMinimumSize(960, 700)
        self.setup_error = ""
        try:
            self.project_behaviors, self.setup_complete = load_labeling_catalog(project_root)
        except Exception as exc:
            self.project_behaviors, self.setup_complete = [], False
            self.setup_error = str(exc)
        self.stack = QStackedWidget(self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.stack)
        self._build_ui()
        self.setup_editor = BehaviorDefinitionEditor(self.project_behaviors, first_open=True)
        self.setup_editor.submitted.connect(self._complete_setup)
        self.setup_editor.canceled.connect(self.reject)
        if self.project_behaviors and not self.setup_complete:
            self.setup_editor.description.setText(
                "Review the project's existing behavior list before continuing. "
                "Save your choices once to use them across project videos. Existing annotations remain saved."
            )
        if self.setup_error:
            self.setup_editor.error.setText(
                "Could not load the saved behavior definitions: " + self.setup_error
            )
            self.setup_editor.save_button.setEnabled(False)
        self.setup_page = QWidget()
        setup_layout = QVBoxLayout(self.setup_page)
        self.setup_editor.setMaximumWidth(1100)
        setup_row = QHBoxLayout()
        setup_row.addStretch()
        setup_row.addWidget(self.setup_editor, 1)
        setup_row.addStretch()
        setup_layout.addLayout(setup_row, 1)
        self.stack.addWidget(self.setup_page)
        self.stack.setCurrentWidget(self.workspace if self.setup_complete else self.setup_page)
        self._shortcuts()
        self._refresh()

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        self.workspace = QWidget()
        self.stack.addWidget(self.workspace)
        layout = QVBoxLayout(self.workspace)
        layout.setContentsMargins(18, 18, 18, 14)
        top = QHBoxLayout()
        title = QLabel("Label Behaviors")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        top.addWidget(title)
        top.addStretch()
        self.manage_behaviors = self._button("Behaviors…", self._manage_behaviors, top)
        self.undo_button = self._button("Undo", self._undo, top)
        self.redo_button = self._button("Redo", self._redo, top)
        self.export_json = self._button("Export JSON…", lambda: self._export(False), top)
        self.export_csv = self._button("Export CSV…", lambda: self._export(True), top)
        layout.addLayout(top)
        sources = QHBoxLayout()
        self.videos = QComboBox()
        self.videos.setMinimumWidth(250)
        for entry in list_project_videos(os.path.join(self.project_root, "videos")):
            if entry.target_exists:
                self.videos.addItem(entry.name, entry.target)
        if not self.videos.count():
            self.videos.addItem("No linked videos — use Browse", None)
        sources.addWidget(self.videos, 1)
        self.open_button = self._button("Open Video", self._open_selected, sources)
        self.browse_button = self._button("Browse…", self._browse, sources)
        self.cancel_open = self._button("Cancel Opening", self._cancel_index, sources)
        self.cancel_open.hide()
        layout.addLayout(sources)
        self.source_label = QLabel("Select a project video or browse to a source recording.")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 10, 0)
        self.scene = QGraphicsScene(self)
        self.view = BehaviorFrameView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setMinimumSize(350, 230)
        video_layout.addWidget(self.view, 1)
        self.time_label = QLabel("00:00:00.000 · Frame 0")
        self.time_label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.time_label.setStyleSheet("font-size: 16px;")
        video_layout.addWidget(self.time_label)
        playback = QHBoxLayout()
        self._button("◀ Frame", lambda: self._step(-1), playback)
        self.play_button = self._button("Play", self._toggle_play, playback)
        self._button("Frame ▶", lambda: self._step(1), playback)
        self.speed = QComboBox()
        for speed in (0.25, 0.5, 1, 2):
            self.speed.addItem(f"{speed:g}×", speed)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._speed_changed)
        playback.addWidget(self.speed)
        playback.addStretch()
        self.go_label = QLabel("Go to time (s)")
        playback.addWidget(self.go_label)
        self.go_frame = SourcePositionSpinBox()
        self.go_frame.editingFinished.connect(
            lambda: self.seek(self._editor_position(self.go_frame.value()))
        )
        playback.addWidget(self.go_frame)
        video_layout.addLayout(playback)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.valueChanged.connect(self._scrubbed)
        video_layout.addWidget(self.scrubber)
        splitter.addWidget(video_panel)
        editor = QWidget()
        editor.setMinimumWidth(300)
        form = QVBoxLayout(editor)
        form.setContentsMargins(6, 0, 6, 0)
        form.setSpacing(4)
        subject_row = QHBoxLayout()
        subject_row.addWidget(QLabel("Animal"))
        self.subject = QComboBox()
        self.subject.currentIndexChanged.connect(self._subject_changed)
        subject_row.addWidget(self.subject, 1)
        self._button("Add…", self._add_subject, subject_row)
        form.addLayout(subject_row)
        behavior_row = QHBoxLayout()
        behavior_row.addWidget(QLabel("Behavior"))
        self.behavior = QComboBox()
        self.behavior.currentIndexChanged.connect(self._behavior_changed)
        behavior_row.addWidget(self.behavior, 1)
        form.addLayout(behavior_row)
        self.definition = QLabel()
        self.definition.setWordWrap(True)
        form.addWidget(self.definition)
        self.editor_title = QLabel("New interval")
        self.editor_title.setStyleSheet("font-weight: 600; font-size: 16px;")
        form.addWidget(self.editor_title)
        bounds = QFormLayout()
        self.start_frame = SourcePositionSpinBox()
        self.last_frame = SourcePositionSpinBox()
        self.start_frame.setMinimumHeight(30)
        self.last_frame.setMinimumHeight(30)
        self.first_label = QLabel("Start (seconds)")
        self.last_label = QLabel("Last frame (seconds)")
        bounds.addRow(self.first_label, self.start_frame)
        bounds.addRow(self.last_label, self.last_frame)
        form.addLayout(bounds)
        marks = QHBoxLayout()
        self._button("Mark Start  [I]", self._mark_start, marks)
        self._button("Mark End  [O]", self._mark_end, marks)
        form.addLayout(marks)
        self.interval_time = QLabel()
        self.interval_time.setWordWrap(True)
        form.addWidget(self.interval_time)
        self.start_frame.valueChanged.connect(self._bounds_changed)
        self.last_frame.valueChanged.connect(self._bounds_changed)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Optional annotation notes")
        self.notes.textEdited.connect(lambda _: setattr(self, "draft", True))
        form.addWidget(self.notes)
        actions = QHBoxLayout()
        self.apply_button = self._button("Add Interval", self._apply_interval, actions)
        self._button("New / Clear", self._clear_editor, actions)
        form.addLayout(actions)
        edits = QHBoxLayout()
        self.delete_button = self._button("Delete", self._delete_interval, edits)
        self.split_button = self._button("Split at Playhead", self._split_interval, edits)
        form.addLayout(edits)
        self.time_label.setToolTip(
            "Space: play/pause · ← / →: step · 1–9: select behavior · I/O: mark boundaries. Markers use the timestamps of the displayed frames."
        )
        form.addStretch()
        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(editor)
        editor_scroll.setMinimumWidth(370)
        editor_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        splitter.addWidget(editor_scroll)
        splitter.setSizes([820, 360])
        content_layout.addWidget(splitter, 3)
        timeline_header = QHBoxLayout()
        timeline_header.addWidget(QLabel("Behavior timeline"))
        timeline_header.addStretch()
        timeline_header.addWidget(QLabel("Zoom around playhead"))
        self.zoom = QComboBox()
        self.zoom.addItems(["1×", "5×", "20×", "100×"])
        self.zoom.currentIndexChanged.connect(self._zoom_changed)
        timeline_header.addWidget(self.zoom)
        content_layout.addLayout(timeline_header)
        self.timeline = BehaviorTimeline()
        self.timeline.seek_requested.connect(self.seek)
        self.timeline.selected.connect(self._select_interval)
        self.timeline.resized.connect(self._resize_interval)
        content_layout.addWidget(self.timeline)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Behavior", "Start", "End boundary", "Duration", "Frames", "Notes"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.setMinimumHeight(140)
        content_layout.addWidget(self.table, 1)
        layout.addWidget(self.content, 1)
        self.status = QLabel("Ready. Saved intervals are autosaved to the active project.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _shortcuts(self):
        bindings = {
            "Space": self._toggle_play,
            "Left": lambda: self._step(-1),
            "Right": lambda: self._step(1),
            "I": self._mark_start,
            "O": self._mark_end,
        }
        for number in range(1, 10):
            bindings[str(number)] = lambda n=number: self.behavior.setCurrentIndex(
                min(n - 1, self.behavior.count() - 1)
            )
        for key, callback in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda cb=callback: self._shortcut(cb))
        for key, callback in (
            (QKeySequence.StandardKey.Undo, self._undo),
            (QKeySequence.StandardKey.Redo, self._redo),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda cb=callback: self._shortcut(cb))

    def _shortcut(self, callback):
        if (
            self.document
            and self.stack.currentWidget() is self.workspace
            and not self.job
            and not isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit, QAbstractSpinBox))
        ):
            callback()

    def _error(self, message):
        self._pause()
        QMessageBox.warning(self, "Behavior Labels", str(message))

    def _open_selected(self):
        path = self.videos.currentData()
        if path:
            self.open_video(path)

    def _browse(self):
        patterns = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video for Behavior Labeling", self.project_root, f"Videos ({patterns})"
        )
        if path:
            self.open_video(path)

    def _discard_draft(self):
        if not self.draft:
            return True
        return (
            QMessageBox.question(
                self,
                "Unapplied interval",
                "Discard the interval edits that have not been added or updated?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            == QMessageBox.StandardButton.Discard
        )

    def open_video(self, path):
        if not self.setup_complete:
            self.stack.setCurrentWidget(self.setup_page)
            return
        if self.job or not self._discard_draft() or not self._save_position():
            return
        self._pause()
        self.draft = False
        self.job = VideoIndexJob(path, self.project_root, self)
        self.job.progress.connect(self.status.setText)
        self.job.ready.connect(self._video_ready)
        self.job.failed.connect(self._error)
        self.job.finished.connect(self._index_finished)
        self.cancel_open.setText("Cancel Opening")
        self.cancel_open.show()
        self.open_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.content.setEnabled(False)
        self.export_csv.setEnabled(False)
        self.export_json.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.manage_behaviors.setEnabled(False)
        self.job.start()

    def _cancel_index(self):
        if self.job:
            self.job.requestInterruption()
            self.status.setText("Canceling…")

    def _index_finished(self):
        job = self.job
        self.job = None
        if job:
            job.deleteLater()
        self.cancel_open.hide()
        self.open_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self._refresh()
        if self.closing:
            self._finish_close()

    def _video_ready(self, index):
        if self.closing:
            return
        reader = None
        try:
            document = load_document(self.project_root, index.source.id)
            if document:
                if (
                    document.video.time_base_num,
                    document.video.time_base_den,
                    document.video.first_pts,
                ) != (
                    index.source.time_base_num,
                    index.source.time_base_den,
                    index.source.first_pts,
                ):
                    raise ValueError("Saved video timing differs from the source.")
                if index.pts:
                    for interval in document.intervals:
                        if interval.start_frame is not None and (
                            interval.start_pts,
                            interval.end_pts,
                        ) != (
                            index.boundary(interval.start_frame),
                            index.boundary(interval.end_frame_exclusive),
                        ):
                            raise ValueError(
                                "Saved interval timestamps do not match the source video."
                            )
                elif document.video.end_pts > index.source.end_pts:
                    index = replace(
                        index, source=replace(index.source, end_pts=document.video.end_pts)
                    )
                document.video = index.source
            else:
                document = new_document(self.project_root, index.source)
            project_behaviors, _ = load_labeling_catalog(self.project_root)
            document = reconcile_behavior_catalog(document, project_behaviors)
            if index.pts:
                document = resolve_frame_annotations(document, index.pts)
            reader = BehaviorVideoReader(index)
            if index.pts:
                current = document.resume_frame or 0
                rgb = reader.frame(current)
            else:
                rgb = reader.frame_time(
                    document.resume_pts
                    if document.resume_pts is not None
                    else index.source.first_pts
                )
                current = reader.current_pts
                document.resume_frame = None
                document.resume_pts = current
            save_document(self.project_root, document)
        except Exception as exc:
            if reader:
                reader.close()
            self._error(exc)
            return
        if self.reader:
            self.reader.close()
        self.reader, self.document, self.index = reader, document, index
        self.project_behaviors = project_behaviors
        self.current = current
        self.current_end_pts = reader.current_end_pts
        self.current_at_end = reader.at_end
        self.history, self.future = [], []
        self.edit_id = None
        indexed = bool(index.pts)
        for spin in (self.go_frame, self.start_frame, self.last_frame):
            spin.blockSignals(True)
            spin.setDecimals(0 if indexed else 9)
            spin.setSingleStep(1 if indexed else 0.1)
            spin.setRange(
                0, len(index.pts) - 1 if indexed else index.source.seconds(index.source.end_pts)
            )
            spin.blockSignals(False)
        self.go_label.setText("Go to frame" if indexed else "Go to time (s)")
        self.first_label.setText("First included frame" if indexed else "Start (seconds)")
        self.last_label.setText("Last included frame" if indexed else "Last frame (seconds)")
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, len(index.pts) - 1 if indexed else 1_000_000)
        self.scrubber.blockSignals(False)
        count = f"{len(index.pts):,} frames  ·  " if indexed else ""
        self.source_label.setText(
            f"{index.source.filename}  ·  {count}{timestamp(index.source.seconds(index.source.end_pts))}"
        )
        self.source_label.setToolTip(index.source.source_path)
        self._show_frame(rgb)
        self._refresh()
        self._clear_editor(force=True)
        extra = (
            " Last frame duration estimated from its predecessor."
            if index.source.end_time_method.startswith("estimated")
            else ""
        )
        self.status.setText("Video ready. Intervals autosave when added or updated." + extra)

    def _show_frame(self, rgb):
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(image))
        self.scene.setSceneRect(0, 0, width, height)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scrubber.blockSignals(True)
        v = self.index.source
        slider = (
            self.current
            if self.index.pts
            else round((self.current - v.first_pts) / (v.end_pts - v.first_pts) * 1_000_000)
        )
        self.scrubber.setValue(slider)
        self.scrubber.blockSignals(False)
        self.go_frame.blockSignals(True)
        self.go_frame.setValue(self._editor_value(self.current))
        self.go_frame.blockSignals(False)
        seconds = v.seconds(self.index.position_pts(self.current))
        suffix = (
            f"  ·  Frame {self.current:,} / {len(self.index.pts) - 1:,}" if self.index.pts else ""
        )
        self.time_label.setText(timestamp(seconds) + suffix)
        self.timeline.current = self.current
        self.timeline.update()

    def _editor_value(self, position):
        return position if self.index.pts else self.index.source.seconds(position)

    def _editor_position(self, value):
        if self.index.pts:
            return int(value)
        v = self.index.source
        return v.first_pts + round(value * v.time_base_den / v.time_base_num)

    def _scrubbed(self, value):
        if not self.index:
            return
        v = self.index.source
        self.seek(
            value
            if self.index.pts
            else v.first_pts + round(value / 1_000_000 * (v.end_pts - v.first_pts))
        )

    def _step(self, direction, *, playback=False):
        if not self.reader:
            return
        if self.index.pts:
            self.seek(self.current + direction, playback=playback)
        else:
            self.seek(
                self.current, playback=playback, direction="next" if direction > 0 else "previous"
            )

    def seek(self, frame, *, playback=False, direction="at"):
        if not self.reader:
            return
        if not playback:
            self._pause()
        frame = max(self.index.first_position, min(self.index.last_position, int(frame)))
        try:
            if self.index.pts:
                rgb = self.reader.frame(frame)
                self.current = frame
            else:
                rgb = self.reader.frame_time(frame, direction=direction)
                self.current = self.reader.current_pts
                self.current_end_pts = self.reader.current_end_pts
                self.current_at_end = self.reader.at_end
                self._extend_duration(self.current_end_pts)
            self._show_frame(rgb)
        except Exception as exc:
            self._pause()
            self._error(exc)
            return
        if not playback:
            self.position_timer.start(1200)

    def _extend_duration(self, observed_end):
        if not self.index.pts and observed_end > self.index.source.end_pts:
            self.index = replace(
                self.index, source=replace(self.index.source, end_pts=observed_end)
            )
            self.document.video = self.index.source
            self.timeline.index = self.index
            for spin in (self.go_frame, self.start_frame, self.last_frame):
                spin.setMaximum(self.index.source.seconds(observed_end))

    def _at_end(self):
        return self.current == self.index.last_position if self.index.pts else self.current_at_end

    def _set_resume(self, document):
        document.video = self.index.source
        document.resume_frame = self.current if self.index.pts else None
        document.resume_pts = self.index.position_pts(self.current)

    def _toggle_play(self):
        if not self.reader:
            return
        if self.playing:
            self._pause()
            self.position_timer.start(100)
        else:
            if self._at_end():
                self.seek(self.index.first_position)
            self.position_timer.stop()
            self.playing = True
            self.play_origin_s = self.index.source.seconds(self.index.position_pts(self.current))
            self.play_clock.start()
            self.play_button.setText("Pause")
            self._schedule_frame()

    def _schedule_frame(self):
        if not self.playing:
            return
        delta = (
            self.index.source.seconds(
                self.index.boundary(self.current + 1) if self.index.pts else self.current_end_pts
            )
            - self.play_origin_s
        )
        self.timer.start(
            max(1, round(1000 * delta / self.speed.currentData() - self.play_clock.elapsed()))
        )

    def _speed_changed(self):
        if self.playing:
            self.play_origin_s = self.index.source.seconds(self.index.position_pts(self.current))
            self.play_clock.restart()
            self._schedule_frame()

    def _advance(self):
        if self._at_end():
            self._pause()
            self._save_position()
            return
        self._step(1, playback=True)
        self._schedule_frame()

    def _pause(self):
        self.playing = False
        self.timer.stop()
        self.play_button.setText("Play")

    def _save_position(self):
        self.position_timer.stop()
        if not self.document:
            return True
        candidate = self.document.copy()
        self._set_resume(candidate)
        try:
            save_document(self.project_root, candidate)
        except Exception as exc:
            self._error(f"Could not save. Keep this window open and retry.\n{exc}")
            return False
        self.document = candidate
        return True

    def _commit(self, candidate, *, catalog_behavior=None):
        self._set_resume(candidate)
        try:
            save_document(
                self.project_root,
                candidate,
                catalog_behaviors=catalog_behavior,
            )
        except Exception as exc:
            self._error(f"Change was not saved or applied.\n{exc}")
            return False
        self.history.append((self.document.copy(), self.project_behaviors.copy()))
        self.history = self.history[-100:]
        self.future.clear()
        self.document = candidate
        if catalog_behavior is not None:
            self.project_behaviors = list(catalog_behavior)
        self.draft = False
        self._refresh()
        self.status.setText(
            "Saved to project · "
            + os.path.join("behavior labels", f"{candidate.video.id[:12]}….json")
        )
        return True

    def _refresh(self):
        active = self.document is not None
        self.manage_behaviors.setEnabled(not self.job)
        self.content.setEnabled(active and not self.job)
        self.export_json.setEnabled(active and not self.job)
        self.export_csv.setEnabled(active and not self.job)

        if not active:
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
            return
        subject, behavior = self.subject.currentData(), self.behavior.currentData()
        first_palette = self.behavior.count() == 0
        self.subject.blockSignals(True)
        self.subject.clear()
        for s in self.document.subjects:
            self.subject.addItem(s.name, s.id)
        self.subject.setCurrentIndex(max(0, self.subject.findData(subject)))
        self.subject.blockSignals(False)
        self.behavior.blockSignals(True)
        self.behavior.clear()
        for n, b in enumerate(self.project_behaviors, 1):
            self.behavior.addItem(f"{n}. {b.name}", b.id)
        selected_interval = next((i for i in self.document.intervals if i.id == self.edit_id), None)
        if selected_interval and self.behavior.findData(selected_interval.behavior_id) < 0:
            saved = next(
                (b for b in self.document.behaviors if b.id == selected_interval.behavior_id), None
            )
            name = saved.name if saved else "Unscorable"
            self.behavior.addItem(f"{name} (saved label)", selected_interval.behavior_id)
        preferred = 0
        previous = self.behavior.findData(behavior)
        self.behavior.setCurrentIndex(preferred if first_palette or previous < 0 else previous)
        self.behavior.blockSignals(False)
        self._behavior_changed(mark_dirty=False)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        lookup = {b.id: b for b in self.document.behaviors}
        v = self.document.video
        selected = -1
        for interval in sorted(self.document.intervals, key=lambda i: i.start_pts):
            if interval.subject_id != self.subject.currentData():
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            b = lookup.get(interval.behavior_id)
            values = [
                b.name if b else "Unscorable",
                timestamp(v.seconds(interval.start_pts)),
                timestamp(v.seconds(interval.end_pts)),
                f"{v.seconds(interval.end_pts) - v.seconds(interval.start_pts):.3f} s",
                f"{interval.start_frame}–{interval.end_frame_exclusive - 1}"
                if interval.start_frame is not None
                else "—",
                interval.notes,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, interval.id)
                if col == 0:
                    item.setForeground(QColor(b.color if b else "#aeb9c4"))
                self.table.setItem(row, col, item)
            if interval.id == self.edit_id:
                selected = row
        if selected >= 0:
            self.table.selectRow(selected)
        self.table.blockSignals(False)
        self.timeline.document = self.document
        self.timeline.index = self.index
        self.timeline.subject_id = self.subject.currentData()
        self.timeline.current = self.current
        self.timeline.selected_id = self.edit_id or ""
        self.timeline.update()
        self.undo_button.setEnabled(bool(self.history) and not self.job)
        self.redo_button.setEnabled(bool(self.future) and not self.job)
        self.delete_button.setEnabled(self.edit_id is not None)
        self.split_button.setEnabled(self.edit_id is not None)

    def _subject_changed(self):
        if self.document:
            if self.draft:
                # Keep drafts tied to their original subject until explicitly cleared.
                self.subject.blockSignals(True)
                self.subject.setCurrentIndex(self.subject.findData(self.timeline.subject_id))
                self.subject.blockSignals(False)
                self._error("Add/update or clear the current interval before changing animals.")
                return
            self._clear_editor(force=True)
            self._refresh()

    def _behavior_changed(self, _index=0, *, mark_dirty=True):
        if not self.document:
            return
        b = next((b for b in self.document.behaviors if b.id == self.behavior.currentData()), None)
        self.definition.setText(
            b.definition or "Add a definition to keep labeling consistent."
            if b
            else "Previously saved Unscorable annotation."
        )
        if mark_dirty and self.edit_id:
            self.draft = True

    def _bounds_changed(self):
        if not self.index:
            return
        self.draft = True
        a = self._editor_position(self.start_frame.value())
        b = self._editor_position(self.last_frame.value())
        v = self.index.source
        start = self.index.boundary(a) if self.index.pts else a
        end = self.index.boundary(b + 1) if self.index.pts else b
        self.interval_time.setText(
            f"{timestamp(v.seconds(start))} → {timestamp(v.seconds(end))}\nThe last selected frame is included."
        )

    def _mark_start(self):
        if self.document:
            self.start_frame.setValue(self._editor_value(self.current))
            if self.last_frame.value() < self._editor_value(self.current):
                self.last_frame.setValue(self._editor_value(self.current))
            self.draft = True

    def _mark_end(self):
        if self.document:
            self._pause()
            self.last_frame.setValue(self._editor_value(self.current))
            self.draft = True

    def _clear_editor(self, _checked=False, *, force=False):
        if not force and not self._discard_draft():
            return
        self.edit_id = None
        self.start_frame.setValue(self._editor_value(self.current))
        self.last_frame.setValue(self._editor_value(self.current))
        self.notes.clear()
        self.editor_title.setText("New interval")
        self.apply_button.setText("Add Interval")
        self._bounds_changed()
        self.draft = False
        self._refresh()

    def _make_interval(self, start, end, *, identity=None, original=None):
        v = self.index.source
        if self.index.pts:
            if not 0 <= start < end <= len(self.index.pts):
                raise ValueError("The end must be at or after the start, inside the video.")
            start_pts, end_pts = self.index.boundary(start), self.index.boundary(end)
            start_frame, end_frame = start, end
        else:
            if not v.first_pts <= start < end <= v.end_pts:
                raise ValueError("The end must be at or after the start, inside the video.")
            start_pts, end_pts = start, end
            start_frame, end_frame = None, None
        if original:
            return replace(
                original,
                start_frame=start_frame,
                end_frame_exclusive=end_frame,
                start_pts=start_pts,
                end_pts=end_pts,
                id=identity or original.id,
                end_time_method=self.reader.end_methods.get(
                    end_pts,
                    original.end_time_method if end_pts == original.end_pts else "source_timestamp",
                ),
            )
        behavior_id = self.behavior.currentData()
        if self.behavior.currentIndex() < 0 or (behavior_id is None and not self.edit_id):
            raise ValueError("Choose a behavior before adding an interval.")
        return Interval(
            identity or new_id(),
            self.subject.currentData(),
            behavior_id,
            "behavior" if behavior_id else "unscorable",
            start_frame,
            end_frame,
            start_pts,
            end_pts,
            self.notes.text(),
            self.reader.end_methods.get(end_pts, "source_timestamp"),
        )

    def _apply_interval(self):
        self._pause()
        try:
            candidate = self.document.copy()
            start = self._editor_position(self.start_frame.value())
            last = self._editor_position(self.last_frame.value())
            if last < start:
                raise ValueError("The last frame must be at or after the start.")
            if self.index.pts:
                end = last + 1
            else:
                start, _ = self.reader.bounds_at(start)
                _, end = self.reader.bounds_at(last)
                self._extend_duration(end)
                candidate.video = self.index.source
            interval = self._make_interval(start, end, identity=self.edit_id)
            candidate.put(interval)
        except ValueError as exc:
            self._error(exc)
            return
        if self._commit(candidate):
            self._clear_editor(force=True)

    def _table_selected(self):
        items = self.table.selectedItems()
        if items:
            self._select_interval(items[0].data(Qt.ItemDataRole.UserRole))

    def _select_interval(self, identity):
        if not self._discard_draft():
            self._refresh()
            return
        interval = next(i for i in self.document.intervals if i.id == identity)
        self.edit_id = identity
        self._refresh()
        self.behavior.setCurrentIndex(self.behavior.findData(interval.behavior_id))
        start = interval.start_frame if self.index.pts else interval.start_pts
        if self.index.pts:
            last = interval.end_frame_exclusive - 1
        else:
            try:
                last, _ = self.reader.bounds_at(interval.end_pts - 1)
            except Exception as exc:
                self._error(exc)
                return
        self.start_frame.setValue(self._editor_value(start))
        self.last_frame.setValue(self._editor_value(last))
        self.notes.setText(interval.notes)
        self.editor_title.setText("Edit interval")
        self.apply_button.setText("Update Interval")
        self._bounds_changed()
        self.draft = False
        self.seek(start)
        self._refresh()

    def _resize_interval(self, identity, start, end):
        if self.edit_id != identity:
            return
        try:
            original = next(i for i in self.document.intervals if i.id == identity)
            if not self.index.pts:
                start, _ = self.reader.bounds_at(start)
                if end >= self.index.source.end_pts:
                    _, end = self.reader.bounds_at(self.index.source.end_pts - 1)
                else:
                    end, _ = self.reader.bounds_at(end)
            candidate = self.document.copy()
            candidate.put(self._make_interval(start, end, original=original))
        except ValueError as exc:
            self._error(exc)
            return
        if self._commit(candidate):
            self._select_interval(identity)

    def _delete_interval(self):
        if self.edit_id:
            candidate = self.document.copy()
            candidate.intervals = [i for i in candidate.intervals if i.id != self.edit_id]
            if self._commit(candidate):
                self._clear_editor(force=True)

    def _split_interval(self):
        if not self.edit_id or not self._discard_draft():
            return
        original = next(i for i in self.document.intervals if i.id == self.edit_id)
        start = original.start_frame if self.index.pts else original.start_pts
        end = original.end_frame_exclusive if self.index.pts else original.end_pts
        if not start < self.current < end:
            self._error(
                "Move the playhead inside the selected interval. The current frame will begin the second interval."
            )
            return
        candidate = self.document.copy()
        candidate.put(self._make_interval(start, self.current, original=original))
        candidate.put(self._make_interval(self.current, end, original=original, identity=new_id()))
        if self._commit(candidate):
            self._clear_editor(force=True)

    def _restore(self, source, destination):
        if not source or not self._discard_draft():
            return
        previous_document, previous_catalog = source[-1]
        candidate = previous_document.copy()
        self._set_resume(candidate)
        try:
            save_document(
                self.project_root,
                candidate,
                catalog_behaviors=previous_catalog
                if previous_catalog != self.project_behaviors
                else None,
            )
        except Exception as exc:
            self._error(exc)
            return
        destination.append((self.document.copy(), self.project_behaviors.copy()))
        source.pop()
        self.project_behaviors = previous_catalog.copy()
        self.document = candidate
        self._clear_editor(force=True)
        self.status.setText("History restored and saved.")

    def _undo(self):
        self._restore(self.history, self.future)

    def _redo(self):
        self._restore(self.future, self.history)

    def _add_subject(self):
        if self.draft:
            self._error("Add/update or clear the draft interval before adding an animal.")
            return
        name, ok = QInputDialog.getText(
            self, "Add Animal", "Animal name (identity is assigned manually):"
        )
        if ok and name.strip():
            candidate = self.document.copy()
            candidate.subjects.append(Subject(new_id(), name.strip()))
            self._commit(candidate)

    def _complete_setup(self, behaviors):
        if self.setup_error:
            return
        try:
            save_behavior_catalog(self.project_root, behaviors)
        except Exception as exc:
            self.setup_editor.error.setText(f"Could not save the behavior list: {exc}")
            return
        self.project_behaviors = list(behaviors)
        self.setup_complete = True
        self.stack.setCurrentWidget(self.workspace)
        self.status.setText("Behaviors saved. Choose a video to begin labeling.")
        self._refresh()

    def _manage_behaviors(self):
        self._pause()
        if self.job or not self._discard_draft():
            return
        if self.document:
            behaviors = self.project_behaviors
            protected = {i.behavior_id for i in self.document.intervals if i.behavior_id}
        else:
            try:
                behaviors, _ = load_labeling_catalog(self.project_root)
            except Exception as exc:
                self._error(exc)
                return
            protected = set()
        dialog = QDialog(self)
        dialog.setWindowTitle("Behavior Definitions")
        dialog.resize(850, 630)
        layout = QVBoxLayout(dialog)
        editor = BehaviorDefinitionEditor(behaviors, protected_ids=protected)
        editor.description.setText(
            "Changes update the project behavior list. Older labels remain saved even if a behavior is removed from the list. "
            "Behaviors used by this video's saved intervals cannot be removed."
            if self.document
            else "These choices will be used when labeling any video in this project. Older labels remain saved."
        )
        layout.addWidget(editor)
        editor.canceled.connect(dialog.reject)

        def save(edited):
            if self.document:
                candidate = reconcile_behavior_catalog(self.document, edited)
                if not self._commit(candidate, catalog_behavior=edited):
                    return
                self._clear_editor(force=True)
            else:
                try:
                    save_behavior_catalog(self.project_root, edited)
                except Exception as exc:
                    editor.error.setText(f"Could not save the behavior list: {exc}")
                    return
            self.project_behaviors = list(edited)
            dialog.accept()

        editor.submitted.connect(save)
        dialog.exec()

    def _zoom_changed(self):
        self.timeline.zoom = (1, 5, 20, 100)[self.zoom.currentIndex()]
        self.timeline.update()

    def _export(self, csv_format):
        self._pause()
        if self.draft:
            self._error("Add/update or clear the draft interval before exporting.")
            return
        suffix = "csv" if csv_format else "json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Behavior Labels",
            os.path.join(
                self.project_root,
                f"{os.path.splitext(self.document.video.filename)[0]}_behaviors.{suffix}",
            ),
            f"{suffix.upper()} (*.{suffix})",
        )
        if not path:
            return
        managed = os.path.realpath(os.path.join(self.project_root, "behavior labels"))
        try:
            inside_managed = os.path.commonpath((managed, os.path.realpath(path))) == managed
        except ValueError:
            inside_managed = False
        if inside_managed:
            self._error("Choose an export folder outside the managed behavior labels directory.")
            return
        if os.path.realpath(path) == os.path.realpath(self.document.video.source_path):
            self._error("Choose an export destination other than the source video.")
            return
        try:
            export_document(self.document, path, csv_format=csv_format)
            self.status.setText(f"Exported {path}")
        except Exception as exc:
            self._error(exc)

    def reject(self):
        if not self._discard_draft() or not self._save_position():
            return
        self._pause()
        if self.job:
            self.closing = True
            self._cancel_index()
            return
        self._finish_close()

    def _finish_close(self):
        self.position_timer.stop()
        if self.reader:
            self.reader.close()
            self.reader = None
        super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.reader:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
