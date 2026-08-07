"""Project-local image corpus and DINO distillation dialog."""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

from PyQt6.QtCore import QProcess, Qt, QTimer
from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from squeakpose.project.paths import ProjectPaths
from squeakpose.workers.process import (
    remove_file_quietly as _remove_file_quietly,
)
from squeakpose.workers.process import (
    request_qprocess_stop,
)
from squeakpose.workers.process import (
    shutdown_qprocess as _shutdown_qprocess,
)
from squeakpose_core import stable_path_id, staging_path_for
from ui_style import (
    ThemedComboBox,
    style_combo_popup,
    train_dialog_stylesheet,
)

APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


def _project_paths(project_root: str) -> ProjectPaths:
    return ProjectPaths.from_root(project_root)


def _distillation_sample_count(
    total_frames: int,
    stride: int,
    max_frames: int = 0,
) -> int:
    total = max(0, int(total_frames))
    step = max(1, int(stride))
    count = (total + step - 1) // step
    return min(count, int(max_frames)) if int(max_frames) > 0 else count


def _refresh_qt_style(widget: QWidget | None) -> None:
    if widget is None:
        return
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class DistillationDialog(QDialog):
    """Prepare a project image corpus and launch DINO distillation."""

    ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    TASK_DEFAULTS = {
        "pose": {
            "label": "Keypoints",
            "student": "ultralytics/yolo26s-pose.pt",
            "run_name": "dinov3-pose",
        },
        "segment": {
            "label": "Segmentation",
            "student": "ultralytics/yolo26s-seg.pt",
            "run_name": "dinov3-segmentation",
        },
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("DINO Distillation")
        self.resize(1050, 760)
        self.setMinimumSize(780, 560)

        self.app_base_dir = os.path.abspath(getattr(parent, "app_base_dir", APP_BASE_DIR))
        self.project_root = os.path.abspath(getattr(parent, "project_root", self.app_base_dir))
        self.paths = _project_paths(self.project_root)
        self.video_paths: list[str] = []
        self.process: Optional[QProcess] = None
        self.cancel_requested = False
        self._current_task = "pose"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Project Distillation")
        title.setObjectName("TrainPanelTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("TrainStatusLabel")
        self.status_label.setProperty("tone", "idle")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        intro = QLabel(
            "Build the unlabeled image corpus only when you are ready, then run Lightly Train "
            "against that corpus. No frames are generated automatically."
        )
        intro.setWordWrap(True)
        intro.setObjectName("TrainHintLabel")
        layout.addWidget(intro)

        tabs = QTabWidget()
        tabs.addTab(self._build_dataset_tab(), "1. Prepare Images")
        tabs.addTab(self._build_run_tab(), "2. Run Distillation")
        layout.addWidget(tabs, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)

        self.run_btn = QPushButton("Start Distillation")
        self.run_btn.clicked.connect(self._start_distillation)
        button_box.addButton(self.run_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self.cancel_btn = QPushButton("Cancel Distillation")
        self.cancel_btn.clicked.connect(self._cancel_process)
        self.cancel_btn.setEnabled(False)
        button_box.addButton(self.cancel_btn, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(button_box)

        self.setStyleSheet(train_dialog_stylesheet())
        self._refresh_dataset_summary()
        self._update_output_path()

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source_label = QLabel("Source videos")
        source_label.setObjectName("TrainPanelTitle")
        layout.addWidget(source_label)

        self.video_list = QListWidget()
        self.video_list.setAlternatingRowColors(True)
        self.video_list.setMinimumHeight(110)
        self.video_list.setToolTip("Only videos listed here will be sampled")
        layout.addWidget(self.video_list)

        video_buttons = QHBoxLayout()
        add_btn = QPushButton("Select Project Videos...")
        add_btn.clicked.connect(self._add_videos)
        video_buttons.addWidget(add_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_videos)
        video_buttons.addWidget(remove_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_videos)
        video_buttons.addWidget(clear_btn)
        video_buttons.addStretch(1)
        layout.addLayout(video_buttons)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        data_row = QHBoxLayout()
        self.data_dir_edit = QLineEdit(self.paths["distillation_unlabeled_images"])
        data_row.addWidget(self.data_dir_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_data_dir)
        data_row.addWidget(browse_btn)
        form.addRow("Image corpus:", data_row)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 100000)
        self.stride_spin.setValue(30)
        self.stride_spin.setSuffix(" frames")
        form.addRow("Sample every:", self.stride_spin)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 10000000)
        self.max_frames_spin.setSpecialValueText("No limit")
        self.max_frames_spin.setValue(10000)
        form.addRow("Maximum per video:", self.max_frames_spin)

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(50, 100)
        self.jpeg_quality_spin.setValue(95)
        form.addRow("JPEG quality:", self.jpeg_quality_spin)
        layout.addLayout(form)

        self.dataset_summary = QLabel("")
        self.dataset_summary.setWordWrap(True)
        self.dataset_summary.setObjectName("TrainHintLabel")
        layout.addWidget(self.dataset_summary)

        create_row = QHBoxLayout()
        self.create_dataset_btn = QPushButton("Create / Update Image Corpus")
        self.create_dataset_btn.clicked.connect(self._create_dataset)
        create_row.addWidget(self.create_dataset_btn)
        refresh_btn = QPushButton("Refresh Count")
        refresh_btn.clicked.connect(self._refresh_dataset_summary)
        create_row.addWidget(refresh_btn)
        create_row.addStretch(1)
        layout.addLayout(create_row)
        layout.addStretch(1)
        return tab

    def _build_run_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.task_combo = ThemedComboBox()
        self.task_combo.addItem("Keypoints", "pose")
        self.task_combo.addItem("Segmentation", "segment")
        style_combo_popup(self.task_combo.view())
        form.addRow("Task:", self.task_combo)

        self.run_name_edit = QLineEdit(self.TASK_DEFAULTS["pose"]["run_name"])
        self.run_name_edit.textChanged.connect(self._update_output_path)
        form.addRow("Run name:", self.run_name_edit)

        self.student_edit = QLineEdit(self.TASK_DEFAULTS["pose"]["student"])
        form.addRow("Student model:", self.student_edit)

        self.teacher_edit = QLineEdit("dinov3/vitb16")
        form.addRow("Teacher model:", self.teacher_edit)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(300)
        form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(64)
        form.addRow("Batch size:", self.batch_spin)

        self.precision_combo = ThemedComboBox()
        self.precision_combo.addItems(["bf16-mixed", "16-mixed", "32-true"])
        style_combo_popup(self.precision_combo.view())
        form.addRow("Precision:", self.precision_combo)

        self.overwrite_check = QCheckBox("Allow replacing an existing run directory")
        form.addRow("Overwrite:", self.overwrite_check)

        self.output_path_label = QLabel("")
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setObjectName("TrainHintLabel")
        form.addRow("Output:", self.output_path_label)
        layout.addLayout(form)

        output_title = QLabel("Distillation Output")
        output_title.setObjectName("TrainPanelTitle")
        layout.addWidget(output_title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setMaximumBlockCount(12000)
        self.log_view.setPlaceholderText("Distillation output will appear here.")
        terminal_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        terminal_font.setPointSize(10)
        self.log_view.setFont(terminal_font)
        layout.addWidget(self.log_view, 1)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        return tab

    def _selected_task(self) -> str:
        task = self.task_combo.currentData()
        return task if task in self.TASK_DEFAULTS else "pose"

    def _on_task_changed(self, _index: int) -> None:
        old_defaults = self.TASK_DEFAULTS[self._current_task]
        new_task = self._selected_task()
        new_defaults = self.TASK_DEFAULTS[new_task]
        if (
            not self.run_name_edit.text().strip()
            or self.run_name_edit.text() == old_defaults["run_name"]
        ):
            self.run_name_edit.setText(new_defaults["run_name"])
        if (
            not self.student_edit.text().strip()
            or self.student_edit.text() == old_defaults["student"]
        ):
            self.student_edit.setText(new_defaults["student"])
        self._current_task = new_task
        self._update_output_path()

    @staticmethod
    def _student_task_mismatch(student: str, task: str) -> bool:
        model_name = os.path.basename(student).lower()
        if task == "segment":
            return "-pose" in model_name
        return "-seg" in model_name or "segment" in model_name

    def _set_status(self, text: str, tone: str = "idle") -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("tone", tone)
        _refresh_qt_style(self.status_label)

    def _add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select project videos",
            self.paths["videos"],
            "Videos (*.mp4 *.mov *.avi *.mkv *.m4v *.mpg *.mpeg *.wmv);;All files (*)",
        )
        for path in paths:
            normalized = os.path.abspath(path)
            if normalized in self.video_paths:
                continue
            self.video_paths.append(normalized)
            self.video_list.addItem(normalized)

    def _remove_selected_videos(self) -> None:
        rows = sorted(
            {self.video_list.row(item) for item in self.video_list.selectedItems()}, reverse=True
        )
        for row in rows:
            self.video_list.takeItem(row)
            del self.video_paths[row]

    def _clear_videos(self) -> None:
        self.video_paths.clear()
        self.video_list.clear()

    def _browse_data_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select unlabeled image corpus",
            self.data_dir_edit.text().strip() or self.paths["distillation"],
        )
        if path:
            self.data_dir_edit.setText(os.path.abspath(path))
            self._refresh_dataset_summary()

    def _image_count(self, root: str) -> int:
        if not os.path.isdir(root):
            return 0
        count = 0
        try:
            for _dirpath, _dirnames, names in os.walk(root):
                count += sum(name.lower().endswith(self.IMAGE_EXTENSIONS) for name in names)
        except OSError:
            return 0
        return count

    def _refresh_dataset_summary(self) -> None:
        data_dir = self.data_dir_edit.text().strip()
        count = self._image_count(data_dir)
        self.dataset_summary.setText(
            f"Current corpus: {count:,} image(s)\n"
            f"Frames are written as JPEG files with source-specific names; existing files are skipped."
        )

    def _probe_selected_videos(self) -> tuple[list[tuple[str, int, int]], list[str]]:
        stride = self.stride_spin.value()
        maximum = self.max_frames_spin.value()
        probes: list[tuple[str, int, int]] = []
        errors: list[str] = []
        for path in self.video_paths:
            cap = _cv2.VideoCapture(path)
            try:
                if not cap or not cap.isOpened():
                    errors.append(f"{os.path.basename(path)}: could not open")
                    continue
                total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total <= 0:
                    errors.append(f"{os.path.basename(path)}: frame count unavailable")
                    continue
                probes.append((path, total, _distillation_sample_count(total, stride, maximum)))
            finally:
                if cap:
                    cap.release()
        return probes, errors

    def _create_dataset(self) -> None:
        if _cv2 is None:
            QMessageBox.warning(
                self, "OpenCV missing", "Run uv sync --locked to restore project dependencies."
            )
            return
        if not self.video_paths:
            QMessageBox.information(
                self, "Select videos", "Select one or more project videos first."
            )
            return

        data_text = self.data_dir_edit.text().strip()
        if not data_text:
            QMessageBox.warning(self, "Image corpus required", "Choose an image corpus directory.")
            return
        data_dir = os.path.abspath(data_text)

        probes, errors = self._probe_selected_videos()
        if not probes:
            QMessageBox.warning(
                self,
                "No readable videos",
                "None of the selected videos could be read.\n\n" + "\n".join(errors),
            )
            return

        estimated = sum(item[2] for item in probes)
        details = (
            f"This will sample up to {estimated:,} frame(s) from {len(probes)} video(s) into:\n"
            f"{data_dir}\n\n"
            f"Sampling interval: every {self.stride_spin.value():,} frame(s)\n"
            f"Existing matching images will be skipped."
        )
        if errors:
            details += f"\n\n{len(errors)} video(s) could not be read and will be skipped."
        decision = QMessageBox.question(
            self,
            "Create distillation image corpus?",
            details,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return

        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self, "Corpus error", f"Could not create the image directory:\n{exc}"
            )
            return

        progress = QProgressDialog("Preparing image corpus...", "Cancel", 0, estimated, self)
        progress.setWindowTitle("Creating Distillation Corpus")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)

        stride = self.stride_spin.value()
        maximum = self.max_frames_spin.value()
        quality = self.jpeg_quality_spin.value()
        handled = 0
        saved = 0
        skipped = 0
        failed: list[str] = []

        for path, total, sample_count in probes:
            if progress.wasCanceled():
                break
            cap = _cv2.VideoCapture(path)
            source_id = stable_path_id(path)
            base = re.sub(
                r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(path))[0]
            ).strip("._")
            base = base or "video"
            try:
                for sample_number, frame_idx in enumerate(range(0, total, stride)):
                    if maximum > 0 and sample_number >= maximum:
                        break
                    if progress.wasCanceled():
                        break

                    out_path = os.path.join(data_dir, f"{base}_{source_id}_f{frame_idx:09d}.jpg")
                    if os.path.exists(out_path):
                        skipped += 1
                    else:
                        cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            failed.append(
                                f"{os.path.basename(path)} frame {frame_idx}: read failed"
                            )
                        else:
                            staged_path = staging_path_for(out_path)
                            try:
                                write_ok = _cv2.imwrite(
                                    staged_path,
                                    frame,
                                    [_cv2.IMWRITE_JPEG_QUALITY, quality],
                                )
                                if not write_ok:
                                    raise OSError("OpenCV could not encode the frame")
                                os.replace(staged_path, out_path)
                                saved += 1
                            except Exception as exc:
                                _remove_file_quietly(staged_path)
                                failed.append(f"{os.path.basename(path)} frame {frame_idx}: {exc}")

                    handled += 1
                    progress.setValue(handled)
                    progress.setLabelText(
                        f"{os.path.basename(path)}: {sample_number + 1:,}/{sample_count:,} samples"
                    )
                    QApplication.processEvents()
            finally:
                cap.release()

        canceled = progress.wasCanceled()
        progress.setValue(min(handled, estimated))
        progress.close()
        self._refresh_dataset_summary()

        summary = f"Saved {saved:,} new image(s); skipped {skipped:,} existing image(s)."
        if canceled:
            summary += "\n\nExtraction was canceled; images already written were kept."
        if failed:
            summary += f"\n\n{len(failed):,} frame(s) failed. First issues:\n" + "\n".join(
                failed[:8]
            )
        QMessageBox.information(self, "Image corpus updated", summary)

    def _run_name(self) -> str:
        return self.run_name_edit.text().strip()

    def _output_dir(self) -> str:
        return os.path.join(self.paths["distillation_runs"], self._run_name())

    def _update_output_path(self) -> None:
        if hasattr(self, "output_path_label"):
            self.output_path_label.setText(self._output_dir())

    def _append_log(self, text: str) -> None:
        clean = self.ANSI_ESCAPE_RE.sub("", text)
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(clean)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _start_distillation(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(
                self, "Distillation running", "A distillation run is already active."
            )
            return

        data_text = self.data_dir_edit.text().strip()
        data_dir = os.path.abspath(data_text) if data_text else ""
        run_name = self._run_name()
        student = self.student_edit.text().strip()
        teacher = self.teacher_edit.text().strip()
        task = self._selected_task()
        script_path = os.path.join(self.app_base_dir, "distillation", "distiller.py")

        if not data_dir or not os.path.isdir(data_dir) or self._image_count(data_dir) == 0:
            QMessageBox.warning(
                self,
                "Image corpus required",
                "Choose a directory containing unlabeled images, or create the corpus in the first tab.",
            )
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
            QMessageBox.warning(
                self,
                "Invalid run name",
                "Use letters, numbers, periods, underscores, or hyphens; start with a letter or number.",
            )
            return
        if not student or not teacher:
            QMessageBox.warning(
                self, "Model required", "Both student and teacher model values are required."
            )
            return
        if self._student_task_mismatch(student, task):
            task_label = self.TASK_DEFAULTS[task]["label"]
            QMessageBox.warning(
                self,
                "Student model task mismatch",
                f"The selected task is {task_label}, but the student model appears to use a different head.\n\n"
                f"Choose a compatible model or switch the task.",
            )
            return
        if not os.path.isfile(script_path):
            QMessageBox.critical(self, "Distiller missing", f"Could not find:\n{script_path}")
            return

        out_dir = self._output_dir()
        if os.path.exists(out_dir) and not self.overwrite_check.isChecked():
            QMessageBox.warning(
                self,
                "Run directory exists",
                f"The output directory already exists:\n{out_dir}\n\n"
                "Choose a new run name or explicitly allow overwrite.",
            )
            return

        args = [
            "-u",
            script_path,
            "--project-root",
            self.project_root,
            "--data-dir",
            data_dir,
            "--run-name",
            run_name,
            "--model",
            student,
            "--task",
            task,
            "--teacher",
            teacher,
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch-size",
            str(self.batch_spin.value()),
            "--precision",
            self.precision_combo.currentText(),
        ]
        if self.overwrite_check.isChecked():
            args.append("--overwrite")

        self.log_view.clear()
        self._append_log(
            f"Project: {self.project_root}\n"
            f"Images: {data_dir} ({self._image_count(data_dir):,} files)\n"
            f"Task: {self.TASK_DEFAULTS[task]['label']}\n"
            f"Student: {student}\n"
            f"Output: {out_dir}\n\n"
        )

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(self.app_base_dir)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._finish_process)
        process.errorOccurred.connect(self._process_error)

        self.process = process
        self.cancel_requested = False
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.create_dataset_btn.setEnabled(False)
        self._set_status("Launching", "running")
        process.start()
        if not process.waitForStarted(1000):
            self._append_log(f"Could not start distillation: {process.errorString()}\n")
            self._finish_process(1, QProcess.ExitStatus.CrashExit)

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._append_log(text)
            self._set_status("Running", "running")

    def _process_error(self, _error) -> None:
        if self.process is not None:
            self._append_log(f"\nProcess error: {self.process.errorString()}\n")

    def _cancel_process(self) -> None:
        if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.cancel_requested = True
        self._set_status("Canceling", "canceled")
        self._append_log("\nCancel requested. Stopping distillation...\n")
        request_qprocess_stop(
            self.process,
            schedule=QTimer.singleShot,
            force_kill=self._kill_process_if_running,
            kill_after_ms=5000,
        )

    def _kill_process_if_running(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _finish_process(self, exit_code: int, exit_status) -> None:
        if self.process is None:
            return
        self._read_process_output()
        canceled = self.cancel_requested
        self.process = None
        self.cancel_requested = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.create_dataset_btn.setEnabled(True)

        if canceled:
            self._set_status("Canceled", "canceled")
            self._append_log("\nDistillation canceled.\n")
            return
        if exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0:
            self._set_status("Failed", "failed")
            self._append_log(f"\nDistillation failed with exit code {exit_code}.\n")
            QMessageBox.critical(self, "Distillation failed", "Review the output log for details.")
            return

        self._set_status("Complete", "complete")
        self._append_log(f"\nDistillation complete. Output: {self._output_dir()}\n")
        QMessageBox.information(
            self,
            "Distillation complete",
            "The distilled model export is now available in the Train Model dialog.",
        )

    def _confirm_stop_for_close(self) -> bool:
        if self.process is None or self.process.state() == QProcess.ProcessState.NotRunning:
            return True
        answer = QMessageBox.question(
            self,
            "Cancel distillation?",
            "Distillation is still running. Cancel it and close this dialog?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        _shutdown_qprocess(self.process)
        self.process = None
        return True

    def reject(self) -> None:
        if self._confirm_stop_for_close():
            super().reject()

    def closeEvent(self, event) -> None:
        if not self._confirm_stop_for_close():
            event.ignore()
            return
        super().closeEvent(event)
