"""Qt-free planning for distillation corpora and worker runs."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from squeakpose.core import stable_path_id, staging_path_for
from squeakpose.project.distillation import normalize_distillation_task

DISTILLATION_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
DISTILLATION_TASK_DEFAULTS = {
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
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class DistillationPlanError(ValueError):
    """Stable validation failure suitable for presentation by a UI."""

    def __init__(self, code: str, title: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.title = title
        self.message = message


@dataclass(frozen=True)
class DistillationCorpusPlan:
    """A deterministic summary of already-probed video inputs."""

    videos: tuple[tuple[str, int, int], ...]
    stride: int
    maximum_per_video: int
    estimated_samples: int


class DistillationVideoReader(Protocol):
    """Minimal random-access video reader used by corpus extraction."""

    def read_frame(self, frame_index: int) -> Any | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DistillationCorpusProgress:
    source_path: str
    sample_number: int
    source_samples: int
    frame_index: int
    handled: int
    estimated_samples: int


@dataclass(frozen=True)
class DistillationCorpusResult:
    saved: int
    skipped: int
    failures: tuple[str, ...]
    canceled: bool
    handled: int
    estimated_samples: int

    @property
    def failed(self) -> int:
        return len(self.failures)


@dataclass(frozen=True)
class DistillationRunPlan:
    """Validated inputs and exact command arguments for one distiller run."""

    project_root: str
    data_dir: str
    output_dir: str
    run_name: str
    task: str
    task_label: str
    student: str
    teacher: str
    image_count: int
    program: str
    arguments: tuple[str, ...]
    working_directory: str


def count_distillation_images(root: str) -> int:
    """Count supported images below ``root``, tolerating unreadable directories."""

    if not os.path.isdir(root):
        return 0
    count = 0
    try:
        for _dirpath, _dirnames, names in os.walk(root):
            count += sum(name.lower().endswith(DISTILLATION_IMAGE_EXTENSIONS) for name in names)
    except OSError:
        return 0
    return count


def distillation_sample_count(
    total_frames: int,
    stride: int,
    maximum_per_video: int = 0,
) -> int:
    """Return how many frames the corpus plan samples from one video."""

    total = max(0, int(total_frames))
    step = max(1, int(stride))
    count = (total + step - 1) // step
    maximum = max(0, int(maximum_per_video))
    return min(count, maximum) if maximum else count


def plan_distillation_corpus(
    video_frame_counts: Iterable[tuple[str, int]],
    *,
    stride: int,
    maximum_per_video: int = 0,
) -> DistillationCorpusPlan:
    """Plan sampling after a UI or adapter has probed the source videos."""

    step = max(1, int(stride))
    maximum = max(0, int(maximum_per_video))
    videos: list[tuple[str, int, int]] = []
    for path, total_frames in video_frame_counts:
        total = max(0, int(total_frames))
        samples = distillation_sample_count(total, step, maximum)
        videos.append((os.path.abspath(path), total, samples))
    return DistillationCorpusPlan(
        videos=tuple(videos),
        stride=step,
        maximum_per_video=maximum,
        estimated_samples=sum(item[2] for item in videos),
    )


def distillation_frame_filename(source_path: str, frame_index: int) -> str:
    """Return the stable, source-specific filename used by existing corpora."""

    base = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        os.path.splitext(os.path.basename(source_path))[0],
    ).strip("._")
    return f"{base or 'video'}_{stable_path_id(source_path)}_f{int(frame_index):09d}.jpg"


def _remove_staged_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def build_distillation_corpus(
    plan: DistillationCorpusPlan,
    *,
    data_dir: str,
    jpeg_quality: int,
    open_video: Callable[[str], DistillationVideoReader],
    write_image: Callable[[str, Any, int], bool],
    is_canceled: Callable[[], bool] = lambda: False,
    on_progress: Callable[[DistillationCorpusProgress], None] | None = None,
) -> DistillationCorpusResult:
    """Sample a planned corpus using injected media and UI adapters.

    Successfully encoded frames are atomically moved from a staging path. A
    cancellation keeps all completed images and stops before the next sample.
    """

    destination = os.path.abspath(data_dir)
    saved = 0
    skipped = 0
    failures: list[str] = []
    handled = 0

    for source_path, total_frames, source_samples in plan.videos:
        if is_canceled():
            break
        try:
            reader = open_video(source_path)
        except Exception as exc:
            failures.append(f"{os.path.basename(source_path)}: {exc}")
            continue
        try:
            for sample_number, frame_index in enumerate(range(0, total_frames, plan.stride)):
                if plan.maximum_per_video > 0 and sample_number >= plan.maximum_per_video:
                    break
                if is_canceled():
                    break

                output_path = os.path.join(
                    destination,
                    distillation_frame_filename(source_path, frame_index),
                )
                if os.path.exists(output_path):
                    skipped += 1
                else:
                    frame = reader.read_frame(frame_index)
                    if frame is None:
                        failures.append(
                            f"{os.path.basename(source_path)} frame {frame_index}: read failed"
                        )
                    else:
                        staged_path = staging_path_for(output_path)
                        try:
                            if not write_image(staged_path, frame, int(jpeg_quality)):
                                raise OSError("image writer could not encode the frame")
                            os.replace(staged_path, output_path)
                            saved += 1
                        except Exception as exc:
                            _remove_staged_file(staged_path)
                            failures.append(
                                f"{os.path.basename(source_path)} frame {frame_index}: {exc}"
                            )

                handled += 1
                if on_progress is not None:
                    on_progress(
                        DistillationCorpusProgress(
                            source_path=source_path,
                            sample_number=sample_number + 1,
                            source_samples=source_samples,
                            frame_index=frame_index,
                            handled=handled,
                            estimated_samples=plan.estimated_samples,
                        )
                    )
        finally:
            reader.close()

    return DistillationCorpusResult(
        saved=saved,
        skipped=skipped,
        failures=tuple(failures),
        canceled=is_canceled(),
        handled=handled,
        estimated_samples=plan.estimated_samples,
    )


def student_task_mismatch(student: str, task: str) -> bool:
    """Return whether a conventional model filename declares another head."""

    model_name = os.path.basename(student).lower()
    normalized_task = normalize_distillation_task(task) or "pose"
    if normalized_task == "segment":
        return "-pose" in model_name
    return "-seg" in model_name or "segment" in model_name


def build_distillation_run_plan(
    *,
    program: str,
    script_path: str,
    app_base_dir: str,
    project_root: str,
    runs_root: str,
    data_dir: str,
    run_name: str,
    student: str,
    teacher: str,
    task: str,
    epochs: int,
    batch_size: int,
    precision: str,
    overwrite: bool = False,
) -> DistillationRunPlan:
    """Validate a run and construct the worker's existing CLI envelope."""

    normalized_data_dir = os.path.abspath(data_dir) if data_dir else ""
    image_count = count_distillation_images(normalized_data_dir)
    if not normalized_data_dir or not os.path.isdir(normalized_data_dir) or image_count == 0:
        raise DistillationPlanError(
            "corpus_required",
            "Image corpus required",
            "Choose a directory containing unlabeled images, or create the corpus in the first tab.",
        )

    clean_run_name = str(run_name).strip()
    if not _RUN_NAME_RE.fullmatch(clean_run_name):
        raise DistillationPlanError(
            "invalid_run_name",
            "Invalid run name",
            "Use letters, numbers, periods, underscores, or hyphens; start with a letter or number.",
        )

    clean_student = str(student).strip()
    clean_teacher = str(teacher).strip()
    if not clean_student or not clean_teacher:
        raise DistillationPlanError(
            "model_required",
            "Model required",
            "Both student and teacher model values are required.",
        )

    normalized_task = normalize_distillation_task(task) or "pose"
    defaults = DISTILLATION_TASK_DEFAULTS[normalized_task]
    if student_task_mismatch(clean_student, normalized_task):
        raise DistillationPlanError(
            "student_task_mismatch",
            "Student model task mismatch",
            f"The selected task is {defaults['label']}, but the student model appears to use a different head.\n\n"
            "Choose a compatible model or switch the task.",
        )

    normalized_script = os.path.abspath(script_path)
    if not os.path.isfile(normalized_script):
        raise DistillationPlanError(
            "distiller_missing",
            "Distiller missing",
            f"Could not find:\n{normalized_script}",
        )

    output_dir = os.path.join(os.path.abspath(runs_root), clean_run_name)
    if os.path.exists(output_dir) and not overwrite:
        raise DistillationPlanError(
            "run_exists",
            "Run directory exists",
            f"The output directory already exists:\n{output_dir}\n\n"
            "Choose a new run name or explicitly allow overwrite.",
        )

    normalized_project = os.path.abspath(project_root)
    arguments = [
        "-u",
        normalized_script,
        "--project-root",
        normalized_project,
        "--data-dir",
        normalized_data_dir,
        "--run-name",
        clean_run_name,
        "--model",
        clean_student,
        "--task",
        normalized_task,
        "--teacher",
        clean_teacher,
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--precision",
        str(precision),
    ]
    if overwrite:
        arguments.append("--overwrite")

    return DistillationRunPlan(
        project_root=normalized_project,
        data_dir=normalized_data_dir,
        output_dir=output_dir,
        run_name=clean_run_name,
        task=normalized_task,
        task_label=defaults["label"],
        student=clean_student,
        teacher=clean_teacher,
        image_count=image_count,
        program=program,
        arguments=tuple(arguments),
        working_directory=os.path.abspath(app_base_dir),
    )
