"""Qt-free filesystem helpers for dataset export and label normalization."""

from __future__ import annotations

import datetime
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from dataset_builder import create_dataset_yaml
from squeakpose_core import (
    atomic_write_text,
    image_stem_collisions,
    normalize_pose_label_lines,
    normalize_segmentation_label_lines,
    remove_path,
)

DATASET_POSE = "pose"
DATASET_SEGMENT = "segment"
DATASET_DETECT = "detect"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class DatasetExportPaths:
    base_dir: str
    images_train_dir: str
    images_val_dir: str
    labels_train_dir: str
    labels_val_dir: str
    dataset_yaml_path: str

    @property
    def split_dirs(self) -> tuple[str, str, str, str]:
        return (
            self.images_train_dir,
            self.images_val_dir,
            self.labels_train_dir,
            self.labels_val_dir,
        )


@dataclass
class DatasetExportResult:
    mode: str
    train_images: list[str]
    val_images: list[str]
    dataset_yaml_path: str
    split_seed: Optional[int] = None
    skipped_images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processed: int = 0
    canceled: bool = False


@dataclass
class LabelNormalizationResult:
    mode: str
    total_files: int
    normalized: int = 0
    untouched: int = 0
    copied_images: int = 0
    quarantined: int = 0
    warnings: list[str] = field(default_factory=list)
    backup_dir: Optional[str] = None
    quarantine_dir: Optional[str] = None
    canceled: bool = False


@dataclass
class ProjectHealthReport:
    project_root: str
    queue_images: int = 0
    stored_images: int = 0
    pose_labels: int = 0
    segmentation_labels: int = 0
    usable_pose_labels: int = 0
    usable_segmentation_labels: int = 0
    unlabeled_images: list[str] = field(default_factory=list)
    orphan_pose_labels: list[str] = field(default_factory=list)
    orphan_segmentation_labels: list[str] = field(default_factory=list)
    queue_collisions: dict[str, list[str]] = field(default_factory=dict)
    stored_collisions: dict[str, list[str]] = field(default_factory=dict)
    likely_duplicate_images: list[tuple[str, str]] = field(default_factory=list)
    temporary_paths: list[str] = field(default_factory=list)
    worker_config_paths: list[str] = field(default_factory=list)


def is_segmentation_mode(mode: str) -> bool:
    return mode in {DATASET_SEGMENT, "segmentation"}


def dataset_export_paths(project_root: str, mode: str) -> DatasetExportPaths:
    subdir = DATASET_SEGMENT if is_segmentation_mode(mode) else mode
    base_dir = os.path.join(project_root, "datasets", subdir)
    return dataset_export_paths_from_base(base_dir)


def dataset_export_paths_from_base(base_dir: str) -> DatasetExportPaths:
    return DatasetExportPaths(
        base_dir=base_dir,
        images_train_dir=os.path.join(base_dir, "images", "train"),
        images_val_dir=os.path.join(base_dir, "images", "val"),
        labels_train_dir=os.path.join(base_dir, "labels", "train"),
        labels_val_dir=os.path.join(base_dir, "labels", "val"),
        dataset_yaml_path=os.path.join(base_dir, "dataset.yaml"),
    )


def list_image_files(images_dir: str) -> list[str]:
    if not os.path.isdir(images_dir):
        return []
    return [
        f
        for f in os.listdir(images_dir)
        if not f.startswith(".") and f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def list_label_files(labels_dir: str) -> list[str]:
    if not os.path.isdir(labels_dir):
        return []
    return [
        f
        for f in os.listdir(labels_dir)
        if not f.startswith(".") and f.lower().endswith(".txt")
    ]


def dataset_dirs_have_files(paths: DatasetExportPaths) -> bool:
    return any(os.path.isdir(d) and bool(os.listdir(d)) for d in paths.split_dirs)


def remove_dataset_split_dirs(paths: DatasetExportPaths) -> None:
    for directory in paths.split_dirs:
        if os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)


def split_train_val_images(images: Iterable[str], train_ratio: float) -> tuple[list[str], list[str]]:
    items = list(images)
    train_count = int(len(items) * train_ratio)
    if train_count <= 0 and len(items) > 0:
        train_count = 1
    if train_count >= len(items) and len(items) > 1:
        train_count = len(items) - 1
    return items[:train_count], items[train_count:]


def partition_images_by_usable_labels(
    images: Iterable[str],
    *,
    labels_dir: str,
    mode: str,
    class_count: int,
    keypoint_count: int = 0,
) -> tuple[list[str], list[str]]:
    """Partition images into exportable and skipped lists for one label format."""
    exportable: list[str] = []
    skipped: list[str] = []
    for image_name in images:
        stem = os.path.splitext(os.path.basename(image_name))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if label_file_has_usable_rows(
            label_path,
            mode=mode,
            class_count=class_count,
            keypoint_count=keypoint_count,
        ):
            exportable.append(image_name)
        else:
            skipped.append(image_name)
    return exportable, skipped


def _copy_label_for_dataset(
    label_src: str,
    label_dst: str,
    base_name: str,
    mode: str,
    class_count: Optional[int] = None,
    keypoint_count: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    if is_segmentation_mode(mode) and class_count is not None:
        try:
            with open(label_src, "r", encoding="utf-8") as label_file:
                lines = [line.strip() for line in label_file if line.strip()]
            normalized, row_warnings, _ = normalize_segmentation_label_lines(
                lines,
                class_count=max(1, int(class_count)),
            )
            warnings = [f"{base_name}.txt: {warning}" for warning in row_warnings]
            if not normalized:
                return warnings, [f"{base_name}.txt: no usable segmentation rows"]
            atomic_write_text(label_dst, "\n".join(normalized) + "\n")
            return warnings, []
        except Exception as exc:
            return [], [f"{base_name}.txt: normalize failed ({exc})"]

    if mode == DATASET_POSE and class_count is not None and keypoint_count is not None:
        try:
            with open(label_src, "r", encoding="utf-8") as label_file:
                lines = [line.strip() for line in label_file if line.strip()]
            normalized, row_warnings, _ = normalize_pose_label_lines(
                lines,
                class_count=max(1, int(class_count)),
                keypoint_count=max(0, int(keypoint_count)),
            )
            warnings = [f"{base_name}.txt: {warning}" for warning in row_warnings]
            if not normalized:
                return warnings, [f"{base_name}.txt: no usable pose rows"]
            atomic_write_text(label_dst, "\n".join(normalized) + "\n")
            return warnings, []
        except Exception as exc:
            return [], [f"{base_name}.txt: normalize failed ({exc})"]

    if mode in {DATASET_POSE, DATASET_SEGMENT} or is_segmentation_mode(mode):
        try:
            shutil.copy2(label_src, label_dst)
        except Exception as exc:
            return [], [f"{base_name}.txt: copy failed ({exc})"]
        return [], []

    warnings: list[str] = []
    errors: list[str] = []
    try:
        det_lines: list[str] = []
        with open(label_src, "r", encoding="utf-8") as lf:
            for raw in lf:
                parts = raw.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    warnings.append(f"{base_name}.txt: insufficient columns for detection")
                    continue
                det_lines.append(" ".join(parts[:5]))
        if det_lines:
            atomic_write_text(label_dst, "\n".join(det_lines) + "\n")
        else:
            errors.append(f"{base_name}.txt: no usable bbox rows")
    except Exception as exc:
        errors.append(f"{base_name}.txt: convert failed ({exc})")
    return warnings, errors


def export_dataset_files(
    *,
    images_all_dir: str,
    labels_all_dir: str,
    paths: DatasetExportPaths,
    train_images: list[str],
    val_images: list[str],
    mode: str,
    class_count: Optional[int] = None,
    keypoint_count: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_requested: Optional[CancelCallback] = None,
) -> DatasetExportResult:
    for directory in paths.split_dirs:
        os.makedirs(directory, exist_ok=True)

    result = DatasetExportResult(
        mode=mode,
        train_images=list(train_images),
        val_images=list(val_images),
        dataset_yaml_path=paths.dataset_yaml_path,
    )
    targets = [
        (train_images, paths.images_train_dir, paths.labels_train_dir),
        (val_images, paths.images_val_dir, paths.labels_val_dir),
    ]

    for group, img_dir, lbl_dir in targets:
        for img_file in group:
            if cancel_requested and cancel_requested():
                result.canceled = True
                return result

            src_img = os.path.join(images_all_dir, img_file)
            dst_img = os.path.join(img_dir, img_file)
            try:
                shutil.copy2(src_img, dst_img)
            except Exception as exc:
                result.errors.append(f"{img_file}: copy image failed ({exc})")
                result.processed += 1
                if progress_callback:
                    progress_callback(result.processed, img_file)
                continue

            base_name = os.path.splitext(img_file)[0]
            label_src = os.path.join(labels_all_dir, f"{base_name}.txt")
            label_dst = os.path.join(lbl_dir, f"{base_name}.txt")
            if os.path.exists(label_src):
                warnings, errors = _copy_label_for_dataset(
                    label_src,
                    label_dst,
                    base_name,
                    mode,
                    class_count=class_count,
                    keypoint_count=keypoint_count,
                )
                result.warnings.extend(warnings)
                result.errors.extend(errors)
            else:
                result.errors.append(f"{base_name}.txt: missing")

            result.processed += 1
            if progress_callback:
                progress_callback(result.processed, img_file)

    return result


_COPY_SUFFIX_RE = re.compile(r"^(?P<base>.+) (?P<number>[2-9][0-9]*)$")


def scan_project_health(
    project_root: str,
    *,
    pose_class_count: int,
    pose_keypoint_count: int,
    segmentation_class_count: int,
) -> ProjectHealthReport:
    """Inspect project files without modifying them."""
    root = os.path.abspath(project_root)
    queue_dir = os.path.join(root, "images_to_label")
    images_dir = os.path.join(root, "images_all")
    pose_dir = os.path.join(root, "labels_all")
    seg_dir = os.path.join(root, "labels_seg_all")

    queue_images = sorted(list_image_files(queue_dir), key=str.casefold)
    stored_images = sorted(list_image_files(images_dir), key=str.casefold)
    pose_labels = sorted(list_label_files(pose_dir), key=str.casefold)
    seg_labels = sorted(list_label_files(seg_dir), key=str.casefold)
    report = ProjectHealthReport(
        project_root=root,
        queue_images=len(queue_images),
        stored_images=len(stored_images),
        pose_labels=len(pose_labels),
        segmentation_labels=len(seg_labels),
        queue_collisions=image_stem_collisions(queue_images),
        stored_collisions=image_stem_collisions(stored_images),
    )

    image_stems = {
        os.path.splitext(name)[0].casefold()
        for name in stored_images
    }
    pose_stems = {
        os.path.splitext(name)[0].casefold()
        for name in pose_labels
    }
    seg_stems = {
        os.path.splitext(name)[0].casefold()
        for name in seg_labels
    }
    report.unlabeled_images = [
        name
        for name in stored_images
        if os.path.splitext(name)[0].casefold() not in pose_stems | seg_stems
    ]
    report.orphan_pose_labels = [
        name
        for name in pose_labels
        if os.path.splitext(name)[0].casefold() not in image_stems
    ]
    report.orphan_segmentation_labels = [
        name
        for name in seg_labels
        if os.path.splitext(name)[0].casefold() not in image_stems
    ]

    report.usable_pose_labels = sum(
        label_file_has_usable_rows(
            os.path.join(pose_dir, name),
            mode=DATASET_POSE,
            class_count=max(1, pose_class_count),
            keypoint_count=max(0, pose_keypoint_count),
        )
        for name in pose_labels
    )
    report.usable_segmentation_labels = sum(
        label_file_has_usable_rows(
            os.path.join(seg_dir, name),
            mode=DATASET_SEGMENT,
            class_count=max(1, segmentation_class_count),
        )
        for name in seg_labels
    )

    names_by_key = {name.casefold(): name for name in stored_images}
    for name in stored_images:
        stem, ext = os.path.splitext(name)
        match = _COPY_SUFFIX_RE.match(stem)
        if not match:
            continue
        original = names_by_key.get(f"{match.group('base')}{ext}".casefold())
        if original:
            report.likely_duplicate_images.append((original, name))

    scan_roots = [
        queue_dir,
        images_dir,
        pose_dir,
        seg_dir,
        os.path.join(root, "datasets"),
        os.path.join(root, "logs"),
        os.path.join(root, "runs"),
    ]
    for scan_root in scan_roots:
        if not os.path.isdir(scan_root):
            continue
        for current, dir_names, file_names in os.walk(scan_root):
            retained_dirs: list[str] = []
            for name in dir_names:
                path = os.path.join(current, name)
                if name.startswith(".") and "-export-" in name:
                    report.temporary_paths.append(path)
                elif name not in {"__pycache__", ".git", ".venv"}:
                    retained_dirs.append(name)
            dir_names[:] = retained_dirs
            for name in file_names:
                if not name.startswith("."):
                    continue
                path = os.path.join(current, name)
                if ".tmp" in name:
                    report.temporary_paths.append(path)
                elif name.endswith(".json") and "config" in name:
                    report.worker_config_paths.append(path)

    report.temporary_paths = sorted(set(report.temporary_paths), key=str.casefold)
    report.worker_config_paths = sorted(set(report.worker_config_paths), key=str.casefold)
    return report


def cleanup_project_temporary_paths(report: ProjectHealthReport) -> list[str]:
    """Remove only transaction staging paths identified by a health scan."""
    errors: list[str] = []
    for path in report.temporary_paths:
        try:
            remove_path(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def format_project_health_summary(report: ProjectHealthReport) -> str:
    """Build a compact human-readable project health report."""
    lines = [
        f"Queue images: {report.queue_images}",
        f"Stored images: {report.stored_images}",
        f"Pose labels: {report.pose_labels} ({report.usable_pose_labels} usable)",
        (
            "Segmentation labels: "
            f"{report.segmentation_labels} ({report.usable_segmentation_labels} usable)"
        ),
        "",
        f"Images without pose or segmentation labels: {len(report.unlabeled_images)}",
        f"Orphan pose labels: {len(report.orphan_pose_labels)}",
        f"Orphan segmentation labels: {len(report.orphan_segmentation_labels)}",
        f"Ambiguous queue stems: {len(report.queue_collisions)}",
        f"Ambiguous stored-image stems: {len(report.stored_collisions)}",
        f"Likely numbered image copies: {len(report.likely_duplicate_images)}",
        f"Temporary transaction paths: {len(report.temporary_paths)}",
        f"Worker config files: {len(report.worker_config_paths)}",
    ]

    detail_groups = (
        ("Unlabeled images", report.unlabeled_images),
        ("Orphan pose labels", report.orphan_pose_labels),
        ("Orphan segmentation labels", report.orphan_segmentation_labels),
        (
            "Likely numbered copies",
            [f"{original} / {copy}" for original, copy in report.likely_duplicate_images],
        ),
        (
            "Temporary transaction paths",
            [os.path.relpath(path, report.project_root) for path in report.temporary_paths],
        ),
    )
    for title, items in detail_groups:
        if not items:
            continue
        lines.extend(["", f"{title}:"])
        lines.extend(str(item) for item in items[:8])
        if len(items) > 8:
            lines.append(f"...{len(items) - 8} more")
    return "\n".join(lines)


def write_dataset_yaml_for_mode(
    base_dir: str,
    mode: str,
    class_names: Iterable[str],
    kp_names: Iterable[str],
    *,
    verbose: bool = True,
    dataset_path: Optional[str] = None,
) -> str:
    if mode == DATASET_POSE:
        return create_dataset_yaml(
            base_dir,
            class_names,
            kp_names,
            verbose=verbose,
            dataset_path=dataset_path,
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write dataset.yaml.") from exc

    cls_list = list(class_names)
    payload = {
        "path": dataset_path or base_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": len(cls_list),
        "names": cls_list,
    }
    if is_segmentation_mode(mode):
        payload["task"] = "segment"
    out_path = os.path.join(base_dir, "dataset.yaml")
    atomic_write_text(out_path, yaml.safe_dump(payload, sort_keys=False))
    return out_path


def format_dataset_export_summary(result: DatasetExportResult) -> str:
    if is_segmentation_mode(result.mode):
        label_format = "Segmentation (mask)"
    elif result.mode == DATASET_POSE:
        label_format = "Pose (keypoints)"
    else:
        label_format = "Detection (bbox)"

    summary = (
        f"Train images: {len(result.train_images)}\n"
        f"Val images: {len(result.val_images)}\n"
        f"Format: {label_format}\n"
        f"dataset.yaml written to: {result.dataset_yaml_path}"
    )
    if result.split_seed is not None:
        summary += f"\nSplit seed: {result.split_seed}"
    if result.skipped_images:
        summary += f"\nSkipped without usable labels: {len(result.skipped_images)}"
    if result.warnings:
        summary += "\n\nWarnings:\n" + "\n".join(result.warnings[:10])
        if len(result.warnings) > 10:
            summary += f"\n...{len(result.warnings) - 10} more"
    if result.errors:
        summary += "\n\nErrors:\n" + "\n".join(result.errors[:10])
        if len(result.errors) > 10:
            summary += f"\n...{len(result.errors) - 10} more"
    return summary


def backup_label_dir(labels_dir: str) -> str:
    src = os.path.abspath(labels_dir)
    parent = os.path.dirname(src)
    name = os.path.basename(src.rstrip(os.sep)) or "labels"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(parent, f"{name}_backup_{timestamp}")
    suffix = 1
    while os.path.exists(backup_path):
        backup_path = os.path.join(parent, f"{name}_backup_{timestamp}_{suffix}")
        suffix += 1
    shutil.copytree(src, backup_path)
    return backup_path


def _quarantine_label_path(labels_dir: str, fname: str, result: LabelNormalizationResult) -> str:
    if result.quarantine_dir is None:
        src = os.path.abspath(labels_dir)
        parent = os.path.dirname(src)
        name = os.path.basename(src.rstrip(os.sep)) or "labels"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        quarantine_path = os.path.join(parent, f"{name}_quarantine_{timestamp}")
        suffix = 1
        while os.path.exists(quarantine_path):
            quarantine_path = os.path.join(parent, f"{name}_quarantine_{timestamp}_{suffix}")
            suffix += 1
        os.makedirs(quarantine_path)
        result.quarantine_dir = quarantine_path
    return os.path.join(result.quarantine_dir, fname)


def _quarantine_unusable_label(
    labels_dir: str,
    fname: str,
    label_path: str,
    result: LabelNormalizationResult,
) -> None:
    if result.backup_dir is None:
        result.backup_dir = backup_label_dir(labels_dir)
    quarantine_path = _quarantine_label_path(labels_dir, fname, result)
    os.replace(label_path, quarantine_path)
    result.quarantined += 1


def label_file_has_usable_rows(
    label_path: str,
    *,
    mode: str,
    class_count: int,
    keypoint_count: int = 0,
) -> bool:
    """Return whether a label file contains at least one usable row."""
    try:
        with open(label_path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
    except Exception:
        return False
    if not lines:
        return False
    if is_segmentation_mode(mode):
        normalized, _, _ = normalize_segmentation_label_lines(lines, class_count=max(1, class_count))
    else:
        normalized, _, _ = normalize_pose_label_lines(
            lines,
            class_count=max(1, class_count),
            keypoint_count=max(0, keypoint_count),
        )
    return bool(normalized)


def _ensure_matching_image(
    *,
    stem: str,
    label_name: str,
    images_all_dir: str,
    images_to_label_dir: str,
) -> tuple[bool, int, list[str]]:
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_all_dir, stem + ext)
        if os.path.exists(candidate):
            return True, 0, []

    for ext in IMAGE_EXTENSIONS:
        src = os.path.join(images_to_label_dir, stem + ext)
        if os.path.exists(src):
            dst = os.path.join(images_all_dir, os.path.basename(src))
            try:
                os.makedirs(images_all_dir, exist_ok=True)
                shutil.copy2(src, dst)
                return True, 1, []
            except Exception as exc:
                return False, 0, [f"{stem}{ext}: copy failed ({exc})"]

    return False, 0, [f"{label_name}: no matching image found in images_all or images_to_label"]


def normalize_label_directory(
    *,
    labels_dir: str,
    images_all_dir: str,
    images_to_label_dir: str,
    mode: str,
    class_count: int,
    keypoint_count: int = 0,
    label_files: Optional[Iterable[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_requested: Optional[CancelCallback] = None,
) -> LabelNormalizationResult:
    files = sorted(list(label_files) if label_files is not None else list_label_files(labels_dir))
    result = LabelNormalizationResult(mode=mode, total_files=len(files))

    for idx, fname in enumerate(files, start=1):
        if cancel_requested and cancel_requested():
            result.canceled = True
            return result

        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, fname)
        try:
            with open(label_path, "r", encoding="utf-8") as lf:
                lines = [ln.strip() for ln in lf if ln.strip()]
        except Exception as exc:
            result.warnings.append(f"{fname}: read error ({exc})")
            if progress_callback:
                progress_callback(idx, fname)
            continue

        if not lines:
            result.warnings.append(f"{fname}: empty file")
            try:
                _quarantine_unusable_label(labels_dir, fname, label_path, result)
            except Exception as exc:
                result.warnings.append(f"{fname}: quarantine error ({exc})")
            if progress_callback:
                progress_callback(idx, fname)
            continue

        if is_segmentation_mode(mode):
            normalized_lines, line_warnings, line_changed = normalize_segmentation_label_lines(
                lines,
                class_count=class_count,
            )
            unusable_msg = f"{fname}: no usable segmentation rows"
        else:
            normalized_lines, line_warnings, line_changed = normalize_pose_label_lines(
                lines,
                class_count=class_count,
                keypoint_count=keypoint_count,
            )
            unusable_msg = f"{fname}: no usable pose rows"
        result.warnings.extend(f"{fname}: {msg}" for msg in line_warnings)

        if not normalized_lines:
            result.warnings.append(unusable_msg)
            try:
                _quarantine_unusable_label(labels_dir, fname, label_path, result)
            except Exception as exc:
                result.warnings.append(f"{fname}: quarantine error ({exc})")
            if progress_callback:
                progress_callback(idx, fname)
            continue

        if line_changed:
            try:
                if result.backup_dir is None:
                    result.backup_dir = backup_label_dir(labels_dir)
                atomic_write_text(label_path, "\n".join(normalized_lines) + "\n")
            except Exception as exc:
                result.warnings.append(f"{fname}: write error ({exc})")
                if progress_callback:
                    progress_callback(idx, fname)
                continue
            result.normalized += 1
        else:
            result.untouched += 1

        _, copied, image_warnings = _ensure_matching_image(
            stem=stem,
            label_name=fname,
            images_all_dir=images_all_dir,
            images_to_label_dir=images_to_label_dir,
        )
        result.copied_images += copied
        result.warnings.extend(image_warnings)

        if progress_callback:
            progress_callback(idx, fname)

    return result


def format_label_normalization_summary(result: LabelNormalizationResult) -> str:
    label_kind = "segmentation label" if is_segmentation_mode(result.mode) else "label"
    if result.normalized == 0 and result.copied_images == 0 and not result.warnings:
        if is_segmentation_mode(result.mode):
            return "All segmentation label files already normalized. No changes made."
        return "All label files already normalized. No changes made."

    parts: list[str] = []
    if result.normalized:
        parts.append(f"Normalized {result.normalized} {label_kind} file(s).")
    if result.backup_dir:
        parts.append(f"Backup written to: {result.backup_dir}")
    if result.quarantined:
        parts.append(f"Quarantined {result.quarantined} unusable {label_kind} file(s).")
    if result.quarantine_dir:
        parts.append(f"Quarantine written to: {result.quarantine_dir}")
    if result.untouched and result.normalized:
        parts.append(f"{result.untouched} file(s) were already normalized.")
    elif result.untouched and not result.normalized:
        parts.append(f"{result.untouched} file(s) already normalized.")
    parts.append(f"Copied {result.copied_images} missing image(s) into images_all.")
    summary = "\n".join(parts)

    if result.warnings:
        summary += "\n\nWarnings:\n" + "\n".join(result.warnings[:10])
        if len(result.warnings) > 10:
            summary += f"\n...{len(result.warnings) - 10} more"
    return summary
