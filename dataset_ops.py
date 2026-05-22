"""Qt-free filesystem helpers for dataset export and label normalization."""

from __future__ import annotations

import datetime
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from dataset_builder import create_dataset_yaml
from squeakpose_core import (
    atomic_write_text,
    normalize_pose_label_lines,
    normalize_segmentation_label_lines,
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
    warnings: list[str] = field(default_factory=list)
    processed: int = 0
    canceled: bool = False


@dataclass
class LabelNormalizationResult:
    mode: str
    total_files: int
    normalized: int = 0
    untouched: int = 0
    copied_images: int = 0
    warnings: list[str] = field(default_factory=list)
    backup_dir: Optional[str] = None
    canceled: bool = False


def is_segmentation_mode(mode: str) -> bool:
    return mode in {DATASET_SEGMENT, "segmentation"}


def dataset_export_paths(project_root: str, mode: str) -> DatasetExportPaths:
    subdir = DATASET_SEGMENT if is_segmentation_mode(mode) else mode
    base_dir = os.path.join(project_root, "datasets", subdir)
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
    return [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]


def list_label_files(labels_dir: str) -> list[str]:
    if not os.path.isdir(labels_dir):
        return []
    return [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]


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


def _copy_label_for_dataset(label_src: str, label_dst: str, base_name: str, mode: str) -> list[str]:
    if mode in {DATASET_POSE, DATASET_SEGMENT} or is_segmentation_mode(mode):
        try:
            shutil.copy2(label_src, label_dst)
        except Exception as exc:
            return [f"{base_name}.txt: copy failed ({exc})"]
        return []

    warnings: list[str] = []
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
            warnings.append(f"{base_name}.txt: no usable bbox rows")
    except Exception as exc:
        warnings.append(f"{base_name}.txt: convert failed ({exc})")
    return warnings


def export_dataset_files(
    *,
    images_all_dir: str,
    labels_all_dir: str,
    paths: DatasetExportPaths,
    train_images: list[str],
    val_images: list[str],
    mode: str,
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
                result.warnings.append(f"{img_file}: copy image failed ({exc})")
                result.processed += 1
                if progress_callback:
                    progress_callback(result.processed, img_file)
                continue

            base_name = os.path.splitext(img_file)[0]
            label_src = os.path.join(labels_all_dir, f"{base_name}.txt")
            label_dst = os.path.join(lbl_dir, f"{base_name}.txt")
            if os.path.exists(label_src):
                result.warnings.extend(_copy_label_for_dataset(label_src, label_dst, base_name, mode))
            else:
                result.warnings.append(f"{base_name}.txt: missing")

            result.processed += 1
            if progress_callback:
                progress_callback(result.processed, img_file)

    return result


def write_dataset_yaml_for_mode(
    base_dir: str,
    mode: str,
    class_names: Iterable[str],
    kp_names: Iterable[str],
    *,
    verbose: bool = True,
) -> str:
    if mode == DATASET_POSE:
        return create_dataset_yaml(base_dir, class_names, kp_names, verbose=verbose)

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write dataset.yaml. Install with `pip install pyyaml`.") from exc

    cls_list = list(class_names)
    payload = {
        "path": base_dir,
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
    if result.warnings:
        summary += "\n\nWarnings:\n" + "\n".join(result.warnings[:10])
        if len(result.warnings) > 10:
            summary += f"\n...{len(result.warnings) - 10} more"
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
