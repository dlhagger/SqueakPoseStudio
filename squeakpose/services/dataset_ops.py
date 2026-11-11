"""Qt-free filesystem helpers for dataset export and label normalization."""

from __future__ import annotations

import datetime
import hashlib
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from squeakpose.core import (
    atomic_write_text,
    normalize_pose_label_lines,
    normalize_segmentation_label_lines,
)
from squeakpose.project.health import ProjectHealthReport as ProjectHealthReport
from squeakpose.project.health import (
    cleanup_project_temporary_paths as cleanup_project_temporary_paths,
)
from squeakpose.project.health import (
    format_project_health_summary as format_project_health_summary,
)
from squeakpose.project.health import scan_project_health as scan_project_health
from squeakpose.services.dataset_yaml import create_dataset_yaml

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
        f for f in os.listdir(labels_dir) if not f.startswith(".") and f.lower().endswith(".txt")
    ]


def dataset_dirs_have_files(paths: DatasetExportPaths) -> bool:
    return any(os.path.isdir(d) and bool(os.listdir(d)) for d in paths.split_dirs)


def remove_dataset_split_dirs(paths: DatasetExportPaths) -> None:
    for directory in paths.split_dirs:
        if os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)


def split_train_val_images(
    images: Iterable[str], train_ratio: float
) -> tuple[list[str], list[str]]:
    items = list(images)
    train_count = int(len(items) * train_ratio)
    if train_count <= 0 and len(items) > 0:
        train_count = 1
    if train_count >= len(items) and len(items) > 1:
        train_count = len(items) - 1
    return items[:train_count], items[train_count:]


_FRAME_NAME_RE = re.compile(r"^(?P<group>.+)_f(?P<index>\d{6,})(?:_\d+)?$")
_SPLIT_FALLBACK_BLOCKS = 10


def _split_group_and_frame(image_name: str) -> tuple[str, Optional[int]]:
    """Return ``(group_key, frame_index)`` for one image name.

    Video exports are named ``{video_base}_{source_id}_f{index:06d}`` (optionally with a
    ``_N`` collision suffix); their group is everything before ``_f<digits>``. Any other
    image is its own group keyed by its stem and has no frame index.
    """

    stem = os.path.splitext(os.path.basename(image_name))[0]
    match = _FRAME_NAME_RE.match(stem)
    if match is None:
        return stem, None
    return match.group("group"), int(match.group("index"))


def _stable_unit_interval(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _assign_units_to_val(
    keys: list[str], val_fraction: float, *, seed: int
) -> tuple[set[str], dict[str, float]]:
    """Pick val keys by stable hash, forcing at least one key onto each side."""

    scores = {key: _stable_unit_interval(seed, key) for key in keys}
    val_keys = {key for key in keys if scores[key] < val_fraction}
    if len(keys) >= 2:
        if not val_keys:
            val_keys = {min(keys, key=lambda key: (scores[key], key))}
        elif len(val_keys) == len(keys):
            val_keys.discard(max(keys, key=lambda key: (scores[key], key)))
    return val_keys, scores


def plan_grouped_train_val_split(
    images: Iterable[str], train_ratio: float, *, seed: int
) -> tuple[list[str], list[str]]:
    """Split images into train/val so frames of one video stay on one side.

    Each group (see :func:`_split_group_and_frame`) goes to val when a stable hash of
    ``f"{seed}:{group}"`` falls below ``1 - train_ratio``. The decision for a group does
    not depend on which other images exist, so adding images never moves existing
    groups. With two or more images both sides are non-empty: when hashing puts every
    group on one side, the group whose hash is closest to the other side is moved; when
    there is only one group, its frames are cut into contiguous frame-index blocks that
    are assigned the same way, keeping val frames temporally apart from train frames.
    Both returned lists are sorted.
    """

    items = sorted(dict.fromkeys(images))
    if len(items) < 2:
        return items, []
    ratio = float(train_ratio)
    if math.isnan(ratio):
        raise ValueError("train_ratio must be a number")
    val_fraction = min(1.0, max(0.0, 1.0 - ratio))

    groups: dict[str, list[tuple[Optional[int], str]]] = {}
    for name in items:
        group, frame_index = _split_group_and_frame(name)
        groups.setdefault(group, []).append((frame_index, name))

    if len(groups) >= 2:
        val_groups, _ = _assign_units_to_val(list(groups), val_fraction, seed=seed)
        train: list[str] = []
        val: list[str] = []
        for group, members in groups.items():
            (val if group in val_groups else train).extend(name for _, name in members)
        return sorted(train), sorted(val)

    # Single group: fall back to contiguous frame-index blocks.
    group, members = next(iter(groups.items()))
    ordered = sorted(members, key=lambda member: (member[0] is None, member[0] or 0, member[1]))
    block_count = min(_SPLIT_FALLBACK_BLOCKS, len(ordered))
    blocks: dict[str, list[str]] = {}
    for position, (_, name) in enumerate(ordered):
        block = position * block_count // len(ordered)
        blocks.setdefault(f"{group}#block{block}", []).append(name)
    val_blocks, _ = _assign_units_to_val(list(blocks), val_fraction, seed=seed)
    train = [name for key, names in blocks.items() if key not in val_blocks for name in names]
    val = [name for key, names in blocks.items() if key in val_blocks for name in names]
    return sorted(train), sorted(val)


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
            normalized_warnings = [f"{base_name}.txt: {warning}" for warning in row_warnings]
            if not normalized:
                return normalized_warnings, [f"{base_name}.txt: no usable segmentation rows"]
            atomic_write_text(label_dst, "\n".join(normalized) + "\n")
            return normalized_warnings, []
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
            normalized_warnings = [f"{base_name}.txt: {warning}" for warning in row_warnings]
            if not normalized:
                return normalized_warnings, [f"{base_name}.txt: no usable pose rows"]
            atomic_write_text(label_dst, "\n".join(normalized) + "\n")
            return normalized_warnings, []
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
    # ``path`` is omitted by default so Ultralytics resolves train/val against the
    # YAML's own directory; an absolute root would keep pointing at the original
    # location after a project is moved or copied.
    payload: dict[str, object] = {}
    if dataset_path:
        payload["path"] = dataset_path
    payload.update(
        {
            "train": "images/train",
            "val": "images/val",
            "nc": len(cls_list),
            "names": cls_list,
        }
    )
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
        normalized, _, _ = normalize_segmentation_label_lines(
            lines, class_count=max(1, class_count)
        )
    else:
        normalized, _, _ = normalize_pose_label_lines(
            lines,
            class_count=max(1, class_count),
            keypoint_count=max(0, keypoint_count),
        )
    return bool(normalized)


def _build_image_stem_index(images_dir: str) -> dict[str, str]:
    """Map case-folded image stems to file names, keeping the first name in sorted order."""

    index: dict[str, str] = {}
    for name in sorted(list_image_files(images_dir)):
        index.setdefault(os.path.splitext(name)[0].casefold(), name)
    return index


def _ensure_matching_image(
    *,
    stem: str,
    label_name: str,
    images_all_dir: str,
    images_to_label_dir: str,
    images_all_index: Optional[dict[str, str]] = None,
    images_to_label_index: Optional[dict[str, str]] = None,
) -> tuple[bool, int, list[str]]:
    if images_all_index is None:
        images_all_index = _build_image_stem_index(images_all_dir)
    wanted = stem.casefold()
    if images_all_index.get(wanted):
        return True, 0, []

    if images_to_label_index is None:
        images_to_label_index = _build_image_stem_index(images_to_label_dir)
    source_name = images_to_label_index.get(wanted, "")
    if source_name:
        src = os.path.join(images_to_label_dir, source_name)
        dst = os.path.join(images_all_dir, source_name)
        try:
            os.makedirs(images_all_dir, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as exc:
            return False, 0, [f"{source_name}: copy failed ({exc})"]
        images_all_index[wanted] = source_name
        return True, 1, []

    return False, 0, [f"{label_name}: no matching image found in images_all or images_to_label"]


def _find_image_for_stem(images_dir: str, stem: str) -> str:
    """Return the image file name whose stem matches ``stem`` case-insensitively."""

    return _build_image_stem_index(images_dir).get(stem.casefold(), "")


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
    # Each image directory is listed once; per-label lookups are then O(1).
    image_indexes: dict[str, dict[str, str]] = {}

    def image_index(directory: str) -> dict[str, str]:
        if directory not in image_indexes:
            image_indexes[directory] = _build_image_stem_index(directory)
        return image_indexes[directory]

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
                class_count=max(1, class_count),
            )
            unusable_msg = f"{fname}: no usable segmentation rows"
        else:
            normalized_lines, line_warnings, line_changed = normalize_pose_label_lines(
                lines,
                class_count=max(1, class_count),
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
            images_all_index=image_index(images_all_dir),
            images_to_label_index=image_index(images_to_label_dir),
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
