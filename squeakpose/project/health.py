"""Qt-free project health inspection and cleanup coordination."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from squeakpose.core import (
    image_stem_collisions,
    normalize_pose_label_lines,
    normalize_segmentation_label_lines,
)
from squeakpose.project.recovery import (
    cleanup_transaction_staging,
    scan_transaction_artifacts,
)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
_COPY_SUFFIX_RE = re.compile(r"^(?P<base>.+) (?P<number>[2-9][0-9]*)$")


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
    restorable_transaction_backups: list[str] = field(default_factory=list)
    preserved_transaction_backups: list[str] = field(default_factory=list)
    worker_config_paths: list[str] = field(default_factory=list)


def _list_image_files(images_dir: str) -> list[str]:
    if not os.path.isdir(images_dir):
        return []
    return [
        name
        for name in os.listdir(images_dir)
        if not name.startswith(".") and name.lower().endswith(_IMAGE_EXTENSIONS)
    ]


def _list_label_files(labels_dir: str) -> list[str]:
    if not os.path.isdir(labels_dir):
        return []
    return [
        name
        for name in os.listdir(labels_dir)
        if not name.startswith(".") and name.lower().endswith(".txt")
    ]


def _label_file_has_usable_rows(
    label_path: str,
    *,
    segmentation: bool,
    class_count: int,
    keypoint_count: int = 0,
) -> bool:
    try:
        with open(label_path, "r", encoding="utf-8") as label_file:
            lines = [line.strip() for line in label_file if line.strip()]
    except Exception:
        return False
    if not lines:
        return False
    if segmentation:
        normalized, _, _ = normalize_segmentation_label_lines(
            lines,
            class_count=max(1, class_count),
        )
    else:
        normalized, _, _ = normalize_pose_label_lines(
            lines,
            class_count=max(1, class_count),
            keypoint_count=max(0, keypoint_count),
        )
    return bool(normalized)


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

    queue_images = sorted(_list_image_files(queue_dir), key=str.casefold)
    stored_images = sorted(_list_image_files(images_dir), key=str.casefold)
    pose_labels = sorted(_list_label_files(pose_dir), key=str.casefold)
    seg_labels = sorted(_list_label_files(seg_dir), key=str.casefold)
    report = ProjectHealthReport(
        project_root=root,
        queue_images=len(queue_images),
        stored_images=len(stored_images),
        pose_labels=len(pose_labels),
        segmentation_labels=len(seg_labels),
        queue_collisions=image_stem_collisions(queue_images),
        stored_collisions=image_stem_collisions(stored_images),
    )

    image_stems = {os.path.splitext(name)[0].casefold() for name in stored_images}
    pose_stems = {os.path.splitext(name)[0].casefold() for name in pose_labels}
    seg_stems = {os.path.splitext(name)[0].casefold() for name in seg_labels}
    report.unlabeled_images = [
        name
        for name in stored_images
        if os.path.splitext(name)[0].casefold() not in pose_stems | seg_stems
    ]
    report.orphan_pose_labels = [
        name for name in pose_labels if os.path.splitext(name)[0].casefold() not in image_stems
    ]
    report.orphan_segmentation_labels = [
        name for name in seg_labels if os.path.splitext(name)[0].casefold() not in image_stems
    ]

    report.usable_pose_labels = sum(
        _label_file_has_usable_rows(
            os.path.join(pose_dir, name),
            segmentation=False,
            class_count=pose_class_count,
            keypoint_count=pose_keypoint_count,
        )
        for name in pose_labels
    )
    report.usable_segmentation_labels = sum(
        _label_file_has_usable_rows(
            os.path.join(seg_dir, name),
            segmentation=True,
            class_count=segmentation_class_count,
        )
        for name in seg_labels
    )

    transaction_report = scan_transaction_artifacts(root)
    report.temporary_paths = transaction_report.staging_paths
    report.restorable_transaction_backups = [
        item.backup_path for item in transaction_report.restorable_backups
    ]
    report.preserved_transaction_backups = [
        item.backup_path for item in transaction_report.preserved_backups
    ]

    names_by_key = {name.casefold(): name for name in stored_images}
    for name in stored_images:
        stem, extension = os.path.splitext(name)
        match = _COPY_SUFFIX_RE.match(stem)
        if not match:
            continue
        original = names_by_key.get(f"{match.group('base')}{extension}".casefold())
        if original:
            report.likely_duplicate_images.append((original, name))

    for current, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names[:] = [
            name
            for name in dir_names
            if name not in {"__pycache__", ".git", ".venv"}
            and not os.path.islink(os.path.join(current, name))
        ]
        for name in file_names:
            if name.startswith(".") and name.endswith(".json") and "config" in name:
                report.worker_config_paths.append(os.path.join(current, name))

    report.worker_config_paths = sorted(set(report.worker_config_paths), key=str.casefold)
    return report


def cleanup_project_temporary_paths(report: ProjectHealthReport) -> list[str]:
    """Remove only transaction staging paths identified by a health scan."""
    return cleanup_transaction_staging(report.project_root).errors


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
        f"Restorable transaction backups: {len(report.restorable_transaction_backups)}",
        f"Preserved transaction backups: {len(report.preserved_transaction_backups)}",
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
        (
            "Restorable transaction backups",
            [
                os.path.relpath(path, report.project_root)
                for path in report.restorable_transaction_backups
            ],
        ),
        (
            "Preserved transaction backups",
            [
                os.path.relpath(path, report.project_root)
                for path in report.preserved_transaction_backups
            ],
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


__all__ = [
    "ProjectHealthReport",
    "cleanup_project_temporary_paths",
    "format_project_health_summary",
    "scan_project_health",
]
