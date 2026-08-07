"""Transactional orchestration for YOLO dataset exports."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable

from dataset_ops import (
    DatasetExportPaths,
    DatasetExportResult,
    dataset_export_paths_from_base,
    export_dataset_files,
    write_dataset_yaml_for_mode,
)
from squeakpose.project.safety import require_path_within_project
from squeakpose_core import commit_staged_paths, remove_path

ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]
Committer = Callable[[list[tuple[str, str]]], None]
logger = logging.getLogger(__name__)


def export_dataset_transaction(
    *,
    project_root: str,
    images_all_dir: str,
    labels_all_dir: str,
    final_paths: DatasetExportPaths,
    train_images: list[str],
    val_images: list[str],
    mode: str,
    classes: list[str],
    keypoint_names: list[str],
    split_seed: int,
    skipped_images: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    committer: Committer = commit_staged_paths,
) -> DatasetExportResult:
    """Build an export in staging and install it only when complete."""
    for purpose, path in (
        ("dataset source images", images_all_dir),
        ("dataset source labels", labels_all_dir),
        ("dataset export", final_paths.base_dir),
        ("dataset YAML", final_paths.dataset_yaml_path),
    ):
        require_path_within_project(
            project_root,
            path,
            purpose=purpose,
            allow_root=False,
        )
    os.makedirs(os.path.dirname(final_paths.base_dir), exist_ok=True)
    staging_base = tempfile.mkdtemp(
        prefix=f".{mode}-export-",
        dir=os.path.dirname(final_paths.base_dir),
    )
    staging_paths = dataset_export_paths_from_base(staging_base)
    try:
        result = export_dataset_files(
            images_all_dir=images_all_dir,
            labels_all_dir=labels_all_dir,
            paths=staging_paths,
            train_images=train_images,
            val_images=val_images,
            mode=mode,
            class_count=len(classes),
            keypoint_count=len(keypoint_names),
            progress_callback=progress_callback,
            cancel_requested=cancel_requested,
        )
        result.split_seed = int(split_seed)
        result.skipped_images = list(skipped_images or [])
        if result.canceled or result.errors:
            logger.warning(
                "Dataset transaction did not reach installation",
                extra={
                    "event": "dataset_transaction_incomplete",
                    "operation": "export_dataset",
                    "source_path": images_all_dir,
                    "target_path": final_paths.base_dir,
                },
            )
            return result

        result.dataset_yaml_path = write_dataset_yaml_for_mode(
            staging_paths.base_dir,
            mode,
            classes,
            keypoint_names,
            dataset_path=final_paths.base_dir,
        )
        os.makedirs(final_paths.base_dir, exist_ok=True)
        committer(
            [
                (
                    os.path.join(staging_base, "images"),
                    os.path.join(final_paths.base_dir, "images"),
                ),
                (
                    os.path.join(staging_base, "labels"),
                    os.path.join(final_paths.base_dir, "labels"),
                ),
                (staging_paths.dataset_yaml_path, final_paths.dataset_yaml_path),
            ]
        )
        result.dataset_yaml_path = final_paths.dataset_yaml_path
        logger.info(
            "Dataset transaction committed",
            extra={
                "event": "dataset_transaction_committed",
                "operation": "export_dataset",
                "source_path": images_all_dir,
                "target_path": final_paths.base_dir,
            },
        )
        return result
    except Exception:  # noqa: BLE001 - exporter and committer are injectable boundaries
        logger.exception(
            "Dataset transaction failed",
            extra={
                "event": "dataset_transaction_failed",
                "operation": "export_dataset",
                "source_path": images_all_dir,
                "target_path": final_paths.base_dir,
            },
        )
        raise
    finally:
        try:
            remove_path(staging_base)
        except OSError:
            logger.warning(
                "Could not remove dataset staging directory",
                exc_info=True,
                extra={
                    "event": "dataset_cleanup_failed",
                    "operation": "export_dataset_cleanup",
                    "target_path": staging_base,
                },
            )
