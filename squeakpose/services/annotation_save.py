"""Transactional persistence for one image annotation."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from squeakpose.core import (
    commit_staged_paths,
    remove_path,
    stage_copy_file,
    stage_text_file,
    staging_path_for,
)
from squeakpose.project.safety import require_path_within_project

Committer = Callable[[list[tuple[str, str]]], None]
OverlayRenderer = Callable[[str], bool]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnnotationSaveRequest:
    project_root: str
    source_image_path: str
    image_output_path: str
    label_output_path: str
    overlay_output_path: str
    label_text: str


@dataclass(frozen=True, slots=True)
class AnnotationSaveResult:
    image_path: str
    label_path: str
    overlay_path: str


def save_annotation_transaction(
    request: AnnotationSaveRequest,
    *,
    render_overlay: OverlayRenderer,
    committer: Committer = commit_staged_paths,
) -> AnnotationSaveResult:
    """Stage and atomically install an image, label, and rendered overlay."""
    if not request.label_text.strip():
        raise ValueError("annotation label text must not be empty")
    for purpose, target in (
        ("saved annotation image", request.image_output_path),
        ("annotation label", request.label_output_path),
        ("annotation preview", request.overlay_output_path),
    ):
        require_path_within_project(
            request.project_root,
            target,
            purpose=purpose,
            allow_root=False,
        )
    source_exists = bool(request.source_image_path) and os.path.isfile(request.source_image_path)
    output_exists = os.path.isfile(request.image_output_path)
    if not source_exists and not output_exists:
        raise FileNotFoundError("annotation source image is missing")

    for target in (
        request.image_output_path,
        request.label_output_path,
        request.overlay_output_path,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)

    staged: list[tuple[str, str]] = []
    try:
        if source_exists and os.path.abspath(request.source_image_path) != os.path.abspath(
            request.image_output_path
        ):
            staged.append(
                (
                    stage_copy_file(
                        request.source_image_path,
                        request.image_output_path,
                    ),
                    request.image_output_path,
                )
            )

        staged_overlay = staging_path_for(request.overlay_output_path)
        staged.append((staged_overlay, request.overlay_output_path))
        if not render_overlay(staged_overlay):
            raise OSError("could not render the annotated preview image")

        staged.append(
            (
                stage_text_file(request.label_output_path, request.label_text),
                request.label_output_path,
            )
        )
        committer(staged)
    except Exception:  # noqa: BLE001 - transaction callbacks may raise arbitrary failures
        logger.exception(
            "Annotation transaction failed",
            extra={
                "event": "annotation_transaction_failed",
                "operation": "save_annotation",
                "source_path": request.source_image_path,
                "target_path": request.label_output_path,
            },
        )
        for staged_path, _ in staged:
            try:
                remove_path(staged_path)
            except OSError:
                logger.warning(
                    "Could not remove staged annotation artifact",
                    exc_info=True,
                    extra={
                        "event": "annotation_cleanup_failed",
                        "operation": "save_annotation_cleanup",
                        "target_path": staged_path,
                    },
                )
        raise

    logger.info(
        "Annotation transaction committed",
        extra={
            "event": "annotation_transaction_committed",
            "operation": "save_annotation",
            "source_path": request.source_image_path,
            "target_path": request.label_output_path,
        },
    )

    return AnnotationSaveResult(
        image_path=request.image_output_path,
        label_path=request.label_output_path,
        overlay_path=request.overlay_output_path,
    )
