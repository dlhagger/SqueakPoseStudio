"""Transactional persistence for one image annotation."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from squeakpose_core import (
    commit_staged_paths,
    remove_path,
    stage_copy_file,
    stage_text_file,
    staging_path_for,
)

Committer = Callable[[list[tuple[str, str]]], None]
OverlayRenderer = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class AnnotationSaveRequest:
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
    source_exists = bool(request.source_image_path) and os.path.isfile(
        request.source_image_path
    )
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
        if (
            source_exists
            and os.path.abspath(request.source_image_path)
            != os.path.abspath(request.image_output_path)
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
        if not render_overlay(staged_overlay):
            remove_path(staged_overlay)
            raise OSError("could not render the annotated preview image")
        staged.append((staged_overlay, request.overlay_output_path))

        staged.append(
            (
                stage_text_file(request.label_output_path, request.label_text),
                request.label_output_path,
            )
        )
        committer(staged)
    except Exception:
        for staged_path, _ in staged:
            try:
                remove_path(staged_path)
            except OSError:
                pass
        raise

    return AnnotationSaveResult(
        image_path=request.image_output_path,
        label_path=request.label_output_path,
        overlay_path=request.overlay_output_path,
    )
