"""Qt-free file coordination for one frame's typed annotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    PoseDocumentSnapshot,
    SegmentationAnnotationDocument,
    SegmentationDocumentSnapshot,
)
from squeakpose.annotation.models import BoundingBox
from squeakpose.annotation.segmentation import polygon_bounds
from squeakpose.annotation.serialization import (
    load_pose_annotations_from_file,
    load_segmentation_annotations_from_file,
    pose_annotation_to_line,
    segmentation_annotation_to_line,
)
from squeakpose.services.annotation_save import AnnotationSaveRequest


@dataclass(frozen=True, slots=True)
class PoseFrameLoadResult:
    """A typed pose document plus the loader's recovery diagnostic."""

    document: PoseAnnotationDocument
    extra_keypoint_rows: int = 0


@dataclass(frozen=True, slots=True)
class SegmentationBoxTransferPlan:
    """A class-name-matched segmentation box ready for a pose document."""

    pose_class_id: int
    segmentation_class_id: int
    class_name: str
    box: BoundingBox


class SegmentationBoxUnavailableError(ValueError):
    """Raised when no unambiguous saved segmentation can supply a pose box."""


def plan_segmentation_box_transfer(
    *,
    pose_class_id: int,
    pose_classes: Sequence[str],
    segmentation_classes: Sequence[str],
    segmentation_document: SegmentationAnnotationDocument,
) -> SegmentationBoxTransferPlan:
    """Match a pose class by name and derive tight bounds from its segmentation."""
    target_id = int(pose_class_id)
    if target_id < 0 or target_id >= len(pose_classes):
        raise SegmentationBoxUnavailableError("Select a valid keypoint class first.")
    class_name = str(pose_classes[target_id]).strip()
    exact_matches = [
        index
        for index, candidate in enumerate(segmentation_classes)
        if str(candidate).strip() == class_name
    ]
    if not exact_matches:
        folded_matches = [
            index
            for index, candidate in enumerate(segmentation_classes)
            if str(candidate).strip().casefold() == class_name.casefold()
        ]
        if len(folded_matches) != 1:
            raise SegmentationBoxUnavailableError(
                f"No matching segmentation class named '{class_name}'."
            )
        source_id = folded_matches[0]
    else:
        source_id = exact_matches[0]

    annotation = segmentation_document.annotation(source_id)
    if annotation is None:
        raise SegmentationBoxUnavailableError(
            f"No saved segmentation exists for '{class_name}' on this image."
        )
    bounds = polygon_bounds(annotation.segments)
    if bounds is None:
        raise SegmentationBoxUnavailableError(
            f"The saved segmentation for '{class_name}' has no usable bounds."
        )
    return SegmentationBoxTransferPlan(
        pose_class_id=target_id,
        segmentation_class_id=source_id,
        class_name=class_name,
        box=BoundingBox(*bounds, class_id=target_id),
    )


def load_pose_document(
    label_file: str,
    *,
    classes_count: int,
    canonical_names: Sequence[str],
    class_keypoint_lookup: Sequence[Sequence[str]],
    image_width: float,
    image_height: float,
) -> PoseFrameLoadResult:
    """Load one pose label file without changing its tolerant row semantics."""
    entries, extra_rows = load_pose_annotations_from_file(
        label_file,
        classes_count=classes_count,
        canonical_names=list(canonical_names),
        class_keypoint_lookup=[list(names) for names in class_keypoint_lookup],
        img_w=image_width,
        img_h=image_height,
    )
    return PoseFrameLoadResult(
        document=PoseAnnotationDocument(entries),
        extra_keypoint_rows=extra_rows,
    )


def load_segmentation_document(
    label_file: str,
    *,
    classes_count: int,
    image_width: float,
    image_height: float,
) -> SegmentationAnnotationDocument:
    """Load one segmentation label file into its typed document owner."""
    entries = load_segmentation_annotations_from_file(
        label_file,
        classes_count=classes_count,
        img_w=image_width,
        img_h=image_height,
    )
    return SegmentationAnnotationDocument(entries)


def serialize_pose_snapshot(
    snapshot: PoseDocumentSnapshot,
    *,
    canonical_names: Sequence[str],
    image_width: float,
    image_height: float,
) -> str:
    """Serialize a detached typed pose snapshot using the canonical row codec."""
    lines = [
        pose_annotation_to_line(
            annotation.as_dict(),
            kp_names=list(canonical_names),
            img_w=image_width,
            img_h=image_height,
        )
        for annotation in snapshot.annotations
    ]
    return _label_text(lines)


def serialize_segmentation_snapshot(
    snapshot: SegmentationDocumentSnapshot,
    *,
    image_width: float,
    image_height: float,
) -> str:
    """Serialize a detached typed segmentation snapshot using the canonical codec."""
    lines = [
        segmentation_annotation_to_line(
            annotation.as_dict(),
            img_w=image_width,
            img_h=image_height,
        )
        for annotation in snapshot.annotations
    ]
    return _label_text(lines)


def build_pose_save_request(
    snapshot: PoseDocumentSnapshot,
    *,
    canonical_names: Sequence[str],
    image_width: float,
    image_height: float,
    project_root: str,
    source_image_path: str,
    image_output_path: str,
    label_output_path: str,
    overlay_output_path: str,
) -> AnnotationSaveRequest:
    """Build the existing transaction input from a detached pose snapshot."""
    return _save_request(
        label_text=serialize_pose_snapshot(
            snapshot,
            canonical_names=canonical_names,
            image_width=image_width,
            image_height=image_height,
        ),
        project_root=project_root,
        source_image_path=source_image_path,
        image_output_path=image_output_path,
        label_output_path=label_output_path,
        overlay_output_path=overlay_output_path,
    )


def build_segmentation_save_request(
    snapshot: SegmentationDocumentSnapshot,
    *,
    image_width: float,
    image_height: float,
    project_root: str,
    source_image_path: str,
    image_output_path: str,
    label_output_path: str,
    overlay_output_path: str,
) -> AnnotationSaveRequest:
    """Build the existing transaction input from a detached segmentation snapshot."""
    return _save_request(
        label_text=serialize_segmentation_snapshot(
            snapshot,
            image_width=image_width,
            image_height=image_height,
        ),
        project_root=project_root,
        source_image_path=source_image_path,
        image_output_path=image_output_path,
        label_output_path=label_output_path,
        overlay_output_path=overlay_output_path,
    )


def _label_text(lines: Sequence[str]) -> str:
    usable = [line for line in lines if line]
    return "\n".join(usable) + ("\n" if usable else "")


def _save_request(
    *,
    label_text: str,
    project_root: str,
    source_image_path: str,
    image_output_path: str,
    label_output_path: str,
    overlay_output_path: str,
) -> AnnotationSaveRequest:
    return AnnotationSaveRequest(
        project_root=project_root,
        source_image_path=source_image_path,
        image_output_path=image_output_path,
        label_output_path=label_output_path,
        overlay_output_path=overlay_output_path,
        label_text=label_text,
    )


__all__ = [
    "PoseFrameLoadResult",
    "SegmentationBoxTransferPlan",
    "SegmentationBoxUnavailableError",
    "build_pose_save_request",
    "build_segmentation_save_request",
    "load_pose_document",
    "load_segmentation_document",
    "plan_segmentation_box_transfer",
    "serialize_pose_snapshot",
    "serialize_segmentation_snapshot",
]
