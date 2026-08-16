"""Qt-free image-queue navigation and deletion planning."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from squeakpose.core import filter_image_stem_collisions
from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.project.safety import require_path_within_project
from squeakpose.services.dataset_ops import (
    DATASET_DETECT,
    DATASET_POSE,
    DATASET_SEGMENT,
    dataset_export_paths,
    list_image_files,
)

LabelUsable = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class ImageQueueScan:
    images: tuple[str, ...]
    collisions: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class ImageQueueProgress:
    labeled: int
    total: int


@dataclass(frozen=True, slots=True)
class ImageDeletionPlan:
    image_name: str
    paths: tuple[str, ...]
    conflicting_names: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return bool(self.image_name) and not self.conflicting_names


@dataclass(frozen=True, slots=True)
class ImageQueueSelection:
    """Detached result of one queue-navigation transition."""

    current_index: int
    matching_indices: tuple[int, ...]

    @property
    def has_match(self) -> bool:
        return bool(self.matching_indices)

    @property
    def position(self) -> int:
        if self.current_index not in self.matching_indices:
            return 0
        return self.matching_indices.index(self.current_index) + 1


class ImageQueueNavigator:
    """Own filtered queue-index transitions without depending on Qt widgets."""

    FILTER_MODES = frozenset(("all", "labeled", "unlabeled"))

    def __init__(
        self,
        images: Iterable[str] = (),
        *,
        current_index: int = 0,
        filter_mode: str = "all",
    ):
        self.images: tuple[str, ...] = ()
        self.current_index = 0
        self.filter_mode = "all"
        self.synchronize(
            images,
            current_index=current_index,
            filter_mode=filter_mode,
        )

    def synchronize(
        self,
        images: Iterable[str],
        *,
        current_index: int | None = None,
        filter_mode: str | None = None,
    ) -> None:
        """Reconcile controller state with a refreshed compatibility queue."""

        if filter_mode is not None:
            if filter_mode not in self.FILTER_MODES:
                raise ValueError(f"unsupported image queue filter: {filter_mode}")
            self.filter_mode = filter_mode
        self.images = tuple(images)
        requested_index = self.current_index if current_index is None else int(current_index)
        self.current_index = min(max(0, requested_index), max(0, len(self.images) - 1))

    def selection(
        self,
        label_dir: str,
        *,
        label_is_usable: LabelUsable,
    ) -> ImageQueueSelection:
        matches = tuple(
            filtered_queue_indices(
                self.images,
                self.filter_mode,
                label_dir,
                label_is_usable=label_is_usable,
            )
        )
        return ImageQueueSelection(self.current_index, matches)

    def set_filter(
        self,
        mode: str,
        label_dir: str,
        *,
        label_is_usable: LabelUsable,
    ) -> ImageQueueSelection:
        if mode not in self.FILTER_MODES:
            raise ValueError(f"unsupported image queue filter: {mode}")
        self.filter_mode = mode
        result = self.selection(label_dir, label_is_usable=label_is_usable)
        if result.matching_indices and self.current_index not in result.matching_indices:
            self.current_index = result.matching_indices[0]
            result = ImageQueueSelection(self.current_index, result.matching_indices)
        return result

    def move(
        self,
        offset: int,
        label_dir: str,
        *,
        label_is_usable: LabelUsable,
    ) -> ImageQueueSelection:
        result = self.selection(label_dir, label_is_usable=label_is_usable)
        matches = result.matching_indices
        if not matches:
            return result
        if self.current_index not in matches:
            self.current_index = matches[0]
        else:
            position = matches.index(self.current_index)
            self.current_index = matches[(position + int(offset)) % len(matches)]
        return ImageQueueSelection(self.current_index, matches)

    def move_to_next_unlabeled(
        self,
        label_dir: str,
        *,
        label_is_usable: LabelUsable,
    ) -> int:
        self.current_index = next_unlabeled_index(
            self.images,
            self.current_index,
            label_dir,
            label_is_usable=label_is_usable,
        )
        return self.current_index


def scan_image_queue(images_dir: str) -> ImageQueueScan:
    """List queue images deterministically, excluding ambiguous label stems."""

    candidates = sorted(list_image_files(images_dir))
    images, collisions = filter_image_stem_collisions(candidates)
    return ImageQueueScan(tuple(images), collisions)


def image_label_path(label_dir: str, image_name: str) -> str:
    stem = os.path.splitext(image_name)[0]
    return os.path.join(label_dir, f"{stem}.txt")


def queue_progress(
    images: Iterable[str],
    label_dir: str,
    *,
    label_is_usable: LabelUsable,
) -> ImageQueueProgress:
    """Count usable labels for the supplied queue images."""

    image_names = list(images)
    labeled = sum(
        1 for image_name in image_names if label_is_usable(image_label_path(label_dir, image_name))
    )
    return ImageQueueProgress(labeled=labeled, total=len(image_names))


def next_unlabeled_index(
    images: Iterable[str],
    start_from: int,
    label_dir: str,
    *,
    label_is_usable: LabelUsable,
) -> int:
    """Return the next cyclically reachable unlabeled index."""

    image_names = list(images)
    total = len(image_names)
    if total == 0:
        return 0
    index = start_from
    for _ in range(total):
        index = (index + 1) % total
        if not label_is_usable(image_label_path(label_dir, image_names[index])):
            return index
    return start_from


def filtered_queue_indices(
    images: Iterable[str],
    mode: str,
    label_dir: str,
    *,
    label_is_usable: LabelUsable,
) -> list[int]:
    """Return indices matching the main-window navigation filter."""

    image_names = list(images)
    if mode == "all":
        return list(range(len(image_names)))
    want_labeled = mode == "labeled"
    return [
        index
        for index, image_name in enumerate(image_names)
        if label_is_usable(image_label_path(label_dir, image_name)) == want_labeled
    ]


def image_stem_conflicts(image_name: str, image_directories: Iterable[str]) -> tuple[str, ...]:
    """Return other project image names that share a case-insensitive label stem."""

    file_name = os.path.basename(image_name)
    normalized_stem = os.path.splitext(file_name)[0].casefold()
    conflicting_names: set[str] = set()
    if not normalized_stem:
        return ()
    for directory in image_directories:
        for candidate in list_image_files(directory):
            if (
                os.path.splitext(candidate)[0].casefold() == normalized_stem
                and candidate != file_name
            ):
                conflicting_names.add(candidate)
    return tuple(sorted(conflicting_names, key=str.casefold))


def plan_image_deletion(
    *,
    project_root: str,
    image_name: str,
    active_image_dir: str,
    image_dir_queue: str,
    image_dir_all: str,
    pose_label_dir: str,
    seg_label_dir: str,
    depth_image_dir: str = "",
    depth_preview_dir: str = "",
) -> ImageDeletionPlan:
    """Build a contained deletion plan and report ambiguous image stems."""

    file_name = os.path.basename(image_name)
    if not file_name:
        return ImageDeletionPlan("", (), ())
    base = os.path.splitext(file_name)[0]
    label_name = f"{base}.txt"

    conflicting_names = image_stem_conflicts(file_name, (image_dir_queue, image_dir_all))

    targets: list[tuple[str, str]] = [
        (active_image_dir, file_name),
        (image_dir_queue, file_name),
        (image_dir_all, file_name),
        (pose_label_dir, label_name),
        (seg_label_dir, label_name),
        (depth_image_dir, f"{base}.npy"),
        (depth_image_dir, f"{base}_depth.json"),
        (depth_preview_dir, f"{base}_depth.png"),
        (os.path.join(project_root, "annotations"), f"{base}_annotated.png"),
        (
            os.path.join(project_root, "annotations", LAYER_KEYPOINTS),
            f"{base}_annotated.png",
        ),
        (
            os.path.join(project_root, "annotations", LAYER_SEGMENTATION),
            f"{base}_annotated.png",
        ),
    ]
    for mode in (DATASET_POSE, DATASET_SEGMENT, DATASET_DETECT):
        dataset_paths = dataset_export_paths(project_root, mode)
        targets.extend(
            (
                (dataset_paths.images_train_dir, file_name),
                (dataset_paths.images_val_dir, file_name),
                (dataset_paths.labels_train_dir, label_name),
                (dataset_paths.labels_val_dir, label_name),
            )
        )

    paths: list[str] = []
    seen: set[str] = set()
    for directory, target_name in targets:
        if not directory:
            continue
        path = require_path_within_project(
            project_root,
            os.path.join(directory, target_name),
            purpose="image deletion target",
            allow_root=False,
        )
        normalized_path = os.path.normcase(path)
        if normalized_path in seen:
            continue
        seen.add(normalized_path)
        paths.append(path)

    return ImageDeletionPlan(
        image_name=file_name,
        paths=tuple(paths),
        conflicting_names=conflicting_names,
    )
