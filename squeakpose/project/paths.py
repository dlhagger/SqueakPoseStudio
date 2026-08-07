"""Canonical filesystem layout for a SqueakPose Studio project."""

from __future__ import annotations

import datetime
import json
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass

from squeakpose.json_io import read_json_file
from squeakpose_core import CURRENT_PROJECT_SCHEMA_VERSION, atomic_write_text

from .safety import require_path_within_project

PROJECT_META_FILE = "squeakpose_project.json"
LAST_PROJECT_STATE_FILE = os.path.join(
    os.path.expanduser("~"),
    ".squeakpose_studio_last_project.json",
)


@dataclass(frozen=True, slots=True)
class ProjectPaths(Mapping[str, str]):
    """Immutable, mapping-compatible collection of project paths."""

    root: str
    videos: str
    images_to_label: str
    images_all: str
    labels_all: str
    labels_seg_all: str
    annotations: str
    annotations_keypoints: str
    annotations_segmentation: str
    depth_maps: str
    depth_images: str
    depth_previews: str
    datasets: str
    runs: str
    distillation: str
    distillation_unlabeled_images: str
    distillation_runs: str
    templates: str
    inference_outputs: str
    inference_keypoints: str
    inference_segmentation: str
    inference_depth: str
    analysis_outputs: str
    analysis_keypoints: str
    analysis_segmentation: str
    analysis_depth: str
    logs: str
    cache: str
    video_prediction_cache: str
    classes_file: str
    keypoints_file: str
    class_keypoints_file: str
    classes_seg_file: str

    @classmethod
    def from_root(cls, project_root: str) -> "ProjectPaths":
        root = os.path.abspath(project_root)
        return cls(
            root=root,
            videos=os.path.join(root, "videos"),
            images_to_label=os.path.join(root, "images_to_label"),
            images_all=os.path.join(root, "images_all"),
            labels_all=os.path.join(root, "labels_all"),
            labels_seg_all=os.path.join(root, "labels_seg_all"),
            annotations=os.path.join(root, "annotations"),
            annotations_keypoints=os.path.join(root, "annotations", "keypoints"),
            annotations_segmentation=os.path.join(root, "annotations", "segmentation"),
            depth_maps=os.path.join(root, "depth maps"),
            depth_images=os.path.join(root, "depth maps", "images"),
            depth_previews=os.path.join(root, "depth maps", "previews"),
            datasets=os.path.join(root, "datasets"),
            runs=os.path.join(root, "runs"),
            distillation=os.path.join(root, "distillation"),
            distillation_unlabeled_images=os.path.join(root, "distillation", "unlabeled_images"),
            distillation_runs=os.path.join(root, "runs", "distillation"),
            templates=os.path.join(root, "templates"),
            inference_outputs=os.path.join(root, "inference outputs"),
            inference_keypoints=os.path.join(root, "inference outputs", "keypoints"),
            inference_segmentation=os.path.join(root, "inference outputs", "segmentation"),
            inference_depth=os.path.join(root, "inference outputs", "depth"),
            analysis_outputs=os.path.join(root, "analysis outputs"),
            analysis_keypoints=os.path.join(root, "analysis outputs", "keypoints"),
            analysis_segmentation=os.path.join(root, "analysis outputs", "segmentation"),
            analysis_depth=os.path.join(root, "analysis outputs", "depth"),
            logs=os.path.join(root, "logs"),
            cache=os.path.join(root, "cache"),
            video_prediction_cache=os.path.join(root, "cache", "video_predictions"),
            classes_file=os.path.join(root, "classes.txt"),
            keypoints_file=os.path.join(root, "keypoints.txt"),
            class_keypoints_file=os.path.join(root, "class_keypoints.json"),
            classes_seg_file=os.path.join(root, "classes_seg.txt"),
        )

    def __getitem__(self, key: str) -> str:
        try:
            value = getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc
        if not isinstance(value, str):
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


PROJECT_DIRECTORY_FIELDS = (
    "videos",
    "images_to_label",
    "images_all",
    "labels_all",
    "labels_seg_all",
    "annotations",
    "annotations_keypoints",
    "annotations_segmentation",
    "depth_maps",
    "depth_images",
    "depth_previews",
    "datasets",
    "runs",
    "distillation",
    "distillation_unlabeled_images",
    "distillation_runs",
    "templates",
    "inference_outputs",
    "inference_keypoints",
    "inference_segmentation",
    "inference_depth",
    "analysis_outputs",
    "analysis_keypoints",
    "analysis_segmentation",
    "analysis_depth",
    "logs",
    "cache",
    "video_prediction_cache",
)


def ensure_project_structure(
    project_root: str,
    *,
    default_segmentation_classes: tuple[str, ...] = ("mouse",),
) -> ProjectPaths:
    """Create missing project entries and return their canonical paths."""
    paths = ProjectPaths.from_root(project_root)
    for field_name in PROJECT_DIRECTORY_FIELDS:
        directory = require_path_within_project(
            paths.root,
            paths[field_name],
            purpose=f"project directory '{field_name}'",
            allow_root=False,
        )
        os.makedirs(directory, exist_ok=True)

    for field_name in (
        "classes_file",
        "keypoints_file",
        "class_keypoints_file",
        "classes_seg_file",
    ):
        require_path_within_project(
            paths.root,
            paths[field_name],
            purpose=f"project file '{field_name}'",
            allow_root=False,
        )

    if not os.path.exists(paths.classes_seg_file):
        atomic_write_text(
            paths.classes_seg_file,
            "".join(f"{name}\n" for name in default_segmentation_classes),
        )

    metadata_path = os.path.join(paths.root, PROJECT_META_FILE)
    metadata_path = require_path_within_project(
        paths.root,
        metadata_path,
        purpose="project metadata path",
        allow_root=False,
    )
    if not os.path.exists(metadata_path):
        payload = {
            "schema_version": CURRENT_PROJECT_SCHEMA_VERSION,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_text(metadata_path, json.dumps(payload, indent=2))
    return paths


def project_window_title(project_root: str) -> str:
    root = os.path.abspath(project_root)
    name = os.path.basename(root.rstrip(os.sep)) or root
    return f"SqueakPose Studio — {name}"


def default_projects_root(*, platform_name: str | None = None) -> str:
    """Return the default parent directory for user projects."""
    current_platform = platform_name or sys.platform
    if current_platform.startswith("linux"):
        xdg_docs = os.environ.get("XDG_DOCUMENTS_DIR", "").strip()
        if xdg_docs:
            return os.path.join(os.path.expanduser(xdg_docs), "SqueakPose Studio Projects")
    return os.path.join(
        os.path.expanduser("~"),
        "Documents",
        "SqueakPose Studio Projects",
    )


def load_last_project(*, state_file: str = LAST_PROJECT_STATE_FILE) -> str | None:
    """Return the last existing project recorded by the launcher."""
    if not os.path.exists(state_file):
        return None
    try:
        data = read_json_file(state_file, max_bytes=64 * 1024, require_object=True)
        path = str(data.get("last_project", "")).strip()
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return None


def save_last_project(
    project_root: str,
    *,
    state_file: str = LAST_PROJECT_STATE_FILE,
) -> None:
    """Atomically remember the last project selected by the launcher."""
    payload = {"last_project": os.path.abspath(project_root)}
    atomic_write_text(state_file, json.dumps(payload, indent=2))
