"""Qt-free active-project session state and preference normalization."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from squeakpose.project.layers import (
    LAYER_DEFINITIONS,
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    layer_worker_mode,
    normalize_layer_id,
    normalize_layer_settings,
)
from squeakpose.project.metadata import ProjectMetadataStore
from squeakpose.project.paths import ProjectPaths
from squeakpose.project.safety import ProjectPathError, require_path_within_project

PathExists = Callable[[str], bool]
BUILTIN_DEPTH_MODEL_REFERENCES = frozenset(f"yolo26{size}-depth.pt" for size in "nslmx")


def is_builtin_model_reference(reference: Any) -> bool:
    return str(reference or "").strip().lower() in BUILTIN_DEPTH_MODEL_REFERENCES


def resolve_model_reference(project_root: str, reference: Any) -> str:
    """Resolve a metadata model reference, rejecting relative project escapes."""

    raw = str(reference or "").strip()
    if not raw:
        return ""
    if is_builtin_model_reference(raw):
        return raw
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return require_path_within_project(
        project_root,
        os.path.join(os.path.abspath(project_root), raw),
        purpose="project-relative model reference",
        allow_root=False,
    )


def store_model_reference(project_root: str, path: Any) -> str:
    """Serialize in-project models relatively and external models absolutely."""

    raw = str(path or "").strip()
    if not raw:
        return ""
    if is_builtin_model_reference(raw):
        return raw
    if not os.path.isabs(raw):
        raw = resolve_model_reference(project_root, raw)
    stored = ProjectMetadataStore(project_root).store_path(raw)
    if stored and not os.path.isabs(stored):
        resolve_model_reference(project_root, stored)
    return stored


@dataclass(frozen=True, slots=True)
class ActiveProjectPaths:
    project_root: str
    image_queue: str
    image_all: str
    label_dir: str
    class_file: str
    keypoint_file: str
    class_keypoints_file: str


@dataclass(frozen=True, slots=True)
class LayerSessionSnapshot:
    layer_id: str
    classes: tuple[str, ...]
    keypoints: tuple[str, ...]
    class_keypoints: tuple[tuple[str, tuple[str, ...]], ...]
    selected_class_id: int
    model_path: str

    def class_keypoint_mapping(self) -> dict[str, list[str]]:
        return {name: list(keypoints) for name, keypoints in self.class_keypoints}

    @property
    def selected_class_name(self) -> str:
        if 0 <= self.selected_class_id < len(self.classes):
            return self.classes[self.selected_class_id]
        return ""


@dataclass(frozen=True, slots=True)
class ProjectSessionSnapshot:
    project_root: str
    active_layer: str
    active_workflow: str
    layer_visibility: tuple[tuple[str, bool], ...]
    layers: tuple[LayerSessionSnapshot, ...]
    assistant_model_path: str

    def layer(self, layer_id: Any) -> LayerSessionSnapshot:
        normalized = normalize_layer_id(layer_id, default="")
        if not normalized:
            raise KeyError(str(layer_id))
        for state in self.layers:
            if state.layer_id == normalized:
                return state
        raise KeyError(normalized)

    @property
    def active(self) -> LayerSessionSnapshot:
        return self.layer(self.active_layer)


@dataclass(frozen=True, slots=True)
class ProjectSessionTransition:
    before: ProjectSessionSnapshot
    after: ProjectSessionSnapshot

    @property
    def changed(self) -> bool:
        return self.before.active_layer != self.after.active_layer


@dataclass(slots=True)
class _LayerSessionState:
    classes: list[str]
    keypoints: list[str]
    class_keypoints: dict[str, list[str]]
    selected_class_id: int = -1
    model_path: str = ""


class ProjectSession:
    """Detached state for one active project and its annotation layers."""

    def __init__(
        self,
        project_root: str,
        *,
        paths: ProjectPaths | None = None,
        active_layer: Any = LAYER_KEYPOINTS,
        pose_classes: Sequence[str] = (),
        pose_keypoints: Sequence[str] = (),
        pose_class_keypoints: Mapping[str, Sequence[str]] | None = None,
        segmentation_classes: Sequence[str] = (),
        selected_class_ids: Mapping[str, int] | None = None,
        layer_model_paths: Mapping[str, str] | None = None,
        layer_visibility: Mapping[str, bool] | None = None,
        layer_settings: Mapping[str, Any] | None = None,
        assistant_model_path: str = "",
    ):
        self.project_root = os.path.abspath(project_root)
        self.paths = paths or ProjectPaths.from_root(self.project_root)
        if os.path.abspath(self.paths.root) != self.project_root:
            raise ValueError("project paths do not belong to the session root")
        pose_classes_clean, pose_keypoints_clean, pose_map = _normalize_pose_schema(
            pose_classes,
            pose_keypoints,
            pose_class_keypoints or {},
        )
        segmentation_classes_clean = _clean_names(segmentation_classes)
        selected = selected_class_ids or {}
        models = layer_model_paths or {}
        self._layers: dict[str, _LayerSessionState] = {
            LAYER_KEYPOINTS: _LayerSessionState(
                classes=pose_classes_clean,
                keypoints=pose_keypoints_clean,
                class_keypoints=pose_map,
                selected_class_id=_normalize_selected_id(
                    selected.get(LAYER_KEYPOINTS, 0), pose_classes_clean
                ),
                model_path=str(models.get(LAYER_KEYPOINTS) or ""),
            ),
            LAYER_SEGMENTATION: _LayerSessionState(
                classes=segmentation_classes_clean,
                keypoints=[],
                class_keypoints={},
                selected_class_id=_normalize_selected_id(
                    selected.get(LAYER_SEGMENTATION, 0), segmentation_classes_clean
                ),
                model_path=str(models.get(LAYER_SEGMENTATION) or ""),
            ),
            LAYER_DEPTH: _LayerSessionState(
                classes=[],
                keypoints=[],
                class_keypoints={},
                selected_class_id=-1,
                model_path=str(models.get(LAYER_DEPTH) or ""),
            ),
        }
        self.active_layer = normalize_layer_id(active_layer)
        self.layer_visibility = {layer_id: True for layer_id in LAYER_DEFINITIONS}
        for layer_id, visible in (layer_visibility or {}).items():
            normalized = normalize_layer_id(layer_id, default="")
            if normalized in self.layer_visibility:
                self.layer_visibility[normalized] = bool(visible)
        self.layer_visibility[self.active_layer] = True
        self.layer_settings = normalize_layer_settings(layer_settings or {})
        self.assistant_model_path = str(assistant_model_path or "")

    @classmethod
    def from_preferences(
        cls,
        project_root: str,
        preferences: Mapping[str, Any],
        *,
        paths: ProjectPaths | None = None,
        pose_classes: Sequence[str] = (),
        pose_keypoints: Sequence[str] = (),
        pose_class_keypoints: Mapping[str, Sequence[str]] | None = None,
        segmentation_classes: Sequence[str] = (),
        selected_class_ids: Mapping[str, int] | None = None,
        path_is_file: PathExists = os.path.isfile,
    ) -> ProjectSession:
        """Build a session from a detached project metadata preference mapping."""

        metadata = dict(preferences) if isinstance(preferences, Mapping) else {}
        settings = normalize_layer_settings(metadata.get("layers"))
        model_paths: dict[str, str] = {}
        for layer_id in LAYER_DEFINITIONS:
            reference = settings[layer_id].get("model_path")
            resolved = _resolve_persisted_reference(project_root, reference)
            model_paths[layer_id] = (
                resolved
                if resolved and (is_builtin_model_reference(resolved) or path_is_file(resolved))
                else ""
            )

        raw_visibility = metadata.get("layer_visibility")
        visibility = raw_visibility if isinstance(raw_visibility, Mapping) else {}
        assistant_reference = metadata.get("sam_model_path")
        if not assistant_reference:
            assistant_reference = settings[LAYER_SEGMENTATION].get("assistant_model_path")
        assistant_path = _resolve_persisted_reference(project_root, assistant_reference)
        if not assistant_path or not path_is_file(assistant_path):
            assistant_path = ""

        return cls(
            project_root,
            paths=paths,
            active_layer=metadata.get("active_layer") or metadata.get("active_workflow"),
            pose_classes=pose_classes,
            pose_keypoints=pose_keypoints,
            pose_class_keypoints=pose_class_keypoints,
            segmentation_classes=segmentation_classes,
            selected_class_ids=selected_class_ids,
            layer_model_paths=model_paths,
            layer_visibility=visibility,
            layer_settings=settings,
            assistant_model_path=assistant_path,
        )

    @property
    def active_workflow(self) -> str:
        return layer_worker_mode(self.active_layer)

    @property
    def active_paths(self) -> ActiveProjectPaths:
        if self.active_layer == LAYER_KEYPOINTS:
            return ActiveProjectPaths(
                self.project_root,
                self.paths.images_to_label,
                self.paths.images_all,
                self.paths.labels_all,
                self.paths.classes_file,
                self.paths.keypoints_file,
                self.paths.class_keypoints_file,
            )
        if self.active_layer == LAYER_SEGMENTATION:
            return ActiveProjectPaths(
                self.project_root,
                self.paths.images_to_label,
                self.paths.images_all,
                self.paths.labels_seg_all,
                self.paths.classes_seg_file,
                "",
                "",
            )
        return ActiveProjectPaths(
            self.project_root,
            self.paths.images_to_label,
            self.paths.images_all,
            self.paths.depth_images,
            "",
            "",
            "",
        )

    @property
    def active_state(self) -> LayerSessionSnapshot:
        return self._layer_snapshot(self.active_layer)

    @property
    def active_model_path(self) -> str:
        return self._layers[self.active_layer].model_path

    def set_model_path(self, layer_id: Any, model_path: str) -> None:
        normalized = normalize_layer_id(layer_id)
        self._layers[normalized].model_path = str(model_path or "")

    def set_layer_visibility(self, layer_id: Any, visible: bool) -> None:
        normalized = normalize_layer_id(layer_id)
        self.layer_visibility[normalized] = (
            True if normalized == self.active_layer else bool(visible)
        )

    def select_class(self, selected: int | str, *, layer_id: Any | None = None) -> int:
        normalized = normalize_layer_id(layer_id or self.active_layer)
        state = self._layers[normalized]
        if isinstance(selected, str):
            try:
                selected_id = state.classes.index(selected)
            except ValueError:
                selected_id = -1
        else:
            selected_id = int(selected)
        state.selected_class_id = _normalize_selected_id(selected_id, state.classes)
        return state.selected_class_id

    def capture_active_state(
        self,
        *,
        classes: Sequence[str] | None = None,
        keypoints: Sequence[str] | None = None,
        class_keypoints: Mapping[str, Sequence[str]] | None = None,
        selected_class_id: int | None = None,
        model_path: str | None = None,
    ) -> LayerSessionSnapshot:
        """Persist detached UI state into the currently active layer."""

        state = self._layers[self.active_layer]
        if self.active_layer == LAYER_KEYPOINTS:
            normalized = _normalize_pose_schema(
                state.classes if classes is None else classes,
                state.keypoints if keypoints is None else keypoints,
                state.class_keypoints if class_keypoints is None else class_keypoints,
            )
            state.classes, state.keypoints, state.class_keypoints = normalized
        elif self.active_layer == LAYER_SEGMENTATION and classes is not None:
            state.classes = _clean_names(classes)
        if self.active_layer != LAYER_DEPTH:
            requested_selection = (
                state.selected_class_id if selected_class_id is None else selected_class_id
            )
            state.selected_class_id = _normalize_selected_id(requested_selection, state.classes)
        if model_path is not None:
            state.model_path = str(model_path or "")
        return self._layer_snapshot(self.active_layer)

    def transition_to(self, layer_id: Any) -> ProjectSessionTransition:
        """Switch layers and return detached before/after session snapshots."""

        before = self.snapshot()
        self.active_layer = normalize_layer_id(layer_id)
        self.layer_visibility[self.active_layer] = True
        after = self.snapshot()
        return ProjectSessionTransition(before=before, after=after)

    def transition_workflow(self, workflow: Any) -> ProjectSessionTransition:
        return self.transition_to(workflow)

    def snapshot(self) -> ProjectSessionSnapshot:
        return ProjectSessionSnapshot(
            project_root=self.project_root,
            active_layer=self.active_layer,
            active_workflow=self.active_workflow,
            layer_visibility=tuple(
                (layer_id, bool(self.layer_visibility[layer_id])) for layer_id in LAYER_DEFINITIONS
            ),
            layers=tuple(self._layer_snapshot(layer_id) for layer_id in LAYER_DEFINITIONS),
            assistant_model_path=self.assistant_model_path,
        )

    def to_preferences(self, *, path_is_file: PathExists = os.path.isfile) -> dict[str, Any]:
        """Return a detached mapping suitable for ``ProjectMetadataStore.update``."""

        settings = normalize_layer_settings(self.layer_settings)
        for layer_id, state in self._layers.items():
            if state.model_path:
                settings[layer_id]["model_path"] = store_model_reference(
                    self.project_root, state.model_path
                )
            else:
                settings[layer_id].pop("model_path", None)

        payload: dict[str, Any] = {
            "active_layer": self.active_layer,
            "active_workflow": self.active_workflow,
            "layers": settings,
            "layer_visibility": {
                layer_id: bool(self.layer_visibility[layer_id]) for layer_id in LAYER_DEFINITIONS
            },
        }
        if self.assistant_model_path and path_is_file(self.assistant_model_path):
            stored_assistant = store_model_reference(self.project_root, self.assistant_model_path)
            payload["sam_model_path"] = stored_assistant
            settings[LAYER_SEGMENTATION]["assistant_model_path"] = stored_assistant
        else:
            payload["sam_model_path"] = None
            settings[LAYER_SEGMENTATION].pop("assistant_model_path", None)
        self.layer_settings = normalize_layer_settings(settings)
        return payload

    def _layer_snapshot(self, layer_id: str) -> LayerSessionSnapshot:
        state = self._layers[layer_id]
        return LayerSessionSnapshot(
            layer_id=layer_id,
            classes=tuple(state.classes),
            keypoints=tuple(state.keypoints),
            class_keypoints=tuple(
                (name, tuple(state.class_keypoints.get(name, ()))) for name in state.classes
            ),
            selected_class_id=state.selected_class_id,
            model_path=state.model_path,
        )


def _resolve_persisted_reference(project_root: str, reference: Any) -> str:
    try:
        return resolve_model_reference(project_root, reference)
    except ProjectPathError:
        return ""


def _clean_names(values: Sequence[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _normalize_pose_schema(
    classes: Sequence[str],
    keypoints: Sequence[str],
    class_keypoints: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    class_names = _clean_names(classes)
    canonical = _clean_names(keypoints)
    mapping: dict[str, list[str]] = {}
    for class_name in class_names:
        selected = _clean_names(class_keypoints.get(class_name, canonical))
        mapping[class_name] = selected or canonical[:]
        for keypoint_name in mapping[class_name]:
            if keypoint_name not in canonical:
                canonical.append(keypoint_name)
    return class_names, canonical, mapping


def _normalize_selected_id(selected: Any, classes: Sequence[str]) -> int:
    if not classes:
        return -1
    try:
        selected_id = int(selected)
    except (TypeError, ValueError):
        selected_id = 0
    return min(max(0, selected_id), len(classes) - 1)
