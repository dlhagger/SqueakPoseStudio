"""Project paths, structure, and metadata services."""

from squeakpose.project.metadata import MetadataReadResult, ProjectMetadataStore
from squeakpose.project.distillation import (
    DISTILLATION_MANIFEST_FILENAME,
    discover_distillation_exports,
    distillation_export_search_roots,
    distillation_run_task,
    distillation_sample_count,
    normalize_distillation_task,
    preferred_distillation_export,
)
from squeakpose.project.paths import (
    ProjectPaths,
    default_projects_root,
    ensure_project_structure,
    load_last_project,
    project_window_title,
    save_last_project,
)
from squeakpose.project.layers import (
    LAYER_DEFINITIONS,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    LayerDefinition,
    layer_definition,
    normalize_layer_id,
)

__all__ = [
    "MetadataReadResult",
    "ProjectMetadataStore",
    "ProjectPaths",
    "LayerDefinition",
    "LAYER_DEFINITIONS",
    "LAYER_KEYPOINTS",
    "LAYER_SEGMENTATION",
    "DISTILLATION_MANIFEST_FILENAME",
    "default_projects_root",
    "discover_distillation_exports",
    "distillation_export_search_roots",
    "distillation_run_task",
    "distillation_sample_count",
    "ensure_project_structure",
    "load_last_project",
    "layer_definition",
    "normalize_layer_id",
    "normalize_distillation_task",
    "project_window_title",
    "preferred_distillation_export",
    "save_last_project",
]
