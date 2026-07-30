"""Project paths, structure, and metadata services."""

from squeakpose.project.metadata import MetadataReadResult, ProjectMetadataStore
from squeakpose.project.distillation import (
    discover_distillation_exports,
    distillation_export_search_roots,
    distillation_sample_count,
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

__all__ = [
    "MetadataReadResult",
    "ProjectMetadataStore",
    "ProjectPaths",
    "default_projects_root",
    "discover_distillation_exports",
    "distillation_export_search_roots",
    "distillation_sample_count",
    "ensure_project_structure",
    "load_last_project",
    "project_window_title",
    "preferred_distillation_export",
    "save_last_project",
]
