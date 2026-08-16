"""Project paths, structure, and metadata services.

Convenience exports are resolved lazily so importing one project submodule does
not initialize recovery, health, metadata, and session modules as a side effect.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "DISTILLATION_MANIFEST_FILENAME": "squeakpose.project.distillation",
    "discover_distillation_exports": "squeakpose.project.distillation",
    "distillation_export_search_roots": "squeakpose.project.distillation",
    "distillation_run_task": "squeakpose.project.distillation",
    "distillation_sample_count": "squeakpose.project.distillation",
    "normalize_distillation_task": "squeakpose.project.distillation",
    "preferred_distillation_export": "squeakpose.project.distillation",
    "ProjectHealthReport": "squeakpose.project.health",
    "cleanup_project_temporary_paths": "squeakpose.project.health",
    "format_project_health_summary": "squeakpose.project.health",
    "scan_project_health": "squeakpose.project.health",
    "LAYER_DEFINITIONS": "squeakpose.project.layers",
    "LAYER_DEPTH": "squeakpose.project.layers",
    "LAYER_KEYPOINTS": "squeakpose.project.layers",
    "LAYER_SEGMENTATION": "squeakpose.project.layers",
    "LayerDefinition": "squeakpose.project.layers",
    "layer_definition": "squeakpose.project.layers",
    "normalize_layer_id": "squeakpose.project.layers",
    "MetadataReadResult": "squeakpose.project.metadata",
    "ProjectMetadataStore": "squeakpose.project.metadata",
    "ProjectPaths": "squeakpose.project.paths",
    "default_projects_root": "squeakpose.project.paths",
    "ensure_project_structure": "squeakpose.project.paths",
    "load_last_project": "squeakpose.project.paths",
    "project_window_title": "squeakpose.project.paths",
    "save_last_project": "squeakpose.project.paths",
    "TransactionBackup": "squeakpose.project.recovery",
    "TransactionRecoveryReport": "squeakpose.project.recovery",
    "TransactionRecoveryResult": "squeakpose.project.recovery",
    "cleanup_transaction_staging": "squeakpose.project.recovery",
    "restore_missing_transaction_targets": "squeakpose.project.recovery",
    "scan_transaction_artifacts": "squeakpose.project.recovery",
    "ProjectLock": "squeakpose.project.safety",
    "ProjectLockedError": "squeakpose.project.safety",
    "ProjectLockInfo": "squeakpose.project.safety",
    "ProjectPathError": "squeakpose.project.safety",
    "break_stale_project_lock": "squeakpose.project.safety",
    "inspect_project_lock": "squeakpose.project.safety",
    "is_path_within_project": "squeakpose.project.safety",
    "require_path_within_project": "squeakpose.project.safety",
    "ActiveProjectPaths": "squeakpose.project.session",
    "LayerSessionSnapshot": "squeakpose.project.session",
    "ProjectSession": "squeakpose.project.session",
    "ProjectSessionSnapshot": "squeakpose.project.session",
    "ProjectSessionTransition": "squeakpose.project.session",
    "is_builtin_model_reference": "squeakpose.project.session",
    "resolve_model_reference": "squeakpose.project.session",
    "store_model_reference": "squeakpose.project.session",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
