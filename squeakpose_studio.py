#!/usr/bin/env python3
"""Backward-compatible imports and launcher for SqueakPose Studio."""

from squeakpose.ui.main_window import *  # noqa: F401,F403
from squeakpose.ui.main_window import (
    _default_projects_root,
    _discover_distillation_exports,
    _distillation_export_search_roots,
    _distillation_sample_count,
    _ensure_project_structure,
    _ensure_qt_plugin_paths,
    _load_last_project,
    _project_paths,
    _project_window_title,
    _qt_app_instance,
    _retain_main_window,
    _save_last_project,
)

if __name__ == "__main__":
    from squeakpose.app import run

    raise SystemExit(run())
