"""Remove generated depth outputs while preserving project inputs and models."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from squeakpose.project.paths import ProjectPaths
from squeakpose.project.safety import ProjectPathError, require_path_within_project


def delete_project_depth_data(project_root: str) -> None:
    """Clear only depth output folders, rejecting redirected managed directories."""
    paths = ProjectPaths.from_root(project_root)
    targets = (paths.depth_maps, paths.inference_depth, paths.analysis_depth)
    # Validate every target before deleting anything. Parent links could redirect
    # even a depth-only path into another layer or outside the project.
    for target in targets:
        require_path_within_project(project_root, target, allow_root=False)
        current = Path(target)
        root = Path(paths.root)
        while current != root:
            if current.is_symlink():
                raise ProjectPathError(f"Depth output directory is a symbolic link: {current}")
            current = current.parent
        if os.path.lexists(target) and not os.path.isdir(target):
            raise ProjectPathError(f"Depth output path is not a directory: {target}")
    for target in targets:
        if os.path.isdir(target):
            # rmtree removes nested links themselves, without following them.
            shutil.rmtree(target)
    for directory in (
        paths.depth_images,
        paths.depth_previews,
        paths.inference_depth,
        paths.analysis_depth,
    ):
        os.makedirs(directory, exist_ok=True)
