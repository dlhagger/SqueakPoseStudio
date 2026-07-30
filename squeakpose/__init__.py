"""Application package for SqueakPose Studio.

The top-level ``squeakpose_studio.py`` module remains the supported launcher
while implementation responsibilities are moved into this package.
"""

from squeakpose.project.paths import ProjectPaths

__all__ = ["ProjectPaths"]
