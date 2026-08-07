"""Application package for SqueakPose Studio.

The top-level ``squeakpose_studio.py`` module remains the supported launcher
while implementation responsibilities are moved into this package.
"""

import logging

from squeakpose.project.paths import ProjectPaths

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ProjectPaths"]
