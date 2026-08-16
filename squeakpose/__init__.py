"""Application package for SqueakPose Studio.

The top-level ``squeakpose_studio.py`` module remains the supported launcher
while implementation responsibilities are moved into this package.
"""

import logging
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

if TYPE_CHECKING:
    from squeakpose.project.paths import ProjectPaths

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ProjectPaths", "__version__"]


def __getattr__(name: str) -> Any:
    """Load convenience exports without eagerly importing the project package."""

    if name == "ProjectPaths":
        from squeakpose.project.paths import ProjectPaths

        return ProjectPaths
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
