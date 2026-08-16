"""Compatibility exports for :mod:`squeakpose.core`.

New package code should import the implementation from ``squeakpose.core``.
This module remains available for scripts and third-party callers.
"""

from squeakpose.core import *  # noqa: F403
from squeakpose.core import os as os
