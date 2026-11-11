"""Compatibility path for the SAM checkpoint catalog.

The catalog is Qt-free, dependency-free data used by project state as well as services,
so its canonical home is :mod:`squeakpose.sam_models` (base layer). This module keeps
``squeakpose.services.sam_models`` imports working.
"""

from __future__ import annotations

from squeakpose.sam_models import (
    DEFAULT_SAM_VARIANT,
    SAM_MODELS,
    SamModel,
    get_sam_model,
    official_sam_variant,
    resolve_sam_variant,
    sam_download_target,
)

__all__ = [
    "DEFAULT_SAM_VARIANT",
    "SAM_MODELS",
    "SamModel",
    "get_sam_model",
    "official_sam_variant",
    "resolve_sam_variant",
    "sam_download_target",
]
