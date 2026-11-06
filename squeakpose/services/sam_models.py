"""Allow-listed official SAM image checkpoints and Qt-free model resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class SamModel:
    key: str
    display_name: str
    repo_id: str
    filename: str
    access_url: str
    approximate_size: str = ""


SAM_MODELS = MappingProxyType(
    {
        "sam3": SamModel(
            "sam3",
            "SAM 3",
            "facebook/sam3",
            "sam3.pt",
            "https://huggingface.co/facebook/sam3",
            "about 3.45 GB",
        ),
        "sam3.1": SamModel(
            "sam3.1",
            "SAM 3.1",
            "facebook/sam3.1",
            "sam3.1_multiplex.pt",
            "https://huggingface.co/facebook/sam3.1",
        ),
    }
)
DEFAULT_SAM_VARIANT = "sam3.1"


def get_sam_model(key: str) -> SamModel:
    try:
        return SAM_MODELS[key]
    except (KeyError, TypeError):
        raise ValueError(f"Unsupported SAM model variant: {key!r}") from None


def official_sam_variant(path: str) -> str | None:
    name = os.path.basename(path).lower()
    return next((model.key for model in SAM_MODELS.values() if model.filename == name), None)


def resolve_sam_variant(project_root: str, variant: str = "", model_path: str = "") -> str:
    """Restore a selector, inferring old projects without changing their actual path."""
    if variant in SAM_MODELS:
        return variant
    inferred = official_sam_variant(model_path)
    if inferred:
        return inferred
    if not model_path:
        for key in (DEFAULT_SAM_VARIANT, *SAM_MODELS):
            if os.path.isfile(os.path.join(project_root, get_sam_model(key).filename)):
                return key
    return DEFAULT_SAM_VARIANT


def sam_download_target(project_root: str, variant: str) -> str:
    return os.path.join(os.path.abspath(project_root), get_sam_model(variant).filename)
