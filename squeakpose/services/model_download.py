"""Model downloads backed by the packaged Hugging Face Hub client."""

from __future__ import annotations

import os
from collections.abc import Callable

from squeakpose.services.sam_models import get_sam_model


def download_sam_weights(
    destination_dir: str,
    model_key: str,
    *,
    downloader: Callable[..., str] | None = None,
) -> str:
    """Download one allow-listed official checkpoint into ``destination_dir``."""
    model = get_sam_model(model_key)
    root = os.path.abspath(str(destination_dir or ""))
    if not destination_dir or not os.path.isdir(root):
        raise ValueError(
            f"{model.display_name} download destination must be an existing directory."
        )

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    downloaded = downloader(repo_id=model.repo_id, filename=model.filename, local_dir=root)
    expected = os.path.join(root, model.filename)
    target = os.path.abspath(str(downloaded)) if downloaded else ""
    if (
        target != expected
        or not os.path.isfile(target)
        or os.path.realpath(target) != os.path.join(os.path.realpath(root), model.filename)
    ):
        raise RuntimeError(
            f"Hugging Face finished without creating {model.filename} in the project."
        )
    return target


def sam_download_error_message(error: BaseException | str, model_key: str) -> str:
    """Return concise, model-specific UI text for common Hub failures."""
    model = get_sam_model(model_key)
    detail = str(error or "").strip()
    lower = detail.lower()
    if any(marker in lower for marker in ("gated", "401", "403", "unauthorized", "forbidden")):
        return (
            f"{model.display_name} is a gated Hugging Face model. Request or accept access at "
            f"{model.access_url}, then sign in with `hf auth login` (or set HF_TOKEN) and retry."
        )
    if "huggingface_hub" in lower and ("no module" in lower or "not found" in lower):
        return "The packaged Hugging Face Hub client is unavailable. Reinstall the application."
    if detail:
        last_line = next(
            (line.strip() for line in reversed(detail.splitlines()) if line.strip()), ""
        )
        return f"Could not download {model.display_name} from Hugging Face: {last_line[:1200]}"
    return f"Could not download {model.display_name} from Hugging Face. Check the connection and retry."
