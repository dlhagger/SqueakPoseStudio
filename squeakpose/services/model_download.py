"""Model downloads backed by the packaged Hugging Face Hub client."""

from __future__ import annotations

import os
from collections.abc import Callable

SAM3_REPO_ID = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"
SAM3_MODEL_URL = f"https://huggingface.co/{SAM3_REPO_ID}"


def download_sam3_weights(
    destination_dir: str,
    *,
    downloader: Callable[..., str] | None = None,
) -> str:
    """Download the official SAM 3 checkpoint into ``destination_dir``."""
    root = os.path.abspath(str(destination_dir or ""))
    if not destination_dir or not os.path.isdir(root):
        raise ValueError("SAM 3 download destination must be an existing directory.")

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    downloaded = downloader(
        repo_id=SAM3_REPO_ID,
        filename=SAM3_FILENAME,
        local_dir=root,
    )
    target = os.path.abspath(str(downloaded or os.path.join(root, SAM3_FILENAME)))
    expected = os.path.join(root, SAM3_FILENAME)
    if target != expected or not os.path.isfile(target):
        raise RuntimeError("Hugging Face finished without creating sam3.pt in the project.")
    return target


def sam3_download_error_message(error: BaseException | str) -> str:
    """Return concise, actionable UI text for common Hub failures."""
    detail = str(error or "").strip()
    lower = detail.lower()
    if any(marker in lower for marker in ("gated", "401", "403", "unauthorized", "forbidden")):
        return (
            "SAM 3 is a gated Hugging Face model. Request or accept access at "
            f"{SAM3_MODEL_URL}, then sign in with `hf auth login` (or set HF_TOKEN) and retry."
        )
    if "huggingface_hub" in lower and ("no module" in lower or "not found" in lower):
        return "The packaged Hugging Face Hub client is unavailable. Reinstall the application."
    if detail:
        last_line = next(
            (line.strip() for line in reversed(detail.splitlines()) if line.strip()), ""
        )
        return f"Could not download SAM 3 from Hugging Face: {last_line[:1200]}"
    return "Could not download SAM 3 from Hugging Face. Check the connection and retry."


__all__ = [
    "SAM3_FILENAME",
    "SAM3_MODEL_URL",
    "SAM3_REPO_ID",
    "download_sam3_weights",
    "sam3_download_error_message",
]
