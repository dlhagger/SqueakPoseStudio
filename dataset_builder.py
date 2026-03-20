"""
Utility helpers for building YOLO-style dataset.yaml files for pose datasets.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

_SIDE_SPLIT_RE = re.compile(r"[\s\-_]+")
_LEFT_TOKENS = {"left", "l"}
_RIGHT_TOKENS = {"right", "r"}


def _extract_side_and_base(name: str) -> tuple[str | None, str | None]:
    """Return ('left'|'right', base_name) when a side token is unambiguous.

    Supported patterns include:
      - left_ear / right_ear
      - ear_left / ear_right
      - bottom_left / bottom_right
      - upper-left-eye / upper-right-eye

    If no side token is present, or side tokens are ambiguous, returns (None, None).
    """
    tokens = [tok for tok in _SIDE_SPLIT_RE.split(name.strip().lower()) if tok]
    if not tokens:
        return None, None

    left_pos = [i for i, tok in enumerate(tokens) if tok in _LEFT_TOKENS]
    right_pos = [i for i, tok in enumerate(tokens) if tok in _RIGHT_TOKENS]

    # Ambiguous (both left and right tokens in one name), skip pairing.
    if left_pos and right_pos:
        return None, None

    if left_pos:
        if len(left_pos) != 1:
            return None, None
        side = "left"
        side_idx = left_pos[0]
    elif right_pos:
        if len(right_pos) != 1:
            return None, None
        side = "right"
        side_idx = right_pos[0]
    else:
        return None, None

    base_tokens = tokens[:side_idx] + tokens[side_idx + 1 :]
    if not base_tokens:
        return None, None

    return side, "_".join(base_tokens)


def _default_flip_indices(kp_names: Iterable[str]) -> list[int]:
    """Return a flip index list for YOLO pose datasets.

    Pairs left/right keypoints based on tokenized names (e.g. ``left_ear`` and
    ``ear_right``). Unpaired names fall back to the identity mapping.
    """
    names = list(kp_names)
    flip = list(range(len(names)))
    left_lookup: dict[str, list[int]] = {}
    right_lookup: dict[str, list[int]] = {}

    for idx, raw in enumerate(names):
        side, base = _extract_side_and_base(raw)
        if side == "left" and base:
            left_lookup.setdefault(base, []).append(idx)
        elif side == "right" and base:
            right_lookup.setdefault(base, []).append(idx)

    # Only iterate left-side keys to avoid right-pass overwrite bugs.
    for base, left_idxs in left_lookup.items():
        right_idxs = right_lookup.get(base, [])
        if not left_idxs or not right_idxs:
            continue
        for li, ri in zip(sorted(left_idxs), sorted(right_idxs)):
            flip[li] = ri
            flip[ri] = li
    return flip


def _flip_index_summary(kp_names: list[str], flip_idx: list[int]) -> str:
    """Build a human-readable summary of resolved flip mappings."""
    lines: list[str] = []
    seen_pairs: set[tuple[int, int]] = set()
    unresolved_sided: list[str] = []

    for idx, name in enumerate(kp_names):
        side, _ = _extract_side_and_base(name)
        if side is not None and flip_idx[idx] == idx:
            unresolved_sided.append(name)

        mate = flip_idx[idx]
        if mate != idx:
            pair = tuple(sorted((idx, mate)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)

    lines.append("Flip index summary:")
    lines.append(f"  flip_idx: {flip_idx}")

    if seen_pairs:
        lines.append("  paired keypoints:")
        for left_idx, right_idx in sorted(seen_pairs):
            left_name = kp_names[left_idx]
            right_name = kp_names[right_idx]
            lines.append(f"    - [{left_idx}] {left_name} <-> [{right_idx}] {right_name}")
    else:
        lines.append("  paired keypoints: none detected")

    if unresolved_sided:
        lines.append("  unresolved sided names (kept as identity):")
        for name in unresolved_sided:
            lines.append(f"    - {name}")

    return "\n".join(lines)


def create_dataset_yaml(
    base_dir: str,
    class_names: Iterable[str],
    kp_names: Iterable[str],
    verbose: bool = True,
) -> str:
    """Create a YOLO pose dataset.yaml file in `base_dir`.

    Returns the path to the written YAML.
    """
    images_train = os.path.join(base_dir, "images", "train")
    images_val = os.path.join(base_dir, "images", "val")
    labels_train = os.path.join(base_dir, "labels", "train")
    labels_val = os.path.join(base_dir, "labels", "val")

    if (
        not os.path.isdir(images_train)
        or not os.path.isdir(images_val)
        or not os.path.isdir(labels_train)
        or not os.path.isdir(labels_val)
    ):
        raise FileNotFoundError(
            "Expected images/train, images/val, labels/train, and labels/val directories to exist."
        )

    kp_list = list(kp_names)
    cls_list = list(class_names)
    flip_idx = _default_flip_indices(kp_list)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write dataset.yaml. Install with `pip install pyyaml`.") from exc

    dataset = {
        "path": base_dir,
        "train": "images/train",
        "val": "images/val",
        "names": cls_list,
        "kpt_shape": [len(kp_list), 3],
        "kp_names": kp_list,
        "flip_idx": flip_idx,
    }

    out_path = os.path.join(base_dir, "dataset.yaml")
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dataset, fh, sort_keys=False)

    if verbose:
        print(_flip_index_summary(kp_list, flip_idx))
        print(f"dataset.yaml written to: {out_path}")

    return out_path


__all__ = ["create_dataset_yaml"]
