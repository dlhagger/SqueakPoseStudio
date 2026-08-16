"""Build YOLO-style dataset YAML files without UI dependencies."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from squeakpose.core import atomic_write_text

_SIDE_SPLIT_RE = re.compile(r"[\s\-_]+")
_LEFT_TOKENS = {"left", "l"}
_RIGHT_TOKENS = {"right", "r"}


def _extract_side_and_base(name: str) -> tuple[str | None, str | None]:
    """Return a side and base name when the side token is unambiguous."""

    tokens = [token for token in _SIDE_SPLIT_RE.split(name.strip().lower()) if token]
    if not tokens:
        return None, None

    left_positions = [index for index, token in enumerate(tokens) if token in _LEFT_TOKENS]
    right_positions = [index for index, token in enumerate(tokens) if token in _RIGHT_TOKENS]
    if left_positions and right_positions:
        return None, None

    if left_positions:
        if len(left_positions) != 1:
            return None, None
        side = "left"
        side_index = left_positions[0]
    elif right_positions:
        if len(right_positions) != 1:
            return None, None
        side = "right"
        side_index = right_positions[0]
    else:
        return None, None

    base_tokens = tokens[:side_index] + tokens[side_index + 1 :]
    if not base_tokens:
        return None, None
    return side, "_".join(base_tokens)


def _default_flip_indices(keypoint_names: Iterable[str]) -> list[int]:
    """Pair unambiguous left/right keypoint names, retaining identity otherwise."""

    names = list(keypoint_names)
    flip = list(range(len(names)))
    left_lookup: dict[str, list[int]] = {}
    right_lookup: dict[str, list[int]] = {}

    for index, raw_name in enumerate(names):
        side, base = _extract_side_and_base(raw_name)
        if side == "left" and base:
            left_lookup.setdefault(base, []).append(index)
        elif side == "right" and base:
            right_lookup.setdefault(base, []).append(index)

    for base, left_indices in left_lookup.items():
        right_indices = right_lookup.get(base, [])
        for left_index, right_index in zip(sorted(left_indices), sorted(right_indices)):
            flip[left_index] = right_index
            flip[right_index] = left_index
    return flip


def _flip_index_summary(keypoint_names: list[str], flip_indices: list[int]) -> str:
    """Build the existing human-readable flip mapping summary."""

    lines: list[str] = []
    seen_pairs: set[tuple[int, int]] = set()
    unresolved_sided: list[str] = []

    for index, name in enumerate(keypoint_names):
        side, _ = _extract_side_and_base(name)
        if side is not None and flip_indices[index] == index:
            unresolved_sided.append(name)

        mate = flip_indices[index]
        if mate != index:
            left_index, right_index = sorted((index, mate))
            seen_pairs.add((left_index, right_index))

    lines.append("Flip index summary:")
    lines.append(f"  flip_idx: {flip_indices}")
    if seen_pairs:
        lines.append("  paired keypoints:")
        for left_index, right_index in sorted(seen_pairs):
            left_name = keypoint_names[left_index]
            right_name = keypoint_names[right_index]
            lines.append(f"    - [{left_index}] {left_name} <-> [{right_index}] {right_name}")
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
    *,
    dataset_path: str | None = None,
) -> str:
    """Create the existing YOLO pose ``dataset.yaml`` in ``base_dir``."""

    required_directories = (
        os.path.join(base_dir, "images", "train"),
        os.path.join(base_dir, "images", "val"),
        os.path.join(base_dir, "labels", "train"),
        os.path.join(base_dir, "labels", "val"),
    )
    if not all(os.path.isdir(path) for path in required_directories):
        raise FileNotFoundError(
            "Expected images/train, images/val, labels/train, and labels/val directories to exist."
        )

    keypoints = list(kp_names)
    classes = list(class_names)
    flip_indices = _default_flip_indices(keypoints)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write dataset.yaml.") from exc

    dataset = {
        "path": dataset_path or base_dir,
        "train": "images/train",
        "val": "images/val",
        "names": classes,
        "kpt_shape": [len(keypoints), 3],
        "kp_names": keypoints,
        "flip_idx": flip_indices,
    }
    output_path = os.path.join(base_dir, "dataset.yaml")
    atomic_write_text(output_path, yaml.safe_dump(dataset, sort_keys=False))

    if verbose:
        print(_flip_index_summary(keypoints, flip_indices))
        print(f"dataset.yaml written to: {output_path}")
    return output_path


__all__ = ["create_dataset_yaml"]
