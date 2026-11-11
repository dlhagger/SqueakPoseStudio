"""Build YOLO-style dataset YAML files without UI dependencies."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

from squeakpose.core import atomic_write_text

_SIDE_SPLIT_RE = re.compile(r"[\s\-_.]+")
# camelCase / PascalCase boundaries: "leftEar" -> "left Ear", "LFPaw" -> "LF Paw",
# "LEar" -> "L Ear".
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_LEFT_TOKENS = {"left", "l"}
_RIGHT_TOKENS = {"right", "r"}
# Two-letter limb/corner codes such as LF/RF (left/right fore), LH/RH (hind), FL/FR
# (front), BL/BR (back), TL/TR (top). The non-side letter stays in the base name.
_SIDE_CODE_MODIFIERS = frozenset("fhbt")
# Minimum length of the rest of a "left..."/"right..." compound such as "leftear".
_COMPOUND_PREFIX_MIN_REST = 2


def _side_name_tokens(name: str) -> list[str]:
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", str(name).strip())
    return [token for token in _SIDE_SPLIT_RE.split(spaced.lower()) if token]


def _token_side(token: str) -> tuple[str | None, str | None]:
    """Classify one token as ``(side, remaining_base_token)``.

    ``remaining_base_token`` is ``None`` when the token is only a side marker.
    """

    if token in _LEFT_TOKENS:
        return "left", None
    if token in _RIGHT_TOKENS:
        return "right", None
    if len(token) == 2:
        first, second = token[0], token[1]
        for side_letter, other in ((first, second), (second, first)):
            if other in _SIDE_CODE_MODIFIERS and side_letter in {"l", "r"}:
                return ("left" if side_letter == "l" else "right"), other
    # Compact compounds such as "leftear"/"rightear". Single-letter prefixes are not
    # accepted here ("rear", "lateral", "light" are ordinary words).
    for word, side in (("left", "left"), ("right", "right")):
        if token.startswith(word) and len(token) - len(word) >= _COMPOUND_PREFIX_MIN_REST:
            return side, token[len(word) :]
    return None, None


def _extract_side_and_base(name: str) -> tuple[str | None, str | None]:
    """Return a side and base name when the side token is unambiguous."""

    tokens = _side_name_tokens(name)
    if not tokens:
        return None, None

    sided = [(index, *_token_side(token)) for index, token in enumerate(tokens)]
    sided = [entry for entry in sided if entry[1] is not None]
    if len(sided) != 1:
        return None, None

    side_index, side, remainder = sided[0]
    base_tokens = list(tokens)
    if remainder:
        base_tokens[side_index] = remainder
    else:
        del base_tokens[side_index]
    if not base_tokens:
        return None, None
    return side, "_".join(base_tokens)


def _looks_sided(name: str) -> bool:
    """Return whether any token of ``name`` carries a left/right marker."""

    return any(_token_side(token)[0] is not None for token in _side_name_tokens(name))


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


def flip_indices_for_keypoints(keypoint_names: Iterable[str]) -> list[int]:
    """Return the ``flip_idx`` list written to ``dataset.yaml`` for these names."""

    return _default_flip_indices(keypoint_names)


def unpaired_sided_keypoints(keypoint_names: Iterable[str]) -> list[str]:
    """Return names that look left/right sided but have no mirrored partner.

    These keep an identity ``flip_idx`` entry, so horizontal-flip augmentation will not
    swap them. Callers can show the list to the user as a warning before training.
    The result preserves the input order.
    """

    names = list(keypoint_names)
    flip_indices = _default_flip_indices(names)
    return [
        name
        for index, name in enumerate(names)
        if flip_indices[index] == index and _looks_sided(name)
    ]


def _flip_index_summary(keypoint_names: list[str], flip_indices: list[int]) -> str:
    """Build the existing human-readable flip mapping summary."""

    lines: list[str] = []
    seen_pairs: set[tuple[int, int]] = set()
    unresolved_sided: list[str] = []

    for index, name in enumerate(keypoint_names):
        if flip_indices[index] == index and _looks_sided(name):
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
    """Create the existing YOLO pose ``dataset.yaml`` in ``base_dir``.

    No ``path`` key is written unless ``dataset_path`` is given explicitly, so the
    dataset root is the YAML's own directory and survives moving or copying the project.
    """

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

    dataset: dict[str, object] = {}
    if dataset_path:
        dataset["path"] = dataset_path
    dataset.update(
        {
            "train": "images/train",
            "val": "images/val",
            "names": classes,
            "kpt_shape": [len(keypoints), 3],
            "kp_names": keypoints,
            "flip_idx": flip_indices,
        }
    )
    output_path = os.path.join(base_dir, "dataset.yaml")
    atomic_write_text(output_path, yaml.safe_dump(dataset, sort_keys=False))

    if verbose:
        print(_flip_index_summary(keypoints, flip_indices))
        print(f"dataset.yaml written to: {output_path}")
    return output_path


def resolve_dataset_root(yaml_path: str, payload: Mapping[str, Any] | None = None) -> str:
    """Return the directory that a dataset YAML's ``train``/``val`` entries are relative to.

    Without a ``path`` key (current exports) the root is the YAML's own directory, which
    matches Ultralytics' fallback. Old exports with an absolute ``path`` are still
    honoured; a relative ``path`` is resolved against the YAML's directory.
    """

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    if payload is None:
        payload = _load_yaml_mapping(yaml_path)
    raw = payload.get("path") if isinstance(payload, Mapping) else None
    if not isinstance(raw, str) or not raw.strip():
        return yaml_dir
    root = os.path.expanduser(raw.strip())
    if not os.path.isabs(root):
        root = os.path.join(yaml_dir, root)
    return os.path.normpath(root)


def stale_dataset_root(yaml_path: str) -> str | None:
    """Return a legacy absolute ``path`` that no longer points at the YAML's directory.

    A project exported before ``path`` was omitted, then moved or copied, keeps the old
    absolute root and would train on the original location's images. Returns ``None``
    when the YAML has no ``path`` or it refers to the YAML's own directory.
    """

    payload = _load_yaml_mapping(yaml_path)
    raw = payload.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    root = resolve_dataset_root(yaml_path, payload)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    if os.path.normcase(os.path.realpath(root)) == os.path.normcase(os.path.realpath(yaml_dir)):
        return None
    return root


def _load_yaml_mapping(yaml_path: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read dataset.yaml.") from exc
    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except OSError, yaml.YAMLError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


__all__ = [
    "create_dataset_yaml",
    "flip_indices_for_keypoints",
    "resolve_dataset_root",
    "stale_dataset_root",
    "unpaired_sided_keypoints",
]
