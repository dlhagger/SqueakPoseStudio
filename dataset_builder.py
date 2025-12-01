"""
Utility helpers for building YOLO-style dataset.yaml files for pose datasets.
"""
from __future__ import annotations

import os
from typing import Iterable

import yaml


def _default_flip_indices(kp_names: Iterable[str]) -> list[int]:
    """Return a flip index list for YOLO pose datasets.

    Pairs left/right keypoints when names contain those tokens; otherwise
    falls back to the identity mapping so data augmentation still works.
    """
    names = list(kp_names)
    n = len(names)
    flip = list(range(n))
    lookup = {}
    for idx, raw in enumerate(names):
        key = raw.lower()
        if "left" in key:
            normalized = key.replace("left", "")
            lookup.setdefault(("left", normalized), []).append(idx)
        elif "right" in key:
            normalized = key.replace("right", "")
            lookup.setdefault(("right", normalized), []).append(idx)
    for (_, normalized), left_idxs in list(lookup.items()):
        right_idxs = lookup.get(("right", normalized), [])
        if not left_idxs or not right_idxs:
            continue
        for li, ri in zip(sorted(left_idxs), sorted(right_idxs)):
            flip[li] = ri
            flip[ri] = li
    return flip


def create_dataset_yaml(base_dir: str, class_names: Iterable[str], kp_names: Iterable[str]) -> str:
    """Create a YOLO pose dataset.yaml file in `base_dir`.

    Returns the path to the written YAML.
    """
    images_train = os.path.join(base_dir, "images", "train")
    images_val = os.path.join(base_dir, "images", "val")

    if not os.path.isdir(images_train) or not os.path.isdir(images_val):
        raise FileNotFoundError("Expected images/train and images/val directories to exist.")

    kp_list = list(kp_names)
    cls_list = list(class_names)

    dataset = {
        "path": base_dir,
        "train": "images/train",
        "val": "images/val",
        "names": cls_list,
        "kpt_shape": [len(kp_list), 3],
        "kp_names": kp_list,
        "flip_idx": _default_flip_indices(kp_list),
    }

    out_path = os.path.join(base_dir, "dataset.yaml")
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dataset, fh, sort_keys=False)
    return out_path


__all__ = ["create_dataset_yaml"]
