"""Core helpers used by SqueakPose Studio.

This module intentionally avoids Qt/Ultralytics imports so it can be unit tested
in isolation.
"""

from __future__ import annotations

from typing import Any, Optional


def find_duplicate_names(names: list[str]) -> list[str]:
    """Return duplicate names in first-seen order."""
    seen: set[str] = set()
    dupes: list[str] = []
    dupes_seen: set[str] = set()
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name in seen and name not in dupes_seen:
            dupes.append(name)
            dupes_seen.add(name)
        seen.add(name)
    return dupes


def effective_prediction_batch(requested_batch: int, device: str) -> int:
    """Resolve a positive chunk size for review-time prediction batching."""
    if requested_batch > 0:
        return requested_batch
    return 8 if (device or "").lower() in {"cuda", "mps"} else 1


class InferenceCsvWriter:
    """Tiny wrapper that tracks incremental CSV row writes."""

    def __init__(self, writer: Any):
        self._writer = writer
        self.rows_written = 0

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self.rows_written += 1


def parse_yolo_pose_label_line(
    line: str,
    *,
    classes_count: int,
    canonical_names: list[str],
    class_keypoint_lookup: list[list[str]],
    img_w: float,
    img_h: float,
) -> tuple[Optional[dict[str, Any]], bool]:
    """Parse one YOLO pose label line without mutating global schema.

    Returns: (parsed_entry_or_none, had_extra_keypoints)
    """
    parts = line.split()
    if len(parts) < 5:
        return None, False

    try:
        cid = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        if cid < 0 or cid >= classes_count:
            return None, False
    except ValueError:
        return None, False

    img_w_safe = float(img_w)
    img_h_safe = float(img_h)
    x = (xc - w / 2.0) * img_w_safe
    y = (yc - h / 2.0) * img_h_safe
    bbox = {"x": x, "y": y, "w": w * img_w_safe, "h": h * img_h_safe}

    kp_data = parts[5:]
    triple_count = len(kp_data) // 3
    usable_count = min(triple_count, len(canonical_names))
    had_extra_keypoints = triple_count > len(canonical_names)

    parsed_keypoints: list[dict[str, Any]] = []
    for canon_idx in range(usable_count):
        base = canon_idx * 3
        try:
            xn = float(kp_data[base])
            yn = float(kp_data[base + 1])
            vis = int(float(kp_data[base + 2]))
        except ValueError:
            break
        name = canonical_names[canon_idx]
        parsed_keypoints.append(
            {
                "idx": canon_idx,
                "canon_idx": canon_idx,
                "name": name,
                "x": xn * img_w_safe,
                "y": yn * img_h_safe,
                "vis": vis,
            }
        )

    class_names = class_keypoint_lookup[cid] if 0 <= cid < len(class_keypoint_lookup) else []
    filtered: list[dict[str, Any]] = []
    kp_by_name = {kp["name"]: kp for kp in parsed_keypoints}
    for idx_cls, name in enumerate(class_names):
        entry = kp_by_name.get(name)
        if entry:
            cp = entry.copy()
            cp["idx"] = idx_cls
            filtered.append(cp)

    return {"class_id": cid, "bbox": bbox, "keypoints": filtered}, had_extra_keypoints

