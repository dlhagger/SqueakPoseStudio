"""Qt-free helpers for reading and writing SqueakPose label rows."""

from __future__ import annotations

from typing import Any, Optional

from squeakpose_core import parse_yolo_pose_label_line


def parse_pose_label_line(
    line: str,
    *,
    classes_count: int,
    canonical_names: list[str],
    class_keypoint_lookup: list[list[str]],
    img_w: float,
    img_h: float,
) -> tuple[Optional[dict[str, Any]], bool]:
    """Parse one YOLO pose row into the app's annotation cache schema."""
    return parse_yolo_pose_label_line(
        line,
        classes_count=classes_count,
        canonical_names=canonical_names,
        class_keypoint_lookup=class_keypoint_lookup,
        img_w=img_w,
        img_h=img_h,
    )


def load_pose_annotations_from_file(
    label_file: str,
    *,
    classes_count: int,
    canonical_names: list[str],
    class_keypoint_lookup: list[list[str]],
    img_w: float,
    img_h: float,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Load a YOLO pose label file into {class_id: annotation_entry}."""
    cache: dict[int, dict[str, Any]] = {}
    extra_rows = 0
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln:
                    continue
                entry, had_extra = parse_pose_label_line(
                    ln,
                    classes_count=classes_count,
                    canonical_names=canonical_names,
                    class_keypoint_lookup=class_keypoint_lookup,
                    img_w=img_w,
                    img_h=img_h,
                )
                if had_extra:
                    extra_rows += 1
                if entry:
                    cache[int(entry["class_id"])] = entry
    except Exception:
        return {}, 0
    return cache, extra_rows


def parse_segmentation_label_line(
    line: str,
    *,
    classes_count: int,
    img_w: float,
    img_h: float,
) -> Optional[dict[str, Any]]:
    """Parse one YOLO segmentation row into the app's annotation cache schema."""
    parts = line.split()
    if len(parts) < 7:
        return None
    try:
        cid = int(parts[0])
    except Exception:
        return None
    if cid < 0 or cid >= classes_count:
        return None

    coord_tokens = parts[1:]
    if len(coord_tokens) % 2 != 0:
        coord_tokens = coord_tokens[:-1]
    if len(coord_tokens) < 6:
        return None

    width = max(1.0, float(img_w))
    height = max(1.0, float(img_h))
    points: list[tuple[float, float]] = []
    for idx in range(0, len(coord_tokens), 2):
        try:
            xn = float(coord_tokens[idx])
            yn = float(coord_tokens[idx + 1])
        except Exception:
            return None
        points.append((xn * width, yn * height))

    if len(points) < 3:
        return None
    return {"class_id": cid, "segments": points}


def load_segmentation_annotations_from_file(
    label_file: str,
    *,
    classes_count: int,
    img_w: float,
    img_h: float,
) -> dict[int, dict[str, Any]]:
    """Load a YOLO segmentation label file into {class_id: annotation_entry}."""
    cache: dict[int, dict[str, Any]] = {}
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                entry = parse_segmentation_label_line(
                    line,
                    classes_count=classes_count,
                    img_w=img_w,
                    img_h=img_h,
                )
                if entry:
                    cache[int(entry["class_id"])] = entry
    except Exception:
        return {}
    return cache


def segmentation_annotation_to_line(entry: dict[str, Any], *, img_w: float, img_h: float) -> str:
    """Serialize one segmentation annotation entry to a YOLO segmentation row."""
    cid = int(entry.get("class_id", 0))
    seg = entry.get("segments", [])
    width = max(1.0, float(img_w))
    height = max(1.0, float(img_h))
    coords: list[str] = []
    for pair in seg:
        try:
            x = float(pair[0])
            y = float(pair[1])
        except Exception:
            continue
        coords.append(f"{x / width:.6f}")
        coords.append(f"{y / height:.6f}")
    if len(coords) < 6:
        return ""
    return f"{cid} " + " ".join(coords)


def pose_annotation_to_line(
    entry: dict[str, Any],
    *,
    kp_names: list[str],
    img_w: float,
    img_h: float,
) -> str:
    """Serialize one pose annotation entry to a YOLO pose row."""
    cid = entry.get("class_id", 0)
    bbox = entry.get("bbox", {})
    width = max(1.0, float(img_w))
    height = max(1.0, float(img_h))
    x = bbox.get("x", 0.0)
    y = bbox.get("y", 0.0)
    w = bbox.get("w", 0.0)
    h = bbox.get("h", 0.0)
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height
    line = f"{cid} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}"

    kp_lookup: dict[int, dict[str, Any]] = {}
    for kp in entry.get("keypoints", []):
        canon_idx = int(kp.get("canon_idx", -1))
        if canon_idx >= 0:
            kp_lookup[canon_idx] = kp

    for idx in range(len(kp_names)):
        kp = kp_lookup.get(idx)
        if not kp:
            line += " 0.000000 0.000000 0"
            continue
        vis = int(kp.get("vis", 2))
        if vis == 0:
            line += " 0.000000 0.000000 0"
        else:
            xn = kp.get("x", 0.0) / width
            yn = kp.get("y", 0.0) / height
            line += f" {xn:.6f} {yn:.6f} {vis}"
    return line
