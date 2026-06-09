"""Core helpers used by SqueakPose Studio.

This module intentionally avoids Qt/Ultralytics imports so it can be unit tested
in isolation.
"""

from __future__ import annotations

import os
import math
import hashlib
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Optional

CURRENT_PROJECT_SCHEMA_VERSION = 2


def normalize_yolo_task(value: Any) -> Optional[str]:
    """Return a canonical Ultralytics task name when recognized."""
    raw = str(value or "").strip().lower()
    aliases = {
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
        "segment": "segment",
        "segmentation": "segment",
        "seg": "segment",
        "detect": "detect",
        "detection": "detect",
        "bbox": "detect",
    }
    return aliases.get(raw)


def model_task_mismatch_message(
    actual_task: Any,
    expected_task: Any,
    *,
    subject: str = "Model",
) -> Optional[str]:
    """Return an actionable error when known model and requested tasks differ."""
    actual = normalize_yolo_task(actual_task)
    expected = normalize_yolo_task(expected_task)
    if actual is None or expected is None or actual == expected:
        return None
    return (
        f"{subject} task mismatch: loaded a '{actual}' model, "
        f"but this operation requires '{expected}'."
    )


def infer_dataset_task(payload: Any) -> Optional[str]:
    """Infer a YOLO dataset task from parsed dataset.yaml contents."""
    if not isinstance(payload, Mapping):
        return None
    explicit = normalize_yolo_task(payload.get("task"))
    if explicit is not None:
        return explicit
    if "kpt_shape" in payload or "kp_names" in payload:
        return "pose"
    if "train" in payload or "val" in payload:
        return "detect"
    return None


def migrate_project_metadata(
    payload: Mapping[str, Any],
    *,
    created_at: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Migrate project metadata to the current schema without dropping extras."""
    migrated = dict(payload)
    changed = False

    try:
        version = int(migrated.get("schema_version", 1))
    except (TypeError, ValueError):
        version = 1
        changed = True

    if version > CURRENT_PROJECT_SCHEMA_VERSION:
        return migrated, changed

    if "active_workflow" not in migrated and "workflow" in migrated:
        migrated["active_workflow"] = migrated.pop("workflow")
        changed = True
    if "sam_model_path" not in migrated and "sam_path" in migrated:
        migrated["sam_model_path"] = migrated.pop("sam_path")
        changed = True

    workflow = str(migrated.get("active_workflow", "") or "").strip().lower()
    normalized_workflow = {
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
        "segment": "segmentation",
        "seg": "segmentation",
        "segmentation": "segmentation",
    }.get(workflow)
    if normalized_workflow and normalized_workflow != migrated.get("active_workflow"):
        migrated["active_workflow"] = normalized_workflow
        changed = True

    if not migrated.get("created_at") and created_at:
        migrated["created_at"] = created_at
        changed = True

    if version != CURRENT_PROJECT_SCHEMA_VERSION:
        migrated["schema_version"] = CURRENT_PROJECT_SCHEMA_VERSION
        changed = True

    return migrated, changed


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


def resolve_default_training_dataset_path(project_root: str) -> str:
    """Pick the best default dataset folder for the training dialog.

    Preference order:
    1. `datasets/pose` if it contains `dataset.yaml`
    2. `datasets/segment` if it contains `dataset.yaml`
    3. `datasets/detect` if it contains `dataset.yaml`
    4. fallback to `datasets` under the project root
    """
    root = os.path.abspath(project_root or os.getcwd())
    datasets_root = os.path.join(root, "datasets")
    candidates = (
        os.path.join(datasets_root, "pose"),
        os.path.join(datasets_root, "segment"),
        os.path.join(datasets_root, "detect"),
        datasets_root,
    )
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "dataset.yaml")):
            return candidate
    return datasets_root


class InferenceCsvWriter:
    """Tiny wrapper that tracks incremental CSV row writes."""

    def __init__(self, writer: Any):
        self._writer = writer
        self.rows_written = 0

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self.rows_written += 1


def atomic_write_text(path: str, text: str, *, encoding: str = "utf-8") -> None:
    """Write text via same-directory temp file and atomic replace."""
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or os.getcwd()
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(abs_path)}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp_path, abs_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def image_stem_collisions(file_names: Iterable[str]) -> dict[str, list[str]]:
    """Return case-insensitive image-stem collisions keyed by normalized stem."""
    grouped: dict[str, list[str]] = {}
    for raw_name in file_names:
        name = os.path.basename(str(raw_name))
        stem = os.path.splitext(name)[0].casefold()
        if stem:
            grouped.setdefault(stem, []).append(name)
    return {
        stem: sorted(names, key=str.casefold)
        for stem, names in grouped.items()
        if len(names) > 1
    }


def filter_image_stem_collisions(file_names: Iterable[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Return files with ambiguous stems removed, plus the collision map."""
    names = list(file_names)
    collisions = image_stem_collisions(names)
    blocked = set(collisions)
    accepted = [
        name
        for name in names
        if os.path.splitext(os.path.basename(name))[0].casefold() not in blocked
    ]
    return accepted, collisions


def stable_path_id(path: str, *, length: int = 8) -> str:
    """Return a short stable identifier for a canonical filesystem path."""
    canonical = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:max(4, int(length))]


def staging_path_for(target_path: str) -> str:
    """Create and return an empty sibling staging file for a target path."""
    target = os.path.abspath(target_path)
    directory = os.path.dirname(target) or os.getcwd()
    os.makedirs(directory, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(target))
    fd, staged_path = tempfile.mkstemp(
        prefix=f".{base}.",
        suffix=f".tmp{ext}",
        dir=directory,
    )
    os.close(fd)
    return staged_path


def stage_text_file(target_path: str, text: str, *, encoding: str = "utf-8") -> str:
    """Write text to a sibling staging file and return its path."""
    staged_path = staging_path_for(target_path)
    try:
        with open(staged_path, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
        return staged_path
    except Exception:
        try:
            os.unlink(staged_path)
        except OSError:
            pass
        raise


def stage_copy_file(source_path: str, target_path: str) -> str:
    """Copy a file to a sibling staging path and return the staged path."""
    staged_path = staging_path_for(target_path)
    try:
        shutil.copy2(source_path, staged_path)
        return staged_path
    except Exception:
        try:
            os.unlink(staged_path)
        except OSError:
            pass
        raise


def remove_path(path: str) -> None:
    """Remove a file, symlink, or directory when it exists."""
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def commit_staged_paths(replacements: Iterable[tuple[str, str]]) -> None:
    """Replace multiple paths as one rollback-capable transaction.

    Replacements are committed in the supplied order. Existing targets are
    moved to sibling backups first. If any replacement fails, all earlier
    replacements are rolled back.
    """
    pairs = [(os.path.abspath(stage), os.path.abspath(target)) for stage, target in replacements]
    if not pairs:
        return

    targets = [target for _, target in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("Transaction targets must be unique")
    for staged_path, _ in pairs:
        if not os.path.lexists(staged_path):
            raise FileNotFoundError(staged_path)

    records: list[dict[str, object]] = []
    try:
        for staged_path, target_path in pairs:
            os.makedirs(os.path.dirname(target_path) or os.getcwd(), exist_ok=True)
            backup_path: Optional[str] = None
            if os.path.lexists(target_path):
                backup_path = f"{target_path}.backup-{uuid.uuid4().hex}"
                os.replace(target_path, backup_path)

            record = {
                "target": target_path,
                "backup": backup_path,
                "installed": False,
            }
            records.append(record)
            os.replace(staged_path, target_path)
            record["installed"] = True
    except Exception as exc:
        rollback_errors: list[str] = []
        for record in reversed(records):
            target_path = str(record["target"])
            backup_path = record["backup"]
            try:
                if bool(record["installed"]) and os.path.lexists(target_path):
                    remove_path(target_path)
                if backup_path and os.path.lexists(str(backup_path)):
                    os.replace(str(backup_path), target_path)
            except Exception as rollback_exc:
                rollback_errors.append(f"{target_path}: {rollback_exc}")
        for staged_path, _ in pairs:
            try:
                remove_path(staged_path)
            except Exception:
                pass
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise RuntimeError(f"{exc}; rollback failed: {detail}") from exc
        raise
    else:
        for record in records:
            backup_path = record["backup"]
            if backup_path:
                try:
                    remove_path(str(backup_path))
                except Exception:
                    pass


def atomic_write_text_files(files: Mapping[str, str], *, encoding: str = "utf-8") -> None:
    """Atomically replace a collection of text files with rollback."""
    staged: list[tuple[str, str]] = []
    try:
        for target_path, text in files.items():
            staged.append((stage_text_file(target_path, text, encoding=encoding), target_path))
        commit_staged_paths(staged)
    except Exception:
        for staged_path, _ in staged:
            try:
                remove_path(staged_path)
            except Exception:
                pass
        raise


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(raw)
    return value


def _parse_class_id(raw: str, class_count: int) -> int:
    value = _finite_float(raw)
    cid = int(round(value))
    if cid < 0 or cid >= class_count:
        raise ValueError(raw)
    return cid


def normalize_pose_label_lines(
    lines: list[str],
    *,
    class_count: int,
    keypoint_count: int,
) -> tuple[list[str], list[str], bool]:
    """Validate and normalize YOLO pose label lines.

    Invalid rows are dropped. Missing keypoints are padded invisible, extra
    keypoints are dropped, and normalized coordinates are clamped to [0, 1].
    """
    warnings: list[str] = []
    normalized: list[str] = []
    clean_lines = [ln.strip() for ln in lines if ln.strip()]
    expected_kp_values = max(0, int(keypoint_count)) * 3

    for line_no, raw_line in enumerate(clean_lines, start=1):
        parts = raw_line.split()
        if len(parts) < 5:
            warnings.append(f"line {line_no} has <5 values")
            continue

        try:
            cid = _parse_class_id(parts[0], class_count)
            bbox_vals = [_finite_float(parts[i]) for i in range(1, 5)]
        except Exception as exc:
            warnings.append(f"line {line_no} parse error ({exc})")
            continue

        xc, yc, width, height = bbox_vals
        if width <= 0.0 or height <= 0.0:
            warnings.append(f"line {line_no} has non-positive bbox size")
            continue

        clamped_bbox = [_clamp01(xc), _clamp01(yc), _clamp01(width), _clamp01(height)]
        if clamped_bbox != bbox_vals:
            warnings.append(f"line {line_no} bbox values were clamped")

        kp_vals_raw = parts[5:]
        if len(kp_vals_raw) > expected_kp_values:
            warnings.append(f"line {line_no} has extra keypoint values")
        elif len(kp_vals_raw) < expected_kp_values:
            warnings.append(f"line {line_no} has missing keypoint values")

        normalized_kp: list[tuple[float, float, int]] = []
        for kp_idx in range(max(0, int(keypoint_count))):
            base_idx = kp_idx * 3
            if base_idx + 2 >= len(kp_vals_raw):
                normalized_kp.append((0.0, 0.0, 0))
                continue
            try:
                xn = _finite_float(kp_vals_raw[base_idx])
                yn = _finite_float(kp_vals_raw[base_idx + 1])
                vis_raw = _finite_float(kp_vals_raw[base_idx + 2])
            except Exception:
                warnings.append(f"line {line_no} keypoint {kp_idx} parse error")
                normalized_kp.append((0.0, 0.0, 0))
                continue

            vis = int(round(vis_raw))
            vis_clamped = max(0, min(2, vis))
            if vis_clamped != vis:
                warnings.append(f"line {line_no} keypoint {kp_idx} visibility was clamped")
            if vis_clamped == 0:
                normalized_kp.append((0.0, 0.0, 0))
                continue

            xn_clamped = _clamp01(xn)
            yn_clamped = _clamp01(yn)
            if xn_clamped != xn or yn_clamped != yn:
                warnings.append(f"line {line_no} keypoint {kp_idx} coordinates were clamped")
            normalized_kp.append((xn_clamped, yn_clamped, vis_clamped))

        line_out = (
            f"{cid} "
            f"{clamped_bbox[0]:.6f} {clamped_bbox[1]:.6f} "
            f"{clamped_bbox[2]:.6f} {clamped_bbox[3]:.6f}"
        )
        for xn, yn, vis in normalized_kp:
            line_out += f" {xn:.6f} {yn:.6f} {vis}"
        normalized.append(line_out)

    changed = normalized != clean_lines
    return normalized, warnings, changed


def normalize_segmentation_label_lines(
    lines: list[str],
    *,
    class_count: int,
) -> tuple[list[str], list[str], bool]:
    """Validate and normalize YOLO segmentation label lines."""
    warnings: list[str] = []
    normalized: list[str] = []
    clean_lines = [ln.strip() for ln in lines if ln.strip()]

    for line_no, raw_line in enumerate(clean_lines, start=1):
        parts = raw_line.split()
        if len(parts) < 7:
            warnings.append(f"line {line_no} has <7 values")
            continue
        try:
            cid = _parse_class_id(parts[0], class_count)
        except Exception:
            warnings.append(f"line {line_no} invalid class id")
            continue

        coord_tokens = parts[1:]
        if len(coord_tokens) % 2 != 0:
            coord_tokens = coord_tokens[:-1]
            warnings.append(f"line {line_no} has odd coordinate count")
        if len(coord_tokens) < 6:
            warnings.append(f"line {line_no} has <3 polygon points")
            continue

        coords: list[str] = []
        points: list[tuple[float, float]] = []
        parse_failed = False
        for cidx in range(0, len(coord_tokens), 2):
            try:
                xn = _finite_float(coord_tokens[cidx])
                yn = _finite_float(coord_tokens[cidx + 1])
            except Exception:
                parse_failed = True
                break
            xn_clamped = _clamp01(xn)
            yn_clamped = _clamp01(yn)
            if xn_clamped != xn or yn_clamped != yn:
                warnings.append(f"line {line_no} polygon coordinates were clamped")
            points.append((xn_clamped, yn_clamped))
            coords.append(f"{xn_clamped:.6f}")
            coords.append(f"{yn_clamped:.6f}")
        if parse_failed or len(coords) < 6:
            warnings.append(f"line {line_no} parse error")
            continue
        if len(set(points)) < 3:
            warnings.append(f"line {line_no} has <3 unique polygon points")
            continue
        area = abs(
            sum(
                points[idx][0] * points[(idx + 1) % len(points)][1]
                - points[(idx + 1) % len(points)][0] * points[idx][1]
                for idx in range(len(points))
            )
            / 2.0
        )
        if area <= 1e-12:
            warnings.append(f"line {line_no} has zero-area polygon")
            continue
        normalized.append(f"{cid} " + " ".join(coords))

    changed = normalized != clean_lines
    return normalized, warnings, changed


def build_segmentation_inference_rows(
    *,
    frame_index: int,
    detections: list[dict[str, Any]],
    class_names: Any = None,
    include_binary_mask: bool = True,
) -> list[dict[str, Any]]:
    """Build segmentation inference rows for one frame."""
    if not detections:
        return [
            {
                "frame": int(frame_index),
                "det": -1,
                "class_id": "",
                "class_name": "",
                "conf": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "mask_polygon": None,
                "binary_mask": None,
            }
        ]

    rows: list[dict[str, Any]] = []
    for det_idx, det in enumerate(detections):
        cls_id = int(det.get("class_id", 0))
        class_name = det.get("class_name")
        if not class_name:
            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id, "")
            elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = ""
        box = det.get("box") or [0.0, 0.0, 0.0, 0.0]
        rows.append(
            {
                "frame": int(frame_index),
                "det": int(det.get("det", det_idx)),
                "class_id": cls_id,
                "class_name": str(class_name or ""),
                "conf": float(det.get("conf", 0.0)),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "mask_polygon": det.get("mask_polygon"),
                "binary_mask": det.get("binary_mask") if include_binary_mask else None,
            }
        )
    return rows


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
