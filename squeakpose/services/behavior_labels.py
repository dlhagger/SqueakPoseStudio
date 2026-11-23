"""Atomic persistence and portable exports for human behavior labels."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import asdict, replace

from squeakpose.annotation.behavior import (
    SCHEMA_VERSION,
    Behavior,
    BehaviorDocument,
    VideoSource,
    validate_behavior_definitions,
)
from squeakpose.core import atomic_write_text, atomic_write_text_files
from squeakpose.json_io import read_json_file
from squeakpose.project.safety import require_path_within_project


def annotation_path(project_root: str, video_id: str) -> str:
    import re

    if not re.fullmatch(r"[0-9a-f]{64}", video_id):
        raise ValueError("Invalid video identity.")
    return require_path_within_project(
        project_root,
        os.path.join(project_root, "behavior labels", f"{video_id}.json"),
        purpose="behavior annotations",
        allow_root=False,
    )


def catalog_path(project_root: str) -> str:
    return require_path_within_project(
        project_root,
        os.path.join(project_root, "behavior labels", "catalog.json"),
        purpose="behavior definitions",
        allow_root=False,
    )


def load_behavior_catalog(project_root: str) -> tuple[list[Behavior], bool]:
    """Return project definitions and whether the user completed initial setup."""
    path = catalog_path(project_root)
    if not os.path.lexists(path):
        return [], False
    data = read_json_file(path, require_object=True)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported behavior catalog version.")
    try:
        behaviors = [Behavior(**row) for row in data["behaviors"]]
        validate_behavior_definitions(behaviors)
        configured = data.get("setup_complete", False)
        if type(configured) is not bool:
            raise ValueError("Invalid behavior setup state.")
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid behavior catalog.") from exc
    return behaviors, configured


def load_labeling_catalog(project_root: str) -> tuple[list[Behavior], bool]:
    """Read active targets, omitting the review categories supplied by older versions."""
    behaviors, configured = load_behavior_catalog(project_root)
    targets = [b for b in behaviors if b.name.strip().casefold() not in {"other", "unscorable"}]
    return targets, configured and bool(targets)


def save_behavior_catalog(project_root: str, behaviors: list[Behavior]) -> None:
    """Confirm project definitions without modifying any saved video annotations."""
    validate_behavior_definitions(behaviors)
    if not any(b.name.strip().casefold() != "other" for b in behaviors):
        raise ValueError("Add at least one behavior you want to label.")
    path = catalog_path(project_root)
    if os.path.lexists(path):
        load_behavior_catalog(project_root)  # Never replace malformed saved data.
    atomic_write_text(
        path,
        json.dumps(
            {
                "schema_version": 1,
                "setup_complete": True,
                "behaviors": [asdict(b) for b in behaviors],
            },
            indent=2,
            ensure_ascii=False,
        ),
    )


def new_document(project_root: str, source: VideoSource) -> BehaviorDocument:
    document = BehaviorDocument.new(source)
    behaviors, _ = load_behavior_catalog(project_root)
    if behaviors:
        document.behaviors = behaviors
    return document


def reconcile_behavior_catalog(
    document: BehaviorDocument, behaviors: list[Behavior]
) -> BehaviorDocument:
    """Use project choices while retaining definitions referenced by older labels."""
    validate_behavior_definitions(behaviors)
    result = document.copy()
    active = {b.id for b in behaviors}
    by_name = {b.name.strip().casefold(): b.id for b in behaviors}
    # A behavior removed and recreated in setup may have a new ID but the same name.
    aliases = {
        b.id: by_name[b.name.strip().casefold()]
        for b in document.behaviors
        if b.id not in active and b.name.strip().casefold() in by_name
    }
    result.intervals = [
        replace(i, behavior_id=aliases[i.behavior_id]) if i.behavior_id in aliases else i
        for i in document.intervals
    ]
    used = {i.behavior_id for i in result.intervals}
    result.behaviors = list(behaviors) + [
        b for b in document.behaviors if b.id in used and b.id not in active
    ]
    result.validate()
    return result


def save_document(
    project_root: str,
    document: BehaviorDocument,
    *,
    previous_behaviors: list[Behavior] | None = None,
    catalog_behaviors: list[Behavior] | None = None,
) -> str:
    payload = document.to_dict()
    path = annotation_path(project_root, document.video.id)
    files = {path: json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)}
    catalog = catalog_path(project_root)
    if catalog_behaviors is not None:
        validate_behavior_definitions(catalog_behaviors)
        _, configured = load_behavior_catalog(project_root)
        files[catalog] = json.dumps(
            {
                "schema_version": 1,
                "setup_complete": configured,
                "behaviors": [asdict(b) for b in catalog_behaviors],
            },
            indent=2,
        )
    elif not os.path.lexists(catalog) or previous_behaviors is not None:
        definitions = new_document(project_root, document.video)
        if not os.path.lexists(catalog):
            definitions.behaviors = document.behaviors.copy()
        if previous_behaviors is not None:
            previous = {b.id: b for b in previous_behaviors}
            updated = {b.id: b for b in document.behaviors}
            changed = {key: b for key, b in updated.items() if b != previous.get(key)}
            removed = previous.keys() - updated.keys()
            definitions.behaviors = [
                changed.get(b.id, b) for b in definitions.behaviors if b.id not in removed
            ]
            existing = {b.id for b in definitions.behaviors}
            definitions.behaviors.extend(b for key, b in changed.items() if key not in existing)
        definitions.validate()
        files[catalog] = json.dumps(
            {
                "schema_version": 1,
                "behaviors": definitions.to_dict()["behaviors"],
                "setup_complete": load_behavior_catalog(project_root)[1],
            },
            indent=2,
        )
    if len(files) == 1:
        atomic_write_text(path, files[path])
    else:
        atomic_write_text_files(files)
    return path


def load_document(project_root: str, video_id: str) -> BehaviorDocument | None:
    path = annotation_path(project_root, video_id)
    if not os.path.lexists(path):
        return None
    document = BehaviorDocument.from_dict(
        read_json_file(path, max_bytes=32 * 1024 * 1024, require_object=True)
    )
    if document.video.id != video_id:
        raise ValueError("Saved annotations belong to a different video.")
    return document


def export_document(document: BehaviorDocument, destination: str, *, csv_format: bool) -> None:
    """JSON is the full record; CSV contains intervals, definitions, and exact timing."""
    document.validate()
    if not csv_format:
        atomic_write_text(destination, json.dumps(document.to_dict(), indent=2, ensure_ascii=False))
        return
    out = io.StringIO(newline="")
    fields = (
        "schema_version",
        "video_id",
        "video_filename",
        "subject_id",
        "subject_name",
        "annotation_id",
        "kind",
        "behavior_id",
        "behavior_name",
        "behavior_definition",
        "start_frame",
        "end_frame_exclusive",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "start_pts",
        "end_pts",
        "time_base_num",
        "time_base_den",
        "first_pts",
        "notes",
        "end_time_method",
    )
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    behaviors = {b.id: b for b in document.behaviors}
    subjects = {s.id: s for s in document.subjects}
    v = document.video
    for i in sorted(document.intervals, key=lambda i: (i.start_pts, i.subject_id)):
        b = behaviors.get(i.behavior_id or "")
        start, end = v.seconds(i.start_pts), v.seconds(i.end_pts)
        writer.writerow(
            dict(
                zip(
                    fields,
                    (
                        SCHEMA_VERSION,
                        v.id,
                        v.filename,
                        i.subject_id,
                        subjects[i.subject_id].name,
                        i.id,
                        i.kind,
                        i.behavior_id or "",
                        b.name if b else "",
                        b.definition if b else "",
                        i.start_frame,
                        i.end_frame_exclusive,
                        f"{start:.9f}",
                        f"{end:.9f}",
                        f"{end - start:.9f}",
                        i.start_pts,
                        i.end_pts,
                        v.time_base_num,
                        v.time_base_den,
                        v.first_pts,
                        i.notes,
                        i.end_time_method,
                    ),
                    strict=True,
                )
            )
        )
    atomic_write_text(destination, out.getvalue())


def resolve_frame_annotations(
    document: BehaviorDocument, points: tuple[int, ...]
) -> BehaviorDocument:
    """Recover exact frame boundaries for annotations made in timestamp-only mode."""
    import bisect
    from dataclasses import replace

    result = document.copy()
    boundaries = (*points, result.video.end_pts)
    if not points or result.video.frame_count != len(points):
        raise ValueError("The source frame lookup is incomplete.")
    intervals = []
    for interval in result.intervals:
        start = bisect.bisect_left(points, interval.start_pts)
        end = bisect.bisect_left(boundaries, interval.end_pts)
        if (
            start >= len(points)
            or points[start] != interval.start_pts
            or end >= len(boundaries)
            or boundaries[end] != interval.end_pts
        ):
            raise ValueError(
                "A saved behavior boundary does not match a source frame. The labels were not changed."
            )
        if interval.start_frame is not None and (
            interval.start_frame,
            interval.end_frame_exclusive,
        ) != (start, end):
            raise ValueError("Saved frame numbers disagree with the source frame lookup.")
        intervals.append(replace(interval, start_frame=start, end_frame_exclusive=end))
    result.intervals = intervals
    if result.resume_pts is not None:
        resume = bisect.bisect_left(points, result.resume_pts)
        if resume >= len(points) or points[resume] != result.resume_pts:
            raise ValueError("The saved playback position does not match a source frame.")
        result.resume_frame = resume
    result.validate()
    return result
