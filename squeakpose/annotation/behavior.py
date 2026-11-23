"""Model-independent human behavior intervals and their interchange contract."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from uuid import uuid4

FORMAT = "squeakpose.behavior_annotations"
SCHEMA_VERSION = 2
COLORS = ("#60a5fa", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#22d3ee")


def new_id() -> str:
    return uuid4().hex


def timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


@dataclass(frozen=True)
class Behavior:
    id: str
    name: str
    color: str
    definition: str = ""


def validate_behavior_definitions(behaviors: list[Behavior]) -> None:
    if not behaviors:
        raise ValueError("Add at least one behavior.")
    ids = [b.id for b in behaviors]
    names = [b.name for b in behaviors]
    if any(not isinstance(value, str) or not value.strip() for value in ids + names):
        raise ValueError("Behavior names and IDs cannot be blank.")
    if len(set(ids)) != len(ids) or len({name.strip().casefold() for name in names}) != len(names):
        raise ValueError("Each behavior needs a unique name and ID.")
    for behavior in behaviors:
        if not isinstance(behavior.color, str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", behavior.color
        ):
            raise ValueError("Invalid behavior color.")
        if not isinstance(behavior.definition, str):
            raise ValueError("Invalid behavior definition.")


@dataclass(frozen=True)
class Subject:
    id: str
    name: str


@dataclass(frozen=True)
class VideoSource:
    id: str  # Fingerprint according to identity_method.
    filename: str
    source_path: str
    frame_count: int | None
    time_base_num: int
    time_base_den: int
    first_pts: int
    end_pts: int
    end_time_method: str
    identity_method: str = "sha256"

    def seconds(self, pts: int) -> float:
        return (pts - self.first_pts) * self.time_base_num / self.time_base_den


@dataclass(frozen=True)
class Interval:
    id: str
    subject_id: str
    behavior_id: str | None
    kind: str  # behavior or unscorable; absent intervals are unreviewed.
    start_frame: int | None
    end_frame_exclusive: int | None
    start_pts: int
    end_pts: int
    notes: str = ""
    end_time_method: str = "source_timestamp"


@dataclass
class BehaviorDocument:
    video: VideoSource
    behaviors: list[Behavior] = field(default_factory=list)
    subjects: list[Subject] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)
    resume_frame: int | None = 0
    resume_pts: int | None = None

    @classmethod
    def new(cls, video: VideoSource) -> BehaviorDocument:
        names = ("Other", "Rearing", "Turning", "Drinking")
        definitions = ("Reviewed: none of the defined target behaviors.", "", "", "")
        return cls(
            video,
            [Behavior(new_id(), n, COLORS[i], definitions[i]) for i, n in enumerate(names)],
            [Subject(new_id(), "Animal 1")],
        )

    def copy(self) -> BehaviorDocument:
        # Records are immutable; histories need not duplicate video data.
        return BehaviorDocument(
            self.video,
            self.behaviors.copy(),
            self.subjects.copy(),
            self.intervals.copy(),
            self.resume_frame,
            self.resume_pts,
        )

    def put(self, interval: Interval) -> None:
        candidate = self.copy()
        candidate.intervals = [i for i in self.intervals if i.id != interval.id] + [interval]
        candidate.validate()
        self.intervals = sorted(candidate.intervals, key=lambda i: (i.subject_id, i.start_pts))

    def validate(self) -> None:
        v = self.video
        if not isinstance(v.id, str) or not re.fullmatch(r"[0-9a-f]{64}", v.id):
            raise ValueError("Invalid source video fingerprint.")
        for value in (v.time_base_num, v.time_base_den):
            if type(value) is not int or value <= 0:
                raise ValueError("Invalid video timing metadata.")
        if type(v.first_pts) is not int or type(v.end_pts) is not int or v.end_pts <= v.first_pts:
            raise ValueError("Invalid source presentation timestamps.")
        for text_value in (v.filename, v.source_path, v.end_time_method):
            if not isinstance(text_value, str) or not text_value:
                raise ValueError("Missing video metadata.")
        if not math.isfinite(v.seconds(v.end_pts)):
            raise ValueError("Invalid video duration.")
        if v.frame_count is not None and (type(v.frame_count) is not int or v.frame_count <= 0):
            raise ValueError("Invalid video frame count.")
        if v.identity_method not in ("sha256", "sampled_sha256_stat_v1"):
            raise ValueError("Unsupported video fingerprint method.")
        if self.resume_frame is not None and (
            type(self.resume_frame) is not int
            or self.resume_frame < 0
            or (v.frame_count is not None and self.resume_frame >= v.frame_count)
        ):
            raise ValueError("Invalid playback position.")
        if self.resume_pts is not None and (
            type(self.resume_pts) is not int or not v.first_pts <= self.resume_pts < v.end_pts
        ):
            raise ValueError("Invalid playback timestamp.")
        if not self.behaviors or not self.subjects:
            raise ValueError("At least one behavior and subject are required.")
        validate_behavior_definitions(self.behaviors)
        for records in (self.subjects,):
            ids = [r.id for r in records]
            names = [r.name for r in records]
            if any(not isinstance(x, str) or not x.strip() for x in ids + names):
                raise ValueError("Names and IDs cannot be blank.")
            if len(set(ids)) != len(ids) or len({n.strip().casefold() for n in names}) != len(
                names
            ):
                raise ValueError("Behavior and subject names and IDs must be unique.")
        behavior_ids = {b.id for b in self.behaviors}
        subject_ids = {s.id for s in self.subjects}
        seen = set()
        for i in self.intervals:
            if not isinstance(i.id, str) or not i.id or i.id in seen:
                raise ValueError("Annotation IDs must be unique and nonempty.")
            seen.add(i.id)
            if i.subject_id not in subject_ids:
                raise ValueError("Annotation refers to an unknown subject.")
            if i.kind not in ("behavior", "unscorable"):
                raise ValueError("Unknown annotation kind.")
            if (i.kind == "behavior" and i.behavior_id not in behavior_ids) or (
                i.kind == "unscorable" and i.behavior_id is not None
            ):
                raise ValueError("Annotation refers to an invalid behavior.")
            if any(type(x) is not int for x in (i.start_pts, i.end_pts)):
                raise ValueError("Frame boundaries and PTS must be integers.")
            if (i.start_frame is None) != (i.end_frame_exclusive is None):
                raise ValueError("Both frame boundaries must be known or both left unknown.")
            if i.start_frame is not None and i.end_frame_exclusive is not None:
                if (
                    type(i.start_frame) is not int
                    or type(i.end_frame_exclusive) is not int
                    or not 0 <= i.start_frame < i.end_frame_exclusive
                    or (v.frame_count is not None and i.end_frame_exclusive > v.frame_count)
                ):
                    raise ValueError(
                        "An interval must include at least one frame inside the video."
                    )
            if not v.first_pts <= i.start_pts < i.end_pts <= v.end_pts:
                raise ValueError("Annotation timestamps are outside the video.")
            if not isinstance(i.notes, str):
                raise ValueError("Annotation notes must be text.")
            if i.end_time_method not in (
                "source_timestamp",
                "next_frame_pts",
                "last_frame_duration",
                "stream_end",
                "estimated_from_previous_frame",
            ):
                raise ValueError("Invalid annotation end-time method.")
        for subject in subject_ids:
            intervals = sorted(
                (i for i in self.intervals if i.subject_id == subject), key=lambda i: i.start_pts
            )
            for left, right in zip(intervals, intervals[1:]):
                if left.end_pts > right.start_pts or (
                    left.end_frame_exclusive is not None
                    and right.start_frame is not None
                    and left.end_frame_exclusive > right.start_frame
                ):
                    raise ValueError(
                        "Intervals overlap for this animal. Adjust or delete the existing label."
                    )

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "frame_convention": "zero_based_start_inclusive_end_exclusive",
            "time_convention": "source_presentation_time_relative_to_first_frame",
            "unannotated_time": "unreviewed",
            **asdict(self),
        }
        for interval in payload["intervals"]:
            start = self.video.seconds(interval["start_pts"])
            end = self.video.seconds(interval["end_pts"])
            interval.update(start_time_s=start, end_time_s=end, duration_s=end - start)
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorDocument:
        if data.get("format") != FORMAT or data.get("schema_version") not in (1, SCHEMA_VERSION):
            raise ValueError("Unsupported behavior annotation format or version.")
        if data.get("frame_convention") != "zero_based_start_inclusive_end_exclusive":
            raise ValueError("Unsupported annotation frame convention.")
        if (
            data.get("time_convention") != "source_presentation_time_relative_to_first_frame"
            or data.get("unannotated_time") != "unreviewed"
        ):
            raise ValueError("Unsupported annotation timing or review convention.")
        try:
            intervals = []
            for raw in data["intervals"]:
                row = dict(raw)
                for name in ("start_time_s", "end_time_s", "duration_s"):
                    row.pop(name, None)
                intervals.append(Interval(**row))
            result = cls(
                VideoSource(**data["video"]),
                [Behavior(**r) for r in data["behaviors"]],
                [Subject(**r) for r in data["subjects"]],
                intervals,
                data.get("resume_frame", 0),
                data.get("resume_pts"),
            )
            result.validate()
            for raw, expected in zip(data["intervals"], result.to_dict()["intervals"], strict=True):
                for name in ("start_time_s", "end_time_s", "duration_s"):
                    if name in raw and (
                        type(raw[name]) not in (int, float)
                        or not math.isclose(raw[name], expected[name], rel_tol=1e-10, abs_tol=1e-9)
                    ):
                        raise ValueError("Annotation seconds disagree with source timestamps.")
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f"Invalid behavior annotation file: {exc}") from exc
        return result
