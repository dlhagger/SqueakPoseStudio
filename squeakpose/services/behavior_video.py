"""Frame-accurate video access through an isolated, OpenCV-free PyAV process."""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from squeakpose.annotation.behavior import VideoSource
from squeakpose.core import atomic_write_text
from squeakpose.json_io import read_json_file
from squeakpose.project.safety import require_path_within_project
from squeakpose.workers.protocol import parse_event_line

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoIndex:
    source: VideoSource
    pts: tuple[int, ...] = ()

    def position_pts(self, position: int) -> int:
        return self.pts[position] if self.pts else position

    @property
    def first_position(self) -> int:
        return 0 if self.pts else self.source.first_pts

    @property
    def last_position(self) -> int:
        return len(self.pts) - 1 if self.pts else self.source.end_pts - 1

    def boundary(self, frame: int) -> int:
        return self.source.end_pts if frame == len(self.pts) else self.pts[frame]


class _VideoProcess:
    def __init__(self, mode: str, path: str):
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "squeakpose.workers.behavior_video", mode, path],
                cwd=str(Path(__file__).resolve().parents[2]),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.errors,
            )
        except BaseException:
            self.errors.close()
            raise

    def event(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            self.errors.seek(0)
            detail = self.errors.read(8192).decode("utf-8", errors="replace").strip()
            raise ValueError(f"Video decoder stopped unexpectedly. {detail}".strip())
        payload = parse_event_line(line.decode("utf-8")).payload
        if payload["event"] == "error":
            raise ValueError(payload.get("message", "Video decoding failed."))
        return payload

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.errors.close()


def _decode_index(path: str, progress, canceled, *, mode="index") -> dict:
    worker = _VideoProcess(mode, path)
    events: queue.Queue = queue.Queue()

    def receive():
        try:
            while True:
                event = worker.event()
                events.put(event)
                if event["event"] == "result":
                    return
        except Exception as exc:
            events.put(exc)

    thread = threading.Thread(target=receive, name="behavior-video-index", daemon=True)
    thread.start()
    try:
        while True:
            if canceled():
                raise InterruptedError("Video opening canceled.")
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(event, Exception):
                raise event
            if event["event"] == "progress":
                progress(event["message"])
            elif event["event"] == "result":
                return event
    finally:
        worker.close()
        thread.join(timeout=2)


def _make_index(data: dict, path: str, identity: str) -> VideoIndex:
    points = data["pts"]
    if (
        not isinstance(points, list)
        or not points
        or any(type(p) is not int for p in points)
        or any(a >= b for a, b in zip(points, points[1:]))
        or type(data["time_base_num"]) is not int
        or type(data["time_base_den"]) is not int
        or data["time_base_num"] <= 0
        or data["time_base_den"] <= 0
        or type(data["end_pts"]) is not int
        or data["end_pts"] <= points[-1]
        or data["end_time_method"]
        not in (
            "last_frame_duration",
            "packet_duration",
            "stream_end",
            "estimated_from_previous_frame",
        )
    ):
        raise ValueError("Invalid frame timestamp index.")
    return VideoIndex(
        VideoSource(
            identity,
            os.path.basename(path),
            path,
            len(points),
            data["time_base_num"],
            data["time_base_den"],
            points[0],
            data["end_pts"],
            data["end_time_method"],
        ),
        tuple(points),
    )


def _check_mp4_complete(path: str, file_size: int) -> None:
    """Reject visibly truncated MP4/MOV boxes without reading or decoding media.

    This is a fast completeness check, not a general container validator. An
    unrecognized layout is left to the decoder, and size-zero boxes extend to EOF.
    """
    with open(path, "rb") as handle:
        first = handle.read(8)
        if len(first) < 8 or first[4:] not in (
            b"ftyp",
            b"styp",
            b"moov",
            b"mdat",
            b"free",
            b"wide",
            b"skip",
        ):
            return
        offset = 0
        while offset + 8 <= file_size:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("The video changed while checking it. Wait for copying to finish.")
            length, kind = struct.unpack(">I4s", header)
            header_size = 8
            if length == 0:
                return
            if length == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    raise ValueError(
                        "The MP4 video has an incomplete block header. Open a complete copy."
                    )
                length = struct.unpack(">Q", extended)[0]
                header_size = 16
            if length < header_size:
                return  # Let the decoder explain an unsupported or malformed layout.
            if offset + length > file_size:
                name = kind.decode("ascii", errors="replace")
                raise ValueError(
                    "This video copy is incomplete. "
                    f"Its MP4 {name} block extends to byte {offset + length:,}, "
                    f"but the file currently ends at byte {file_size:,}. "
                    "Finish downloading or copying the complete original, then open it again. "
                    "No partial timestamp index was saved."
                )
            offset += length


def index_video(
    path: str,
    progress: Callable[[str], None] = lambda _: None,
    canceled: Callable[[], bool] = lambda: False,
    *,
    project_root: str | None = None,
) -> VideoIndex:
    """Index actual PTS once per content hash, including VFR/B-frame recordings.

    Rehash on each open so edited files never reuse an obsolete timestamp map.
    The optional cache is disposable; annotation files remain independent.
    """
    path = os.path.abspath(path)
    before = os.stat(path)
    if canceled():
        raise InterruptedError("Video opening canceled.")
    progress("Checking video file completeness…")
    _check_mp4_complete(path, before.st_size)
    digest = hashlib.sha256()
    progress("Identifying source video… 0%")
    with open(path, "rb") as handle:
        total = 0
        previous_percent = -1
        while block := handle.read(4 * 1024 * 1024):
            if canceled():
                raise InterruptedError("Video opening canceled.")
            digest.update(block)
            total += len(block)
            percent = min(100, int(total * 100 / max(1, before.st_size)))
            if percent != previous_percent:
                progress(f"Identifying source video… {percent}%")
                previous_percent = percent
    if canceled():
        raise InterruptedError("Video opening canceled.")
    identity = digest.hexdigest()
    cache_path = None
    cached = None
    if project_root:
        try:
            cache_path = require_path_within_project(
                project_root,
                os.path.join(project_root, "cache", "behavior_video_indexes", f"{identity}.json"),
                purpose="behavior timestamp cache",
                allow_root=False,
            )
            if os.path.lexists(cache_path):
                data = read_json_file(cache_path, max_bytes=128 * 1024 * 1024, require_object=True)
                if data.get("version") == 1 and data.get("video_id") == identity:
                    cached = _make_index(data, path, identity)
        except OSError, ValueError, KeyError, TypeError:
            logger.warning("Ignoring unusable behavior timestamp cache", exc_info=True)
    if cached is not None:
        progress("Using saved exact frame timestamps…")
        result = cached
    else:
        progress("Preparing exact timestamps (first open)… Saved for future openings.")
        data = _decode_index(path, progress, canceled)
        result = _make_index(data, path, identity)
    if canceled():
        raise InterruptedError("Video opening canceled.")
    after = os.stat(path)
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(
            "The video changed while opening it. Try again when recording has finished."
        )
    if cached is None and cache_path:
        try:
            atomic_write_text(cache_path, json.dumps({"version": 1, "video_id": identity, **data}))
        except OSError, ValueError:
            logger.warning("Could not save behavior timestamp cache", exc_info=True)
            progress("Exact timestamps ready; cache could not be saved.")
    return result


def open_behavior_video(
    path: str,
    progress: Callable[[str], None] = lambda _: None,
    canceled: Callable[[], bool] = lambda: False,
    *,
    project_root: str | None = None,
) -> VideoIndex:
    """Open from existing timing metadata; never enumerate all packets or frames.

    A bounded head/tail fingerprint plus size and modification time identifies new
    sources. Legacy full-file identities are verified only when reusing old labels.
    """
    path = os.path.abspath(path)
    before = os.stat(path)
    if canceled():
        raise InterruptedError("Video opening canceled.")
    progress("Checking video file…")
    _check_mp4_complete(path, before.st_size)
    digest = hashlib.sha256(b"squeakpose-sampled-source-v1")
    digest.update(f"{before.st_size}:{before.st_mtime_ns}".encode("ascii"))
    with open(path, "rb") as handle:
        digest.update(handle.read(1024 * 1024))
        handle.seek(max(0, before.st_size - 1024 * 1024))
        digest.update(handle.read(1024 * 1024))
    identity, method = digest.hexdigest(), "sampled_sha256_stat_v1"
    if project_root:
        # Preserve existing annotations when transitioning from full-file hashes.
        directory = require_path_within_project(
            project_root, os.path.join(project_root, "behavior labels"), allow_root=False
        )
        current = os.path.join(directory, f"{identity}.json")
        if os.path.isdir(directory) and not os.path.lexists(current):
            legacy_ids = set()
            for candidate in Path(directory).glob("*.json"):
                if candidate.name == "catalog.json":
                    continue
                data = read_json_file(
                    str(candidate), max_bytes=32 * 1024 * 1024, require_object=True
                )
                source = data.get("video", {})
                if source.get("identity_method", "sha256") == "sha256" and (
                    source.get("source_path") == path
                    or source.get("filename") == os.path.basename(path)
                ):
                    legacy_ids.add(source.get("id"))
            if legacy_ids:
                progress("Verifying source for existing labels…")
                full = hashlib.sha256()
                with open(path, "rb") as handle:
                    while block := handle.read(4 * 1024 * 1024):
                        if canceled():
                            raise InterruptedError("Video opening canceled.")
                        full.update(block)
                if full.hexdigest() in legacy_ids:
                    identity, method = full.hexdigest(), "sha256"
    progress("Opening video from its stored timestamps…")
    data = _decode_index(path, progress, canceled, mode="probe")
    after = os.stat(path)
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("The video changed while opening it. Wait for copying to finish.")
    if canceled():
        raise InterruptedError("Video opening canceled.")
    return VideoIndex(
        VideoSource(
            identity,
            os.path.basename(path),
            path,
            None,
            data["time_base_num"],
            data["time_base_den"],
            data["first_pts"],
            data["end_pts"],
            data["end_time_method"],
            method,
        )
    )


def open_frame_video(
    path: str,
    progress: Callable[[str], None] = lambda _: None,
    canceled: Callable[[], bool] = lambda: False,
    *,
    project_root: str | None = None,
) -> VideoIndex:
    """Rebuild an in-memory frame lookup on each open, without decoding all images."""
    before = os.stat(path)
    preview = open_behavior_video(path, progress, canceled, project_root=project_root)
    progress("Reading frame lookup from the video container…")
    data = _decode_index(preview.source.source_path, progress, canceled, mode="packets")
    result = _make_index(data, preview.source.source_path, preview.source.id)
    result = replace(
        result, source=replace(result.source, identity_method=preview.source.identity_method)
    )
    after = os.stat(path)
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("The video changed while opening. Wait for copying to finish.")
    if canceled():
        raise InterruptedError("Video opening canceled.")
    return result


class BehaviorVideoReader:
    """Read verified RGB frames without loading PyAV beside OpenCV in the GUI."""

    def __init__(self, index: VideoIndex):
        self.index = index
        self.last_index = -1
        self.worker = _VideoProcess("read" if index.pts else "time", index.source.source_path)
        self.current_pts = index.source.first_pts
        self.current_end_pts = index.source.first_pts
        self.at_end = False
        self.bounds_cache: dict[int, tuple[int, int]] = {}
        self.end_methods: dict[int, str] = {}
        try:
            if self.worker.event()["event"] != "ready":
                raise ValueError("Video decoder did not initialize.")
        except BaseException:
            self.close()
            raise

    def frame_time(self, target: int, *, direction: str = "at", render: bool = True):
        process = self.worker.process
        assert process.stdin is not None and process.stdout is not None
        request = {"pts": int(target), "direction": direction, "render": render}
        process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        process.stdin.flush()
        event = self.worker.event()
        if event["event"] not in ("frame", "bounds"):
            raise ValueError("Video decoder returned an unexpected response.")
        self.current_pts, self.current_end_pts = event["pts"], event["end_pts"]
        self.at_end = event["at_end"]
        if self.current_end_pts <= self.current_pts:
            raise ValueError("The decoded frame has invalid timing.")
        if len(self.bounds_cache) >= 4096:
            self.bounds_cache.clear()
            self.end_methods.clear()
        self.end_methods[self.current_end_pts] = event["end_time_method"]
        self.bounds_cache[self.current_pts] = (self.current_pts, self.current_end_pts)
        if not render:
            return self.current_pts, self.current_end_pts
        return self._pixels(event)

    def bounds_at(self, target: int) -> tuple[int, int]:
        if target in self.bounds_cache:
            return self.bounds_cache[target]
        return self.frame_time(target, render=False)

    def _pixels(self, event):
        process = self.worker.process
        assert process.stdout is not None
        width, height = event["width"], event["height"]
        if not (0 < width <= 32768 and 0 < height <= 32768 and width * height <= 100_000_000):
            raise ValueError("Unsupported decoded frame size.")
        expected = width * height * 3
        pixels = process.stdout.read(expected)
        if len(pixels) != expected:
            raise ValueError("Video decoder returned an incomplete frame.")
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)

    def frame(self, number: int):
        if not 0 <= number < len(self.index.pts):
            raise ValueError("Frame outside video.")
        process = self.worker.process
        assert process.stdin is not None and process.stdout is not None
        request = {"pts": self.index.pts[number], "sequential": number == self.last_index + 1}
        process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        process.stdin.flush()
        event = self.worker.event()
        if event["event"] != "frame":
            raise ValueError("Video decoder returned an unexpected response.")
        self.last_index = number
        return self._pixels(event)

    def frame_at_seconds(self, seconds: float) -> int:
        v = self.index.source
        pts = v.first_pts + seconds * v.time_base_den / v.time_base_num
        return max(0, min(len(self.index.pts) - 1, bisect.bisect_right(self.index.pts, pts) - 1))

    def close(self) -> None:
        self.worker.close()
