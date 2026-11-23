"""Isolated PyAV decoder: never import OpenCV or the services package here.

Control messages use the shared JSON event envelope. A ``frame`` header is
followed by exactly ``width * height * 3`` raw RGB bytes, then the next event.
"""

from __future__ import annotations

import json
import sys
import time

import av

from squeakpose.workers.protocol import write_event


def index(path: str) -> dict:
    points: list[int] = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("This file contains no video stream.")
        stream = container.streams.video[0]
        base = stream.time_base
        if base is None:
            raise ValueError("The video has no reliable timestamp time base.")
        duration = 0
        last_report = time.monotonic()
        for frame in container.decode(stream):
            if frame.pts is None or (points and frame.pts <= points[-1]):
                raise ValueError(
                    "The video has missing or non-increasing timestamps; re-encode it before labeling."
                )
            points.append(frame.pts)
            duration = int(frame.duration or 0)
            if time.monotonic() - last_report >= 0.5:
                percent = ""
                if stream.duration and stream.duration > 0:
                    fraction = (frame.pts - points[0]) / stream.duration
                    percent = f"{min(99, max(0, int(fraction * 100)))}% · "
                write_event(
                    {
                        "event": "progress",
                        "message": f"Preparing exact timestamps (first open)… {percent}{len(points):,} frames",
                    }
                )
                last_report = time.monotonic()
        if not points:
            raise ValueError("The video has no decodable frames.")
        method = "last_frame_duration"
        if duration <= 0:
            end = (stream.start_time or 0) + (stream.duration or 0)
            if end > points[-1]:
                duration = end - points[-1]
                method = "stream_end"
            elif len(points) > 1:
                duration = points[-1] - points[-2]
                method = "estimated_from_previous_frame"
            else:
                raise ValueError("Cannot determine the last frame's duration.")
        return {
            "pts": points,
            "time_base_num": base.numerator,
            "time_base_den": base.denominator,
            "end_pts": points[-1] + duration,
            "end_time_method": method,
        }


def packet_index(path: str) -> dict:
    """Index presentation-order samples in frame-packet video containers.

    B-frame packets arrive in decode order, so sort by PTS before assigning the
    same zero-based presentation-order frame indices used by inference.
    """
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("This file contains no video stream.")
        stream = container.streams.video[0]
        codec = stream.codec_context.codec.name
        if codec not in {
            "h264",
            "hevc",
            "av1",
            "libdav1d",
            "ffv1",
            "mjpeg",
            "vp8",
            "vp9",
        } or not any(
            name in container.format.name for name in ("mov", "mp4", "matroska", "webm", "avi")
        ):
            raise ValueError(
                "Fast frame indexing is not supported for this video format. "
                "Use an MP4 or MKV with H.264, HEVC, AV1, VP8/9, MJPEG, or FFV1 video."
            )
        base = stream.time_base
        if not base:
            raise ValueError("The video has no source time base.")
        points = []
        last_pts = None
        last_duration = 0
        reported = time.monotonic()
        for packet in container.demux(stream):
            if not packet.size or packet.is_discard:
                continue
            if packet.pts is None or packet.is_corrupt:
                raise ValueError(
                    "Cannot establish exact frame numbers: missing timing or a corrupt video packet."
                )
            points.append(packet.pts)
            if last_pts is None or packet.pts > last_pts:
                last_pts, last_duration = packet.pts, int(packet.duration or 0)
            if time.monotonic() - reported >= 0.5:
                write_event(
                    {
                        "event": "progress",
                        "message": f"Reading frame lookup (no image decoding)… {len(points):,} frames",
                    }
                )
                reported = time.monotonic()
        if not points:
            raise ValueError("This video contains no timestamped frames.")
        points.sort()
        if any(a == b for a, b in zip(points, points[1:])):
            raise ValueError(
                "Repeated presentation timestamps prevent an unambiguous frame lookup."
            )
        if stream.frames and stream.frames != len(points):
            raise ValueError("The frame count in the container disagrees with its video samples.")
        method = "packet_duration"
        if last_duration <= 0:
            end = (stream.start_time or 0) + (stream.duration or 0)
            if end > points[-1]:
                last_duration, method = end - points[-1], "stream_end"
            elif len(points) > 1:
                last_duration, method = points[-1] - points[-2], "estimated_from_previous_frame"
            else:
                raise ValueError("The final video frame has no duration.")
        return {
            "pts": points,
            "time_base_num": base.numerator,
            "time_base_den": base.denominator,
            "end_pts": points[-1] + last_duration,
            "end_time_method": method,
            "index_method": "presentation_order_packets_v1",
        }


class Decoder:
    def __init__(self, path: str):
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.frames = iter(self.container.decode(self.stream))

    def frame(self, target: int, sequential: bool):
        if not sequential:
            self.container.seek(target, stream=self.stream, backward=True, any_frame=False)
            self.frames = iter(self.container.decode(self.stream))
        for frame in self.frames:
            if frame.pts == target:
                return frame.to_ndarray(format="rgb24")
            if frame.pts is not None and frame.pts > target:
                break
        # Some demuxers seek imprecisely. Verify a sequential fallback instead
        # of displaying an incorrect frame under an annotation timestamp.
        self.container.close()
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.frames = iter(self.container.decode(self.stream))
        for frame in self.frames:
            if frame.pts == target:
                return frame.to_ndarray(format="rgb24")
        raise ValueError("Could not decode the requested source frame.")

    def close(self):
        self.container.close()


class TimestampDecoder:
    """Decode around a requested PTS, keeping one-frame lookahead for exact stops."""

    def __init__(self, path: str):
        self.container = av.open(path)
        if not self.container.streams.video:
            self.container.close()
            raise ValueError("This file contains no video stream.")
        self.stream = self.container.streams.video[0]
        self.frames = iter(self.container.decode(self.stream))
        self.current = None
        self.following = None
        self.previous_pts = None

    def _read(self):
        frame = next(self.frames, None)
        if frame is not None and frame.pts is None:
            raise ValueError("This video has a frame without a presentation timestamp.")
        return frame

    def _pair(self, current, following):
        if following is not None and following.pts <= current.pts:
            raise ValueError("This video has non-increasing frame timestamps.")
        self.current, self.following = current, following

    def locate(self, target: int, direction: str = "at"):
        if direction == "next":
            if self.current is None or target != self.current.pts:
                self.locate(target)
            if self.following is not None:
                self.previous_pts = self.current.pts
                self._pair(self.following, self._read())
            return self.current
        if direction == "previous":
            target -= 1
        if (
            self.current is not None
            and self.current.pts <= target
            and (self.following is not None and target < self.following.pts)
        ):
            return self.current
        self.container.seek(target, stream=self.stream, backward=True, any_frame=False)
        self.frames = iter(self.container.decode(self.stream))
        chosen = self._read()
        if chosen is None:
            raise ValueError("No frame is available at the requested time.")
        # The first frame can be after a time before the recording's start.
        # Otherwise seek farther back when a demuxer landed after the target.
        if chosen.pts > target and target > (self.stream.start_time or 0):
            step = max(1, int(1 / self.stream.time_base))
            seek_at = target
            while chosen.pts > target and seek_at > (self.stream.start_time or 0):
                seek_at = max(self.stream.start_time or 0, target - step)
                self.container.seek(seek_at, stream=self.stream, backward=True, any_frame=False)
                self.frames = iter(self.container.decode(self.stream))
                chosen = self._read()
                if chosen is None:
                    raise ValueError("No frame is available at the requested time.")
                step *= 2
        self.previous_pts = None
        following = self._read()
        while following is not None and following.pts <= target:
            if following.pts <= chosen.pts:
                raise ValueError("This video has non-increasing frame timestamps.")
            self.previous_pts = chosen.pts
            chosen, following = following, self._read()
        self._pair(chosen, following)
        return chosen

    def bounds(self) -> dict:
        frame = self.current
        if frame is None:
            raise ValueError("No frame has been selected.")
        if self.following is not None:
            end = self.following.pts
            method = "next_frame_pts"
        else:
            duration = int(frame.duration or 0)
            method = "last_frame_duration"
            if duration <= 0 and self.stream.duration:
                duration = (self.stream.start_time or 0) + self.stream.duration - frame.pts
                method = "stream_end"
            if duration <= 0:
                if self.previous_pts is None:
                    raise ValueError("The final frame has no reliable duration.")
                duration = frame.pts - self.previous_pts
                method = "estimated_from_previous_frame"
            end = frame.pts + duration
        return {
            "pts": frame.pts,
            "end_pts": end,
            "at_end": self.following is None,
            "end_time_method": method,
        }

    def close(self):
        self.container.close()


def probe(path: str) -> dict:
    decoder = TimestampDecoder(path)
    try:
        stream = decoder.stream
        base = stream.time_base
        if base is None or base <= 0:
            raise ValueError("This video has no reliable timestamp time base.")
        frame = decoder.locate(stream.start_time or 0)
        if stream.duration and stream.duration > 0:
            end = (
                stream.start_time if stream.start_time is not None else frame.pts
            ) + stream.duration
            method = "stream_duration"
        elif decoder.container.duration and decoder.container.duration > 0:
            # Matroska Duration is measured from segment timestamp zero, so
            # adding a nonzero track start would count that offset twice.
            offset = (
                0
                if "matroska" in decoder.container.format.name
                else (decoder.container.start_time or 0)
            )
            end = round((decoder.container.duration + offset) / av.time_base / base)
            method = "container_duration"
        else:
            raise ValueError(
                "This recording has no duration metadata. Immediate timestamp-based opening "
                "is unavailable; a timestamp scan is required for this file."
            )
        if end <= frame.pts:
            raise ValueError("The recording's duration metadata is invalid.")
        return {
            "first_pts": frame.pts,
            "end_pts": end,
            "time_base_num": base.numerator,
            "time_base_den": base.denominator,
            "end_time_method": method,
        }
    finally:
        decoder.close()


def main() -> int:
    try:
        mode, path = sys.argv[1:]
        if mode == "packets":
            write_event({"event": "result", **packet_index(path)})
        elif mode == "probe":
            write_event({"event": "result", **probe(path)})
        elif mode == "time":
            timed = TimestampDecoder(path)
            try:
                write_event({"event": "ready"})
                for line in sys.stdin:
                    request = json.loads(line)
                    frame = timed.locate(request["pts"], request.get("direction", "at"))
                    bounds = timed.bounds()
                    if not request.get("render", True):
                        write_event({"event": "bounds", **bounds})
                        continue
                    rgb = frame.to_ndarray(format="rgb24")
                    height, width, _ = rgb.shape
                    write_event({"event": "frame", "width": width, "height": height, **bounds})
                    sys.stdout.buffer.write(rgb.tobytes())
                    sys.stdout.buffer.flush()
            finally:
                timed.close()
        elif mode == "index":
            write_event({"event": "result", **index(path)})
        elif mode == "read":
            decoder = Decoder(path)
            try:
                write_event({"event": "ready"})
                for line in sys.stdin:
                    request = json.loads(line)
                    rgb = decoder.frame(request["pts"], request["sequential"])
                    height, width, _ = rgb.shape
                    write_event({"event": "frame", "width": width, "height": height})
                    sys.stdout.buffer.write(rgb.tobytes())
                    sys.stdout.buffer.flush()
            finally:
                decoder.close()
        else:
            raise ValueError("Unknown video worker mode.")
        return 0
    except Exception as exc:
        write_event({"event": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
