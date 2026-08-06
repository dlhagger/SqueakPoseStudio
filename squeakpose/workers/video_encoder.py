"""Isolated PyAV encoder process used by analysis video exports."""

from __future__ import annotations

import argparse
from fractions import Fraction
import sys
from typing import Optional

import av
import numpy as np


def _read_exact(stream, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_stdin(output_path: str, fps: float, width: int, height: int) -> int:
    encoded_width = width + width % 2
    encoded_height = height + height % 2
    frame_bytes = width * height * 3
    container = None
    try:
        container = av.open(
            output_path,
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        stream = container.add_stream(
            "libx264",
            rate=Fraction(str(float(fps))).limit_denominator(100_000),
        )
        stream.width = encoded_width
        stream.height = encoded_height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "21", "preset": "medium"}

        sys.stdout.buffer.write(b"READY\n")
        sys.stdout.buffer.flush()

        while True:
            raw = _read_exact(sys.stdin.buffer, frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise ValueError(
                    f"Received an incomplete video frame ({len(raw)} of {frame_bytes} bytes)."
                )
            array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            if encoded_width != width or encoded_height != height:
                array = np.pad(
                    array,
                    ((0, encoded_height - height), (0, encoded_width - width), (0, 0)),
                    mode="edge",
                )
            frame = av.VideoFrame.from_ndarray(array, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return 0
    except Exception as exc:
        print(f"PyAV H.264 encoding failed: {exc}", file=sys.stderr, flush=True)
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Encode raw BGR frames as H.264 MP4.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    args = parser.parse_args(argv)
    return encode_stdin(args.output, args.fps, args.width, args.height)


if __name__ == "__main__":
    raise SystemExit(main())
