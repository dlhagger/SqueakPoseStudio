"""Frame-based behavior labels retain source-frame numbering without full decoding."""

import subprocess
import sys
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.annotation.behavior import BehaviorDocument
from squeakpose.services.behavior_labels import (
    resolve_frame_annotations,
    save_document,
)
from squeakpose.services.behavior_video import (
    BehaviorVideoReader,
    index_video,
    open_behavior_video,
    open_frame_video,
)
from tests.test_behavior_labeler import make_video
from tests.test_behavior_labels import sample_interval


class FrameLookupTests(unittest.TestCase):
    def test_packet_lookup_matches_decoded_frames_for_vfr_and_b_frames(self):
        with TemporaryDirectory() as root:
            for hevc in (False, True):
                with self.subTest(hevc=hevc):
                    path = Path(root, "bframes.mp4" if hevc else "vfr.mkv")
                    make_video(path, hevc=hevc)
                    actual = open_frame_video(str(path), project_root=root)
                    decoded = index_video(str(path))
                    self.assertEqual(actual.pts, decoded.pts)
                    self.assertEqual(actual.source.frame_count, 8)
                    self.assertEqual(actual.source.end_pts, decoded.source.end_pts)
                    reader = BehaviorVideoReader(actual)
                    try:
                        for frame in (7, 2, 3, 0, 5):
                            self.assertAlmostEqual(
                                float(reader.frame(frame).mean()), frame * 30, delta=3
                            )
                    finally:
                        reader.close()
                    self.assertFalse(Path(root, "cache").exists())
                    with patch(
                        "squeakpose.services.behavior_video._decode_index",
                        wraps=__import__(
                            "squeakpose.services.behavior_video", fromlist=["_decode_index"]
                        )._decode_index,
                    ) as job:
                        reopened = open_frame_video(str(path), project_root=root)
                        self.assertEqual(reopened.pts, actual.pts)
                        self.assertEqual(
                            [call.kwargs.get("mode") for call in job.call_args_list],
                            ["probe", "packets"],
                        )
                    self.assertFalse(Path(root, "cache").exists())

    def test_packet_lookup_does_not_decode_images(self):
        with TemporaryDirectory() as root:
            path = Path(root, "source.mp4")
            make_video(path, hevc=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent("""
                    import sys
                    from squeakpose.workers import behavior_video as worker
                    original = worker.av.open
                    class Container:
                        def __init__(self, *args, **kwargs): self.inner = original(*args, **kwargs)
                        def __enter__(self): return self
                        def __exit__(self, *args): self.inner.close()
                        def __getattr__(self, key): return getattr(self.inner, key)
                        def decode(self, *args, **kwargs): raise AssertionError('Full decoding forbidden')
                        def demux(self, stream):
                            for packet in self.inner.demux(stream):
                                class Packet:
                                    def __getattr__(self, key): return getattr(packet, key)
                                    def decode(self): raise AssertionError('Packet decoding forbidden')
                                yield Packet()
                    worker.av.open = Container
                    result = worker.packet_index(sys.argv[1])
                    assert len(result['pts']) == 8
                    assert result['pts'] == sorted(result['pts'])
                """),
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_timestamp_only_labels_resolve_to_frame_indices_without_rounding(self):
        with TemporaryDirectory() as root:
            path = Path(root, "vfr.mkv")
            make_video(path)
            quick = open_behavior_video(str(path))
            doc = BehaviorDocument.new(quick.source)
            doc.put(
                replace(
                    sample_interval(doc),
                    start_frame=None,
                    end_frame_exclusive=None,
                    start_pts=40,
                    end_pts=280,
                )
            )
            doc.resume_pts, doc.resume_frame = 160, None
            save_document(root, doc)
            indexed = open_frame_video(str(path), project_root=root)
            self.assertEqual(indexed.source.id, doc.video.id)
            doc.video = indexed.source
            mapped = resolve_frame_annotations(doc, indexed.pts)
            self.assertEqual(
                (mapped.intervals[0].start_frame, mapped.intervals[0].end_frame_exclusive), (1, 4)
            )
            self.assertEqual(mapped.resume_frame, 3)
            bad = doc.copy()
            bad.intervals = [replace(doc.intervals[0], start_pts=41)]
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_frame_annotations(bad, indexed.pts)
