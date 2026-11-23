import json
import os
import struct
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from squeakpose.services.behavior_labels import load_document
from squeakpose.services.behavior_video import BehaviorVideoReader, index_video
from squeakpose.ui.behavior_labeler import BehaviorLabelerDialog


def make_video(path, *, hevc=False, start_pts=0):
    # Keep PyAV out of the test runner too: it already imports OpenCV via the UI.
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            from fractions import Fraction
            import av
            import numpy as np
            with av.open(sys.argv[1], "w") as output:
                hevc = sys.argv[2] == "hevc"
                stream = output.add_stream("libx265" if hevc else "ffv1", rate=25)
                if hevc:
                    stream.options = {"x265-params": "bframes=4:b-adapt=0:keyint=240:scenecut=0:log-level=error:pools=none"}
                stream.width, stream.height = 96, 64
                stream.pix_fmt = "yuv420p" if hevc else "bgr0"
                stream.time_base = Fraction(1, 1000)
                stream.codec_context.time_base = Fraction(1, 1000)
                for number, pts in enumerate((0, 40, 120, 160, 280, 320, 400, 440)):
                    rgb = np.full((64, 96, 3), number * 30, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts, frame.time_base = pts + int(sys.argv[3]), Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            if hevc:
                with av.open(sys.argv[1]) as check:
                    assert any(frame.pict_type == 3 for frame in check.decode(video=0))
        """),
            str(path),
            "hevc" if hevc else "ffv1",
            str(start_pts),
        ],
        check=True,
        capture_output=True,
    )


class BehaviorVideoTests(unittest.TestCase):
    def test_vfr_timestamps_and_random_seeks_match_displayed_frames(self):
        with TemporaryDirectory() as root:
            path = Path(root, "source.mkv")
            make_video(path)
            index = index_video(str(path))
            self.assertEqual(index.source.frame_count, 8)
            self.assertAlmostEqual(index.source.seconds(index.pts[2]), 0.12)
            reader = BehaviorVideoReader(index)
            try:
                for frame in (0, 1, 7, 3, 4, 0, 6):
                    self.assertTrue(np.all(reader.frame(frame) == frame * 30))
                self.assertEqual(reader.frame_at_seconds(0.2), 3)
            finally:
                reader.close()
            moved = path.with_name("renamed.mkv")
            path.rename(moved)
            self.assertEqual(index_video(str(moved)).source.id, index.source.id)

    def test_truncated_mp4_rejected_before_hashing_or_decoding(self):
        with TemporaryDirectory() as root:
            path = Path(root, "incomplete.mp4")
            make_video(path, hevc=True)
            content = path.read_bytes()
            offset = 0
            while offset + 8 <= len(content):
                length, kind = struct.unpack(">I4s", content[offset : offset + 8])
                if kind == b"mdat":
                    path.write_bytes(content[: offset + 8 + (length - 8) // 2])
                    break
                offset += length
            else:
                self.fail("HEVC fixture contains no media block")
            with (
                patch("squeakpose.services.behavior_video.hashlib.sha256") as digest,
                patch("squeakpose.services.behavior_video._decode_index") as decode,
            ):
                with self.assertRaisesRegex(ValueError, "video copy is incomplete"):
                    index_video(str(path), project_root=root)
                digest.assert_not_called()
                decode.assert_not_called()
            self.assertFalse(Path(root, "cache").exists())

    def test_extended_mp4_box_detects_missing_bytes_and_size_zero_is_allowed(self):
        from squeakpose.services.behavior_video import _check_mp4_complete

        with TemporaryDirectory() as root:
            path = Path(root, "video.mp4")
            prefix = struct.pack(">I4s", 8, b"ftyp")
            path.write_bytes(prefix + struct.pack(">I4sQ", 1, b"mdat", 8_000_000_000))
            with self.assertRaisesRegex(ValueError, "8,000,000,008"):
                _check_mp4_complete(str(path), path.stat().st_size)
            path.write_bytes(prefix + struct.pack(">I4s", 0, b"mdat") + b"data")
            _check_mp4_complete(str(path), path.stat().st_size)

    def test_hevc_b_frames_seek_in_presentation_order(self):
        with TemporaryDirectory() as root:
            path = Path(root, "source.mp4")
            make_video(path, hevc=True)
            index = index_video(str(path), project_root=root)
            self.assertEqual(index.source.frame_count, 8)
            self.assertAlmostEqual(index.source.seconds(index.pts[2]), 0.12)
            reader = BehaviorVideoReader(index)
            try:
                for number in (7, 0, 4, 2, 3, 1, 6):
                    self.assertAlmostEqual(float(reader.frame(number).mean()), number * 30, delta=3)
            finally:
                reader.close()

    def test_timestamp_cache_reused_after_rename_and_rebuilt_if_corrupt(self):
        with TemporaryDirectory() as root:
            path = Path(root, "source.mkv")
            make_video(path)
            first = index_video(str(path), project_root=root)
            moved = path.with_name("renamed.mkv")
            path.rename(moved)
            with patch("squeakpose.services.behavior_video._decode_index") as decode:
                cached = index_video(str(moved), project_root=root)
                decode.assert_not_called()
            self.assertEqual(cached.pts, first.pts)
            self.assertEqual(cached.source.source_path, str(moved))
            cache = Path(root, "cache", "behavior_video_indexes", f"{first.source.id}.json")
            data = json.loads(cache.read_text())
            data["pts"].reverse()
            cache.write_text(json.dumps(data))
            rebuilt = index_video(str(moved), project_root=root)
            self.assertEqual(rebuilt.pts, first.pts)
            self.assertEqual(json.loads(cache.read_text())["pts"], list(first.pts))

    def test_changed_video_cannot_reuse_old_timestamp_map(self):
        with TemporaryDirectory() as root:
            path = Path(root, "source.mkv")
            make_video(path)
            first = index_video(str(path), project_root=root)
            with path.open("ab") as handle:
                handle.write(b"changed content")
            with patch(
                "squeakpose.services.behavior_video._decode_index",
                side_effect=RuntimeError("cache miss"),
            ) as decode:
                with self.assertRaisesRegex(RuntimeError, "cache miss"):
                    index_video(str(path), project_root=root)
                decode.assert_called_once()
            self.assertTrue(
                Path(root, "cache", "behavior_video_indexes", f"{first.source.id}.json").exists()
            )

    def test_cancel_terminates_decoder_and_does_not_cache_partial_index(self):
        from squeakpose.services.behavior_video import _decode_index, _VideoProcess

        with TemporaryDirectory() as root:
            path = Path(root, "source.mkv")
            make_video(path)
            worker = _VideoProcess("index", str(path))
            with patch("squeakpose.services.behavior_video._VideoProcess", return_value=worker):
                with self.assertRaises(InterruptedError):
                    _decode_index(str(path), lambda _: None, lambda: True)
            self.assertIsNotNone(worker.process.poll())
            self.assertTrue(worker.process.stdout.closed)
            self.assertFalse(Path(root, "cache").exists())

    def test_gui_process_never_loads_pyav(self):
        with TemporaryDirectory() as root:
            path = Path(root, "source.mkv")
            make_video(path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent("""
                    import sys
                    import cv2
                    from squeakpose.ui.behavior_labeler import BehaviorLabelerDialog
                    from squeakpose.services.behavior_video import index_video, BehaviorVideoReader
                    reader = BehaviorVideoReader(index_video(sys.argv[1]))
                    try:
                        assert reader.frame(3).shape == (64, 96, 3)
                        assert 'av' not in sys.modules
                    finally:
                        reader.close()
                        reader.close()
                    assert reader.worker.process.poll() is not None
                """),
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Class AVF", result.stderr)

    def test_cancel_and_invalid_video(self):
        with TemporaryDirectory() as root:
            path = Path(root, "video.mkv")
            make_video(path)
            with self.assertRaises(InterruptedError):
                index_video(str(path), canceled=lambda: True)
            path.write_bytes(b"not a video")
            with self.assertRaises(Exception):
                index_video(str(path))


class BehaviorLabelerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["behavior-labeler-test"])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = self.temp.name
        self.path = Path(self.root, "source.mkv")
        make_video(self.path)
        self.index = index_video(str(self.path))
        from squeakpose.annotation.behavior import BehaviorDocument
        from squeakpose.services.behavior_labels import save_behavior_catalog

        save_behavior_catalog(self.root, BehaviorDocument.new(self.index.source).behaviors)
        self.dialog = BehaviorLabelerDialog(self.root)
        self.warnings = patch.object(
            QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok
        )
        self.warning = self.warnings.start()
        self.dialog._video_ready(self.index)

    def tearDown(self):
        self.dialog.draft = False
        self.dialog.reject()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.warnings.stop()
        self.temp.cleanup()

    def add_interval(self, start=1, end=2):
        self.dialog.behavior.setCurrentIndex(1)
        self.dialog.seek(start)
        self.dialog._mark_start()
        self.dialog.seek(end)
        self.dialog._mark_end()
        self.dialog._apply_interval()

    def test_label_edit_split_undo_and_reopen(self):
        self.add_interval(1, 4)
        doc = load_document(self.root, self.index.source.id)
        self.assertEqual(
            (doc.intervals[0].start_frame, doc.intervals[0].end_frame_exclusive), (1, 5)
        )
        self.assertAlmostEqual(doc.video.seconds(doc.intervals[0].end_pts), 0.32)
        identity = doc.intervals[0].id
        self.dialog._select_interval(identity)
        self.dialog.seek(3)
        self.dialog._split_interval()
        self.assertEqual(len(self.dialog.document.intervals), 2)
        self.dialog._undo()
        self.assertEqual(len(self.dialog.document.intervals), 1)
        self.dialog._redo()
        self.assertEqual(len(self.dialog.document.intervals), 2)
        self.dialog._video_ready(self.index)
        self.assertEqual(len(self.dialog.document.intervals), 2)
        self.assertEqual(self.dialog.current, 3)
        self.assertFalse(self.warning.called)

    def test_overlap_and_save_failure_leave_labels_unchanged(self):
        self.add_interval()
        self.add_interval(2, 3)
        self.assertEqual(len(self.dialog.document.intervals), 1)
        self.assertTrue(self.warning.called)
        self.dialog._clear_editor(force=True)
        with patch(
            "squeakpose.ui.behavior_labeler.save_document", side_effect=OSError("disk full")
        ):
            self.add_interval(4, 5)
        self.assertTrue(self.dialog.draft)
        self.assertEqual(len(self.dialog.document.intervals), 1)
        self.assertEqual(len(load_document(self.root, self.index.source.id).intervals), 1)

    def test_resize_and_delete(self):
        self.add_interval()
        identity = self.dialog.document.intervals[0].id
        self.dialog._select_interval(identity)
        self.dialog._resize_interval(identity, 0, 4)
        self.assertEqual(self.dialog.document.intervals[0].start_frame, 0)
        self.assertEqual(self.dialog.last_frame.value(), 3)
        self.dialog._delete_interval()
        self.assertEqual(self.dialog.document.intervals, [])
        self.dialog._undo()
        self.assertEqual(len(self.dialog.document.intervals), 1)

    def test_async_open_and_close_during_indexing_release_worker(self):
        self.dialog.open_video(str(self.path))
        deadline = time.monotonic() + 5
        while self.dialog.job is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.dialog.job)
        self.assertTrue(self.dialog.content.isEnabled())
        self.assertTrue(self.dialog.index.pts)
        self.assertEqual(self.dialog.go_label.text(), "Go to frame")
        self.dialog.open_video(str(self.path))
        self.dialog.reject()
        deadline = time.monotonic() + 5
        while self.dialog.job is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.dialog.job)
        self.assertIsNone(self.dialog.reader)


if __name__ == "__main__":
    unittest.main()
