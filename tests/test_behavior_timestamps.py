"""Timestamp-first opening and labeling must not depend on a whole-video scan."""

import csv
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from squeakpose.annotation.behavior import BehaviorDocument
from squeakpose.services.behavior_labels import export_document, load_document, save_document
from squeakpose.services.behavior_video import BehaviorVideoReader, index_video, open_behavior_video
from squeakpose.ui.behavior_labeler import BehaviorLabelerDialog
from tests.test_behavior_labeler import make_video
from tests.test_behavior_labels import sample_document, sample_interval


class TimestampVideoTests(unittest.TestCase):
    def test_probe_decodes_only_two_frames_and_never_demuxes_entire_video(self):
        with TemporaryDirectory() as root:
            path = Path(root, "video.mkv")
            make_video(path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent("""
                    import sys
                    from squeakpose.workers import behavior_video as worker
                    original = worker.av.open
                    count = 0
                    class LimitedContainer:
                        def __init__(self, *args, **kwargs):
                            self.container = original(*args, **kwargs)
                        def __getattr__(self, name):
                            return getattr(self.container, name)
                        def demux(self, *args, **kwargs):
                            raise AssertionError('Opening must not scan packets')
                        def decode(self, *args, **kwargs):
                            global count
                            for frame in self.container.decode(*args, **kwargs):
                                count += 1
                                assert count <= 2, 'Opening decoded more than the first two frames'
                                yield frame
                    worker.av.open = LimitedContainer
                    result = worker.probe(sys.argv[1])
                    assert result['first_pts'] == 0
                    assert result['end_pts'] == 480
                    assert count == 2
                """),
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_time_seeks_and_steps_use_actual_vfr_frame_boundaries(self):
        with TemporaryDirectory() as root:
            path = Path(root, "video.mkv")
            make_video(path)
            index = open_behavior_video(str(path), project_root=root)
            self.assertFalse(index.pts)
            self.assertIsNone(index.source.frame_count)
            self.assertFalse(Path(root, "cache").exists())
            reader = BehaviorVideoReader(index)
            try:
                for target, expected, end, intensity in (
                    (100, 40, 120, 30),
                    (140, 120, 160, 60),
                    (479, 440, 480, 210),
                    (0, 0, 40, 0),
                ):
                    image = reader.frame_time(target)
                    self.assertEqual((reader.current_pts, reader.current_end_pts), (expected, end))
                    self.assertEqual(float(image.mean()), intensity)
                reader.frame_time(0, direction="previous")
                self.assertEqual(reader.current_pts, 0)
                reader.bounds_at(280)  # Boundary queries must not disturb frame stepping.
                reader.frame_time(0, direction="next")
                self.assertEqual(reader.current_pts, 40)
                reader.frame_time(40, direction="previous")
                self.assertEqual(reader.current_pts, 0)
                reader.frame_time(440)
                self.assertTrue(reader.at_end)
            finally:
                reader.close()

    def test_hevc_time_seeks_match_original_frame_timestamps(self):
        with TemporaryDirectory() as root:
            path = Path(root, "video.mp4")
            make_video(path, hevc=True)
            index = open_behavior_video(str(path))
            reader = BehaviorVideoReader(index)
            try:
                for seconds, level in ((0.28, 120), (0.12, 60), (0.0, 0), (0.44, 210)):
                    pts = round(seconds * index.source.time_base_den / index.source.time_base_num)
                    rgb = reader.frame_time(pts)
                    self.assertEqual(reader.current_pts, pts)
                    self.assertAlmostEqual(float(rgb.mean()), level, delta=3)
            finally:
                reader.close()

    def test_nonzero_source_start_is_not_added_twice_to_mkv_duration(self):
        with TemporaryDirectory() as root:
            path = Path(root, "offset.mkv")
            make_video(path, start_pts=2000)
            index = open_behavior_video(str(path))
            self.assertEqual((index.source.first_pts, index.source.end_pts), (2000, 2480))
            reader = BehaviorVideoReader(index)
            try:
                reader.frame_time(2050)
                self.assertEqual(reader.current_pts, 2040)
                self.assertAlmostEqual(index.source.seconds(reader.current_pts), 0.04)
            finally:
                reader.close()

    def test_legacy_identity_and_labels_remain_available(self):
        with TemporaryDirectory() as root:
            path = Path(root, "video.mkv")
            make_video(path)
            old_index = index_video(str(path))
            document = BehaviorDocument.new(old_index.source)
            save_document(root, document)
            fast = open_behavior_video(str(path), project_root=root)
            self.assertEqual(fast.source.id, old_index.source.id)
            self.assertIsNotNone(load_document(root, fast.source.id))

    def test_new_identity_survives_rename_and_detects_changed_source(self):
        with TemporaryDirectory() as root:
            path = Path(root, "video.mkv")
            make_video(path)
            first = open_behavior_video(str(path))
            renamed = path.with_name("renamed.mkv")
            path.rename(renamed)
            self.assertEqual(open_behavior_video(str(renamed)).source.id, first.source.id)
            stat = renamed.stat()
            os.utime(renamed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
            self.assertNotEqual(open_behavior_video(str(renamed)).source.id, first.source.id)

    def test_version_one_annotations_load_without_reinterpreting_frames(self):
        doc = sample_document()
        doc.put(sample_interval(doc))
        raw = doc.to_dict()
        raw["schema_version"] = 1
        raw["video"].pop("identity_method")
        raw.pop("resume_pts")
        loaded = BehaviorDocument.from_dict(raw)
        self.assertEqual(loaded.intervals, doc.intervals)
        self.assertEqual(loaded.to_dict()["schema_version"], 2)


class TimestampLabelerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["timestamp-labeler-test"])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = self.temp.name
        self.path = Path(self.root, "source.mkv")
        make_video(self.path)
        self.index = open_behavior_video(str(self.path), project_root=self.root)
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

    def label(self, start, last):
        self.dialog.behavior.setCurrentIndex(1)
        self.dialog.seek(start)
        self.dialog._mark_start()
        self.dialog.seek(last)
        self.dialog._mark_end()
        self.dialog._apply_interval()

    def test_timestamp_label_split_undo_export_and_reopen(self):
        self.label(40, 280)
        interval = self.dialog.document.intervals[0]
        self.assertEqual((interval.start_pts, interval.end_pts), (40, 320))
        self.assertIsNone(interval.start_frame)
        self.assertIsNone(interval.end_frame_exclusive)
        self.dialog._select_interval(interval.id)
        self.dialog.seek(160)
        self.dialog._split_interval()
        self.assertEqual(
            [(i.start_pts, i.end_pts) for i in self.dialog.document.intervals],
            [(40, 160), (160, 320)],
        )
        self.dialog._undo()
        self.assertEqual(len(self.dialog.document.intervals), 1)
        self.dialog._redo()
        self.assertEqual(len(self.dialog.document.intervals), 2)
        output = Path(self.root, "labels.csv")
        export_document(self.dialog.document, str(output), csv_format=True)
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["start_frame"], "")
        self.assertEqual(rows[0]["schema_version"], "2")
        self.assertEqual(rows[0]["start_time_s"], "0.040000000")
        self.dialog._video_ready(open_behavior_video(str(self.path), project_root=self.root))
        self.assertEqual(self.dialog.current, 160)
        self.assertEqual(len(self.dialog.document.intervals), 2)
        self.assertFalse(self.warning.called, self.warning.call_args_list)

    def test_last_frame_included_overlap_rejected_and_boundary_resize(self):
        self.label(440, 440)
        interval = self.dialog.document.intervals[0]
        self.assertEqual((interval.start_pts, interval.end_pts), (440, 480))
        self.dialog._select_interval(interval.id)
        self.dialog._resize_interval(interval.id, 400, 480)
        self.assertEqual(self.dialog.document.intervals[0].start_pts, 400)
        self.dialog._clear_editor(force=True)
        self.label(400, 440)
        self.assertEqual(len(self.dialog.document.intervals), 1)
        self.assertTrue(self.warning.called)
        saved = load_document(self.root, self.index.source.id)
        self.assertEqual(saved.intervals[0].end_pts, 480)

    def test_manual_seconds_snap_to_real_frames_and_playback_ends(self):
        self.dialog.start_frame.setValue(0.10)
        self.dialog.last_frame.setValue(0.15)
        self.dialog._apply_interval()
        interval = self.dialog.document.intervals[0]
        self.assertEqual((interval.start_pts, interval.end_pts), (40, 160))
        self.dialog.seek(0)
        self.dialog._step(1)
        self.assertEqual(self.dialog.current, 40)
        self.dialog._step(-1)
        self.assertEqual(self.dialog.current, 0)
        self.dialog._toggle_play()
        QTest.qWait(700)
        self.assertEqual(self.dialog.current, 440)
        self.assertFalse(self.dialog.playing)
        self.assertFalse(self.warning.called, self.warning.call_args_list)
