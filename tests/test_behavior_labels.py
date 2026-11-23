import csv
import io
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.annotation.behavior import (
    Behavior,
    BehaviorDocument,
    Interval,
    Subject,
    VideoSource,
    timestamp,
)
from squeakpose.services.behavior_labels import (
    annotation_path,
    export_document,
    load_document,
    new_document,
    reconcile_behavior_catalog,
    save_document,
)


def sample_document():
    source = VideoSource(
        "a" * 64, "mouse.mp4", "/source/mouse.mp4", 10, 1, 1000, 2000, 2600, "last_frame_duration"
    )
    return BehaviorDocument.new(source)


def sample_interval(document, **overrides):
    interval = Interval(
        "first",
        document.subjects[0].id,
        document.behaviors[1].id,
        "behavior",
        1,
        3,
        2040,
        2200,
        "observed, manually",
    )
    return replace(interval, **overrides)


class BehaviorLabelTests(unittest.TestCase):
    def test_catalog_reconciliation_handles_recreated_names_and_keeps_old_targets(self):
        doc = sample_document()
        interval = sample_interval(doc)
        doc.put(interval)
        rearing = Behavior("new-rearing-id", "Rearing", "#aabbcc", "New criteria")
        freezing = Behavior("freezing", "Freezing", "#112233")
        reconciled = reconcile_behavior_catalog(doc, [doc.behaviors[0], rearing, freezing])
        self.assertEqual(reconciled.intervals, [replace(interval, behavior_id=rearing.id)])
        self.assertEqual(reconciled.behaviors, [doc.behaviors[0], rearing, freezing])
        self.assertEqual(doc.intervals, [interval])
        self.assertEqual(len(doc.behaviors), 4)

    def test_overlap_rejected_without_mutation_adjacent_and_other_animals_allowed(self):
        doc = sample_document()
        original = sample_interval(doc)
        doc.put(original)
        with self.assertRaisesRegex(ValueError, "overlap"):
            doc.put(replace(original, id="overlap", start_frame=2, start_pts=2100))
        self.assertEqual(doc.intervals, [original])
        doc.put(
            replace(
                original,
                id="adjacent",
                start_frame=3,
                end_frame_exclusive=4,
                start_pts=2200,
                end_pts=2250,
            )
        )
        doc.subjects.append(Subject("second-animal", "Animal 2"))
        doc.put(replace(original, id="second", subject_id="second-animal"))
        self.assertEqual(len(doc.intervals), 3)

    def test_single_frame_including_last_frame_and_reversed_boundaries(self):
        doc = sample_document()
        doc.put(
            sample_interval(
                doc, start_frame=9, end_frame_exclusive=10, start_pts=2550, end_pts=2600
            )
        )
        with self.assertRaises(ValueError):
            doc.put(sample_interval(doc, start_frame=5, end_frame_exclusive=5))

    def test_other_unscorable_and_unreviewed_remain_distinct(self):
        doc = sample_document()
        doc.put(sample_interval(doc, behavior_id=doc.behaviors[0].id))
        doc.put(
            sample_interval(
                doc,
                id="unknown",
                kind="unscorable",
                behavior_id=None,
                start_frame=4,
                end_frame_exclusive=5,
                start_pts=2300,
                end_pts=2350,
            )
        )
        restored = BehaviorDocument.from_dict(doc.to_dict())
        self.assertEqual(len(restored.intervals), 2)
        self.assertEqual(restored.to_dict()["unannotated_time"], "unreviewed")
        self.assertEqual(restored.intervals[1].kind, "unscorable")

    def test_roundtrip_and_csv_preserve_vfr_timing_and_quoted_notes(self):
        with TemporaryDirectory() as root:
            doc = sample_document()
            doc.put(sample_interval(doc))
            save_document(root, doc)
            self.assertEqual(load_document(root, doc.video.id), doc)
            for csv_format in (True, False):
                output = Path(root, "export.csv" if csv_format else "export.json")
                export_document(doc, str(output), csv_format=csv_format)
                if csv_format:
                    row = list(csv.DictReader(io.StringIO(output.read_text())))[0]
                    self.assertEqual(row["start_time_s"], "0.040000000")
                    self.assertEqual(row["duration_s"], "0.160000000")
                    self.assertEqual(row["end_frame_exclusive"], "3")
                    self.assertEqual(row["notes"], "observed, manually")
                else:
                    self.assertEqual(
                        BehaviorDocument.from_dict(json.loads(output.read_text())), doc
                    )

    def test_moved_source_keeps_identity_and_original_survives_failed_save(self):
        with TemporaryDirectory() as root:
            doc = sample_document()
            save_document(root, doc)
            doc.video = replace(doc.video, source_path="/new/renamed.mp4", filename="renamed.mp4")
            save_document(root, doc)
            doc.put(sample_interval(doc))
            with patch("squeakpose.core.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_document(root, doc)
            saved = load_document(root, doc.video.id)
            self.assertEqual(saved.video.filename, "renamed.mp4")
            self.assertEqual(saved.intervals, [])

    def test_invalid_file_is_never_replaced_and_path_escape_rejected(self):
        with TemporaryDirectory() as root, TemporaryDirectory() as outside:
            doc = sample_document()
            path = Path(save_document(root, doc))
            path.write_text('{"schema_version":999}')
            with self.assertRaises(ValueError):
                load_document(root, doc.video.id)
            self.assertIn("999", path.read_text())
            with self.assertRaises(ValueError):
                annotation_path(root, "../escape")
            path.unlink()
            Path(root, "behavior labels", "catalog.json").unlink()
            Path(root, "behavior labels").rmdir()
            os.symlink(outside, Path(root, "behavior labels"), target_is_directory=True)
            with self.assertRaises(ValueError):
                save_document(root, doc)

    def test_schema_and_reference_validation(self):
        for change in (
            {"behavior_id": "missing"},
            {"subject_id": "missing"},
            {"start_frame": True},
            {"kind": "unreviewed"},
            {"end_pts": 99999},
        ):
            doc = sample_document()
            with self.subTest(change=change), self.assertRaises(ValueError):
                doc.put(sample_interval(doc, **change))
        self.assertEqual(timestamp(3661.234), "01:01:01.234")

    def test_project_vocabulary_reused_without_rewriting_existing_video_definitions(self):
        with TemporaryDirectory() as root:
            first = sample_document()
            save_document(root, first)
            second = new_document(root, replace(first.video, id="b" * 64))
            self.assertEqual(second.behaviors, first.behaviors)
            previous = second.behaviors.copy()
            second.behaviors[1] = replace(second.behaviors[1], definition="Both front paws lifted.")
            save_document(root, second, previous_behaviors=previous)
            third = new_document(root, replace(first.video, id="c" * 64))
            self.assertEqual(third.behaviors[1].definition, "Both front paws lifted.")
            self.assertEqual(load_document(root, first.video.id).behaviors[1].definition, "")
            # Undoing a definition edit restores the shared vocabulary as well.
            before_undo = second.behaviors.copy()
            second.behaviors = previous
            save_document(root, second, previous_behaviors=before_undo)
            self.assertEqual(new_document(root, first.video).behaviors, first.behaviors)

    def test_inconsistent_exported_seconds_are_rejected(self):
        doc = sample_document()
        doc.put(sample_interval(doc))
        raw = doc.to_dict()
        raw["intervals"][0]["start_time_s"] = 123
        with self.assertRaisesRegex(ValueError, "disagree"):
            BehaviorDocument.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
