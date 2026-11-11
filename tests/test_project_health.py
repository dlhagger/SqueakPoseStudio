import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import dataset_ops
from squeakpose.project import (
    ProjectHealthReport,
    cleanup_project_temporary_paths,
    format_project_health_summary,
    scan_project_health,
)
from squeakpose.project.health import _COPY_SUFFIX_RE


class ProjectHealthTests(unittest.TestCase):
    def test_copy_suffix_matches_every_counter_from_two_upwards(self):
        for number in (*range(2, 25), 99, 100, 101):
            match = _COPY_SUFFIX_RE.match(f"frame {number}")
            self.assertIsNotNone(match, f"frame {number}")
            self.assertEqual(match.group("base"), "frame")
            self.assertEqual(match.group("number"), str(number))
        for stem in ("frame", "frame 0", "frame 1", "frame 01", "frame 010", "frame1"):
            self.assertIsNone(_COPY_SUFFIX_RE.match(stem), stem)

    def test_scan_reports_copies_numbered_ten_to_nineteen(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_all = root / "images_all"
            images_all.mkdir()
            (images_all / "frame.png").write_bytes(b"image")
            for number in range(10, 20):
                (images_all / f"frame {number}.png").write_bytes(b"image")

            report = scan_project_health(
                tmp,
                pose_class_count=1,
                pose_keypoint_count=1,
                segmentation_class_count=1,
            )

            self.assertEqual(
                sorted(report.likely_duplicate_images),
                sorted(("frame.png", f"frame {number}.png") for number in range(10, 20)),
            )

    def test_dataset_ops_reexports_project_health_api(self):
        self.assertIs(dataset_ops.ProjectHealthReport, ProjectHealthReport)
        self.assertIs(dataset_ops.scan_project_health, scan_project_health)
        self.assertIs(
            dataset_ops.cleanup_project_temporary_paths,
            cleanup_project_temporary_paths,
        )
        self.assertIs(dataset_ops.format_project_health_summary, format_project_health_summary)

    def test_scan_and_cleanup_use_project_health_owner(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in (
                "images_to_label",
                "images_all",
                "labels_all",
                "labels_seg_all",
            ):
                (root / directory).mkdir()
            staged = root / "images_all" / ".frame.abcdefgh.tmp.png"
            staged.write_bytes(b"staged")
            (root / "images_all" / "frame.png").write_bytes(b"image")

            report = scan_project_health(
                tmp,
                pose_class_count=1,
                pose_keypoint_count=1,
                segmentation_class_count=1,
            )

            self.assertEqual(report.temporary_paths, [str(staged)])
            self.assertEqual(report.unlabeled_images, ["frame.png"])
            self.assertIn("Stored images: 1", format_project_health_summary(report))
            self.assertEqual(cleanup_project_temporary_paths(report), [])
            self.assertFalse(staged.exists())


if __name__ == "__main__":
    unittest.main()
