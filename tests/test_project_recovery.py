import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.project.recovery import (
    cleanup_transaction_staging,
    restore_missing_transaction_targets,
    scan_transaction_artifacts,
)
from squeakpose_core import commit_staged_paths


class ProjectTransactionRecoveryTests(unittest.TestCase):
    def test_sole_backup_restores_missing_file_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / f"classes.txt.backup-{'a' * 32}"
            target = root / "classes.txt"
            backup.write_text("mouse\n", encoding="utf-8")

            report = scan_transaction_artifacts(tmp)
            self.assertEqual(
                [item.backup_path for item in report.restorable_backups], [str(backup)]
            )

            result = restore_missing_transaction_targets(tmp)

            self.assertEqual(result.restored_paths, [str(target)])
            self.assertEqual(target.read_text(encoding="utf-8"), "mouse\n")
            self.assertFalse(backup.exists())

    def test_sole_backup_restores_missing_directory_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / f"images.backup-{'b' * 32}"
            target = root / "images"
            backup.mkdir()
            (backup / "frame.png").write_bytes(b"image")

            result = restore_missing_transaction_targets(tmp)

            self.assertEqual(result.restored_paths, [str(target)])
            self.assertEqual((target / "frame.png").read_bytes(), b"image")

    def test_existing_target_or_multiple_backups_are_preserved(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "classes.txt"
            existing.write_text("current\n", encoding="utf-8")
            existing_backup = root / f"classes.txt.backup-{'c' * 32}"
            existing_backup.write_text("old\n", encoding="utf-8")
            first = root / f"keypoints.txt.backup-{'d' * 32}"
            second = root / f"keypoints.txt.backup-{'e' * 32}"
            first.write_text("first\n", encoding="utf-8")
            second.write_text("second\n", encoding="utf-8")

            report = scan_transaction_artifacts(tmp)
            preserved = {item.backup_path for item in report.preserved_backups}

            self.assertEqual(preserved, {str(existing_backup), str(first), str(second)})
            self.assertEqual(report.restorable_backups, [])
            result = restore_missing_transaction_targets(tmp)
            self.assertEqual(result.restored_paths, [])
            self.assertEqual(existing.read_text(encoding="utf-8"), "current\n")
            self.assertTrue(existing_backup.exists())
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_cleanup_removes_only_exact_generated_staging_names(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged_file = root / ".classes.abcdefgh.tmp.txt"
            staged_dir = root / ".pose-export-abcdefgh"
            staged_file.write_text("staged", encoding="utf-8")
            staged_dir.mkdir()
            (staged_dir / "data.txt").write_text("staged", encoding="utf-8")
            lookalikes = [
                root / ".classes.short.tmp.txt",
                root / ".keep.tmp",
                root / ".pose-export-user-files",
            ]
            lookalikes[0].write_text("keep", encoding="utf-8")
            lookalikes[1].write_text("keep", encoding="utf-8")
            lookalikes[2].mkdir()

            report = scan_transaction_artifacts(tmp)
            self.assertEqual(set(report.staging_paths), {str(staged_file), str(staged_dir)})

            result = cleanup_transaction_staging(tmp)

            self.assertEqual(set(result.removed_staging_paths), {str(staged_file), str(staged_dir)})
            self.assertFalse(staged_file.exists())
            self.assertFalse(staged_dir.exists())
            self.assertTrue(all(path.exists() for path in lookalikes))

    def test_scan_does_not_follow_symlinked_directories(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            outside_stage = Path(outside) / ".classes.abcdefgh.tmp.txt"
            outside_stage.write_text("outside", encoding="utf-8")
            link = Path(tmp) / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            report = scan_transaction_artifacts(tmp)

            self.assertEqual(report.staging_paths, [])
            cleanup_transaction_staging(tmp)
            self.assertTrue(outside_stage.exists())

    def test_restore_and_cleanup_failures_are_reported_without_deleting_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / f"classes.txt.backup-{'f' * 32}"
            staging = root / ".classes.abcdefgh.tmp.txt"
            backup.write_text("old", encoding="utf-8")
            staging.write_text("new", encoding="utf-8")

            with patch("squeakpose.project.recovery.os.replace", side_effect=OSError("denied")):
                restored = restore_missing_transaction_targets(tmp)
            with patch("squeakpose.project.recovery.remove_path", side_effect=OSError("busy")):
                cleaned = cleanup_transaction_staging(tmp)

            self.assertEqual(len(restored.errors), 1)
            self.assertEqual(len(cleaned.errors), 1)
            self.assertTrue(backup.exists())
            self.assertTrue(staging.exists())

    def test_failed_rollback_backup_is_recovered_on_next_start(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "classes.txt"
            stage = root / ".classes.abcdefgh.tmp.txt"
            target.write_text("old\n", encoding="utf-8")
            stage.write_text("new\n", encoding="utf-8")
            real_replace = os.replace

            def fail_install_and_rollback(src, dst):
                if Path(src) == stage and Path(dst) == target:
                    raise OSError("injected install failure")
                if ".backup-" in str(src) and Path(dst) == target:
                    raise OSError("injected rollback failure")
                return real_replace(src, dst)

            with patch("squeakpose_core.os.replace", side_effect=fail_install_and_rollback):
                with self.assertRaises(RuntimeError):
                    commit_staged_paths([(str(stage), str(target))])

            self.assertFalse(target.exists())
            report = scan_transaction_artifacts(tmp)
            self.assertEqual(len(report.restorable_backups), 1)

            result = restore_missing_transaction_targets(tmp)

            self.assertEqual(result.restored_paths, [str(target)])
            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")


if __name__ == "__main__":
    unittest.main()
