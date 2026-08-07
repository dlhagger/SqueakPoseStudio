import json
import os
import socket
import unittest
from dataclasses import asdict, replace
from tempfile import TemporaryDirectory

from squeakpose.project.paths import ensure_project_structure
from squeakpose.project.safety import (
    PROJECT_LOCK_FILENAME,
    ProjectLock,
    ProjectLockedError,
    ProjectPathError,
    break_stale_project_lock,
    canonical_path,
    inspect_project_lock,
    is_path_within_project,
    require_path_within_project,
)


class ProjectPathSafetyTests(unittest.TestCase):
    def test_containment_accepts_children_and_rejects_parent_paths(self):
        with TemporaryDirectory() as tmp:
            child = os.path.join(tmp, "labels", "frame.txt")
            outside = os.path.join(os.path.dirname(tmp), "outside.txt")

            self.assertTrue(is_path_within_project(tmp, child))
            self.assertFalse(is_path_within_project(tmp, outside))
            self.assertEqual(require_path_within_project(tmp, child), os.path.abspath(child))
            with self.assertRaises(ProjectPathError):
                require_path_within_project(tmp, outside)

    def test_containment_rejects_symlink_that_resolves_outside_project(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            link = os.path.join(tmp, "labels_all")
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable")

            self.assertFalse(is_path_within_project(tmp, os.path.join(link, "frame.txt")))
            with self.assertRaises(ProjectPathError):
                ensure_project_structure(tmp)

    def test_canonical_path_resolves_relative_segments(self):
        with TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "..", "b")
            self.assertEqual(canonical_path(nested), canonical_path(os.path.join(tmp, "b")))

    def test_project_structure_rejects_metadata_symlink_outside_project(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            outside_metadata = os.path.join(outside, "metadata.json")
            with open(outside_metadata, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 2}, handle)
            link = os.path.join(tmp, "squeakpose_project.json")
            try:
                os.symlink(outside_metadata, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable")

            with self.assertRaises(ProjectPathError):
                ensure_project_structure(tmp)


class ProjectLockTests(unittest.TestCase):
    def test_second_writer_is_rejected_until_owner_releases(self):
        with TemporaryDirectory() as tmp:
            first = ProjectLock(tmp, version="test").acquire()
            try:
                with self.assertRaises(ProjectLockedError) as raised:
                    ProjectLock(tmp).acquire()
                self.assertFalse(raised.exception.stale)
                self.assertEqual(raised.exception.info.pid, os.getpid())
            finally:
                first.release()

            second = ProjectLock(tmp).acquire()
            second.release()
            self.assertFalse(os.path.exists(os.path.join(tmp, PROJECT_LOCK_FILENAME)))

    def test_release_does_not_remove_a_lock_owned_by_another_token(self):
        with TemporaryDirectory() as tmp:
            lock = ProjectLock(tmp).acquire()
            replacement = replace(lock.info, token="replacement-token")
            with open(lock.path, "w", encoding="utf-8") as handle:
                json.dump(asdict(replacement), handle)

            with self.assertLogs("squeakpose.project.safety", level="ERROR"):
                lock.release()

            self.assertTrue(os.path.isfile(lock.path))
            os.unlink(lock.path)

    def test_proven_stale_lock_requires_explicit_break(self):
        with TemporaryDirectory() as tmp:
            lock = ProjectLock(tmp)
            stale = replace(lock.info, pid=999_999_999, hostname=socket.gethostname())
            with open(lock.path, "w", encoding="utf-8") as handle:
                json.dump(asdict(stale), handle)

            status = inspect_project_lock(tmp)
            self.assertIsNotNone(status)
            self.assertTrue(status[1])
            with self.assertRaises(ProjectLockedError) as raised:
                lock.acquire()
            self.assertTrue(raised.exception.stale)

            removed = break_stale_project_lock(tmp)
            self.assertEqual(removed.token, stale.token)
            lock.acquire()
            lock.release()

    def test_invalid_lock_is_not_automatically_treated_as_stale(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, PROJECT_LOCK_FILENAME)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not json")

            status = inspect_project_lock(tmp)
            self.assertEqual(status, (None, False))
            with self.assertRaises(ProjectLockedError) as raised:
                break_stale_project_lock(tmp)
            self.assertFalse(raised.exception.stale)


if __name__ == "__main__":
    unittest.main()
