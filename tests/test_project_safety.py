import json
import os
import socket
import unittest
from dataclasses import asdict, replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.project import safety
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
            except OSError, NotImplementedError:
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
            except OSError, NotImplementedError:
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

    def test_breaking_stale_lock_leaves_no_claim_files_behind(self):
        with TemporaryDirectory() as tmp:
            lock = ProjectLock(tmp)
            stale = replace(lock.info, pid=999_999_999, hostname=socket.gethostname())
            with open(lock.path, "w", encoding="utf-8") as handle:
                json.dump(asdict(stale), handle)

            removed = break_stale_project_lock(tmp)

            self.assertEqual(removed, stale)
            self.assertEqual(os.listdir(tmp), [])

    def test_breaking_stale_lock_detects_replacement_before_claim(self):
        with TemporaryDirectory() as tmp:
            lock = ProjectLock(tmp)
            stale = replace(lock.info, pid=999_999_999, hostname=socket.gethostname())
            with open(lock.path, "w", encoding="utf-8") as handle:
                json.dump(asdict(stale), handle)
            live = ProjectLock(tmp)
            real_rename = os.rename

            def replace_then_rename(src, dst, *args, **kwargs):
                os.unlink(lock.path)
                live.acquire()
                return real_rename(src, dst, *args, **kwargs)

            with patch("squeakpose.project.safety.os.rename", side_effect=replace_then_rename):
                with self.assertRaises(ProjectLockedError) as raised:
                    break_stale_project_lock(tmp)

            self.assertFalse(raised.exception.stale)
            self.assertEqual(raised.exception.info, live.info)
            self.assertEqual(os.listdir(tmp), [PROJECT_LOCK_FILENAME])
            status = inspect_project_lock(tmp)
            self.assertEqual(status, (live.info, False))
            live.release()
            self.assertEqual(os.listdir(tmp), [])

    def _break_while_lock_changes_hands(self, tmp, *, link_error, rename_fails_on_restore):
        lock = ProjectLock(tmp)
        stale = replace(lock.info, pid=999_999_999, hostname=socket.gethostname())
        with open(lock.path, "w", encoding="utf-8") as handle:
            json.dump(asdict(stale), handle)
        live = ProjectLock(tmp)
        real_rename = os.rename
        calls = []

        def rename(src, dst, *args, **kwargs):
            calls.append((src, dst))
            if len(calls) == 1:
                os.unlink(lock.path)
                live.acquire()
            elif rename_fails_on_restore:
                raise PermissionError(1, "Operation not permitted")
            return real_rename(src, dst, *args, **kwargs)

        with (
            patch("squeakpose.project.safety.os.rename", side_effect=rename),
            patch("squeakpose.project.safety.os.link", side_effect=link_error),
        ):
            with self.assertRaises(ProjectLockedError) as raised:
                break_stale_project_lock(tmp)
        return live, raised.exception

    def test_breaking_stale_lock_restores_live_lock_without_hard_links(self):
        for error in (
            PermissionError(1, "Operation not permitted"),
            OSError(45, "Operation not supported"),
        ):
            with self.subTest(error=error), TemporaryDirectory() as tmp:
                live, exc = self._break_while_lock_changes_hands(
                    tmp, link_error=error, rename_fails_on_restore=False
                )

                self.assertIsNone(exc.claimed_path)
                self.assertEqual(os.listdir(tmp), [PROJECT_LOCK_FILENAME])
                self.assertEqual(inspect_project_lock(tmp), (live.info, False))
                live.release()
                self.assertEqual(os.listdir(tmp), [])

    def test_breaking_stale_lock_never_deletes_live_lock_it_cannot_restore(self):
        with TemporaryDirectory() as tmp:
            live, exc = self._break_while_lock_changes_hands(
                tmp,
                link_error=PermissionError(1, "Operation not permitted"),
                rename_fails_on_restore=True,
            )

            self.assertIsNotNone(exc.claimed_path)
            self.assertIn(exc.claimed_path, str(exc))
            self.assertEqual(exc.info, live.info)
            self.assertTrue(os.path.isfile(exc.claimed_path))
            names = os.listdir(tmp)
            self.assertEqual(names, [os.path.basename(exc.claimed_path)])
            with open(exc.claimed_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["token"], live.info.token)
            live.acquired = False

    def test_breaking_stale_lock_does_not_clobber_lock_created_during_restore(self):
        with TemporaryDirectory() as tmp:
            lock = ProjectLock(tmp)
            stale = replace(lock.info, pid=999_999_999, hostname=socket.gethostname())
            with open(lock.path, "w", encoding="utf-8") as handle:
                json.dump(asdict(stale), handle)
            live = ProjectLock(tmp)
            newcomer = ProjectLock(tmp)
            real_rename = os.rename
            calls = []

            def rename(src, dst, *args, **kwargs):
                calls.append((src, dst))
                if len(calls) == 1:
                    os.unlink(lock.path)
                    live.acquire()
                return real_rename(src, dst, *args, **kwargs)

            def link(src, dst):
                newcomer.acquire()
                raise PermissionError(1, "Operation not permitted")

            with (
                patch("squeakpose.project.safety.os.rename", side_effect=rename),
                patch("squeakpose.project.safety.os.link", side_effect=link),
            ):
                with self.assertRaises(ProjectLockedError):
                    break_stale_project_lock(tmp)

            self.assertEqual(len(calls), 1)
            self.assertEqual(inspect_project_lock(tmp), (newcomer.info, False))
            newcomer.release()
            live.acquired = False

    def test_breaking_stale_lock_reports_lock_that_disappeared(self):
        with TemporaryDirectory() as tmp:
            lock = ProjectLock(tmp)
            stale = replace(lock.info, pid=999_999_999, hostname=socket.gethostname())
            with open(lock.path, "w", encoding="utf-8") as handle:
                json.dump(asdict(stale), handle)

            def vanish_then_rename(src, dst, *args, **kwargs):
                os.unlink(lock.path)
                raise FileNotFoundError(src)

            with patch("squeakpose.project.safety.os.rename", side_effect=vanish_then_rename):
                with self.assertRaises(ProjectLockedError) as raised:
                    break_stale_project_lock(tmp)

            self.assertIsNone(raised.exception.info)
            self.assertFalse(raised.exception.stale)
            self.assertEqual(os.listdir(tmp), [])

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


class _FakeExitCode:
    def __init__(self):
        self.value = 0


class _FakeKernel32:
    """Stand-in for kernel32 exposing only what _windows_pid_is_running uses."""

    exit_code_type = _FakeExitCode

    def __init__(self, *, handle=1, last_error=0, exit_code=259, query_ok=True):
        self.handle = handle
        self.last_error = last_error
        self.exit_code = exit_code
        self.query_ok = query_ok
        self.closed = []
        self.opened = []

    def OpenProcess(self, access, inherit, pid):
        self.opened.append((access, inherit, pid))
        return self.handle

    def GetExitCodeProcess(self, handle, exit_code):
        exit_code.value = self.exit_code
        return self.query_ok

    def CloseHandle(self, handle):
        self.closed.append(handle)

    def get_last_error(self):
        return self.last_error

    @staticmethod
    def byref(value):
        return value


class WindowsProcessProbeTests(unittest.TestCase):
    def test_running_process_is_detected_and_handle_closed(self):
        kernel32 = _FakeKernel32(exit_code=259)
        self.assertTrue(safety._windows_pid_is_running(4242, kernel32))
        self.assertEqual(kernel32.opened, [(0x1000, False, 4242)])
        self.assertEqual(kernel32.closed, [1])

    def test_exited_process_is_not_running(self):
        kernel32 = _FakeKernel32(exit_code=0)
        self.assertFalse(safety._windows_pid_is_running(4242, kernel32))
        self.assertEqual(kernel32.closed, [1])

    def test_access_denied_means_the_process_exists(self):
        kernel32 = _FakeKernel32(handle=0, last_error=5)
        self.assertTrue(safety._windows_pid_is_running(4242, kernel32))
        self.assertEqual(kernel32.closed, [])

    def test_missing_process_is_not_running(self):
        kernel32 = _FakeKernel32(handle=0, last_error=87)
        self.assertFalse(safety._windows_pid_is_running(4242, kernel32))

    def test_windows_never_uses_os_kill_signal_zero(self):
        with (
            patch.object(safety.os, "name", "nt"),
            patch.object(safety, "_windows_pid_is_running", return_value=True) as probe,
            patch.object(safety.os, "kill") as kill,
        ):
            self.assertTrue(safety._pid_is_running(4242))
        probe.assert_called_once_with(4242)
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
