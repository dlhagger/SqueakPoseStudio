import errno
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.services import video_library
from squeakpose.services.video_library import (
    add_video_links,
    list_project_videos,
    remove_video_link,
    rename_video_link,
    resolve_project_video_paths,
    retarget_video_link,
)


def _symlinks_supported() -> bool:
    with TemporaryDirectory() as tmp:
        try:
            os.symlink(os.path.join(tmp, "target"), os.path.join(tmp, "link"))
        except OSError, NotImplementedError:
            return False
    return True


def _refuse_symlinks(*_args, **_kwargs):
    # What Windows reports without administrator rights or Developer Mode.
    raise PermissionError(errno.EPERM, "A required privilege is not held by the client")


class VideoLibraryServiceTests(unittest.TestCase):
    @unittest.skipUnless(_symlinks_supported(), "symlinks need admin or Developer Mode")
    def test_add_list_rename_retarget_and_remove_link(self):
        with TemporaryDirectory() as project, TemporaryDirectory() as sources:
            videos_dir = Path(project) / "videos"
            first = Path(sources) / "session.mp4"
            second = Path(sources) / "replacement.mov"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            created = add_video_links(str(videos_dir), [str(first)])
            self.assertEqual([entry.name for entry in created], ["session.mp4"])
            link = videos_dir / "session.mp4"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), first.resolve())

            entries = list_project_videos(str(videos_dir))
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0].is_link)
            self.assertTrue(entries[0].target_exists)

            renamed = rename_video_link(str(videos_dir), "session.mp4", "experiment")
            self.assertEqual(Path(renamed).name, "experiment.mp4")
            self.assertEqual(Path(renamed).resolve(), first.resolve())

            retarget_video_link(str(videos_dir), "experiment.mp4", str(second))
            self.assertEqual(Path(renamed).resolve(), second.resolve())

            remove_video_link(str(videos_dir), "experiment.mp4")
            self.assertFalse(os.path.lexists(renamed))
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_pointer_file_fallback_supports_the_full_link_lifecycle(self):
        with (
            TemporaryDirectory() as project,
            TemporaryDirectory() as sources,
            patch.object(video_library.os, "symlink", side_effect=_refuse_symlinks),
        ):
            videos_dir = Path(project) / "videos"
            first = Path(sources) / "session.mp4"
            second = Path(sources) / "replacement.mov"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            created = add_video_links(str(videos_dir), [str(first), str(first)])
            self.assertEqual([entry.name for entry in created], ["session.mp4"])
            pointer = videos_dir / "session.mp4"
            self.assertFalse(pointer.is_symlink())
            raw = pointer.read_bytes()
            self.assertTrue(raw.startswith(b"XSym\n"))
            self.assertEqual(len(raw), video_library.XSYM_FILE_SIZE)

            entries = list_project_videos(str(videos_dir))
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0].is_link)
            self.assertTrue(entries[0].target_exists)
            self.assertEqual(os.path.realpath(entries[0].path), os.path.realpath(first))
            self.assertEqual(
                resolve_project_video_paths(str(videos_dir), [str(pointer)]),
                [entries[0].path],
            )

            renamed = rename_video_link(str(videos_dir), "session.mp4", "experiment")
            self.assertEqual(Path(renamed).name, "experiment.mp4")
            retarget_video_link(str(videos_dir), "experiment.mp4", str(second))
            entry = list_project_videos(str(videos_dir))[0]
            self.assertEqual(os.path.realpath(entry.target), os.path.realpath(second))
            self.assertEqual(sorted(os.listdir(videos_dir)), ["experiment.mp4"])

            remove_video_link(str(videos_dir), "experiment.mp4")
            self.assertFalse(os.path.lexists(renamed))
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_pointer_to_missing_video_is_a_broken_link_not_a_playable_file(self):
        with TemporaryDirectory() as project:
            videos_dir = Path(project) / "videos"
            videos_dir.mkdir()
            with patch.object(video_library.os, "symlink", side_effect=_refuse_symlinks):
                video_library._create_link(
                    str(Path(project) / "gone.mp4"), str(videos_dir / "gone.mp4")
                )

            entry = list_project_videos(str(videos_dir))[0]
            self.assertTrue(entry.is_link)
            self.assertFalse(entry.target_exists)
            remove_video_link(str(videos_dir), "gone.mp4")
            self.assertEqual(os.listdir(videos_dir), [])

    def test_resolve_video_path_follows_pointer_files_and_stops_on_loops(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp, "session.mp4")
            source.write_bytes(b"video")
            pointer = Path(tmp, "pointer.mp4")
            video_library._write_pointer_file(str(pointer), str(source))
            first = Path(tmp, "first.mp4")
            second = Path(tmp, "second.mp4")
            video_library._write_pointer_file(str(first), str(second))
            video_library._write_pointer_file(str(second), str(first))

            resolve = video_library.resolve_video_path
            self.assertEqual(resolve(str(pointer)), os.path.realpath(source))
            self.assertEqual(resolve(str(source)), os.path.realpath(source))
            self.assertIn(
                resolve(str(first)),
                {os.path.realpath(first), os.path.realpath(second)},
            )

    def test_other_link_errors_are_not_hidden_by_the_fallback(self):
        with TemporaryDirectory() as project, TemporaryDirectory() as sources:
            source = Path(sources) / "session.mp4"
            source.write_bytes(b"video")
            failure = OSError(errno.EIO, "disk error")
            with (
                patch.object(video_library.os, "symlink", side_effect=failure),
                self.assertRaises(OSError),
            ):
                add_video_links(str(Path(project) / "videos"), [str(source)])

    def test_project_video_resolution_preserves_non_library_paths(self):
        with TemporaryDirectory() as project, TemporaryDirectory() as sources:
            videos_dir = Path(project) / "videos"
            videos_dir.mkdir()
            source = Path(sources) / "external.mp4"
            source.write_bytes(b"video")

            self.assertEqual(
                resolve_project_video_paths(str(videos_dir), [str(source)]),
                [str(source)],
            )

    def test_duplicate_names_get_numbered_and_same_source_is_not_linked_twice(self):
        with (
            TemporaryDirectory() as project,
            TemporaryDirectory() as first_dir,
            TemporaryDirectory() as second_dir,
        ):
            videos_dir = Path(project) / "videos"
            first = Path(first_dir) / "session.mp4"
            second = Path(second_dir) / "session.mp4"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            created = add_video_links(str(videos_dir), [str(first), str(second), str(first)])
            self.assertEqual([entry.name for entry in created], ["session.mp4", "session 2.mp4"])

    def test_broken_link_is_visible_and_regular_file_is_never_removed(self):
        with TemporaryDirectory() as project:
            videos_dir = Path(project) / "videos"
            videos_dir.mkdir()
            regular = videos_dir / "local.mp4"
            regular.write_bytes(b"video")
            broken = videos_dir / "missing.mp4"
            video_library._create_link(str(Path(project) / "does-not-exist.mp4"), str(broken))

            entries = {entry.name: entry for entry in list_project_videos(str(videos_dir))}
            self.assertFalse(entries["local.mp4"].is_link)
            self.assertFalse(entries["missing.mp4"].target_exists)
            with self.assertRaises(ValueError):
                remove_video_link(str(videos_dir), regular.name)
            self.assertTrue(regular.exists())

    def test_copied_macos_alias_relocates_to_neighboring_source_video(self):
        with TemporaryDirectory() as imported:
            import_root = Path(imported)
            project = import_root / "Studio Project"
            videos_dir = project / "videos"
            videos_dir.mkdir(parents=True)
            source = import_root / "C12345" / "C12345_trimmed.mp4"
            source.parent.mkdir()
            source.write_bytes(b"real video")
            alias = videos_dir / source.name
            alias.write_bytes(
                b"XSym\n0070\ndigest\n"
                b"/Users/old/GPe_Terminal_Reanalysis/C12345/C12345_trimmed.mp4\n"
            )

            entries = list_project_videos(str(videos_dir))

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].path, str(source))
            self.assertEqual(entries[0].target, str(source))
            self.assertTrue(entries[0].is_link)
            self.assertTrue(entries[0].target_exists)
            self.assertEqual(
                resolve_project_video_paths(str(videos_dir), [str(alias)]),
                [str(source)],
            )

    def test_rename_rejects_paths_and_collisions(self):
        with TemporaryDirectory() as project, TemporaryDirectory() as sources:
            videos_dir = Path(project) / "videos"
            source = Path(sources) / "session.mp4"
            source.write_bytes(b"video")
            add_video_links(str(videos_dir), [str(source)])
            (videos_dir / "taken.mp4").write_bytes(b"other")

            with self.assertRaises(ValueError):
                rename_video_link(str(videos_dir), "session.mp4", "../escape.mp4")
            with self.assertRaises(FileExistsError):
                rename_video_link(str(videos_dir), "session.mp4", "taken.mp4")


if __name__ == "__main__":
    unittest.main()
