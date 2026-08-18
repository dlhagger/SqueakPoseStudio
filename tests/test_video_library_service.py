import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.services.video_library import (
    add_video_links,
    list_project_videos,
    remove_video_link,
    rename_video_link,
    retarget_video_link,
)


class VideoLibraryServiceTests(unittest.TestCase):
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

    def test_duplicate_names_get_numbered_and_same_source_is_not_linked_twice(self):
        with TemporaryDirectory() as project, TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
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
            broken.symlink_to(Path(project) / "does-not-exist.mp4")

            entries = {entry.name: entry for entry in list_project_videos(str(videos_dir))}
            self.assertFalse(entries["local.mp4"].is_link)
            self.assertFalse(entries["missing.mp4"].target_exists)
            with self.assertRaises(ValueError):
                remove_video_link(str(videos_dir), regular.name)
            self.assertTrue(regular.exists())

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
