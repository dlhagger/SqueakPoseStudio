import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.project.safety import ProjectPathError, is_path_within_project
from squeakpose.services.image_queue import (
    ImageQueueNavigator,
    filtered_queue_indices,
    next_unlabeled_index,
    plan_image_deletion,
    queue_progress,
    scan_image_queue,
)


class ImageQueueServiceTests(unittest.TestCase):
    def test_navigator_owns_filter_and_cyclic_index_transitions(self):
        images = ["first.jpg", "second.png", "third.webp"]
        usable = {os.path.join("labels", "second.txt")}
        navigator = ImageQueueNavigator(images, current_index=2)

        selection = navigator.set_filter("labeled", "labels", label_is_usable=usable.__contains__)
        self.assertEqual(selection.current_index, 1)
        self.assertEqual(selection.matching_indices, (1,))
        self.assertEqual(selection.position, 1)

        navigator.set_filter("unlabeled", "labels", label_is_usable=usable.__contains__)
        self.assertEqual(navigator.current_index, 0)
        self.assertEqual(
            navigator.move(-1, "labels", label_is_usable=usable.__contains__).current_index,
            2,
        )
        self.assertEqual(
            navigator.move(1, "labels", label_is_usable=usable.__contains__).current_index,
            0,
        )

    def test_navigator_reconciles_queue_changes_and_empty_filters(self):
        navigator = ImageQueueNavigator(["first.jpg", "second.png"], current_index=1)
        navigator.synchronize(["first.jpg"], current_index=9)
        self.assertEqual(navigator.current_index, 0)

        selection = navigator.set_filter("labeled", "labels", label_is_usable=lambda _path: False)
        self.assertFalse(selection.has_match)
        self.assertEqual(selection.position, 0)
        self.assertEqual(
            navigator.move(1, "labels", label_is_usable=lambda _path: False),
            selection,
        )

        with self.assertRaises(ValueError):
            navigator.set_filter("missing", "labels", label_is_usable=lambda _path: False)

    def test_scan_excludes_case_insensitive_stem_collisions(self):
        with TemporaryDirectory() as tmp:
            for name in ("frame.jpg", "frame.png", "Mouse.JPG", "mouse.jpeg", "unique.webp"):
                Path(tmp, name).write_bytes(b"image")
            Path(tmp, ".hidden.png").write_bytes(b"hidden")
            Path(tmp, "notes.txt").write_text("not an image", encoding="utf-8")

            scan = scan_image_queue(tmp)

            self.assertEqual(scan.images, ("unique.webp",))
            self.assertEqual(scan.collisions["frame"], ["frame.jpg", "frame.png"])
            self.assertEqual(scan.collisions["mouse"], ["mouse.jpeg", "Mouse.JPG"])

    def test_progress_and_navigation_delegate_label_usability(self):
        images = ["first.jpg", "second.png", "third.webp"]
        usable = {os.path.join("labels", "first.txt"), os.path.join("labels", "third.txt")}
        is_usable = usable.__contains__

        progress = queue_progress(images, "labels", label_is_usable=is_usable)

        self.assertEqual((progress.labeled, progress.total), (2, 3))
        self.assertEqual(
            next_unlabeled_index(images, 0, "labels", label_is_usable=is_usable),
            1,
        )
        self.assertEqual(
            next_unlabeled_index(images, 1, "labels", label_is_usable=is_usable),
            1,
        )
        self.assertEqual(next_unlabeled_index([], 4, "labels", label_is_usable=is_usable), 0)

    def test_filtered_indices_preserve_queue_positions(self):
        images = ["first.jpg", "second.png", "third.webp"]
        usable = {os.path.join("labels", "second.txt")}
        is_usable = usable.__contains__

        self.assertEqual(
            filtered_queue_indices(images, "all", "labels", label_is_usable=is_usable),
            [0, 1, 2],
        )
        self.assertEqual(
            filtered_queue_indices(images, "labeled", "labels", label_is_usable=is_usable),
            [1],
        )
        self.assertEqual(
            filtered_queue_indices(images, "unlabeled", "labels", label_is_usable=is_usable),
            [0, 2],
        )

    def test_deletion_plan_is_deduplicated_contained_and_detects_conflicts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "images_to_label"
            images_all = root / "images_all"
            queue.mkdir()
            images_all.mkdir()
            (queue / "frame001.jpg").write_bytes(b"queue")
            (images_all / "FRAME001.png").write_bytes(b"conflict")

            plan = plan_image_deletion(
                project_root=str(root),
                image_name="../frame001.jpg",
                active_image_dir=str(queue),
                image_dir_queue=str(queue),
                image_dir_all=str(images_all),
                pose_label_dir=str(root / "labels_all"),
                seg_label_dir=str(root / "labels_seg_all"),
                depth_image_dir=str(root / "depth maps" / "images"),
                depth_preview_dir=str(root / "depth maps" / "previews"),
            )

            self.assertEqual(plan.image_name, "frame001.jpg")
            self.assertEqual(plan.conflicting_names, ("FRAME001.png",))
            self.assertFalse(plan.safe)
            self.assertEqual(len(plan.paths), len(set(plan.paths)))
            self.assertTrue(
                all(
                    is_path_within_project(str(root), path, allow_root=False) for path in plan.paths
                )
            )
            self.assertIn(str(queue / "frame001.jpg"), plan.paths)
            self.assertIn(str(root / "labels_all" / "frame001.txt"), plan.paths)
            self.assertIn(
                str(root / "datasets" / "segment" / "labels" / "val" / "frame001.txt"),
                plan.paths,
            )

    def test_deletion_plan_rejects_a_target_outside_the_project(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            root = Path(tmp)

            with self.assertRaises(ProjectPathError):
                plan_image_deletion(
                    project_root=str(root),
                    image_name="frame001.jpg",
                    active_image_dir=outside,
                    image_dir_queue=str(root / "images_to_label"),
                    image_dir_all=str(root / "images_all"),
                    pose_label_dir=str(root / "labels_all"),
                    seg_label_dir=str(root / "labels_seg_all"),
                )


if __name__ == "__main__":
    unittest.main()
