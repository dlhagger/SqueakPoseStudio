import unittest

from squeakpose.services.dataset_ops import (
    _split_group_and_frame,
    plan_grouped_train_val_split,
)


def _frames(video: str, indices, *, ext: str = ".png") -> list[str]:
    return [f"{video}_f{index:06d}{ext}" for index in indices]


def _group_of(name: str) -> str:
    return _split_group_and_frame(name)[0]


class GroupKeyTests(unittest.TestCase):
    def test_video_export_names_share_a_group(self):
        self.assertEqual(
            _split_group_and_frame("cage1_ab12cd34_f000120.png"), ("cage1_ab12cd34", 120)
        )
        self.assertEqual(
            _split_group_and_frame("cage1_ab12cd34_f000120_2.png"), ("cage1_ab12cd34", 120)
        )
        self.assertEqual(
            _split_group_and_frame("my_video_f00_ab12_f000000007.jpg"), ("my_video_f00_ab12", 7)
        )

    def test_other_names_are_their_own_group(self):
        self.assertEqual(_split_group_and_frame("IMG_0001.jpg"), ("IMG_0001", None))
        self.assertEqual(_split_group_and_frame("clip_f12.png"), ("clip_f12", None))


class GroupedSplitTests(unittest.TestCase):
    def _many_videos(self, count: int = 30, frames: int = 20) -> list[str]:
        images: list[str] = []
        for video in range(count):
            images.extend(_frames(f"video{video}_src{video:04d}", range(0, frames * 5, 5)))
        return images

    def test_frames_of_one_video_stay_on_one_side(self):
        images = self._many_videos()

        train, val = plan_grouped_train_val_split(images, 0.8, seed=42)

        self.assertTrue(train and val)
        self.assertEqual(sorted(train + val), sorted(images))
        self.assertFalse({_group_of(n) for n in train} & {_group_of(n) for n in val})
        self.assertEqual(train, sorted(train))
        self.assertEqual(val, sorted(val))

    def test_split_is_deterministic_and_input_order_independent(self):
        images = self._many_videos()

        first = plan_grouped_train_val_split(images, 0.8, seed=7)
        second = plan_grouped_train_val_split(list(reversed(images)), 0.8, seed=7)

        self.assertEqual(first, second)

    def test_seed_changes_assignment(self):
        images = self._many_videos()

        self.assertNotEqual(
            plan_grouped_train_val_split(images, 0.8, seed=1),
            plan_grouped_train_val_split(images, 0.8, seed=2),
        )

    def test_adding_images_does_not_move_existing_assignments(self):
        images = self._many_videos()
        train, val = plan_grouped_train_val_split(images, 0.8, seed=42)

        extra = self._many_videos(count=5, frames=3)
        extra = [name.replace("video", "newvideo") for name in extra]
        extra += ["standalone_photo.jpg", "IMG_9999.png"]
        # New frames from an already-known video must follow that video.
        known_video = _group_of(val[0])
        extra += [f"{known_video}_f999999.png"]
        grown_train, grown_val = plan_grouped_train_val_split(images + extra, 0.8, seed=42)

        self.assertTrue(set(train) <= set(grown_train))
        self.assertTrue(set(val) <= set(grown_val))
        self.assertIn(f"{known_video}_f999999.png", grown_val)

    def test_val_fraction_roughly_tracks_ratio(self):
        images = [f"photo_{index:04d}.jpg" for index in range(2000)]

        _, val = plan_grouped_train_val_split(images, 0.8, seed=0)

        self.assertGreater(len(val), 300)
        self.assertLess(len(val), 500)

    def test_single_video_falls_back_to_contiguous_blocks(self):
        images = _frames("only_abcd", range(100))

        train, val = plan_grouped_train_val_split(images, 0.8, seed=3)

        self.assertTrue(train and val)
        self.assertEqual(sorted(train + val), sorted(images))
        # Each side is made of contiguous 10-frame blocks.
        val_indices = sorted(_split_group_and_frame(name)[1] for name in val)
        val_blocks = {index // 10 for index in val_indices}
        self.assertEqual(len(val_indices), 10 * len(val_blocks))
        train_blocks = {_split_group_and_frame(name)[1] // 10 for name in train}
        self.assertFalse(val_blocks & train_blocks)

    def test_single_video_fallback_is_stable_for_same_input(self):
        images = _frames("only_abcd", range(0, 500, 7))

        self.assertEqual(
            plan_grouped_train_val_split(images, 0.7, seed=11),
            plan_grouped_train_val_split(list(reversed(images)), 0.7, seed=11),
        )

    def test_tiny_inputs(self):
        self.assertEqual(plan_grouped_train_val_split([], 0.8, seed=0), ([], []))
        self.assertEqual(plan_grouped_train_val_split(["a.png"], 0.8, seed=0), (["a.png"], []))
        for pair in (["a.png", "b.png"], _frames("vid_x", [1, 2])):
            for seed in range(20):
                train, val = plan_grouped_train_val_split(pair, 0.8, seed=seed)
                self.assertEqual(len(train), 1)
                self.assertEqual(len(val), 1)
                self.assertEqual(sorted(train + val), sorted(pair))

    def test_duplicates_are_collapsed(self):
        train, val = plan_grouped_train_val_split(["a.png", "a.png", "b.png"], 0.5, seed=0)

        self.assertEqual(sorted(train + val), ["a.png", "b.png"])

    def test_ratio_extremes_still_produce_both_sides(self):
        images = self._many_videos(count=6, frames=4)
        for ratio in (0.0, 0.01, 0.99, 1.0, 1.5, -0.5):
            with self.subTest(ratio=ratio):
                train, val = plan_grouped_train_val_split(images, ratio, seed=5)
                self.assertTrue(train)
                self.assertTrue(val)
                self.assertFalse({_group_of(n) for n in train} & {_group_of(n) for n in val})
        train, _ = plan_grouped_train_val_split(images, 1.0, seed=5)
        self.assertEqual(len({_group_of(n) for n in train}), 5)
        _, val = plan_grouped_train_val_split(images, 0.0, seed=5)
        self.assertEqual(len({_group_of(n) for n in val}), 5)

    def test_single_video_ratio_extremes(self):
        images = _frames("solo_1234", range(30))
        for ratio in (0.0, 1.0):
            with self.subTest(ratio=ratio):
                train, val = plan_grouped_train_val_split(images, ratio, seed=9)
                self.assertTrue(train and val)
                self.assertEqual(len(train) + len(val), 30)


if __name__ == "__main__":
    unittest.main()
