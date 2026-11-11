import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

import dataset_builder
import dataset_ops
from squeakpose.services.dataset_yaml import (
    create_dataset_yaml,
    flip_indices_for_keypoints,
    resolve_dataset_root,
    stale_dataset_root,
    unpaired_sided_keypoints,
)


def _make_split_dirs(base: Path) -> None:
    for rel in (
        ("images", "train"),
        ("images", "val"),
        ("labels", "train"),
        ("labels", "val"),
    ):
        base.joinpath(*rel).mkdir(parents=True, exist_ok=True)


class DatasetBuilderTests(unittest.TestCase):
    def test_legacy_and_dataset_ops_aliases_preserve_package_function_identity(self):
        self.assertIs(dataset_builder.create_dataset_yaml, create_dataset_yaml)
        self.assertIs(dataset_ops.create_dataset_yaml, create_dataset_yaml)
        self.assertEqual(create_dataset_yaml.__module__, "squeakpose.services.dataset_yaml")
        self.assertFalse(hasattr(dataset_builder, "_default_flip_indices"))

    def test_create_dataset_yaml_writes_expected_pose_metadata(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for rel in (
                ("images", "train"),
                ("images", "val"),
                ("labels", "train"),
                ("labels", "val"),
            ):
                (base.joinpath(*rel)).mkdir(parents=True, exist_ok=True)

            out_path = create_dataset_yaml(
                base_dir=str(base),
                class_names=["mouse"],
                kp_names=["nose", "left_ear", "right_ear"],
                verbose=False,
            )

            self.assertEqual(out_path, str(base / "dataset.yaml"))
            with open(out_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)

            self.assertEqual(data["names"], ["mouse"])
            self.assertNotIn("path", data)
            self.assertEqual(resolve_dataset_root(out_path), str(base))
            self.assertIsNone(stale_dataset_root(out_path))
            self.assertEqual(data["kpt_shape"], [3, 3])
            self.assertEqual(data["kp_names"], ["nose", "left_ear", "right_ear"])
            self.assertEqual(data["flip_idx"], [0, 2, 1])

    def test_create_dataset_yaml_requires_all_split_directories(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "images" / "train").mkdir(parents=True, exist_ok=True)
            (base / "images" / "val").mkdir(parents=True, exist_ok=True)
            # Intentionally omit labels/train and labels/val.

            with self.assertRaises(FileNotFoundError):
                create_dataset_yaml(
                    base_dir=str(base),
                    class_names=["mouse"],
                    kp_names=["nose"],
                    verbose=False,
                )

    def test_create_dataset_yaml_can_reference_final_path_from_staging_directory(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for rel in (
                ("images", "train"),
                ("images", "val"),
                ("labels", "train"),
                ("labels", "val"),
            ):
                (base.joinpath(*rel)).mkdir(parents=True, exist_ok=True)

            out_path = create_dataset_yaml(
                base_dir=str(base),
                class_names=["mouse"],
                kp_names=["nose"],
                verbose=False,
                dataset_path="/final/project/datasets/pose",
            )

            data = yaml.safe_load(Path(out_path).read_text(encoding="utf-8"))
            self.assertEqual(data["path"], "/final/project/datasets/pose")

    def test_package_writer_retains_atomic_write_patch_seam(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for rel in (
                ("images", "train"),
                ("images", "val"),
                ("labels", "train"),
                ("labels", "val"),
            ):
                (base.joinpath(*rel)).mkdir(parents=True, exist_ok=True)

            with patch("squeakpose.services.dataset_yaml.atomic_write_text") as atomic_write:
                output = create_dataset_yaml(
                    str(base),
                    ["mouse"],
                    ["left_ear", "right_ear"],
                    verbose=False,
                )

            self.assertEqual(output, str(base / "dataset.yaml"))
            atomic_write.assert_called_once()
            path, payload = atomic_write.call_args.args
            self.assertEqual(path, output)
            self.assertEqual(yaml.safe_load(payload)["flip_idx"], [1, 0])

    def test_moved_project_yaml_resolves_to_its_new_location(self):
        with TemporaryDirectory() as tmp:
            original = Path(tmp, "original", "datasets", "pose")
            _make_split_dirs(original)
            create_dataset_yaml(str(original), ["mouse"], ["nose"], verbose=False)
            moved = Path(tmp, "copy", "datasets", "pose")
            moved.parent.mkdir(parents=True)
            original.rename(moved)

            moved_yaml = str(moved / "dataset.yaml")
            self.assertEqual(resolve_dataset_root(moved_yaml), str(moved))
            self.assertIsNone(stale_dataset_root(moved_yaml))

    def test_legacy_absolute_path_is_still_honoured_and_reported_when_stale(self):
        with TemporaryDirectory() as tmp:
            here = Path(tmp, "here")
            here.mkdir()
            legacy = here / "dataset.yaml"
            legacy.write_text(
                yaml.safe_dump({"path": "/elsewhere/datasets/pose", "train": "images/train"}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_dataset_root(str(legacy)), "/elsewhere/datasets/pose")
            self.assertEqual(stale_dataset_root(str(legacy)), "/elsewhere/datasets/pose")

            legacy.write_text(yaml.safe_dump({"path": str(here)}), encoding="utf-8")
            self.assertEqual(resolve_dataset_root(str(legacy)), str(here))
            self.assertIsNone(stale_dataset_root(str(legacy)))

            legacy.write_text(yaml.safe_dump({"path": "sub"}), encoding="utf-8")
            self.assertEqual(resolve_dataset_root(str(legacy)), str(here / "sub"))

    def test_unreadable_yaml_resolves_to_its_directory(self):
        with TemporaryDirectory() as tmp:
            broken = Path(tmp, "dataset.yaml")
            broken.write_text("path: [unclosed\n", encoding="utf-8")

            self.assertEqual(resolve_dataset_root(str(broken)), tmp)
            self.assertIsNone(stale_dataset_root(str(broken)))


class FlipIndexNamingTests(unittest.TestCase):
    def assertPairs(self, names, expected):
        self.assertEqual(flip_indices_for_keypoints(names), expected, names)
        self.assertEqual(unpaired_sided_keypoints(names), [], names)

    def test_separator_styles(self):
        for names in (
            ["nose", "left_ear", "right_ear"],
            ["nose", "left ear", "right ear"],
            ["nose", "left-ear", "right-ear"],
            ["nose", "Left.Ear", "Right.Ear"],
            ["nose", "ear_left", "ear_right"],
            ["nose", "l_ear", "r_ear"],
            ["nose", "ear_L", "ear_R"],
        ):
            with self.subTest(names=names):
                self.assertPairs(names, [0, 2, 1])

    def test_camel_case_styles(self):
        for names in (
            ["nose", "LeftEar", "RightEar"],
            ["nose", "leftEar", "rightEar"],
            ["nose", "earLeft", "earRight"],
            ["nose", "LEar", "REar"],
            ["nose", "LeftEar", "right_ear"],
        ):
            with self.subTest(names=names):
                self.assertPairs(names, [0, 2, 1])

    def test_compact_compound_styles(self):
        self.assertPairs(["nose", "leftear", "rightear"], [0, 2, 1])
        self.assertPairs(["leftforepaw", "rightforepaw"], [1, 0])

    def test_limb_codes(self):
        self.assertPairs(["LF_paw", "RF_paw", "LH_paw", "RH_paw"], [1, 0, 3, 2])
        self.assertPairs(["FL paw", "FR paw", "HL paw", "HR paw"], [1, 0, 3, 2])
        self.assertPairs(["LFpaw", "RFpaw"], [1, 0])
        # Fore and hind codes never pair with each other.
        self.assertEqual(flip_indices_for_keypoints(["LF_paw", "RH_paw"]), [0, 1])

    def test_no_false_pairs(self):
        for names in (
            ["light", "right"],
            ["lateral", "rear"],
            ["rear_paw", "lear_paw"],
            ["upright", "bright"],
            ["rightness", "tail"],
            ["left_ear", "ear_l"],
            ["left_ear", "right_eye"],
            ["left_right_mid", "right_left_mid"],
        ):
            with self.subTest(names=names):
                self.assertEqual(flip_indices_for_keypoints(names), list(range(len(names))))

    def test_rear_legs_with_sides_pair(self):
        self.assertPairs(["rear_left_paw", "rear_right_paw", "lateral_tail"], [1, 0, 2])

    def test_unpaired_sided_keypoints_are_reported(self):
        names = ["nose", "left_ear", "RightEye", "LF_paw", "tail", "light", "left_hip", "RightHip"]

        self.assertEqual(unpaired_sided_keypoints(names), ["left_ear", "RightEye", "LF_paw"])
        self.assertEqual(flip_indices_for_keypoints(names)[6:], [7, 6])

    def test_duplicate_sided_name_left_over_is_reported(self):
        names = ["left_ear", "left_ear", "right_ear"]

        self.assertEqual(flip_indices_for_keypoints(names), [2, 1, 0])
        self.assertEqual(unpaired_sided_keypoints(names), ["left_ear"])


if __name__ == "__main__":
    unittest.main()
