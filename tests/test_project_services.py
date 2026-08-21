import json
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.project.distillation import (
    DISTILLATION_MANIFEST_FILENAME,
    discover_distillation_exports,
    distillation_export_search_roots,
    distillation_run_task,
)
from squeakpose.project.metadata import ProjectMetadataStore
from squeakpose.project.paths import (
    ProjectPaths,
    ensure_project_structure,
    load_last_project,
    project_window_title,
    save_last_project,
)


class ProjectPathTests(unittest.TestCase):
    def test_project_paths_support_attributes_and_legacy_mapping_access(self):
        with TemporaryDirectory() as tmp:
            paths = ProjectPaths.from_root(tmp)

            self.assertEqual(paths.images_all, paths["images_all"])
            self.assertEqual(paths.root, os.path.abspath(tmp))
            self.assertEqual(paths.as_dict()["labels_seg_all"], paths.labels_seg_all)
            with self.assertRaises(KeyError):
                _ = paths["missing"]

    def test_ensure_project_structure_creates_canonical_entries(self):
        with TemporaryDirectory() as tmp:
            paths = ensure_project_structure(
                tmp,
                default_segmentation_classes=("mouse", "rat"),
            )

            self.assertTrue(os.path.isdir(paths.distillation_unlabeled_images))
            self.assertTrue(os.path.isdir(paths.analysis_outputs))
            self.assertTrue(os.path.isdir(paths.analysis_settings))
            self.assertTrue(os.path.isdir(paths.analysis_video_settings))
            self.assertTrue(os.path.isdir(paths.analysis_keypoints))
            self.assertTrue(os.path.isdir(paths.analysis_segmentation))
            self.assertTrue(os.path.isdir(paths.inference_keypoints))
            self.assertTrue(os.path.isdir(paths.inference_segmentation))
            self.assertTrue(os.path.isdir(paths.annotations_keypoints))
            self.assertTrue(os.path.isdir(paths.annotations_segmentation))
            self.assertTrue(os.path.isdir(paths.depth_images))
            self.assertTrue(os.path.isdir(paths.depth_previews))
            self.assertTrue(os.path.isdir(paths.inference_depth))
            self.assertTrue(os.path.isdir(paths.analysis_depth))
            self.assertTrue(os.path.isdir(paths.cache))
            self.assertTrue(os.path.isdir(paths.video_prediction_cache))
            with open(paths.classes_seg_file, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "mouse\nrat\n")
            with open(os.path.join(tmp, "squeakpose_project.json"), "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            self.assertIn("schema_version", metadata)
            self.assertIn("created_at", metadata)

    def test_last_project_state_ignores_missing_project(self):
        with TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "last.json")
            project = os.path.join(tmp, "project")
            os.makedirs(project)
            save_last_project(project, state_file=state_file)
            self.assertEqual(load_last_project(state_file=state_file), project)
            os.rmdir(project)
            self.assertIsNone(load_last_project(state_file=state_file))

    def test_project_window_title_uses_directory_name(self):
        self.assertEqual(
            project_window_title("/tmp/example-project"),
            "SqueakPose Studio — example-project",
        )

    def test_distillation_exports_are_filtered_by_declared_task(self):
        with TemporaryDirectory() as tmp:
            runs_root = os.path.join(tmp, "runs", "distillation")
            pose_run = os.path.join(runs_root, "custom-pose-run")
            segment_run = os.path.join(runs_root, "custom-mask-run")
            for run_dir, task in ((pose_run, "pose"), (segment_run, "segment")):
                exported_dir = os.path.join(run_dir, "exported_models")
                os.makedirs(exported_dir, exist_ok=True)
                with open(os.path.join(exported_dir, "exported_last.pt"), "wb") as fh:
                    fh.write(b"weights")
                with open(
                    os.path.join(run_dir, DISTILLATION_MANIFEST_FILENAME),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    json.dump({"task": task}, fh)

            roots = distillation_export_search_roots(tmp)
            pose_exports = discover_distillation_exports(roots, task="keypoints")
            segment_exports = discover_distillation_exports(roots, task="segmentation")

            self.assertEqual(len(pose_exports), 1)
            self.assertIn("custom-pose-run", pose_exports[0][0])
            self.assertEqual(len(segment_exports), 1)
            self.assertIn("custom-mask-run", segment_exports[0][0])
            self.assertEqual(distillation_run_task(segment_run), "segment")


class ProjectMetadataStoreTests(unittest.TestCase):
    def test_failed_metadata_migration_leaves_original_file_intact(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "squeakpose_project.json")
            original = {"created_at": "earlier", "active_workflow": "pose"}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(original, fh)

            with patch(
                "squeakpose.project.metadata.atomic_write_text",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaises(OSError):
                    ProjectMetadataStore(tmp).read()

            with open(path, "r", encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), original)

    def test_update_preserves_unknown_fields_and_removes_none_values(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "squeakpose_project.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "schema_version": 2,
                        "created_at": "earlier",
                        "unknown": {"keep": True},
                        "sam_model_path": "old.pt",
                    },
                    fh,
                )
            store = ProjectMetadataStore(tmp)

            result = store.update(
                {
                    "active_workflow": "segmentation",
                    "sam_model_path": None,
                }
            )

            self.assertEqual(result.data["unknown"], {"keep": True})
            self.assertNotIn("sam_model_path", result.data)
            self.assertEqual(result.data["active_workflow"], "segmentation")
            self.assertEqual(result.data["active_layer"], "segmentation")

    def test_corrupt_metadata_is_preserved_and_reported(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "squeakpose_project.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{bad json")

            with self.assertLogs("squeakpose.project.metadata", level="WARNING") as logs:
                result = ProjectMetadataStore(tmp).read()

            self.assertEqual(result.data, {})
            self.assertTrue(os.path.isfile(result.recovery_path))
            self.assertIn("Expecting property name", result.recovery_error)
            self.assertFalse(os.path.exists(path))
            self.assertTrue(any("Invalid project metadata" in line for line in logs.output))

    def test_project_relative_paths_round_trip(self):
        with TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "models", "sam.pt")
            store = ProjectMetadataStore(tmp)

            serialized = store.store_path(nested)

            self.assertEqual(serialized, os.path.join("models", "sam.pt"))
            self.assertEqual(store.resolve_path(serialized), nested)


if __name__ == "__main__":
    unittest.main()
