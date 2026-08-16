import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.project import (
    LAYER_DEPTH,
    LAYER_KEYPOINTS,
    LAYER_SEGMENTATION,
    ProjectPathError,
    ProjectSession,
    is_builtin_model_reference,
    resolve_model_reference,
    store_model_reference,
)


class ProjectModelReferenceTests(unittest.TestCase):
    def test_model_references_round_trip_inside_and_outside_project(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            root = Path(tmp)
            model = root / "models" / "pose.pt"
            model.parent.mkdir()
            model.write_bytes(b"model")
            external = Path(outside) / "external.pt"
            external.write_bytes(b"external")

            self.assertEqual(resolve_model_reference(tmp, "models/pose.pt"), str(model))
            self.assertEqual(store_model_reference(tmp, str(model)), "models/pose.pt")
            self.assertEqual(store_model_reference(tmp, "models/pose.pt"), "models/pose.pt")
            self.assertEqual(resolve_model_reference(tmp, str(external)), str(external))
            self.assertEqual(store_model_reference(tmp, str(external)), str(external))

    def test_builtin_depth_references_remain_symbolic(self):
        self.assertTrue(is_builtin_model_reference("YOLO26N-DEPTH.PT"))
        self.assertEqual(
            resolve_model_reference("/project", "yolo26n-depth.pt"), "yolo26n-depth.pt"
        )
        self.assertEqual(store_model_reference("/project", "yolo26x-depth.pt"), "yolo26x-depth.pt")

    def test_relative_model_reference_cannot_escape_project(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ProjectPathError):
                resolve_model_reference(tmp, "../outside.pt")
            with self.assertRaises(ProjectPathError):
                store_model_reference(tmp, "../outside.pt")


class ProjectSessionPreferenceTests(unittest.TestCase):
    def test_preferences_normalize_layers_models_visibility_and_assistant(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            pose_model = models / "pose.pt"
            segment_model = models / "segment.pt"
            assistant_model = models / "sam.pt"
            for path in (pose_model, segment_model, assistant_model):
                path.write_bytes(b"model")

            session = ProjectSession.from_preferences(
                tmp,
                {
                    "active_workflow": "segment",
                    "layers": {
                        "pose": {"model_path": "models/pose.pt", "custom": "kept"},
                        "segmentation": {
                            "model_path": "models/segment.pt",
                            "assistant_model_path": "models/sam.pt",
                        },
                        "depth": {"model_path": "yolo26n-depth.pt"},
                    },
                    "layer_visibility": {"keypoints": False, "depth": False},
                },
                pose_classes=["mouse"],
                pose_keypoints=["nose"],
                pose_class_keypoints={"mouse": ["nose"]},
                segmentation_classes=["mouse", "object"],
                selected_class_ids={"segmentation": 1},
            )

            self.assertEqual(session.active_layer, LAYER_SEGMENTATION)
            self.assertEqual(session.active_workflow, "segmentation")
            self.assertEqual(session.active_paths.label_dir, str(root / "labels_seg_all"))
            self.assertEqual(session.active_paths.class_file, str(root / "classes_seg.txt"))
            self.assertEqual(session.active_state.selected_class_name, "object")
            snapshot = session.snapshot()
            self.assertEqual(snapshot.layer(LAYER_KEYPOINTS).model_path, str(pose_model))
            self.assertEqual(snapshot.layer(LAYER_SEGMENTATION).model_path, str(segment_model))
            self.assertEqual(snapshot.layer(LAYER_DEPTH).model_path, "yolo26n-depth.pt")
            self.assertEqual(snapshot.assistant_model_path, str(assistant_model))
            self.assertTrue(dict(snapshot.layer_visibility)[LAYER_SEGMENTATION])
            self.assertFalse(dict(snapshot.layer_visibility)[LAYER_KEYPOINTS])
            self.assertEqual(session.layer_settings[LAYER_KEYPOINTS]["custom"], "kept")

    def test_missing_and_escaping_persisted_models_are_ignored(self):
        with TemporaryDirectory() as tmp:
            session = ProjectSession.from_preferences(
                tmp,
                {
                    "layers": {
                        "keypoints": {"model_path": "missing.pt"},
                        "segmentation": {"model_path": "../escape.pt"},
                    },
                    "sam_model_path": "../escape-sam.pt",
                },
                pose_classes=["mouse"],
                segmentation_classes=["mouse"],
            )

            self.assertEqual(session.snapshot().layer(LAYER_KEYPOINTS).model_path, "")
            self.assertEqual(session.snapshot().layer(LAYER_SEGMENTATION).model_path, "")
            self.assertEqual(session.assistant_model_path, "")

    def test_preferences_round_trip_as_detached_relative_mappings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            pose_model = models / "pose.pt"
            assistant_model = models / "sam.pt"
            pose_model.write_bytes(b"pose")
            assistant_model.write_bytes(b"sam")

            session = ProjectSession(
                tmp,
                active_layer="pose",
                pose_classes=["mouse"],
                pose_keypoints=["nose"],
                pose_class_keypoints={"mouse": ["nose"]},
                segmentation_classes=["mouse"],
                layer_settings={"keypoints": {"custom": {"value": 1}}},
                assistant_model_path=str(assistant_model),
            )
            session.set_model_path(LAYER_KEYPOINTS, str(pose_model))
            session.set_model_path(LAYER_DEPTH, "yolo26s-depth.pt")
            session.set_layer_visibility(LAYER_SEGMENTATION, False)

            preferences = session.to_preferences()

            self.assertEqual(preferences["active_layer"], LAYER_KEYPOINTS)
            self.assertEqual(preferences["active_workflow"], "pose")
            self.assertEqual(preferences["layers"][LAYER_KEYPOINTS]["model_path"], "models/pose.pt")
            self.assertEqual(preferences["layers"][LAYER_DEPTH]["model_path"], "yolo26s-depth.pt")
            self.assertEqual(preferences["sam_model_path"], "models/sam.pt")
            self.assertEqual(
                preferences["layers"][LAYER_SEGMENTATION]["assistant_model_path"],
                "models/sam.pt",
            )
            self.assertFalse(preferences["layer_visibility"][LAYER_SEGMENTATION])

            preferences["layers"][LAYER_KEYPOINTS]["custom"] = "changed externally"
            self.assertEqual(
                session.to_preferences()["layers"][LAYER_KEYPOINTS]["custom"],
                {"value": 1},
            )

            restored = ProjectSession.from_preferences(
                tmp,
                session.to_preferences(),
                pose_classes=["mouse"],
                pose_keypoints=["nose"],
                segmentation_classes=["mouse"],
            )
            self.assertEqual(restored.active_model_path, str(pose_model))
            self.assertEqual(restored.assistant_model_path, str(assistant_model))


class ProjectSessionTransitionTests(unittest.TestCase):
    def test_capture_and_transition_preserve_independent_layer_state(self):
        with TemporaryDirectory() as tmp:
            session = ProjectSession(
                tmp,
                pose_classes=["mouse"],
                pose_keypoints=["nose"],
                pose_class_keypoints={"mouse": ["nose"]},
                segmentation_classes=["animal", "object"],
            )
            session.capture_active_state(
                classes=["mouse", "rat"],
                keypoints=["nose"],
                class_keypoints={"mouse": ["nose", "tail"], "rat": ["nose"]},
                selected_class_id=1,
                model_path="/models/pose.pt",
            )
            pose_snapshot = session.snapshot()

            transition = session.transition_workflow("segmentation")

            self.assertTrue(transition.changed)
            self.assertEqual(transition.before, pose_snapshot)
            self.assertEqual(transition.after.active_layer, LAYER_SEGMENTATION)
            self.assertEqual(transition.after.active.classes, ("animal", "object"))
            captured_pose = transition.after.layer(LAYER_KEYPOINTS)
            self.assertEqual(captured_pose.classes, ("mouse", "rat"))
            self.assertEqual(captured_pose.keypoints, ("nose", "tail"))
            self.assertEqual(captured_pose.selected_class_name, "rat")
            self.assertEqual(captured_pose.model_path, "/models/pose.pt")

            session.capture_active_state(
                classes=["animal", "object", "arena"],
                selected_class_id=2,
                model_path="/models/segment.pt",
            )
            depth_transition = session.transition_to("depth")
            self.assertEqual(depth_transition.after.active.classes, ())
            self.assertEqual(
                session.active_paths.label_dir, os.path.join(tmp, "depth maps", "images")
            )
            self.assertEqual(session.active_paths.class_file, "")

            self.assertEqual(pose_snapshot.layer(LAYER_KEYPOINTS).classes, ("mouse", "rat"))
            self.assertEqual(
                session.snapshot().layer(LAYER_SEGMENTATION).selected_class_name,
                "arena",
            )

    def test_class_selection_is_clamped_per_layer(self):
        session = ProjectSession(
            "/project",
            pose_classes=["mouse", "rat"],
            pose_keypoints=["nose"],
            segmentation_classes=["animal"],
        )

        self.assertEqual(session.select_class(99), 1)
        self.assertEqual(session.select_class("mouse"), 0)
        session.transition_to(LAYER_SEGMENTATION)
        self.assertEqual(session.select_class("missing"), 0)
        session.transition_to(LAYER_DEPTH)
        self.assertEqual(session.select_class(2), -1)


if __name__ == "__main__":
    unittest.main()
