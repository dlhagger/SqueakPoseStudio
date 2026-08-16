import importlib
import json
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

QApplication = importlib.import_module("PyQt6.QtWidgets").QApplication
studio = importlib.import_module("squeakpose_studio")
layers = importlib.import_module("squeakpose.project.layers")
ProjectSession = importlib.import_module("squeakpose.project.session").ProjectSession


class ProjectSessionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication(["project-session-test"])
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def _prepare_project(self, root: Path, *, pose=("mouse", "rat"), seg=("body", "tail")):
        paths = studio._ensure_project_structure(str(root))
        Path(paths["classes_file"]).write_text("\n".join(pose) + "\n", encoding="utf-8")
        Path(paths["keypoints_file"]).write_text("nose\ntail_base\n", encoding="utf-8")
        Path(paths["class_keypoints_file"]).write_text(
            json.dumps({name: ["nose", "tail_base"] for name in pose}) + "\n",
            encoding="utf-8",
        )
        Path(paths["classes_seg_file"]).write_text("\n".join(seg) + "\n", encoding="utf-8")
        return paths

    def _open_window(self, paths):
        stack = ExitStack()
        stack.enter_context(patch("squeakpose.ui.main_window._auto_device", return_value="cpu"))
        stack.enter_context(
            patch("squeakpose.ui.main_window.LabelingApp._restart_prediction_worker")
        )
        stack.enter_context(patch("squeakpose.ui.main_window.QMessageBox.warning"))
        stack.enter_context(patch("squeakpose.ui.main_window.QMessageBox.information"))
        window = studio.LabelingApp(
            paths["images_to_label"],
            paths["labels_all"],
            paths["classes_file"],
            paths["keypoints_file"],
            project_root=paths["root"],
            force_initial_setup=False,
        )
        return stack, window

    def test_layer_switch_restores_per_layer_selection_schema_and_model(self):
        with TemporaryDirectory() as tmp:
            paths = self._prepare_project(Path(tmp))
            pose_model = Path(tmp) / "pose.pt"
            seg_model = Path(tmp) / "segment.pt"
            pose_model.touch()
            seg_model.touch()
            stack, window = self._open_window(paths)
            with stack:
                try:
                    self.assertIsInstance(window._project_session, ProjectSession)
                    window.class_selector.setCurrentIndex(1)
                    window.predict_model_path = str(pose_model)
                    window.layer_model_paths[layers.LAYER_KEYPOINTS] = str(pose_model)

                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    self.assertEqual(window.classes, ["body", "tail"])
                    window.class_selector.setCurrentIndex(1)
                    window.predict_model_path = str(seg_model)
                    window.layer_model_paths[layers.LAYER_SEGMENTATION] = str(seg_model)

                    window._switch_layer(layers.LAYER_KEYPOINTS)
                    self.assertEqual(window.classes, ["mouse", "rat"])
                    self.assertEqual(window.kp_names, ["nose", "tail_base"])
                    self.assertEqual(window.class_selector.currentIndex(), 1)
                    self.assertEqual(window.predict_model_path, str(pose_model))

                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    self.assertEqual(window.classes, ["body", "tail"])
                    self.assertEqual(window.kp_names, [])
                    self.assertEqual(window.class_selector.currentIndex(), 1)
                    self.assertEqual(window.predict_model_path, str(seg_model))
                finally:
                    window.close()

    def test_opening_another_project_does_not_leak_active_layer_or_models(self):
        with TemporaryDirectory() as tmp:
            first_paths = self._prepare_project(Path(tmp) / "first")
            second_paths = self._prepare_project(
                Path(tmp) / "second", pose=("vole",), seg=("silhouette",)
            )
            first_model = Path(first_paths["root"]) / "segment.pt"
            first_model.touch()
            meta_path = Path(first_paths["root"]) / "squeakpose_project.json"
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "active_layer": layers.LAYER_SEGMENTATION,
                    "active_workflow": "segmentation",
                    "layers": {
                        layers.LAYER_SEGMENTATION: {"model_path": "segment.pt"},
                    },
                    "layer_visibility": {layers.LAYER_DEPTH: False},
                }
            )
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")

            first_stack, first = self._open_window(first_paths)
            with first_stack:
                try:
                    self.assertEqual(first.active_layer, layers.LAYER_SEGMENTATION)
                    self.assertEqual(first.predict_model_path, str(first_model))
                finally:
                    first.close()

            second_stack, second = self._open_window(second_paths)
            with second_stack:
                try:
                    self.assertEqual(second.active_layer, layers.LAYER_KEYPOINTS)
                    self.assertEqual(second.classes, ["vole"])
                    self.assertIsNone(second.predict_model_path)
                    self.assertNotEqual(second._project_session.project_root, first_paths["root"])
                finally:
                    second.close()


if __name__ == "__main__":
    unittest.main()
