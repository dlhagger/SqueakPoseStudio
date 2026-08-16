import importlib
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

qt_core = importlib.import_module("PyQt6.QtCore")
qt_gui = importlib.import_module("PyQt6.QtGui")
qt_widgets = importlib.import_module("PyQt6.QtWidgets")
studio = importlib.import_module("squeakpose_studio")
documents = importlib.import_module("squeakpose.annotation.documents")
graphics = importlib.import_module("squeakpose.annotation.graphics")
layers = importlib.import_module("squeakpose.project.layers")
pose = importlib.import_module("squeakpose.annotation.pose")


class PoseStateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["pose-state-test"]
        )
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def _open_window(self, root: Path):
        paths = studio._ensure_project_structure(str(root))
        Path(paths["classes_file"]).write_text("mouse\nrat\n", encoding="utf-8")
        Path(paths["keypoints_file"]).write_text("nose\ntail\n", encoding="utf-8")
        Path(paths["class_keypoints_file"]).write_text(
            '{"mouse": ["nose", "tail"], "rat": ["nose", "tail"]}\n',
            encoding="utf-8",
        )
        Path(paths["classes_seg_file"]).write_text("body\n", encoding="utf-8")
        for name, color in (("frame001.png", 0xFFE7EBEF), ("frame002.png", 0xFFCCDDEE)):
            image = qt_gui.QImage(40, 30, qt_gui.QImage.Format.Format_RGB32)
            image.fill(color)
            self.assertTrue(image.save(str(Path(paths["images_to_label"]) / name)))

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
        window._jump_to_next_pending_class = lambda: None
        return stack, window

    def _complete_active_pose(self, window):
        window.add_bbox(qt_core.QRectF(4, 5, 20, 15))
        window.add_keypoint(qt_core.QPointF(8, 9))
        window.add_keypoint(qt_core.QPointF(18, 16))
        self.assertTrue(window._cache_active_annotation())

    def test_class_switch_visibility_and_undo_use_pose_state_with_legacy_mirrors(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_active_pose(window)
                    self.assertIsInstance(window.pose_edit_state, pose.PoseEditState)
                    self.assertIsInstance(window.annotation_cache, documents.PoseAnnotationDocument)
                    self.assertIs(window.current_box, window.pose_edit_state.box)
                    self.assertIs(window.bboxes[0], window.current_box)
                    self.assertIs(window.current_kps, window.kps)
                    self.assertEqual(window.current_class_id, 0)
                    self.assertEqual(len(window.current_kps), 2)

                    window.class_selector.setCurrentIndex(1)
                    self.assertEqual(window.pose_edit_state.active_class_id, 1)
                    self.assertIsNone(window.current_box)
                    window.class_selector.setCurrentIndex(0)
                    self.assertEqual(window.pose_edit_state.active_class_id, 0)
                    self.assertEqual(len(window.current_kps), 2)

                    nose_item = next(
                        item
                        for item in window.scene.items()
                        if isinstance(item, graphics.KeypointItem) and item.kp.name == "nose"
                    )
                    nose_item.setSelected(True)
                    original_visibility = nose_item.visibility
                    window.toggle_selected_visibility()
                    self.assertNotEqual(
                        window.pose_edit_state.keypoints["nose"].visibility,
                        original_visibility,
                    )
                    window.undo()
                    self.assertEqual(
                        window.pose_edit_state.keypoints["nose"].visibility,
                        original_visibility,
                    )
                    self.assertEqual(
                        window.annotation_cache.annotation(0).keypoints[0].visibility,
                        original_visibility,
                    )
                finally:
                    window.close()

    def test_template_and_image_layer_resets_keep_domain_and_rendering_in_sync(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_active_pose(window)
                    window.save_template_for_current_class()
                    template_path = Path(window._template_path_for_class("mouse"))
                    self.assertTrue(template_path.is_file())

                    window._clear_class_items(0, drop_cache=True)
                    window.pose_edit_state.select_class(
                        0,
                        ["nose", "tail"],
                        canonical_names=["nose", "tail"],
                    )
                    window.apply_template_for_current_class()
                    self.assertTrue(window.pose_edit_state.is_complete)
                    self.assertIs(window.current_box, window.bboxes[0])
                    self.assertEqual(len(window.annotation_cache.export_annotations()), 1)

                    window.current_idx = 1
                    window.load_image()
                    self.assertFalse(window.pose_edit_state.can_undo)
                    self.assertIsNone(window.current_box)
                    self.assertEqual(window.annotation_cache.export_annotations(), ())

                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    self.assertIsNone(window.pose_edit_state.active_class_id)
                    self.assertIsNone(window.current_box)
                    self.assertEqual(window.current_kps, [])
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
