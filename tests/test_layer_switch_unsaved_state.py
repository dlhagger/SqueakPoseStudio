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
layers = importlib.import_module("squeakpose.project.layers")


class LayerSwitchUnsavedStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["layer-switch-unsaved-test"]
        )
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def _open_window(self, root: Path, *, segmentation_class: str = "body"):
        paths = studio._ensure_project_structure(str(root))
        Path(paths["classes_file"]).write_text("mouse\n", encoding="utf-8")
        Path(paths["keypoints_file"]).write_text("nose\ntail\n", encoding="utf-8")
        Path(paths["class_keypoints_file"]).write_text(
            '{"mouse": ["nose", "tail"]}\n',
            encoding="utf-8",
        )
        Path(paths["classes_seg_file"]).write_text(
            f"{segmentation_class}\n",
            encoding="utf-8",
        )
        image = qt_gui.QImage(40, 30, qt_gui.QImage.Format.Format_RGB32)
        image.fill(0xFFE7EBEF)
        self.assertTrue(image.save(str(Path(paths["images_to_label"]) / "frame.png")))

        stack = ExitStack()
        stack.enter_context(patch("squeakpose.ui.main_window._auto_device", return_value="cpu"))
        stack.enter_context(
            patch("squeakpose.ui.main_window.LabelingApp._restart_prediction_worker")
        )
        stack.enter_context(
            patch(
                "squeakpose.ui.main_window.QMessageBox.question",
                side_effect=AssertionError("layer switching must not invent a save prompt"),
            )
        )
        window = studio.LabelingApp(
            paths["images_to_label"],
            paths["labels_all"],
            paths["classes_file"],
            paths["keypoints_file"],
            project_root=paths["root"],
            force_initial_setup=False,
        )
        window._jump_to_next_pending_class = lambda: None
        return stack, window, paths

    @staticmethod
    def _complete_pose(window) -> None:
        window.add_bbox(qt_core.QRectF(4, 5, 20, 15))
        window.add_keypoint(qt_core.QPointF(8, 9))
        window.add_keypoint(qt_core.QPointF(18, 16))
        assert window._cache_active_annotation()

    @staticmethod
    def _accept_segmentation(window) -> None:
        window._add_seg_prompt(qt_core.QPointF(4.0, 5.0), positive=True)
        window.seg_edit_state.set_preview(
            [(2.0, 2.0), (16.0, 2.0), (10.0, 18.0)],
            score=0.8,
        )
        window._accept_segmentation_preview()

    def test_pose_switch_discards_unsaved_cache_but_restores_an_explicit_save(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_pose(window)
                    self.assertEqual(len(window.annotation_cache.export_annotations()), 1)
                    label_path = Path(paths["labels_all"]) / "frame.txt"
                    self.assertFalse(label_path.exists())

                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    window._switch_layer(layers.LAYER_KEYPOINTS)

                    self.assertFalse(label_path.exists())
                    self.assertEqual(window.annotation_cache.export_annotations(), ())
                    self.assertIsNone(window.pose_edit_state.box)
                    self.assertEqual(window.pose_edit_state.keypoints, {})

                    self._complete_pose(window)
                    self.assertTrue(window.save_labels())
                    self.assertTrue(label_path.is_file())

                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    window._switch_layer(layers.LAYER_KEYPOINTS)

                    self.assertEqual(len(window.annotation_cache.export_annotations()), 1)
                    self.assertIsNotNone(window.pose_edit_state.box)
                    self.assertTrue(window.pose_edit_state.is_complete)
                finally:
                    window.close()

    def test_saved_segmentation_box_can_replace_pose_box_without_moving_keypoints(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(
                Path(tmp),
                segmentation_class="Mouse",
            )
            with stack:
                try:
                    self._complete_pose(window)
                    original_keypoints = {
                        name: (entry.kp.x, entry.kp.y, entry.visibility)
                        for name, entry in window.pose_edit_state.keypoints.items()
                    }
                    Path(paths["labels_seg_all"], "frame.txt").write_text(
                        "0 0.050000 0.100000 0.750000 0.100000 "
                        "0.750000 0.800000 0.050000 0.800000\n",
                        encoding="utf-8",
                    )
                    window._refresh_segmentation_box_action()
                    self.assertTrue(window.use_segmentation_box_btn.isEnabled())

                    window.use_segmentation_box_btn.click()

                    box = window.pose_edit_state.box
                    self.assertIsNotNone(box)
                    self.assertEqual((box.x, box.y, box.w, box.h), (2.0, 3.0, 28.0, 21.0))
                    self.assertEqual(
                        {
                            name: (entry.kp.x, entry.kp.y, entry.visibility)
                            for name, entry in window.pose_edit_state.keypoints.items()
                        },
                        original_keypoints,
                    )
                    annotation = window.annotation_cache.annotation(0)
                    self.assertIsNotNone(annotation)
                    self.assertEqual(annotation.box, (2.0, 3.0, 28.0, 21.0))

                    window.undo()
                    restored = window.pose_edit_state.box
                    self.assertIsNotNone(restored)
                    self.assertEqual(
                        (restored.x, restored.y, restored.w, restored.h),
                        (4.0, 5.0, 20.0, 15.0),
                    )
                    self.assertEqual(
                        {
                            name: (entry.kp.x, entry.kp.y, entry.visibility)
                            for name, entry in window.pose_edit_state.keypoints.items()
                        },
                        original_keypoints,
                    )
                finally:
                    window.close()

    def test_segmentation_switch_discards_prompts_preview_and_unsaved_acceptance(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp))
            with stack:
                try:
                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    window._add_seg_prompt(qt_core.QPointF(3.0, 4.0), positive=False)
                    window.seg_edit_state.set_preview(
                        [(1.0, 1.0), (8.0, 1.0), (4.0, 8.0)],
                        score=0.4,
                    )
                    with patch("squeakpose.ui.main_window.QMessageBox.information") as information:
                        self.assertFalse(window.save_labels())
                    information.assert_called_once()
                    self.assertEqual(information.call_args.args[1], "Pending preview")

                    window._switch_layer(layers.LAYER_KEYPOINTS)
                    window._switch_layer(layers.LAYER_SEGMENTATION)

                    self.assertEqual(window.seg_prompt_points, [])
                    self.assertEqual(window.seg_preview_points, [])
                    self.assertEqual(window.seg_edit_state.accepted_masks, {})

                    self._accept_segmentation(window)
                    self.assertIn(0, window.seg_edit_state.accepted_masks)
                    label_path = Path(paths["labels_seg_all"]) / "frame.txt"
                    self.assertFalse(label_path.exists())

                    window._switch_layer(layers.LAYER_KEYPOINTS)
                    window._switch_layer(layers.LAYER_SEGMENTATION)

                    self.assertFalse(label_path.exists())
                    self.assertEqual(window.seg_edit_state.accepted_masks, {})
                    self.assertEqual(window.annotation_cache.snapshot(), {})

                    self._accept_segmentation(window)
                    self.assertTrue(window.save_labels())
                    self.assertTrue(label_path.is_file())

                    window._switch_layer(layers.LAYER_KEYPOINTS)
                    window._switch_layer(layers.LAYER_SEGMENTATION)

                    self.assertIn(0, window.seg_edit_state.accepted_masks)
                    self.assertIn(0, window.annotation_cache)
                    self.assertGreaterEqual(
                        len(window.seg_edit_state.accepted_masks[0]["segments"]),
                        3,
                    )
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
