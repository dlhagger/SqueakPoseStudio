import importlib
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from unittest.mock import patch

from squeakpose.annotation.segmentation import mask_to_polygon

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

QApplication = importlib.import_module("PyQt6.QtWidgets").QApplication
QImage = importlib.import_module("PyQt6.QtGui").QImage
QPointF = importlib.import_module("PyQt6.QtCore").QPointF
studio = importlib.import_module("squeakpose_studio")
project_safety = importlib.import_module("squeakpose.project.safety")
layers = importlib.import_module("squeakpose.project.layers")


class SegmentationStateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication(["squeakpose-seg-state-test"])
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def test_state_tracks_acceptance_and_resets_across_images_and_layers(self):
        with TemporaryDirectory() as tmp:
            paths = studio._ensure_project_structure(tmp)
            Path(paths["classes_file"]).write_text("mouse\n", encoding="utf-8")
            Path(paths["keypoints_file"]).write_text("nose\n", encoding="utf-8")
            Path(paths["class_keypoints_file"]).write_text(
                '{"mouse": ["nose"]}\n', encoding="utf-8"
            )
            Path(paths["classes_seg_file"]).write_text("mouse\n", encoding="utf-8")
            for name in ("frame001.png", "frame002.png"):
                image = QImage(20, 16, QImage.Format.Format_RGB32)
                image.fill(0xFFE7EBEF)
                self.assertTrue(image.save(str(Path(paths["images_to_label"]) / name)))
            Path(paths["labels_seg_all"], "frame001.txt").write_text(
                "0 0.1 0.1 0.8 0.1 0.5 0.8\n",
                encoding="utf-8",
            )

            window = None
            modal_patches = [
                patch(
                    f"squeakpose.ui.main_window.QMessageBox.{method}",
                    side_effect=AssertionError(f"unexpected modal QMessageBox.{method}"),
                )
                for method in ("critical", "information", "question", "warning")
            ]
            with ExitStack() as stack:
                for modal_patch in modal_patches:
                    stack.enter_context(modal_patch)
                stack.enter_context(
                    patch("squeakpose.ui.main_window._auto_device", return_value="cpu")
                )
                try:
                    window = studio.LabelingApp(
                        paths["images_to_label"],
                        paths["labels_all"],
                        paths["classes_file"],
                        paths["keypoints_file"],
                        project_root=paths["root"],
                        force_initial_setup=False,
                    )
                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    self.qt_app.processEvents()

                    self.assertEqual(window.seg_edit_state.selected_target, 0)
                    self.assertIn(0, window.seg_edit_state.accepted_masks)
                    self.assertEqual(
                        window.annotation_cache[0], window.seg_edit_state.accepted_masks[0]
                    )

                    # Main-window compatibility methods delegate raster geometry to
                    # the Qt-free segmentation module while retaining the active
                    # scene item and annotation-cache behavior.
                    target = window._seg_edit_target_item()
                    original_points = window._extract_seg_item_points(target)
                    mask = window._seg_mask_from_points(original_points)
                    self.assertEqual(mask.shape, (16, 20))
                    self.assertGreaterEqual(
                        len(
                            mask_to_polygon(
                                mask,
                                cv2_module=importlib.import_module("cv2"),
                                anchor_points=original_points,
                                max_points=1200,
                            )
                        ),
                        3,
                    )
                    window.set_mode("segedit")
                    anchor = QPointF(*original_points[0])
                    self.assertTrue(window._start_seg_brush(anchor, add=True))
                    self.assertTrue(
                        window._apply_seg_brush(
                            QPointF(19.0, 15.0),
                            add=True,
                            prev_scene_pos=anchor,
                        )
                    )
                    edited_points = window._extract_seg_item_points(target)
                    self.assertNotEqual(edited_points, original_points)
                    self.assertEqual(
                        window.annotation_cache[0]["segments"],
                        edited_points,
                    )
                    window._finish_seg_brush()

                    window._add_seg_prompt(QPointF(3.0, 4.0), positive=True)
                    window.seg_edit_state.set_preview(
                        [(2.0, 2.0), (10.0, 2.0), (6.0, 10.0)], score=0.7
                    )
                    window._accept_segmentation_preview()

                    self.assertEqual(window.seg_prompt_points, [])
                    self.assertEqual(window.seg_preview_points, [])
                    self.assertEqual(
                        window.annotation_cache[0], window.seg_edit_state.accepted_masks[0]
                    )
                    self.assertEqual(window.annotation_cache[0]["score"], 0.7)

                    window.seg_edit_state.push_undo_snapshot()
                    window.current_idx = 1
                    window.load_image()

                    self.assertEqual(window.annotation_cache.snapshot(), {})
                    self.assertEqual(window.seg_edit_state.accepted_masks, {})
                    self.assertEqual(window.seg_prompt_points, [])
                    self.assertEqual(window.seg_preview_points, [])
                    self.assertFalse(window.seg_edit_state.can_undo)
                    self.assertEqual(window.seg_edit_state.selected_target, 0)

                    window.seg_edit_state.add_prompt(5.0, 5.0)
                    window._switch_layer(layers.LAYER_KEYPOINTS)

                    self.assertEqual(window.seg_prompt_points, [])
                    self.assertEqual(window.seg_edit_state.accepted_masks, {})
                    self.assertIsNone(window.seg_edit_state.selected_target)
                finally:
                    if window is not None:
                        window.close()
                        self.qt_app.processEvents()

            self.assertFalse(Path(project_safety.project_lock_path(tmp)).exists())

    def test_legacy_fields_lazily_create_and_delegate_to_state(self):
        window = studio.LabelingApp.__new__(studio.LabelingApp)

        window.seg_prompt_points = [(1, 2, 1)]
        window.seg_preview_points = [(0, 0), (2, 0), (0, 2)]
        window.seg_preview_score = 0.5

        self.assertIs(window.seg_prompt_points, window.seg_edit_state.prompt_points)
        self.assertEqual(window.seg_prompt_labels, [1])
        self.assertIs(window.seg_preview_points, window.seg_edit_state.preview_points)
        self.assertEqual(window.seg_edit_state.preview_score, 0.5)


if __name__ == "__main__":
    unittest.main()
