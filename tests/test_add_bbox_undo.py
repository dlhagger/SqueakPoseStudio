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
graphics = importlib.import_module("squeakpose.annotation.graphics")


class AddBoxUndoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["add-bbox-undo-test"]
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
        image = qt_gui.QImage(40, 30, qt_gui.QImage.Format.Format_RGB32)
        image.fill(0xFFE7EBEF)
        self.assertTrue(image.save(str(Path(paths["images_to_label"]) / "frame001.png")))

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

    def test_redrawing_a_box_over_a_finished_annotation_can_be_undone(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    window.add_bbox(qt_core.QRectF(4, 5, 20, 15))
                    window.add_keypoint(qt_core.QPointF(8, 9))
                    window.add_keypoint(qt_core.QPointF(18, 16))
                    self.assertTrue(window.pose_edit_state.is_complete)
                    original_box = window.pose_edit_state.box

                    window.add_bbox(qt_core.QRectF(1, 1, 10, 10))
                    self.assertEqual(window.pose_edit_state.box.x, 1.0)
                    self.assertEqual(window.pose_edit_state.keypoints, {})
                    self.assertTrue(window.pose_edit_state.can_undo)

                    window.undo()

                    state = window.pose_edit_state
                    self.assertEqual(state.box, original_box)
                    self.assertEqual(set(state.keypoints), {"nose", "tail"})
                    self.assertTrue(state.is_complete)
                    box_items = [
                        item
                        for item in window.scene.items()
                        if isinstance(item, graphics.BoxItem) and item.bbox.class_id == 0
                    ]
                    self.assertEqual(len(box_items), 1)
                    self.assertEqual(box_items[0].bbox.x, original_box.x)
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
