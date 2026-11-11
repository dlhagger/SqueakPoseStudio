"""Main-window regressions: stale canvas/image state, read-only analysis, saves, export."""

import importlib
import os
import struct
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

qt_core = importlib.import_module("PyQt6.QtCore")
qt_gui = importlib.import_module("PyQt6.QtGui")
qt_widgets = importlib.import_module("PyQt6.QtWidgets")
QTest = importlib.import_module("PyQt6.QtTest").QTest
np = importlib.import_module("numpy")
studio = importlib.import_module("squeakpose_studio")
main_window = importlib.import_module("squeakpose.ui.main_window")
graphics = importlib.import_module("squeakpose.annotation.graphics")
documents = importlib.import_module("squeakpose.annotation.documents")
layers = importlib.import_module("squeakpose.project.layers")

Qt = qt_core.Qt
StandardButton = qt_widgets.QMessageBox.StandardButton


def _write_png(path: Path, width: int = 40, height: int = 30) -> None:
    image = qt_gui.QImage(width, height, qt_gui.QImage.Format.Format_RGB32)
    image.fill(0xFFE7EBEF)
    assert image.save(str(path)), path


def _write_jpeg_with_exif_orientation(
    path: Path, width: int, height: int, orientation: int
) -> bool:
    """Write a JPEG carrying a hand-built EXIF APP1 Orientation tag."""

    image = qt_gui.QImage(width, height, qt_gui.QImage.Format.Format_RGB32)
    image.fill(0xFF336699)
    buffer = qt_core.QBuffer()
    buffer.open(qt_core.QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "JPG"):
        return False
    data = bytes(buffer.data())
    tiff = (
        b"MM\x00\x2a"
        + struct.pack(">I", 8)
        + struct.pack(">H", 1)
        + struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
        + struct.pack(">I", 0)
    )
    payload = b"Exif\x00\x00" + tiff
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    insert_at = 2
    if data[2:4] == b"\xff\xe0":  # keep the JFIF APP0 segment first
        insert_at = 4 + struct.unpack(">H", data[4:6])[0]
    path.write_bytes(data[:insert_at] + segment + data[insert_at:])
    return True


class MainWindowRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["main-window-regression-test"]
        )
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def _open_window(
        self,
        root: Path,
        *,
        images: dict[str, bytes | None] | None = None,
        classes: str = "mouse\n",
        class_keypoints: str = '{"mouse": ["nose", "tail"]}\n',
    ):
        paths = studio._ensure_project_structure(str(root))
        Path(paths["classes_file"]).write_text(classes, encoding="utf-8")
        Path(paths["keypoints_file"]).write_text("nose\ntail\n", encoding="utf-8")
        Path(paths["class_keypoints_file"]).write_text(class_keypoints, encoding="utf-8")
        Path(paths["classes_seg_file"]).write_text("body\n", encoding="utf-8")
        if images is None:
            images = {"frame001.png": None}
        for name, content in images.items():
            target = Path(paths["images_to_label"]) / name
            if content is None:
                _write_png(target)
            else:
                target.write_bytes(content)

        stack = ExitStack()
        stack.enter_context(patch("squeakpose.ui.main_window._auto_device", return_value="cpu"))
        stack.enter_context(
            patch("squeakpose.ui.main_window.LabelingApp._restart_prediction_worker")
        )
        self.warning = stack.enter_context(patch("squeakpose.ui.main_window.QMessageBox.warning"))
        stack.enter_context(patch("squeakpose.ui.main_window.QMessageBox.information"))
        stack.enter_context(
            patch(
                "squeakpose.ui.main_window.QMessageBox.question",
                side_effect=AssertionError("unexpected confirmation prompt"),
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
        stack.callback(self.qt_app.processEvents)
        stack.callback(window.close)
        return stack, window, paths

    @staticmethod
    def _complete_pose(window, x: float = 4.0) -> None:
        window.add_bbox(qt_core.QRectF(x, 5, 12, 15))
        window.add_keypoint(qt_core.QPointF(x + 2, 9))
        window.add_keypoint(qt_core.QPointF(x + 8, 16))

    @staticmethod
    def _accept_segmentation(window) -> None:
        window._add_seg_prompt(qt_core.QPointF(4.0, 5.0), positive=True)
        window.seg_edit_state.set_preview(
            [(2.0, 2.0), (16.0, 2.0), (10.0, 18.0)],
            score=0.8,
        )
        window._accept_segmentation_preview()

    @staticmethod
    def _box_items(window, class_id: int) -> list:
        return [
            item
            for item in window.scene.items()
            if isinstance(item, graphics.BoxItem) and item.bbox.class_id == class_id
        ]

    # Item 1 -------------------------------------------------------------------------

    def test_clearing_the_last_image_drops_view_owned_scene_items(self):
        with TemporaryDirectory() as tmp:
            stack, window, _paths = self._open_window(Path(tmp))
            with stack:
                view = window.view
                view._ensure_seg_brush_cursor_ring()
                view._update_crosshairs(qt_core.QPointF(5, 5))
                self.assertIsNotNone(view._seg_brush_cursor_ring)

                window._clear_loaded_image_state()

                self.assertIsNone(view._seg_brush_cursor_ring)
                self.assertIsNone(view._crosshair_v)
                self.assertEqual((window.img_w, window.img_h), (0, 0))
                view._ensure_seg_brush_cursor_ring()
                self.assertIs(view._seg_brush_cursor_ring.scene(), window.scene)

    def test_navigating_while_dragging_a_box_discards_the_temporary_rect(self):
        with TemporaryDirectory() as tmp:
            stack, window, _paths = self._open_window(Path(tmp))
            with stack:
                view = window.view
                view._start_pos = qt_core.QPointF(1, 1)
                temp_rect = qt_widgets.QGraphicsRectItem(1, 1, 5, 5)
                window.scene.addItem(temp_rect)
                view._temp_rect = temp_rect

                window.load_image()

                self.assertIsNone(view._temp_rect)
                self.assertIsNone(view._start_pos)

    # Items 2 and 3 ------------------------------------------------------------------

    def test_failed_image_load_clears_previous_annotations_and_refuses_save(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(
                Path(tmp),
                images={"frame001.png": None, "frame002.png": b"not an image"},
            )
            with stack:
                window._switch_layer(layers.LAYER_SEGMENTATION)
                self._accept_segmentation(window)
                self.assertTrue(window.annotation_cache.all_entries())

                window.current_idx = 1
                window.load_image()

                self.assertEqual(window.current_image_path, "")
                self.assertEqual((window.img_w, window.img_h), (0, 0))
                self.assertIsInstance(
                    window.annotation_cache, documents.SegmentationAnnotationDocument
                )
                self.assertFalse(window.annotation_cache.all_entries())

                self.warning.reset_mock()
                self.assertFalse(window.save_labels())
                self.warning.assert_called_once()
                self.assertEqual(self.warning.call_args.args[1], "No image loaded")
                self.assertFalse(Path(paths["labels_seg_all"], "frame002.txt").exists())
                self.assertFalse(Path(paths["images_all"], "frame002.png").exists())

    def test_frames_exported_into_an_empty_queue_are_displayed(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp), images={})
            with stack:
                self.assertEqual(window.current_image_path, "")
                self.assertEqual((window.img_w, window.img_h), (0, 0))

                image_path = Path(paths["images_to_label"]) / "clip_f000001.png"
                _write_png(image_path, 32, 24)
                window.refresh_image_list()

                self.assertEqual(window.current_image_path, str(image_path))
                self.assertEqual((window.img_w, window.img_h), (32, 24))

    # Item 4 -------------------------------------------------------------------------

    def test_oriented_loader_requests_exif_auto_transform(self):
        class FakeReader:
            instances: list = []

            def __init__(self, path):
                self.path = path
                self.auto_transform = None
                FakeReader.instances.append(self)

            def setAutoTransform(self, enabled):
                self.auto_transform = enabled

            def read(self):
                image = qt_gui.QImage(3, 2, qt_gui.QImage.Format.Format_RGB32)
                image.fill(0)
                return image

        with patch("squeakpose.ui.main_window.QImageReader", FakeReader):
            pixmap = main_window.load_oriented_pixmap("/project/images_to_label/a.jpg")

        self.assertEqual(len(FakeReader.instances), 1)
        self.assertIs(FakeReader.instances[0].auto_transform, True)
        self.assertEqual((pixmap.width(), pixmap.height()), (3, 2))
        self.assertTrue(main_window.load_oriented_pixmap("/missing/file.jpg").isNull())

    def test_rotated_phone_jpeg_is_labeled_in_the_exif_transformed_frame(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jpg"
            if not _write_jpeg_with_exif_orientation(source, 40, 20, orientation=6):
                self.skipTest("Qt JPEG plugin unavailable")
            pixmap = main_window.load_oriented_pixmap(str(source))
            self.assertEqual((pixmap.width(), pixmap.height()), (20, 40))

            project = Path(tmp) / "project"
            project.mkdir()
            stack, window, _paths = self._open_window(
                project, images={"phone.jpg": source.read_bytes()}
            )
            with stack:
                self.assertEqual((window.img_w, window.img_h), (20, 40))
                self.assertEqual(
                    (window.scene.sceneRect().width(), window.scene.sceneRect().height()),
                    (20.0, 40.0),
                )

    # Item 5 -------------------------------------------------------------------------

    def test_arrow_keys_and_space_reach_the_window_while_the_canvas_has_focus(self):
        with TemporaryDirectory() as tmp:
            stack, window, _paths = self._open_window(
                Path(tmp), images={"frame001.png": None, "frame002.png": None}
            )
            with stack:
                with (
                    patch.object(window, "next_index") as next_index,
                    patch.object(window, "prev_index") as prev_index,
                ):
                    QTest.keyPress(window.view, Qt.Key.Key_Right)
                    QTest.keyPress(window.view, Qt.Key.Key_Left)
                next_index.assert_called_once()
                prev_index.assert_called_once()

                window.set_mode("bbox")
                QTest.keyPress(window.view, Qt.Key.Key_Space)
                self.assertTrue(window.view._space_pan_active)
                self.assertEqual(
                    window.view.dragMode(), qt_widgets.QGraphicsView.DragMode.ScrollHandDrag
                )
                QTest.keyRelease(window.view, Qt.Key.Key_Space)
                self.assertFalse(window.view._space_pan_active)
                self.assertEqual(window.mode, "bbox")
                self.assertEqual(window.view.dragMode(), qt_widgets.QGraphicsView.DragMode.NoDrag)

    # Item 7 -------------------------------------------------------------------------

    def test_analysis_read_only_survives_navigation_and_blocks_canvas_edits(self):
        with TemporaryDirectory() as tmp:
            stack, window, _paths = self._open_window(
                Path(tmp), images={"frame001.png": None, "frame002.png": None}
            )
            with stack:
                window._set_analysis_read_only(True)
                stack.enter_context(patch("squeakpose.ui.main_window.CongratsPopup"))
                try:
                    window.next_index()
                    self.assertEqual(window.mode, "panzoom")
                    window.prev_index()
                    self.assertEqual(window.mode, "panzoom")
                    window.skip_to_next_unlabeled()
                    self.assertEqual(window.mode, "panzoom")

                    window.add_bbox(qt_core.QRectF(4, 5, 20, 15))
                    self.assertEqual(self._box_items(window, 0), [])
                    self.assertIsNone(window.pose_edit_state.box)

                    def shortcuts_for(text):
                        sequence = qt_gui.QKeySequence(text)
                        return [
                            shortcut
                            for shortcut in window.findChildren(qt_gui.QShortcut)
                            if shortcut.key() == sequence
                        ]

                    for text in ("P", "N"):
                        found = shortcuts_for(text)
                        self.assertTrue(found, text)
                        self.assertTrue(all(shortcut.isEnabled() for shortcut in found), text)
                    self.assertFalse(any(shortcut.isEnabled() for shortcut in shortcuts_for("S")))
                finally:
                    window._set_analysis_read_only(False)

    # Item 8 -------------------------------------------------------------------------

    def test_segmentation_classes_are_unchanged_when_the_class_file_cannot_be_written(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp))
            with stack:
                window._switch_layer(layers.LAYER_SEGMENTATION)
                self.warning.reset_mock()
                with (
                    patch(
                        "squeakpose.ui.main_window.QInputDialog.getMultiLineText",
                        return_value=("body\nextra", True),
                    ),
                    patch(
                        "squeakpose.ui.main_window.atomic_write_text",
                        side_effect=OSError("disk full"),
                    ),
                ):
                    window._open_seg_class_manager()

                self.assertEqual(window.seg_classes, ["body"])
                self.assertEqual(window.classes, ["body"])
                self.assertEqual(
                    Path(paths["classes_seg_file"]).read_text(encoding="utf-8"), "body\n"
                )
                self.assertEqual(self.warning.call_args.args[1], "File write error")

    # Item 9 -------------------------------------------------------------------------

    def test_removing_every_annotation_offers_to_delete_the_saved_label(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp))
            with stack:
                self._complete_pose(window)
                self.assertTrue(window.save_labels())
                label = Path(paths["labels_all"], "frame001.txt")
                overlay = Path(paths["root"], "annotations", "keypoints", "frame001_annotated.png")
                self.assertTrue(label.is_file())
                self.assertTrue(overlay.is_file())

                boxes = self._box_items(window, 0)
                self.assertEqual(len(boxes), 1)
                boxes[0].setSelected(True)
                window.delete_selected()
                self.assertIsNone(window.pose_edit_state.box)

                with patch(
                    "squeakpose.ui.main_window.QMessageBox.question",
                    return_value=StandardButton.No,
                ) as question:
                    self.assertFalse(window.save_labels())
                question.assert_called_once()
                self.assertIn("frame001.png", question.call_args.args[2])
                self.assertTrue(label.is_file())

                with patch(
                    "squeakpose.ui.main_window.QMessageBox.question",
                    return_value=StandardButton.Yes,
                ):
                    self.assertTrue(window.save_labels())
                self.assertFalse(label.exists())
                self.assertFalse(overlay.exists())
                self.assertTrue(Path(paths["images_all"], "frame001.png").is_file())
                leftovers = [
                    name for name in os.listdir(paths["labels_all"]) if name.startswith("frame001")
                ]
                self.assertEqual(leftovers, [])

    def test_clearing_the_selected_class_saves_the_remaining_classes(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(
                Path(tmp),
                classes="mouse\nrat\n",
                class_keypoints='{"mouse": ["nose", "tail"], "rat": ["nose", "tail"]}\n',
            )
            with stack:
                self._complete_pose(window, x=4.0)
                window.class_selector.setCurrentIndex(1)
                self._complete_pose(window, x=20.0)
                self.assertTrue(window.save_labels())
                label = Path(paths["labels_all"], "frame001.txt")
                self.assertEqual(len(label.read_text(encoding="utf-8").splitlines()), 2)

                rat_boxes = self._box_items(window, 1)
                self.assertEqual(len(rat_boxes), 1)
                rat_boxes[0].setSelected(True)
                window.delete_selected()

                with patch(
                    "squeakpose.ui.main_window.QMessageBox.question",
                    side_effect=AssertionError("remaining classes must save without a prompt"),
                ):
                    self.assertTrue(window.save_labels())
                rows = label.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0].startswith("0 "))

    def test_transactional_removal_restores_earlier_files_when_a_later_move_fails(self):
        with TemporaryDirectory() as tmp:
            first = Path(tmp, "first.txt")
            second = Path(tmp, "second.png")
            first.write_text("label\n", encoding="utf-8")
            second.write_bytes(b"overlay")
            real_replace = os.replace

            def failing_replace(source, target):
                if os.fspath(source) == str(second):
                    raise PermissionError("locked")
                return real_replace(source, target)

            with patch("squeakpose.ui.main_window.os.replace", side_effect=failing_replace):
                with self.assertRaises(PermissionError):
                    main_window._remove_files_transactionally([str(first), str(second)])

            self.assertEqual(first.read_text(encoding="utf-8"), "label\n")
            self.assertEqual(second.read_bytes(), b"overlay")
            self.assertEqual(sorted(os.listdir(tmp)), ["first.txt", "second.png"])

    # Item 11 ------------------------------------------------------------------------

    def test_depth_maps_are_read_without_keeping_the_file_mapped(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "frame.npy")
            np.save(path, np.arange(6, dtype=np.float32).reshape(2, 3))

            depth = main_window._read_depth_array(str(path))

            self.assertNotIsInstance(depth, np.memmap)
            self.assertEqual(depth.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
            replacement = Path(tmp, "staged.npy")
            np.save(replacement, np.zeros((2, 3), dtype=np.float32))
            os.replace(replacement, path)
            self.assertEqual(depth.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    # Item 12 ------------------------------------------------------------------------

    def test_export_uses_the_grouped_train_val_split(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp))
            with stack:
                names = ["vidA_f000001.png", "vidA_f000002.png", "vidB_f000001.png"]
                for name in names:
                    _write_png(Path(paths["images_all"], name))
                    Path(paths["labels_all"], name.replace(".png", ".txt")).write_text(
                        "0 0.5 0.5 0.2 0.2 0.5 0.5 2 0.55 0.55 2\n", encoding="utf-8"
                    )
                planned = (["vidA_f000001.png", "vidA_f000002.png"], ["vidB_f000001.png"])
                with (
                    patch(
                        "squeakpose.ui.main_window.QInputDialog.getDouble",
                        return_value=(0.8, True),
                    ),
                    patch(
                        "squeakpose.ui.main_window.QInputDialog.getInt",
                        return_value=(7, True),
                    ),
                    patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=StandardButton.Yes,
                    ),
                    patch(
                        "squeakpose.ui.main_window.plan_grouped_train_val_split",
                        return_value=planned,
                    ) as plan,
                    patch(
                        "squeakpose.ui.main_window.export_dataset_transaction",
                        return_value=None,
                    ) as export,
                ):
                    window.export_dataset()

                plan.assert_called_once()
                self.assertEqual(sorted(plan.call_args.args[0]), names)
                self.assertEqual(plan.call_args.args[1], 0.8)
                self.assertEqual(plan.call_args.kwargs, {"seed": 7})
                export.assert_called_once()
                self.assertEqual(export.call_args.kwargs["train_images"], planned[0])
                self.assertEqual(export.call_args.kwargs["val_images"], planned[1])
                self.assertEqual(export.call_args.kwargs["split_seed"], 7)

    def test_export_summary_warns_about_unpaired_sided_keypoints(self):
        with TemporaryDirectory() as tmp:
            stack, window, paths = self._open_window(Path(tmp))
            with stack:
                _write_png(Path(paths["images_all"], "frame001.png"))
                Path(paths["labels_all"], "frame001.txt").write_text(
                    "0 0.5 0.5 0.2 0.2 0.5 0.5 2 0.55 0.55 2\n", encoding="utf-8"
                )
                window.kp_names = ["nose", "left_ear"]
                result = SimpleNamespace(canceled=False, errors=[])
                with (
                    patch(
                        "squeakpose.ui.main_window.QInputDialog.getDouble",
                        return_value=(0.8, True),
                    ),
                    patch(
                        "squeakpose.ui.main_window.QInputDialog.getInt",
                        return_value=(0, True),
                    ),
                    patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=StandardButton.Yes,
                    ),
                    patch(
                        "squeakpose.ui.main_window.export_dataset_transaction",
                        return_value=result,
                    ),
                    patch(
                        "squeakpose.ui.main_window.format_dataset_export_summary",
                        return_value="Exported 1 image.",
                    ),
                    patch("squeakpose.ui.main_window.QMessageBox.information") as information,
                ):
                    window.export_dataset()

                information.assert_called_once()
                self.assertEqual(information.call_args.args[1], "Dataset exported")
                summary = information.call_args.args[2]
                self.assertTrue(summary.startswith("Exported 1 image."))
                self.assertIn("horizontal-flip", summary)
                self.assertIn("left_ear", summary)
                self.assertNotIn("nose", summary)

    # Item 18 ------------------------------------------------------------------------

    def test_startup_cleanup_prompt_mentions_analysis_staging_folders(self):
        with TemporaryDirectory() as tmp:
            staging = Path(tmp, "analysis outputs", "session", ".combined.analysis-ab12cd34.tmp")
            staging.mkdir(parents=True)
            with patch(
                "squeakpose.ui.main_window.QMessageBox.question",
                return_value=StandardButton.No,
            ) as question:
                main_window._recover_project_transactions_for_ui(tmp)

            question.assert_called_once()
            self.assertIn("analysis staging folder", question.call_args.args[2])
            self.assertTrue(staging.is_dir())


class SaveRefusalWithoutImageTests(unittest.TestCase):
    def test_save_labels_refuses_when_no_image_is_loaded(self):
        app = Mock()
        app.images = ["frame001.jpg"]
        app.current_image_path = ""
        app.img_w = 0
        app.img_h = 0
        with patch("squeakpose.ui.main_window.QMessageBox.warning") as warning:
            self.assertFalse(studio.LabelingApp.save_labels(app))
        warning.assert_called_once()
        app._cache_active_annotation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
