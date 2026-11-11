"""Canvas view and box-item behavior that would otherwise crash or misplace edits."""

import importlib
import os
import unittest
from tempfile import gettempdir
from types import SimpleNamespace
from unittest.mock import Mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

qt_core = importlib.import_module("PyQt6.QtCore")
qt_gui = importlib.import_module("PyQt6.QtGui")
qt_widgets = importlib.import_module("PyQt6.QtWidgets")
QTest = importlib.import_module("PyQt6.QtTest").QTest
graphics = importlib.import_module("squeakpose.annotation.graphics")
models = importlib.import_module("squeakpose.annotation.models")

Qt = qt_core.Qt
QPointF = qt_core.QPointF
QEvent = qt_core.QEvent


def _fake_app(mode: str = "bbox", **extra):
    values = {
        "mode": mode,
        "seg_brush_radius": 8,
        "_is_seg_workflow": lambda: mode in {"segment", "segedit"},
        "_is_seg_edit_tool_brush": lambda: True,
        "_is_depth_layer": lambda: False,
        "add_bbox": Mock(),
        "add_keypoint": Mock(),
        "_add_seg_prompt": Mock(),
        "set_mode": Mock(),
        "update_status_bar": Mock(),
        "update_zoom_label": Mock(),
    }
    values.update(extra)
    return SimpleNamespace(**values)


def _mouse_event(kind, x: float = 10.0, y: float = 10.0):
    buttons = (
        Qt.MouseButton.NoButton
        if kind == QEvent.Type.MouseButtonRelease
        else Qt.MouseButton.LeftButton
    )
    return qt_gui.QMouseEvent(
        kind,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


class LabelViewSceneItemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["label-view-graphics-test"]
        )

    def setUp(self):
        self.scene = qt_widgets.QGraphicsScene(0, 0, 100, 80)
        self.app = _fake_app("segedit")
        self.view = graphics.LabelView(self.scene, self.app)
        self.view.resize(200, 160)

    def tearDown(self):
        self.view.close()
        self.view.deleteLater()

    def test_forget_scene_items_removes_view_owned_items_before_scene_clear(self):
        self.view._update_seg_brush_cursor(QPointF(20, 20))
        self.view._update_crosshairs(QPointF(10, 10))
        temp_rect = qt_widgets.QGraphicsRectItem(1, 1, 5, 5)
        self.scene.addItem(temp_rect)
        self.view._temp_rect = temp_rect
        self.view._start_pos = QPointF(1, 1)
        self.view._seg_brush_active = True
        self.assertIsNotNone(self.view._seg_brush_cursor_ring)
        self.assertIsNotNone(self.view._crosshair_v)

        self.view.forget_scene_items()

        self.assertIsNone(self.view._seg_brush_cursor_ring)
        self.assertIsNone(self.view._crosshair_v)
        self.assertIsNone(self.view._crosshair_h)
        self.assertIsNone(self.view._temp_rect)
        self.assertIsNone(self.view._start_pos)
        self.assertFalse(self.view._seg_brush_active)
        self.assertEqual(self.scene.items(), [])

        self.scene.clear()
        # The next hover recreates the ring in the live scene instead of touching a
        # deleted C++ item.
        self.view._update_seg_brush_cursor(QPointF(30, 30))
        ring = self.view._seg_brush_cursor_ring
        self.assertIsNotNone(ring)
        self.assertIs(ring.scene(), self.scene)

    def test_items_deleted_by_an_external_scene_clear_are_recreated_not_touched(self):
        self.view._update_seg_brush_cursor(QPointF(20, 20))
        self.view._update_crosshairs(QPointF(10, 10))

        self.scene.clear()

        self.view._hide_seg_brush_cursor()
        self.view._update_seg_brush_cursor(QPointF(25, 25))
        self.assertIs(self.view._seg_brush_cursor_ring.scene(), self.scene)
        self.view._remove_crosshairs()
        self.assertIsNone(self.view._crosshair_v)


class LabelViewInputRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["label-view-graphics-test"]
        )

    def _view(self, mode: str, **extra):
        scene = qt_widgets.QGraphicsScene(0, 0, 100, 80)
        app = _fake_app(mode, **extra)
        view = graphics.LabelView(scene, app)
        view.resize(200, 160)
        self.addCleanup(view.deleteLater)
        self.addCleanup(view.close)
        return view, app, scene

    def test_arrow_and_space_keys_are_forwarded_to_the_window(self):
        # Qt deletes key events after dispatch, so record key() while each event is alive.
        pressed_keys: list = []
        released_keys: list = []
        view, _app, _scene = self._view(
            "bbox",
            keyPressEvent=lambda event: pressed_keys.append(event.key()),
            keyReleaseEvent=lambda event: released_keys.append(event.key()),
        )

        keys = (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
        for key in keys:
            QTest.keyPress(view, key)
        self.assertEqual(pressed_keys, list(keys))

        QTest.keyRelease(view, Qt.Key.Key_Space)
        self.assertEqual(released_keys, [Qt.Key.Key_Space])

        QTest.keyPress(view, Qt.Key.Key_A)
        self.assertEqual(len(pressed_keys), len(keys))

    def test_space_held_pans_instead_of_starting_a_box_or_placing_points(self):
        for mode, action in (
            ("bbox", None),
            ("keypoint", "add_keypoint"),
            ("segment", "_add_seg_prompt"),
        ):
            with self.subTest(mode=mode):
                view, app, _scene = self._view(mode)
                view.setDragMode(qt_widgets.QGraphicsView.DragMode.NoDrag)
                view.set_space_pan_active(True)
                self.assertEqual(view.dragMode(), qt_widgets.QGraphicsView.DragMode.ScrollHandDrag)
                self.assertFalse(view.isInteractive())

                view.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress))
                self.assertTrue(view.is_space_panning())
                self.assertIsNone(view._start_pos)
                if action is not None:
                    getattr(app, action).assert_not_called()

                view.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, 30, 30))
                self.assertIsNone(view._temp_rect)
                view.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, 30, 30))
                self.assertFalse(view.is_space_panning())
                app.add_bbox.assert_not_called()

                view.set_space_pan_active(False)
                self.assertTrue(view.isInteractive())

    def test_releasing_space_mid_drag_restores_the_tool_when_the_drag_ends(self):
        view, app, _scene = self._view("keypoint")
        view.set_space_pan_active(True)
        view.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress))

        view.set_space_pan_active(False)
        self.assertFalse(view.isInteractive())
        app.set_mode.assert_not_called()

        view.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease))
        self.assertTrue(view.isInteractive())
        app.set_mode.assert_called_once_with("keypoint")

    def test_space_held_right_press_does_not_add_a_negative_prompt(self):
        view, app, _scene = self._view("segment")
        view.set_space_pan_active(True)
        event = qt_gui.QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(10, 10),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.mousePressEvent(event)
        app._add_seg_prompt.assert_not_called()
        self.assertFalse(view.is_space_panning())

    def test_without_space_a_left_press_still_places_a_keypoint(self):
        view, app, _scene = self._view("keypoint")
        view.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress))
        app.add_keypoint.assert_called_once()


class BoxItemResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["label-view-graphics-test"]
        )

    def setUp(self):
        self.scene = qt_widgets.QGraphicsScene(0, 0, 100, 80)
        self.bbox = models.BoundingBox(10.0, 10.0, 30.0, 20.0, 0)
        self.item = graphics.BoxItem(self.bbox, "mouse")
        self.scene.addItem(self.item)

    def _drag(self, edges: int, dx: float, dy: float) -> None:
        self.item._press_rect = qt_core.QRectF(self.item.rect())
        self.item._press_item_pos = QPointF(self.item.pos())
        self.item._apply_resize(edges, QPointF(dx, dy))

    def _edges(self) -> tuple[float, float, float, float]:
        return (
            self.bbox.x,
            self.bbox.y,
            self.bbox.x + self.bbox.w,
            self.bbox.y + self.bbox.h,
        )

    def test_dragging_right_edge_past_the_image_clamps_without_moving_left_edge(self):
        self._drag(graphics.BoxItem.RIGHT, 200, 0)
        left, top, right, bottom = self._edges()
        self.assertAlmostEqual(left, 10.0)
        self.assertAlmostEqual(right, 100.0)
        self.assertAlmostEqual((top, bottom), (10.0, 30.0))

    def test_dragging_left_edge_past_the_image_clamps_without_moving_right_edge(self):
        self._drag(graphics.BoxItem.LEFT, -50, 0)
        left, _top, right, _bottom = self._edges()
        self.assertAlmostEqual(left, 0.0)
        self.assertAlmostEqual(right, 40.0)

    def test_dragging_an_edge_past_the_opposite_edge_stops_at_minimum_size(self):
        self._drag(graphics.BoxItem.LEFT, 100, 0)
        left, _top, right, _bottom = self._edges()
        self.assertAlmostEqual(right, 40.0)
        self.assertAlmostEqual(left, 40.0 - graphics.BoxItem.MIN_W)

        self.setUp()
        self._drag(graphics.BoxItem.TOP, 0, 100)
        _left, top, _right, bottom = self._edges()
        self.assertAlmostEqual(bottom, 30.0)
        self.assertAlmostEqual(top, 30.0 - graphics.BoxItem.MIN_H)

    def test_dragging_bottom_right_corner_past_the_image_clamps_both_edges(self):
        self._drag(graphics.BoxItem.BOTTOM | graphics.BoxItem.RIGHT, 500, 500)
        self.assertEqual(self._edges(), (10.0, 10.0, 100.0, 80.0))


if __name__ == "__main__":
    unittest.main()
