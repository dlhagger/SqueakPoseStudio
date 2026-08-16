import importlib
import os
import unittest
from tempfile import gettempdir

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

QApplication = importlib.import_module("PyQt6.QtWidgets").QApplication
QWidget = importlib.import_module("PyQt6.QtWidgets").QWidget
canvas_presentation = importlib.import_module("squeakpose.ui.canvas_presentation")


class CanvasPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["canvas-presentation-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.canvas = QWidget()
        self.canvas.resize(1000, 700)
        self.presenter = canvas_presentation.CanvasHudPresenter(self.canvas)
        self.presenter.show_context()

    def tearDown(self):
        self.canvas.close()

    def test_apply_preserves_layer_legend_and_zoom_presentation(self):
        state = canvas_presentation.CanvasPresentationState(
            editing="Editing: Keypoints",
            references="References: Segmentation",
            mode="keypoint",
            zoom_scale=1.375,
        )
        self.presenter.apply(state)

        self.assertEqual(self.presenter.layer_context.editing_label.text(), state.editing)
        self.assertEqual(self.presenter.layer_context.reference_label.text(), state.references)
        self.assertEqual(self.presenter.legend.title_label.text(), "Keypoint Visibility")
        self.assertEqual(
            self.presenter.legend.legend_label.text(),
            canvas_presentation.KEYPOINT_LEGEND_TEXT,
        )
        self.assertEqual(self.presenter.zoom.value_label.text(), "Zoom: 137%")
        self.assertFalse(self.presenter.legend.isHidden())
        self.assertTrue(self.presenter.zoom.isHidden())

    def test_modes_control_only_overlay_visibility(self):
        self.presenter.set_mode("panzoom")
        self.assertTrue(self.presenter.legend.isHidden())
        self.assertFalse(self.presenter.zoom.isHidden())

        self.presenter.set_mode("bbox")
        self.assertTrue(self.presenter.legend.isHidden())
        self.assertTrue(self.presenter.zoom.isHidden())
        self.assertFalse(self.presenter.layer_context.isHidden())

    def test_layout_preserves_hot_corner_contract(self):
        self.presenter.set_mode("keypoint")
        self.presenter.layout_overlays(viewport_width=1000, viewport_height=700)

        self.assertEqual(self.presenter.layer_context.pos().x(), 10)
        self.assertEqual(self.presenter.layer_context.pos().y(), 10)
        self.assertEqual(self.presenter.legend.pos().x(), 10)
        self.assertGreaterEqual(self.presenter.legend.pos().y(), 10)
        self.assertGreaterEqual(self.presenter.legend.width(), 250)
        self.assertLessEqual(self.presenter.legend.width(), 420)

    def test_presenter_has_no_scene_or_application_dependency(self):
        self.assertFalse(hasattr(self.presenter, "scene"))
        self.assertFalse(hasattr(self.presenter, "app"))
        self.assertIs(self.presenter.parent(), self.canvas)


if __name__ == "__main__":
    unittest.main()
