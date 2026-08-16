import importlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

qt_gui = importlib.import_module("PyQt6.QtGui")
qt_widgets = importlib.import_module("PyQt6.QtWidgets")

from squeakpose.annotation.depth import (  # noqa: E402
    DepthArtifactLoadResult,
    DepthArtifactPlan,
    DepthAssistantState,
    DepthProbe,
)
from squeakpose.ui.depth_panel import DepthRangePanel  # noqa: E402
from squeakpose.ui.depth_presentation import (  # noqa: E402
    DepthPreviewPresenter,
    decide_depth_preview,
)


class DepthPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["depth-presentation-test"]
        )
        cls.app.setQuitOnLastWindowClosed(False)

    @staticmethod
    def _artifacts(path: str, *, available: bool) -> DepthArtifactLoadResult:
        return DepthArtifactLoadResult(
            plan=DepthArtifactPlan(
                image_name="frame.png",
                map_path="/depth/frame.npy",
                metadata_path="/depth/frame_depth.json",
                preview_path=path,
                expected_shape=(6, 8),
            ),
            preview_available=available,
        )

    def test_preview_decisions_preserve_original_depth_and_overlay_status(self):
        missing = self._artifacts("/missing.png", available=False)
        original = decide_depth_preview(missing, "original")
        depth_missing = decide_depth_preview(missing, "depth")
        overlay = decide_depth_preview(
            self._artifacts("/depth/frame_depth.png", available=True),
            "overlay",
        )

        self.assertTrue(original.show_original)
        self.assertFalse(original.show_preview)
        self.assertIn("Original image displayed", original.status_message)
        self.assertFalse(depth_missing.show_preview)
        self.assertIn("Select Predict", depth_missing.status_message)
        self.assertTrue(overlay.show_preview)
        self.assertEqual(overlay.preview_opacity, 0.55)
        self.assertEqual(overlay.status_message, "Saved depth overlay displayed.")

    def test_presenter_scales_preview_and_tracks_scene_item(self):
        with TemporaryDirectory() as tmp:
            preview_path = str(Path(tmp, "frame_depth.png"))
            image = qt_gui.QImage(4, 3, qt_gui.QImage.Format.Format_RGB32)
            image.fill(0xFF123456)
            self.assertTrue(image.save(preview_path))
            scene = qt_widgets.QGraphicsScene()
            tracked = []
            presenter = DepthPreviewPresenter(scene, track_item=tracked.append)

            result = presenter.present_preview(
                self._artifacts(preview_path, available=True),
                mode="overlay",
                image_width=8,
                image_height=6,
            )

        self.assertIs(result.preview_item, presenter.preview_item)
        self.assertEqual(result.preview_item.pixmap().size().width(), 8)
        self.assertEqual(result.preview_item.pixmap().size().height(), 6)
        self.assertEqual(result.preview_item.opacity(), 0.55)
        self.assertEqual(result.preview_item.zValue(), 1.0)
        self.assertEqual(tracked, [result.preview_item])
        self.assertEqual(result.status_message, "Saved depth overlay displayed.")
        self.assertIn(result.preview_item, scene.items())

        presenter.present_preview(
            self._artifacts(preview_path, available=True),
            mode="original",
            image_width=8,
            image_height=6,
        )
        self.assertIsNone(presenter.preview_item)
        self.assertEqual(scene.items(), [])

    def test_undecodable_existing_preview_silently_keeps_original(self):
        scene = qt_widgets.QGraphicsScene()
        presenter = DepthPreviewPresenter(
            scene,
            pixmap_loader=lambda _path: qt_gui.QPixmap(),
        )

        result = presenter.present_preview(
            self._artifacts("/corrupt.png", available=True),
            mode="depth",
            image_width=8,
            image_height=6,
        )

        self.assertTrue(result.decision.show_preview)
        self.assertIsNone(result.preview_item)
        self.assertEqual(result.status_message, "")
        self.assertEqual(scene.items(), [])

    def test_presenter_projects_range_probe_text_and_clear_state_to_panel(self):
        panel = DepthRangePanel()
        presenter = DepthPreviewPresenter(qt_widgets.QGraphicsScene(), range_view=panel)
        state = DepthAssistantState(
            metadata={"p02_depth": 0.5, "p98_depth": 2.5, "median_depth": 1.25},
            probes=[DepthProbe(2, 3, 1.5, True), DepthProbe(4, 5, 2.0, True)],
        )

        presenter.present_state(state)

        self.assertIn("0.500–2.500 m", panel.range_label.text())
        self.assertIn("1. (2, 3): 1.500 m", panel.probe_label.text())
        self.assertIn("Δ last two: 0.500 m", panel.probe_label.text())
        self.assertTrue(panel.clear_btn.isEnabled())

        state.clear_probes()
        presenter.present_state(state)
        self.assertFalse(panel.clear_btn.isEnabled())
        panel.close()

    def test_probe_markers_match_number_position_value_and_layer_visibility(self):
        scene = qt_widgets.QGraphicsScene()
        presenter = DepthPreviewPresenter(scene)
        probes = [
            {"x": 10, "y": 12, "depth": 1.25, "valid": True},
            {"x": 4, "y": 6, "depth": None, "valid": False},
        ]

        items = presenter.present_probe_markers(probes, active_depth_layer=True)
        text_items = [
            item for item in items if isinstance(item, qt_widgets.QGraphicsSimpleTextItem)
        ]
        marker_items = [item for item in items if isinstance(item, qt_widgets.QGraphicsEllipseItem)]

        self.assertEqual(len(items), 4)
        self.assertEqual([item.text() for item in text_items], ["1 · 1.250 m", "2 · invalid"])
        self.assertEqual((marker_items[0].pos().x(), marker_items[0].pos().y()), (10.5, 12.5))
        self.assertTrue(
            marker_items[0].flags()
            & qt_widgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.assertEqual(marker_items[0].zValue(), 20.0)

        self.assertEqual(
            presenter.present_probe_markers(probes, active_depth_layer=False),
            (),
        )
        self.assertEqual(scene.items(), [])

    def test_clear_tolerates_scene_clear_deleting_owned_wrappers(self):
        scene = qt_widgets.QGraphicsScene()
        presenter = DepthPreviewPresenter(scene)
        presenter.present_probe_markers(
            [DepthProbe(1, 2, 3.0, True)],
            active_depth_layer=True,
        )
        scene.clear()

        presenter.clear()

        self.assertEqual(presenter.probe_items, ())


if __name__ == "__main__":
    unittest.main()
