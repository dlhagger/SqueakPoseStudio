import os
import unittest
from tempfile import gettempdir

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSimpleTextItem

from squeakpose.annotation.models import BoundingBox, Keypoint
from squeakpose.project.layers import LAYER_DEPTH, LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.ui.canvas_scene_presenter import (
    CanvasScenePresenter,
    PoseReferenceKeypoint,
)


class CanvasScenePresenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["canvas-scene-presenter-test"])

    def setUp(self):
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(QRectF(0, 0, 100, 80))
        self.tracked = []
        self.untracked = []
        self.presenter = CanvasScenePresenter(
            self.scene,
            track_item=self.tracked.append,
            untrack_item=self.untracked.append,
        )

    def test_segmentation_mask_preserves_frame_badge_and_preview_style(self):
        item = self.presenter.add_segmentation_mask(
            2,
            [(5, 1), (30, 1), (18, 25)],
            label_text="body",
            preview=True,
        )

        self.assertIsNotNone(item)
        self.assertIs(item.scene(), self.scene)
        self.assertEqual(item.pen().width(), 2)
        self.assertEqual(item.pen().style(), Qt.PenStyle.DashLine)
        self.assertEqual(item.brush().color().alpha(), 52)
        self.assertEqual(item.zValue(), 4.5)
        self.assertFalse(item.flags() & item.GraphicsItemFlag.ItemIsSelectable)
        self.assertEqual(item.seg_points, [(5.0, 1.0), (30.0, 1.0), (18.0, 25.0)])
        self.assertEqual(item.seg_label_item.text(), "body (preview)")
        self.assertGreater(item.seg_label_bg.rect().top(), item.path().boundingRect().bottom())
        self.assertEqual(self.tracked, [item])

        self.assertTrue(
            self.presenter.update_segmentation_geometry(
                item,
                [(10, 10), (40, 10), (25, 35)],
            )
        )
        self.assertEqual(item.path().boundingRect(), QRectF(10, 10, 30, 25))
        self.assertFalse(self.presenter.update_segmentation_geometry(item, [(1, 1), (2, 2)]))

    def test_prompt_markers_own_exact_items_and_tracking_lifecycle(self):
        positive = self.presenter.add_prompt_marker(12, 14, positive=True)
        negative = self.presenter.add_prompt_marker(22, 24, positive=False)

        self.assertEqual(len(positive), 1)
        self.assertEqual(len(negative), 3)
        self.assertEqual(positive[0].zValue(), 8.0)
        self.assertEqual([item.zValue() for item in negative], [8.0, 8.1, 8.1])
        self.assertEqual(len(self.presenter.prompt_items), 4)
        self.assertEqual(len(self.tracked), 4)

        owned = list(self.presenter.prompt_items)
        self.presenter.clear_prompts()
        self.assertEqual(self.presenter.prompt_items, [])
        self.assertEqual(self.untracked, owned)
        self.assertTrue(all(item.scene() is None for item in owned))

    def test_depth_probe_rendering_preserves_labels_positions_and_palette(self):
        items = self.presenter.render_depth_probes(
            [
                {"x": 4, "y": 6, "depth": 1.25},
                {"x": 20, "y": 10, "depth": None},
            ]
        )

        self.assertEqual(len(items), 4)
        first_marker, first_text, second_marker, second_text = items
        self.assertEqual(first_marker.pos().x(), 4.5)
        self.assertEqual(first_marker.pos().y(), 6.5)
        self.assertEqual(first_marker.pen().color(), QColor("#73d7ff"))
        self.assertEqual(first_marker.zValue(), 20.0)
        self.assertEqual(first_text.text(), "1 · 1.250 m")
        self.assertEqual(second_marker.pen().color(), QColor("#ffd166"))
        self.assertEqual(second_text.text(), "2 · invalid")
        self.assertTrue(first_text.flags() & first_text.GraphicsItemFlag.ItemIgnoresTransformations)

        self.presenter.clear_depth_probes()
        self.assertEqual(self.presenter.depth_probe_items, [])
        self.assertTrue(all(item.scene() is None for item in items))

    def test_reference_and_depth_pixmaps_are_scaled_and_noninteractive(self):
        source = QPixmap(5, 4)
        source.fill(QColor("#112233"))
        depth = self.presenter.add_depth_reference(
            source,
            layer_id=LAYER_DEPTH,
            image_width=20,
            image_height=16,
        )
        display = self.presenter.add_depth_display(
            source,
            image_width=20,
            image_height=16,
            overlay=True,
        )
        segmentation = self.presenter.add_segmentation_reference(
            1,
            [(2, 3), (10, 3), (6, 12)],
            label_text="tail",
            layer_id=LAYER_SEGMENTATION,
        )

        self.assertEqual((depth.pixmap().width(), depth.pixmap().height()), (20, 16))
        self.assertEqual(depth.opacity(), 0.42)
        self.assertEqual(depth.zValue(), 0.5)
        self.assertEqual(depth.reference_layer_id, LAYER_DEPTH)
        self.assertEqual(depth.acceptedMouseButtons(), Qt.MouseButton.NoButton)
        self.assertEqual(display.opacity(), 0.55)
        self.assertEqual(display.zValue(), 1.0)
        self.assertEqual(segmentation.opacity(), 0.50)
        self.assertFalse(segmentation.seg_label_item.isVisible())
        self.assertIsInstance(segmentation.seg_label_item, QGraphicsSimpleTextItem)

        references = list(self.presenter.reference_items)
        self.presenter.clear_references()
        self.assertTrue(all(item.scene() is None for item in references))
        self.assertIs(display.scene(), self.scene)

    def test_pose_reference_uses_existing_graphics_items_and_explicit_label_text(self):
        box, keypoints = self.presenter.add_pose_reference(
            BoundingBox(2, 3, 30, 20, 0),
            [
                PoseReferenceKeypoint(
                    Keypoint(8, 9, 0, "nose"),
                    visibility=2,
                    label_text="nose · 1.234 m",
                )
            ],
            class_name="mouse",
            layer_id=LAYER_KEYPOINTS,
            keypoint_radius=4,
            keypoint_font_px=10,
            show_keypoint_labels=True,
        )

        self.assertEqual(box.reference_layer_id, LAYER_KEYPOINTS)
        self.assertEqual(box.opacity(), 0.52)
        self.assertFalse(box.flags() & box.GraphicsItemFlag.ItemIsMovable)
        self.assertEqual(len(keypoints), 1)
        self.assertEqual(keypoints[0].text_item.text(), "nose · 1.234 m")
        self.assertEqual(keypoints[0].opacity(), 0.90)
        self.assertTrue(keypoints[0].text_item.isVisible())

    def test_external_scene_clear_can_reset_owned_item_lists_without_touching_wrappers(self):
        self.presenter.add_prompt_marker(1, 2, positive=True)
        self.presenter.render_depth_probes([{"x": 2, "y": 3, "depth": 1.0}])
        self.scene.clear()

        self.presenter.forget_scene_items()

        self.assertEqual(self.presenter.prompt_items, [])
        self.assertEqual(self.presenter.depth_probe_items, [])
        self.assertEqual(self.presenter.reference_items, [])


if __name__ == "__main__":
    unittest.main()
