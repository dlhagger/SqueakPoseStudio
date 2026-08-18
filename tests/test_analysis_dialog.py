import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QApplication

from analysis_dialog import FrameAnnotationView


class FrameAnnotationViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["analysis-dialog-test"])

    def setUp(self):
        self.view = FrameAnnotationView()
        self.view.resize(600, 400)
        frame = QPixmap(1000, 500)
        frame.fill(QColor("#000000"))
        self.view.set_frame(frame, 1000, 500)

    def tearDown(self):
        self.view.close()

    def test_zoom_keeps_cursor_anchored_in_image_coordinates(self):
        anchor = QPointF(450, 200)
        before = self.view._widget_to_image(anchor)

        self.view.set_zoom(2.0, anchor)
        after = self.view._widget_to_image(anchor)

        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(after[0], before[0])
        self.assertAlmostEqual(after[1], before[1])

    def test_segmentation_polygon_is_painted_over_frame(self):
        self.view.set_segmentation_polygons(
            [[(400.0, 150.0), (600.0, 150.0), (600.0, 350.0), (400.0, 350.0)]]
        )

        image = self.view.grab().toImage()
        center = image.pixelColor(300, 200)

        self.assertGreater(center.green(), 20)
        self.assertGreater(center.blue(), 20)


if __name__ == "__main__":
    unittest.main()
