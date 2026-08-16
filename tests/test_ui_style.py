import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QFrame, QGraphicsDropShadowEffect

import ui_style
from squeakpose.ui import style


class UiStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["squeakpose-style-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_root_compatibility_exports_preserve_identity(self):
        public_names = (
            "COLORS",
            "ThemedComboBox",
            "analysis_dialog_stylesheet",
            "app_stylesheet",
            "apply_panel_shadow",
            "combo_popup_stylesheet",
            "hud_stylesheet",
            "launcher_stylesheet",
            "sidebar_stylesheet",
            "style_combo_popup",
            "train_dialog_stylesheet",
        )

        for name in public_names:
            with self.subTest(name=name):
                self.assertIs(getattr(ui_style, name), getattr(style, name))

    def test_stylesheet_helpers_keep_expected_sections(self):
        self.assertIn("QMainWindow, QDialog", style.app_stylesheet("Arial", "Helvetica"))
        self.assertIn("PrimaryLauncherButton", style.launcher_stylesheet())
        self.assertIn("SidebarContent", style.sidebar_stylesheet())
        self.assertIn("TrainStatusLabel", style.train_dialog_stylesheet())
        self.assertIn("AnalysisPanel", style.analysis_dialog_stylesheet())
        self.assertIn(style.COLORS["input_bg"], style.combo_popup_stylesheet())

    def test_widget_helpers_apply_theme_without_showing_widgets(self):
        panel = QFrame()
        style.apply_panel_shadow(panel)
        self.assertIsInstance(panel.graphicsEffect(), QGraphicsDropShadowEffect)

        combo = style.ThemedComboBox()
        self.assertIn("combobox-popup", combo.styleSheet())
        self.assertEqual(combo.view().objectName(), "ComboPopup")
        self.assertIn(style.COLORS["input_bg"], combo.view().styleSheet())


if __name__ == "__main__":
    unittest.main()
