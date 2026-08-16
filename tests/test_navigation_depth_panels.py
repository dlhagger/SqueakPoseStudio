import importlib
import os
import unittest
from tempfile import gettempdir

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

QApplication = importlib.import_module("PyQt6.QtWidgets").QApplication
canvas_hud = importlib.import_module("squeakpose.ui.canvas_hud")
depth_panel = importlib.import_module("squeakpose.ui.depth_panel")
layers = importlib.import_module("squeakpose.project.layers")
navigation_panel = importlib.import_module("squeakpose.ui.navigation_panel")


class NavigationDepthPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["navigation-depth-panel-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_navigation_panel_routes_actions_and_layer_state(self):
        events = []
        callbacks = navigation_panel.NavigationPanelCallbacks(
            filter_changed=lambda value: events.append(("filter", value)),
            layer_changed=lambda value: events.append(("layer", value)),
            visibility_changed=lambda layer, visible: events.append(("visibility", layer, visible)),
            previous=lambda: events.append(("previous",)),
            next=lambda: events.append(("next",)),
            complete=lambda: events.append(("complete",)),
            skip=lambda: events.append(("skip",)),
            save=lambda: events.append(("save",)),
            delete_image=lambda: events.append(("delete",)),
        )
        panel = navigation_panel.NavigationPanel(
            active_filter="unlabeled",
            layer_visibility={layers.LAYER_SEGMENTATION: False},
            callbacks=callbacks,
        )
        panel.show()
        self.app.processEvents()

        self.assertEqual(panel.filter_combo.currentText(), "Unlabeled")
        self.assertFalse(panel.segmentation_visibility_btn.isChecked())
        self.assertIs(panel.nav_grid.itemAtPosition(0, 0).widget(), panel.previous_btn)
        self.assertIs(panel.nav_grid.itemAtPosition(1, 2).widget(), panel.delete_image_btn)
        panel.filter_combo.setCurrentIndex(1)
        panel.layer_selector.setCurrentIndex(
            panel.layer_selector.findData(layers.LAYER_SEGMENTATION)
        )
        panel.segmentation_visibility_btn.setChecked(True)
        panel.previous_btn.click()
        panel.next_btn.click()
        panel.complete_btn.click()
        panel.skip_btn.click()
        panel.save_btn.click()
        panel.delete_image_btn.click()

        self.assertEqual(
            events,
            [
                ("filter", "labeled"),
                ("layer", layers.LAYER_SEGMENTATION),
                ("visibility", layers.LAYER_SEGMENTATION, True),
                ("previous",),
                ("next",),
                ("complete",),
                ("skip",),
                ("save",),
                ("delete",),
            ],
        )

        panel.set_active_layer(layers.LAYER_DEPTH)
        self.assertFalse(panel.save_btn.isEnabled())
        self.assertFalse(panel.complete_btn.isEnabled())
        self.assertTrue(panel.depth_visibility_btn.property("activeLayer"))
        panel.close()

    def test_depth_panels_preserve_controls_and_callbacks(self):
        events = []
        display_callbacks = depth_panel.DepthDisplayCallbacks(
            mode_changed=lambda mode: events.append(("mode", mode)),
            clear_probes=lambda: events.append(("clear_probes",)),
        )
        display = depth_panel.DepthDisplayPanel(
            mode="overlay",
            callbacks=display_callbacks,
        )
        depth_range = depth_panel.DepthRangePanel(callbacks=display_callbacks)
        model = depth_panel.DepthModelPanel(
            depth_panel.DepthModelCallbacks(
                select_model=lambda path: events.append(("model", path)),
                choose_model=lambda: events.append(("choose",)),
            )
        )

        self.assertEqual(display.mode_combo.currentData(), "overlay")
        display.mode_combo.setCurrentIndex(display.mode_combo.findData("original"))
        depth_range.set_range_text("Range text")
        depth_range.set_probe_text("Probe text", can_clear=True)
        depth_range.clear_btn.click()
        model.official_model_btn.menu().actions()[0].trigger()
        model.choose_model_btn.click()
        model.clear_model_btn.click()
        model.set_model_status("Custom model", tooltip="/model.pt", can_clear=True)

        self.assertEqual(depth_range.range_label.text(), "Range text")
        self.assertEqual(depth_range.probe_label.text(), "Probe text")
        self.assertEqual(model.status_label.toolTip(), "/model.pt")
        self.assertEqual(
            events,
            [
                ("mode", "original"),
                ("clear_probes",),
                ("model", "yolo26n-depth.pt"),
                ("choose",),
                ("model", ""),
            ],
        )
        for panel in (display, depth_range, model):
            panel.close()

    def test_layer_context_hud_owns_only_text_presentation(self):
        hud = canvas_hud.LayerContextHud()
        hud.show()
        hud.set_context(editing="Editing: Segmentation", references="References: Keypoints")
        self.app.processEvents()

        self.assertEqual(hud.editing_label.text(), "Editing: Segmentation")
        self.assertEqual(hud.reference_label.text(), "References: Keypoints")
        self.assertFalse(hud.reference_label.isHidden())
        hud.set_context(editing="Editing: Depth")
        self.assertTrue(hud.reference_label.isHidden())
        self.assertFalse(hasattr(hud, "scene"))
        hud.close()


if __name__ == "__main__":
    unittest.main()
