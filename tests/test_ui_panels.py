import importlib
import os
import unittest
from tempfile import gettempdir

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

QApplication = importlib.import_module("PyQt6.QtWidgets").QApplication
annotation_panel = importlib.import_module("squeakpose.ui.annotation_panel")
operation_panel = importlib.import_module("squeakpose.ui.operation_panel")
layers = importlib.import_module("squeakpose.project.layers")


class UiPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["squeakpose-panel-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_annotation_panel_is_constructible_and_routes_explicit_callbacks(self):
        events = []
        panel = annotation_panel.AnnotationPanel(
            ["mouse", "rat"],
            callbacks=annotation_panel.AnnotationPanelCallbacks(
                mode_changed=lambda mode: events.append(("mode", mode)),
                class_changed=lambda class_id: events.append(("class", class_id)),
                manage_classes=lambda: events.append(("manage", None)),
                use_segmentation_box=lambda: events.append(("seg_box", None)),
            ),
        )
        panel.show()
        self.app.processEvents()

        self.assertEqual(panel.objectName(), "ToolPanel")
        self.assertEqual(panel.class_selector.currentText(), "mouse")
        self.assertFalse(panel.bbox_btn.isHidden())
        self.assertTrue(panel.segment_btn.isHidden())

        panel.bbox_btn.click()
        panel.class_selector.setCurrentIndex(1)
        panel.manage_classes_btn.click()
        panel.set_segmentation_box_available(True)
        panel.use_segmentation_box_btn.click()
        panel.set_progress("3 / 12 labeled")

        self.assertEqual(
            events,
            [
                ("mode", "bbox"),
                ("class", 1),
                ("manage", None),
                ("seg_box", None),
            ],
        )
        self.assertEqual(panel.active_mode, "bbox")
        self.assertEqual(panel.progress_label.text(), "3 / 12 labeled")

        panel.set_layer(layers.LAYER_SEGMENTATION)
        self.assertTrue(panel.bbox_btn.isHidden())
        self.assertTrue(panel.use_segmentation_box_btn.isHidden())
        self.assertFalse(panel.segment_btn.isHidden())
        self.assertEqual(panel.panzoom_btn.text(), "Pan (1)")
        self.assertIn("Segmentation", panel.predict_btn.toolTip())

        panel.set_layer(layers.LAYER_DEPTH)
        self.assertTrue(panel.class_controls_frame.isHidden())
        self.assertFalse(panel.predict_btn.isHidden())
        self.assertIn("depth map", panel.predict_btn.toolTip())
        panel.close()

    def test_segmentation_tools_panel_exposes_state_without_model_dependency(self):
        events = []
        panel = annotation_panel.SegmentationToolsPanel(
            callbacks=annotation_panel.SegmentationToolsCallbacks(
                load_model=lambda: events.append("load"),
                download_model=lambda: events.append("download"),
                run=lambda: events.append("run"),
                accept=lambda: events.append("accept"),
                reset=lambda: events.append("reset"),
            ),
            brush_radius=12,
        )

        self.assertEqual(panel.brush_size_label.text(), "Brush: 12px")
        self.assertTrue(panel.load_btn.isEnabled())
        self.assertFalse(panel.run_btn.isEnabled())
        self.assertFalse(panel.accept_btn.isEnabled())
        panel.load_btn.click()
        panel.download_btn.click()

        panel.set_state(model_loaded=True, prompt_count=2, has_preview=False)
        self.assertTrue(panel.run_btn.isEnabled())
        self.assertTrue(panel.reset_btn.isEnabled())
        panel.run_btn.click()

        panel.set_state(model_loaded=True, prompt_count=2, has_preview=True)
        self.assertTrue(panel.accept_btn.isEnabled())
        panel.accept_btn.click()
        panel.reset_btn.click()
        self.assertEqual(events, ["load", "download", "run", "accept", "reset"])
        panel.close()

    def test_operation_panels_preserve_layer_aware_controls_and_callbacks(self):
        events = []
        callbacks = operation_panel.OperationCallbacks(
            video_review=lambda: events.append("video"),
            analysis=lambda: events.append("analysis"),
            validate_labels=lambda: events.append("validate"),
            export_dataset=lambda: events.append("export"),
            project_health=lambda: events.append("health"),
            train=lambda: events.append("train"),
            distill=lambda: events.append("distill"),
            project_models=lambda: events.append("models"),
            inference=lambda: events.append("inference"),
            apply_template=lambda: events.append("apply"),
            save_template=lambda: events.append("save"),
        )
        video = operation_panel.VideoOperationsPanel(callbacks.video_review)
        analysis = operation_panel.AnalysisOperationsPanel(callbacks.analysis)
        dataset = operation_panel.DatasetOperationsPanel(callbacks)
        models = operation_panel.ModelOperationsPanel(callbacks)
        for panel in (video, analysis, dataset, models):
            panel.show()
        self.app.processEvents()

        video.review_btn.click()
        analysis.analysis_btn.click()
        dataset.validate_btn.click()
        dataset.health_btn.click()
        models.models_btn.click()
        models.inference_btn.click()
        self.assertEqual(
            events,
            ["video", "analysis", "validate", "health", "models", "inference"],
        )

        dataset.set_layer(layers.LAYER_SEGMENTATION)
        self.assertIn("Segmentation", dataset.title_label.text())
        self.assertFalse(dataset.validate_btn.isHidden())
        self.assertFalse(dataset.distillation_btn.isEnabled())
        models.set_layer(layers.LAYER_SEGMENTATION)
        self.assertTrue(models.apply_template_btn.isHidden())
        self.assertFalse(models.inference_btn.isHidden())

        dataset.set_layer(layers.LAYER_DEPTH)
        analysis.set_layer(layers.LAYER_DEPTH)
        models.set_layer(layers.LAYER_DEPTH)
        self.assertEqual(dataset.title_label.text(), "Project Tools")
        self.assertTrue(dataset.validate_btn.isHidden())
        self.assertFalse(dataset.health_btn.isHidden())
        self.assertTrue(analysis.isHidden())
        self.assertEqual(models.title_label.text(), "Project Inference")
        self.assertTrue(models.models_btn.isHidden())
        self.assertEqual(models.inference_btn.text(), "Run Inference")
        for panel in (video, analysis, dataset, models):
            panel.close()


if __name__ == "__main__":
    unittest.main()
