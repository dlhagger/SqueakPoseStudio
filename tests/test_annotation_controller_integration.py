import importlib
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

qt_core = importlib.import_module("PyQt6.QtCore")
qt_gui = importlib.import_module("PyQt6.QtGui")
qt_widgets = importlib.import_module("PyQt6.QtWidgets")
studio = importlib.import_module("squeakpose_studio")
layers = importlib.import_module("squeakpose.project.layers")
depth_controller = importlib.import_module("squeakpose.ui.depth_controller")
annotation_panel = importlib.import_module("squeakpose.ui.annotation_panel")
canvas_hud = importlib.import_module("squeakpose.ui.canvas_hud")
canvas_scene = importlib.import_module("squeakpose.ui.canvas_scene_presenter")
depth_panel = importlib.import_module("squeakpose.ui.depth_panel")
depth_presentation = importlib.import_module("squeakpose.ui.depth_presentation")
navigation_panel = importlib.import_module("squeakpose.ui.navigation_panel")
operation_panel = importlib.import_module("squeakpose.ui.operation_panel")
pose_controller = importlib.import_module("squeakpose.ui.pose_controller")
segmentation_controller = importlib.import_module("squeakpose.ui.segmentation_controller")
segmentation_assistant = importlib.import_module("squeakpose.annotation.segmentation_assistant")
sam_service = importlib.import_module("squeakpose.services.sam_assistant")
numpy = importlib.import_module("numpy")


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class _SamAssistantController:
    def __init__(self, _parent, **_kwargs):
        self.status_changed = _Signal()
        self.busy_changed = _Signal()
        self.event_received = _Signal()
        self.decision_ready = _Signal()
        self.terminal = _Signal()
        self.is_busy = False
        self.session = None
        self.restart_requests = []
        self.prompt_requests = []
        self.shutdown_calls = 0

    def restart_model(self, **kwargs):
        self.restart_requests.append(kwargs)
        return False

    def submit_prompt(self, **kwargs):
        self.prompt_requests.append(kwargs)
        self.is_busy = True
        self.busy_changed.emit(True)
        return len(self.prompt_requests)

    def cancel(self):
        self.is_busy = False
        self.busy_changed.emit(False)
        return True

    def shutdown(self):
        self.shutdown_calls += 1
        return True

    def complete(self, decision):
        self.is_busy = False
        self.busy_changed.emit(False)
        self.decision_ready.emit(decision)


class AnnotationControllerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["annotation-controller-integration-test"]
        )
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def test_window_composes_and_rebinds_annotation_controllers(self):
        with TemporaryDirectory() as tmp:
            paths = studio._ensure_project_structure(tmp)
            Path(paths["classes_file"]).write_text("mouse\n", encoding="utf-8")
            Path(paths["keypoints_file"]).write_text("nose\n", encoding="utf-8")
            Path(paths["class_keypoints_file"]).write_text(
                '{"mouse": ["nose"]}\n',
                encoding="utf-8",
            )
            Path(paths["classes_seg_file"]).write_text("body\n", encoding="utf-8")
            image = qt_gui.QImage(20, 16, qt_gui.QImage.Format.Format_RGB32)
            image.fill(0xFFE7EBEF)
            self.assertTrue(image.save(str(Path(paths["images_to_label"]) / "frame.png")))
            Path(tmp, "project-sam3-last.pth").touch()
            Path(tmp, "sam3.pt").touch()

            window = None
            with ExitStack() as stack:
                stack.enter_context(
                    patch("squeakpose.ui.main_window._auto_device", return_value="cpu")
                )
                stack.enter_context(
                    patch("squeakpose.ui.main_window.LabelingApp._restart_prediction_worker")
                )
                stack.enter_context(
                    patch(
                        "squeakpose.ui.main_window.SamAssistantController",
                        _SamAssistantController,
                    )
                )
                warning = stack.enter_context(
                    patch("squeakpose.ui.main_window.QMessageBox.warning")
                )
                information = stack.enter_context(
                    patch("squeakpose.ui.main_window.QMessageBox.information")
                )
                try:
                    window = studio.LabelingApp(
                        paths["images_to_label"],
                        paths["labels_all"],
                        paths["classes_file"],
                        paths["keypoints_file"],
                        project_root=paths["root"],
                        force_initial_setup=False,
                    )
                    window._jump_to_next_pending_class = lambda: None

                    self.assertEqual(
                        [
                            os.path.basename(path)
                            for path in window._sam3_model_candidates_in_project_root()
                        ],
                        ["sam3.pt", "project-sam3-last.pth"],
                    )

                    self.assertIsInstance(
                        window.annotation_panel,
                        annotation_panel.AnnotationPanel,
                    )
                    self.assertIsInstance(
                        window._scene_presenter,
                        canvas_scene.CanvasScenePresenter,
                    )
                    self.assertIsInstance(
                        window._depth_preview_presenter,
                        depth_presentation.DepthPreviewPresenter,
                    )
                    self.assertIs(window.class_selector, window.annotation_panel.class_selector)
                    self.assertIs(window.predict_btn, window.annotation_panel.predict_btn)
                    self.assertIs(window.progress_label, window.annotation_panel.progress_label)
                    self.assertIsInstance(
                        window.navigation_panel,
                        navigation_panel.NavigationPanel,
                    )
                    self.assertIs(
                        window.workflow_selector,
                        window.navigation_panel.layer_selector,
                    )
                    self.assertIs(window.save_btn, window.navigation_panel.save_btn)
                    self.assertIs(
                        window.keypoints_visibility_check,
                        window.navigation_panel.keypoints_visibility_btn,
                    )
                    self.assertIsInstance(
                        window.depth_display_frame,
                        depth_panel.DepthDisplayPanel,
                    )
                    self.assertIs(
                        window.depth_display_combo,
                        window.depth_display_frame.mode_combo,
                    )
                    self.assertIsInstance(
                        window.depth_range_frame,
                        depth_panel.DepthRangePanel,
                    )
                    self.assertIsInstance(
                        window.depth_assistant_frame,
                        depth_panel.DepthModelPanel,
                    )
                    self.assertIsInstance(
                        window.layer_context_frame,
                        canvas_hud.LayerContextHud,
                    )
                    canvas_presentation = importlib.import_module(
                        "squeakpose.ui.canvas_presentation"
                    )
                    self.assertIsInstance(
                        window.canvas_hud_presenter,
                        canvas_presentation.CanvasHudPresenter,
                    )
                    self.assertIs(
                        window.layer_context_frame,
                        window.canvas_hud_presenter.layer_context,
                    )
                    self.assertIs(window.legend_frame, window.canvas_hud_presenter.legend)
                    self.assertIs(window.zoom_frame, window.canvas_hud_presenter.zoom)
                    self.assertIs(
                        window.layer_editing_label,
                        window.layer_context_frame.editing_label,
                    )
                    window.set_mode("panzoom")
                    self.assertFalse(window.zoom_frame.isHidden())
                    self.assertTrue(window.legend_frame.isHidden())
                    window.view.scale(1.25, 1.25)
                    window.update_zoom_label()
                    self.assertEqual(window.zoom_label.text(), "Zoom: 125%")
                    window.set_mode("keypoint")
                    self.assertFalse(window.legend_frame.isHidden())
                    self.assertTrue(window.zoom_frame.isHidden())
                    window._layout_hot_corners()
                    window._layout_overlays()
                    self.assertEqual(window.layer_context_frame.pos(), qt_core.QPoint(10, 10))
                    self.assertEqual(window.legend_frame.pos().x(), 10)
                    self.assertIsInstance(
                        window.seg_tools_frame,
                        annotation_panel.SegmentationToolsPanel,
                    )
                    self.assertIs(window.sam_run_btn, window.seg_tools_frame.run_btn)
                    self.assertIsInstance(
                        window.top_right_frame,
                        operation_panel.VideoOperationsPanel,
                    )
                    self.assertIsInstance(
                        window.bottom_right_frame,
                        operation_panel.ModelOperationsPanel,
                    )
                    self.assertIsInstance(
                        window.bottom_left_frame,
                        operation_panel.DatasetOperationsPanel,
                    )
                    self.assertIsInstance(
                        window.analysis_frame,
                        operation_panel.AnalysisOperationsPanel,
                    )
                    self.assertIs(
                        window.right_sidebar_layout.itemAt(0).widget(),
                        window.top_right_frame,
                    )
                    self.assertIs(
                        window.right_sidebar_layout.itemAt(1).widget(),
                        window.bottom_right_frame,
                    )
                    self.assertIs(
                        window.right_sidebar_layout.itemAt(2).widget(),
                        window.bottom_left_frame,
                    )

                    self.assertIsInstance(
                        window._pose_controller,
                        pose_controller.PoseAnnotationController,
                    )
                    self.assertIs(window._pose_controller.document, window.annotation_cache)
                    self.assertIs(window._pose_controller.state, window.pose_edit_state)
                    window.add_bbox(qt_core.QRectF(2, 3, 12, 9))
                    window.add_keypoint(qt_core.QPointF(5, 6))
                    self.assertTrue(window.annotation_cache.is_complete(0))

                    window._switch_layer(layers.LAYER_SEGMENTATION)
                    self.assertEqual(
                        window.annotation_panel.active_layer,
                        layers.LAYER_SEGMENTATION,
                    )
                    self.assertEqual(
                        window.navigation_panel.layer_selector.currentData(),
                        layers.LAYER_SEGMENTATION,
                    )
                    self.assertFalse(window.segment_btn.isHidden())
                    self.assertTrue(window.bbox_btn.isHidden())
                    self.assertIsInstance(
                        window._segmentation_controller,
                        segmentation_controller.SegmentationAnnotationController,
                    )
                    self.assertIs(
                        window._segmentation_controller.document,
                        window.annotation_cache,
                    )
                    self.assertIs(window._segmentation_controller.state, window.seg_edit_state)
                    window._set_segmentation_cache_entry(
                        0,
                        {
                            "class_id": 0,
                            "segments": [(1, 1), (10, 1), (5, 10)],
                            "score": 0.7,
                        },
                    )
                    self.assertTrue(window.annotation_cache.is_complete(0))
                    self.assertEqual(
                        window._segmentation_controller.state.accepted_masks[0]["score"],
                        0.7,
                    )

                    sam_controller = window._sam_assistant_controller
                    self.assertEqual(sam_controller.restart_requests[-1]["device"], "cpu")
                    sam_controller.event_received.emit(
                        {
                            "event": "loaded",
                            "model_path": window._sam_worker_model_path,
                        }
                    )
                    window._segmentation_controller.select_target(0)
                    window._segmentation_controller.add_prompt(4, 5, positive=True)
                    window._run_sam_segmentation()
                    submitted = sam_controller.prompt_requests[-1]
                    self.assertEqual(submitted["model_path"], window.sam_model_path)
                    self.assertEqual(submitted["device"], "cpu")
                    self.assertEqual(
                        submitted["prompt"].predict_kwargs(),
                        {
                            "source": window.current_image_path,
                            "points": [[4.0, 5.0]],
                            "labels": [1],
                            "verbose": False,
                        },
                    )
                    sam_controller.complete(
                        sam_service.SamAssistantDecision(
                            "apply",
                            request_id=1,
                            result=segmentation_assistant.SamContourResult(
                                ((1.0, 1.0), (12.0, 1.0), (6.0, 12.0)),
                                0.88,
                            ),
                        )
                    )
                    self.assertEqual(window.seg_preview_score, 0.88)
                    self.assertEqual(len(window.seg_preview_points), 3)
                    information.reset_mock()
                    window._sam_request_class_id = 0
                    window._handle_sam_worker_decision(
                        sam_service.SamAssistantDecision(
                            "apply",
                            request_id=2,
                            failure="no_masks",
                        )
                    )
                    self.assertEqual(information.call_args.args[1], "No masks")
                    warning.reset_mock()
                    window._sam_request_class_id = 0
                    window._handle_sam_worker_decision(
                        sam_service.SamAssistantDecision(
                            "error",
                            request_id=3,
                            error_message="worker failure",
                        )
                    )
                    self.assertEqual(warning.call_args.args[1], "SAM inference error")
                    self.assertIn("worker failure", warning.call_args.args[2])

                    analysis_plan = SimpleNamespace(
                        project_root="/planned/project",
                        app_base_dir="/planned/app",
                        layer_id=layers.LAYER_SEGMENTATION,
                    )
                    with (
                        patch(
                            "squeakpose.ui.main_window.plan_analysis_dialog",
                            return_value=analysis_plan,
                        ) as planner,
                        patch("squeakpose.ui.main_window.AnalysisDialog") as dialog_type,
                    ):
                        window.open_analysis_dialog()
                    planner.assert_called_once_with(
                        project_root=window.project_root,
                        app_base_dir=window.app_base_dir,
                        layer_id=layers.LAYER_SEGMENTATION,
                    )
                    dialog_type.assert_called_once_with(
                        window,
                        project_root="/planned/project",
                        app_base_dir="/planned/app",
                        layer_id=layers.LAYER_SEGMENTATION,
                    )
                    dialog_type.return_value.exec.assert_not_called()
                    dialog_type.return_value.show.assert_called_once_with()
                    self.assertTrue(window._analysis_read_only)
                    self.assertFalse(window.annotation_panel.isEnabled())
                    self.assertFalse(window.save_btn.isEnabled())
                    self.assertFalse(window.open_project_action.isEnabled())
                    self.assertTrue(window.view.isEnabled())
                    window._analysis_dialog = None
                    window._set_analysis_read_only(False)
                    self.assertFalse(window._analysis_read_only)
                    self.assertTrue(window.annotation_panel.isEnabled())
                    self.assertTrue(window.save_btn.isEnabled())
                    self.assertTrue(window.open_project_action.isEnabled())

                    training_plan = SimpleNamespace(
                        default_dataset="/planned/dataset",
                        default_task="segment",
                        layer_id=layers.LAYER_SEGMENTATION,
                    )
                    with (
                        patch(
                            "squeakpose.ui.main_window.plan_training_dialog",
                            return_value=training_plan,
                        ) as planner,
                        patch("squeakpose.ui.main_window.TrainDialog") as dialog_type,
                    ):
                        window.open_train_dialog()
                    planner.assert_called_once_with(
                        project_root=window.project_root,
                        layer_id=layers.LAYER_SEGMENTATION,
                    )
                    dialog_type.assert_called_once_with(
                        window,
                        default_dataset="/planned/dataset",
                        default_task="segment",
                        layer_id=layers.LAYER_SEGMENTATION,
                    )

                    video_plan = SimpleNamespace(
                        active_schema={
                            "kp_names": [],
                            "classes": ["body"],
                            "class_keypoints": {},
                        },
                        workflow="segmentation",
                        layer_id=layers.LAYER_SEGMENTATION,
                        model_paths={layers.LAYER_SEGMENTATION: "segment.pt"},
                        layer_schemas={layers.LAYER_SEGMENTATION: {}},
                    )
                    with (
                        patch("squeakpose.ui.main_window._cv2", object()),
                        patch(
                            "squeakpose.ui.main_window.plan_video_review_dialog",
                            return_value=video_plan,
                        ) as planner,
                        patch("squeakpose.ui.main_window.VideoReviewDialog") as dialog_type,
                    ):
                        window.open_video_reviewer()
                    planner.assert_called_once()
                    dialog_type.assert_called_once_with(
                        window,
                        "cpu",
                        [],
                        ["body"],
                        class_keypoints={},
                        workflow="segmentation",
                        layer_id=layers.LAYER_SEGMENTATION,
                        model_paths=video_plan.model_paths,
                        layer_schemas=video_plan.layer_schemas,
                    )

                    depth_map_path = Path(window.depth_image_dir, "frame.npy")
                    numpy.save(depth_map_path, numpy.full((16, 20), 1.25))
                    Path(window.depth_image_dir, "frame_depth.json").write_text(
                        '{"p02_depth": 1.0, "p98_depth": 2.0, '
                        '"median_depth": 1.25, "units": "estimated_meters"}',
                        encoding="utf-8",
                    )
                    depth_preview = qt_gui.QImage(
                        20,
                        16,
                        qt_gui.QImage.Format.Format_RGB32,
                    )
                    depth_preview.fill(0xFF333333)
                    self.assertTrue(
                        depth_preview.save(str(Path(window.depth_preview_dir, "frame_depth.png")))
                    )

                    window._switch_layer(layers.LAYER_DEPTH)
                    self.assertTrue(window.class_controls_frame.isHidden())
                    self.assertFalse(window.save_btn.isEnabled())
                    self.assertFalse(window.complete_btn.isEnabled())
                    self.assertEqual(window.bottom_left_frame.title_label.text(), "Project Tools")
                    self.assertTrue(window.analysis_frame.isHidden())
                    self.assertTrue(window.load_model_btn.isHidden())
                    self.assertIsInstance(
                        window._depth_controller,
                        depth_controller.DepthAssistantController,
                    )
                    self.assertEqual(window._depth_controller.depth_map.shape, (16, 20))
                    self.assertEqual(
                        window._depth_controller.state.metadata["median_depth"],
                        1.25,
                    )
                    self.assertIn("1.000–2.000 m", window.depth_range_label.text())
                    window._depth_controller.load_image(
                        "frame.png",
                        depth_map=[[1.25]],
                    )
                    self.assertTrue(window._probe_depth_at(qt_core.QPointF(0, 0)))
                    self.assertEqual(window._depth_probes[0]["depth"], 1.25)
                    self.assertEqual(window._depth_controller.state.probes[0].depth, 1.25)
                    self.assertEqual(len(window._depth_preview_presenter.probe_items), 2)
                    self.assertIsNotNone(window._depth_preview_presenter.preview_item)
                finally:
                    if window is not None:
                        window.close()
                        self.qt_app.processEvents()


if __name__ == "__main__":
    unittest.main()
