"""Exercise actual Qt labeling paths without model weights or source project edits."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests import test_pose_state_integration as pose_fixture

qt_core = pose_fixture.qt_core
qt_widgets = pose_fixture.qt_widgets


class MultiAnimalUITests(unittest.TestCase):
    _open_window = pose_fixture.PoseStateIntegrationTests._open_window
    _complete_active_pose = pose_fixture.PoseStateIntegrationTests._complete_active_pose

    @classmethod
    def setUpClass(cls):
        cls.qt_app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(["multi-animal"])
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def test_pose_add_select_edit_save_reload_delete_preserves_both_animals(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self.assertFalse(window._multi_animal_enabled())
                    self._complete_active_pose(window)
                    window.annotation_panel.multi_animal_check.setChecked(True)
                    window._add_animal_instance()
                    self.assertIsNone(window.current_box)
                    window.add_bbox(qt_core.QRectF(22, 3, 12, 15))
                    window.add_keypoint(qt_core.QPointF(24, 5))
                    window.add_keypoint(qt_core.QPointF(28, 14))
                    window.pose_edit_state.push_undo_snapshot()
                    window.pose_edit_state.set_visibility("nose", 0)
                    window._store_pose_edit_state()
                    window.undo()
                    self.assertEqual(window.pose_edit_state.keypoints["nose"].visibility, 2)
                    self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                    self.assertTrue(window._is_fully_labeled())
                    self.assertTrue(window.save_labels())
                    text = Path(window.label_dir, "frame001.txt").read_text()
                    self.assertEqual(len(text.splitlines()), 2)
                    window.load_image()
                    self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                    self.assertEqual(window._class_box_item(0)._label.text(), "mouse")
                    labels = [
                        item.text()
                        for item in window.scene.items()
                        if isinstance(item, qt_widgets.QGraphicsSimpleTextItem)
                    ]
                    self.assertEqual(labels.count("mouse"), 2)
                    self.assertNotIn("mouse 1", labels)
                    self.assertNotIn("mouse 2", labels)
                    self.assertEqual(
                        window.annotation_panel.instance_selector.itemText(0), "mouse 1"
                    )
                    self.assertEqual(
                        window.annotation_panel.instance_selector.itemText(1), "mouse 2"
                    )
                    reference_points = [
                        item
                        for item in window.scene.items()
                        if isinstance(item, pose_fixture.graphics.KeypointItem)
                        and getattr(item, "reference_layer_id", "") == "annotation_instance"
                    ]
                    self.assertEqual(
                        {item.text_item.text() for item in reference_points}, {"nose", "tail"}
                    )
                    self.assertTrue(all(item.text_item.isVisible() for item in reference_points))
                    window._bump_kp_font(2)
                    self.assertTrue(
                        all(
                            item.text_item.font().pixelSize() == window.kp_font_px
                            for item in reference_points
                        )
                    )
                    window._toggle_kp_labels()
                    self.assertTrue(
                        all(not item.text_item.isVisible() for item in reference_points)
                    )
                    window._toggle_kp_labels()
                    window.annotation_panel.instance_selector.setCurrentIndex(0)
                    self.assertEqual(window._class_box_item(0)._label.text(), "mouse")
                    self.assertAlmostEqual(window.current_box.x, 4)
                    self.assertEqual(len(window.current_kps), 2)
                    self.assertTrue(window.save_labels())
                    self.assertEqual(Path(window.label_dir, "frame001.txt").read_text(), text)
                    with patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=qt_widgets.QMessageBox.StandardButton.Yes,
                    ):
                        window._delete_animal_instance()
                    self.assertEqual(len(window.annotation_cache.all_entries()), 1)
                    self.assertTrue(window.save_labels())
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_segmentation_add_accept_select_save_reload(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    window._switch_layer("segmentation")
                    window.annotation_panel.multi_animal_check.setChecked(True)
                    for index, polygon in enumerate(
                        ([(2, 2), (12, 2), (2, 12)], [(20, 2), (30, 2), (20, 12)])
                    ):
                        if index:
                            window._add_animal_instance()
                        window.seg_edit_state.set_preview(polygon, 0.9)
                        window._accept_segmentation_preview()
                    self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                    self.assertTrue(window.save_labels())
                    window.load_image()
                    window.annotation_panel.instance_selector.setCurrentIndex(0)
                    self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                    self.assertTrue(window.save_labels())
                    self.assertEqual(
                        len(Path(window.label_dir, "frame001.txt").read_text().splitlines()), 2
                    )
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_multi_animal_prediction_applies_both_pose_and_segmentation_instances(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    window.annotation_panel.multi_animal_check.setChecked(True)
                    for layer in ("keypoints", "segmentation"):
                        if layer != window.active_layer:
                            window._switch_layer(layer)
                        geometry = (
                            {"xyxy": [2, 3, 12, 18], "keypoints": [[4, 5, 0.9], [8, 12, 0.9]]}
                            if layer == "keypoints"
                            else {"segments": [[2, 3], [12, 3], [2, 18]]}
                        )
                        prediction = {
                            "layer_id": layer,
                            "detections": [
                                {"class_id": 0, "confidence": 0.9, **geometry},
                                {"class_id": 0, "confidence": 0.8, **geometry},
                            ],
                        }
                        window._apply_prediction_payload(prediction)
                        self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                        self.assertTrue(window.save_labels())
                        window.load_image()
                        self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def _openai_proposal(self, window, count, keypoint_names=("nose", "tail")):
        from squeakpose.services.openai_labeling import parse_pose_proposal

        return parse_pose_proposal(
            {
                "instances": [
                    {
                        "bounding_box": {
                            "x_min": 50 + index * 180,
                            "y_min": 100,
                            "x_max": 150 + index * 180,
                            "y_max": 500,
                        },
                        "keypoints": [
                            {
                                "name": name,
                                "x": 80 + index * 180,
                                "y": 200,
                                "visibility": 2,
                                "confidence": 0.9,
                            }
                            for name in keypoint_names
                        ],
                    }
                    for index in range(count)
                ]
            },
            source_image=window.current_image_path,
            model="mock-openai",
            class_name="mouse",
            keypoint_names=keypoint_names,
            image_width=window.img_w,
            image_height=window.img_h,
        )

    def test_openai_five_instances_replace_only_requested_class_and_save_editable_rows(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_active_pose(window)
                    window.class_selector.setCurrentIndex(1)
                    self._complete_active_pose(window)
                    other_class = window.annotation_cache.annotation(1)
                    window.class_selector.setCurrentIndex(0)
                    proposal = self._openai_proposal(window, 5)
                    with patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=qt_widgets.QMessageBox.StandardButton.Yes,
                    ):
                        window._apply_openai_pose_proposal(proposal)
                    entries = window.annotation_cache.all_entries()
                    self.assertEqual(sum(entry["class_id"] == 0 for entry in entries), 5)
                    self.assertEqual(window.annotation_cache.annotation(1), other_class)
                    self.assertTrue(window._multi_animal_enabled())
                    self.assertEqual(window.annotation_panel.instance_selector.count(), 5)
                    window.annotation_panel.instance_selector.setCurrentIndex(0)
                    self.assertEqual(len(window.current_kps), 2)
                    self.assertAlmostEqual(window.current_box.x, 2)
                    self.assertFalse(Path(window.label_dir, "frame001.txt").exists())
                    self.assertTrue(window.save_labels())
                    window.load_image()
                    self.assertEqual(len(window.annotation_cache.all_entries()), 6)
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_openai_zero_rejected_and_stale_schema_keep_existing_annotations(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_active_pose(window)
                    before = window.annotation_cache.typed_snapshot()
                    with patch("squeakpose.ui.main_window.QMessageBox.question") as confirm:
                        window._apply_openai_pose_proposal(self._openai_proposal(window, 0))
                        confirm.assert_not_called()
                    self.assertEqual(window.annotation_cache.typed_snapshot(), before)
                    with patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=qt_widgets.QMessageBox.StandardButton.No,
                    ):
                        window._apply_openai_pose_proposal(self._openai_proposal(window, 5))
                    self.assertEqual(window.annotation_cache.typed_snapshot(), before)
                    with patch("squeakpose.ui.main_window.QMessageBox.question") as confirm:
                        window._apply_openai_pose_proposal(
                            self._openai_proposal(window, 1, ("nose",))
                        )
                        confirm.assert_not_called()
                    self.assertEqual(window.annotation_cache.typed_snapshot(), before)
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_openai_one_instance_keeps_single_instance_workflow(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    with patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=qt_widgets.QMessageBox.StandardButton.Yes,
                    ):
                        window._apply_openai_pose_proposal(self._openai_proposal(window, 1))
                    self.assertEqual(len(window.annotation_cache.all_entries()), 1)
                    self.assertFalse(window._multi_animal_enabled())
                    self.assertEqual(len(window.current_kps), 2)
                    self.assertTrue(window.save_labels())
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def _write_transfer_masks(self, window):
        window.seg_classes = ["mouse"]
        Path(window.seg_label_dir, "frame001.txt").write_text(
            "0 0.55 0.1 0.85 0.1 0.85 0.6 0.55 0.6\n0 0.1 0.1 0.4 0.1 0.4 0.6 0.1 0.6\n"
        )
        window._refresh_segmentation_box_action()

    def test_create_from_masks_preserves_unplaced_instances_and_requires_all_keypoints_before_save(
        self,
    ):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._write_transfer_masks(window)
                    window.annotation_panel.create_from_segmentation_btn.click()
                    self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                    self.assertEqual(len(window.current_kps), 0)
                    self.assertFalse(window._is_fully_labeled())
                    window._cycle_instance()
                    window.add_keypoint(qt_core.QPointF(8, 8))
                    window._cycle_instance()
                    self.assertEqual(len(window.current_kps), 0)
                    window.add_keypoint(qt_core.QPointF(24, 8))
                    window.add_keypoint(qt_core.QPointF(28, 15))
                    self.assertFalse(window.save_labels())
                    window._cycle_instance()
                    self.assertEqual(len(window.current_kps), 1)
                    window.add_keypoint(qt_core.QPointF(12, 15))
                    self.assertTrue(window._is_fully_labeled())
                    self.assertTrue(window.save_labels())
                    window.load_image()
                    self.assertEqual(len(window.annotation_cache.all_entries()), 2)
                    self.assertTrue(
                        all(
                            len(entry["keypoints"]) == 2
                            for entry in window.annotation_cache.all_entries()
                        )
                    )
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_replace_mask_boxes_automatically_preserves_points_without_a_dialog(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_active_pose(window)
                    window._add_animal_instance()
                    window.add_bbox(qt_core.QRectF(23, 3, 12, 15))
                    window.add_keypoint(qt_core.QPointF(24, 5))
                    window.add_keypoint(qt_core.QPointF(28, 14))
                    self._write_transfer_masks(window)
                    window._cache_active_annotation()
                    before = window.annotation_cache.typed_snapshot()
                    with patch(
                        "squeakpose.ui.main_window.SegmentationBoxReviewDialog.exec",
                        side_effect=AssertionError("Clear geometry must not open a chooser"),
                    ):
                        window.use_segmentation_box_btn.click()
                    after = window.annotation_cache.typed_snapshot()
                    self.assertEqual(
                        [a.keypoints for a in after.annotations],
                        [a.keypoints for a in before.annotations],
                    )
                    self.assertEqual([a.box[0] for a in after.annotations], [4, 22])
                    self.assertTrue(window.save_labels())
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_segmentation_box_review_prevents_duplicate_mask_assignment(self):
        from squeakpose.ui.segmentation_transfer_dialog import SegmentationBoxReviewDialog

        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._write_transfer_masks(window)
                    plans = window._segmentation_box_transfer_plans()
                    entries = [
                        {
                            "bbox": {"x": p.box.x, "y": p.box.y, "w": p.box.w, "h": p.box.h},
                            "keypoints": [],
                        }
                        for p in plans
                    ]
                    dialog = SegmentationBoxReviewDialog(
                        entries, plans, (0, 1), window.current_image_path
                    )
                    dialog.show()
                    self.qt_app.processEvents()
                    button = dialog.buttons.button(qt_widgets.QDialogButtonBox.StandardButton.Ok)
                    self.assertTrue(button.isEnabled())
                    dialog.selectors[0].setCurrentIndex(1)
                    self.assertFalse(button.isEnabled())
                    dialog.selectors[1].setCurrentIndex(0)
                    self.assertTrue(button.isEnabled())
                    self.assertEqual(dialog.assignments, (1, 0))
                    dialog.reject()
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_transfer_preserves_partial_keypoints_and_create_cancel_keeps_drafts(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    window.add_bbox(qt_core.QRectF(4, 5, 20, 15))
                    window.add_keypoint(qt_core.QPointF(8, 9))
                    self._write_transfer_masks(window)
                    before = window.annotation_cache.typed_snapshot()
                    with patch(
                        "squeakpose.ui.main_window.QMessageBox.question",
                        return_value=qt_widgets.QMessageBox.StandardButton.No,
                    ) as confirm:
                        window.annotation_panel.create_from_segmentation_btn.click()
                        confirm.assert_called_once()
                    self.assertEqual(window.annotation_cache.typed_snapshot(), before)
                    self.assertEqual(len(window.current_kps), 1)
                    source = Path(window.seg_label_dir, "frame001.txt")
                    source.write_text(source.read_text().splitlines()[0] + "\n")
                    window.use_segmentation_box_btn.click()
                    self.assertEqual(len(window.current_kps), 1)
                    self.assertEqual((window.current_kps[0].x, window.current_kps[0].y), (8, 9))
                    self.assertAlmostEqual(window.current_box.x, 22)
                    self.assertFalse(window.save_labels())
                    window.add_keypoint(qt_core.QPointF(28, 14))
                    self.assertTrue(window.save_labels())
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()

    def test_replacement_requests_review_only_when_mask_geometry_is_indistinguishable(self):
        with TemporaryDirectory() as tmp:
            stack, window = self._open_window(Path(tmp))
            with stack:
                try:
                    self._complete_active_pose(window)
                    window._add_animal_instance()
                    window.add_bbox(qt_core.QRectF(23, 3, 12, 15))
                    window.add_keypoint(qt_core.QPointF(24, 5))
                    window.add_keypoint(qt_core.QPointF(28, 14))
                    self._write_transfer_masks(window)
                    source = Path(window.seg_label_dir, "frame001.txt")
                    first_mask = source.read_text().splitlines()[0]
                    source.write_text(first_mask + "\n" + first_mask + "\n")
                    window._cache_active_annotation()
                    before = window.annotation_cache.typed_snapshot()
                    with patch(
                        "squeakpose.ui.main_window.SegmentationBoxReviewDialog.exec",
                        return_value=qt_widgets.QDialog.DialogCode.Rejected,
                    ) as chooser:
                        window.use_segmentation_box_btn.click()
                        chooser.assert_called_once()
                    self.assertEqual(window.annotation_cache.typed_snapshot(), before)
                finally:
                    window._dirty = False
                    window.close()
                    self.qt_app.processEvents()
