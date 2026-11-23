"""First-open behavior setup, project reuse, and preservation of existing labels."""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from squeakpose.annotation.behavior import Behavior, BehaviorDocument, new_id
from squeakpose.services.behavior_labels import (
    catalog_path,
    load_behavior_catalog,
    load_document,
    new_document,
    save_behavior_catalog,
    save_document,
)
from squeakpose.services.behavior_video import index_video
from squeakpose.ui.behavior_labeler import BehaviorLabelerDialog
from squeakpose.ui.behavior_setup import BehaviorDefinitionEditor
from tests.test_behavior_labeler import make_video
from tests.test_behavior_labels import sample_document, sample_interval


class BehaviorSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["behavior-setup-test"])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = self.temp.name
        self.dialog = BehaviorLabelerDialog(self.root)
        self.extra_dialogs = []

    def tearDown(self):
        for dialog in [self.dialog, *self.extra_dialogs]:
            dialog.draft = False
            dialog.reject()
            dialog.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def define_rearing(self, editor=None):
        editor = editor or self.dialog.setup_editor
        editor.table.item(0, 0).setText("Rearing")
        editor.table.item(0, 1).setText(
            "Both front paws leave the floor. Stop when a front paw touches down."
        )
        editor._submit()

    def test_project_choices_replace_old_video_dropdown_without_losing_labels(self):
        path = Path(self.root, "video.mkv")
        make_video(path)
        index = index_video(str(path))
        old = BehaviorDocument.new(index.source)
        drinking = old.behaviors[3]
        interval = sample_interval(
            old,
            behavior_id=drinking.id,
            start_frame=0,
            end_frame_exclusive=2,
            start_pts=index.boundary(0),
            end_pts=index.boundary(2),
        )
        old.put(interval)
        save_document(self.root, old)
        editor = self.dialog.setup_editor
        editor.table.item(0, 0).setText("Rearing")
        editor.add_row(Behavior(new_id(), "Freezing", "#00cc88"))
        editor._submit()
        with patch.object(QMessageBox, "warning") as warning:
            self.dialog._video_ready(index)
        self.assertFalse(warning.called, warning.call_args_list)

        def choices():
            return [self.dialog.behavior.itemText(i) for i in range(self.dialog.behavior.count())]

        expected = ["1. Rearing", "2. Freezing"]
        self.assertEqual(choices(), expected)
        self.assertEqual(self.dialog.document.intervals, [interval])
        self.dialog.behavior.setCurrentIndex(1)
        self.dialog.seek(3)
        self.dialog._clear_editor(force=True)
        self.dialog._apply_interval()
        self.assertEqual(len(self.dialog.document.intervals), 2)
        freezing_id = self.dialog.project_behaviors[1].id
        self.assertEqual(self.dialog.document.intervals[1].behavior_id, freezing_id)
        self.dialog._select_interval(interval.id)
        self.assertEqual(self.dialog.behavior.currentText(), "Drinking (saved label)")
        self.dialog.notes.setText("Reviewed old annotation")
        self.dialog._apply_interval()
        self.assertEqual(
            self.dialog.document.intervals[0], replace(interval, notes="Reviewed old annotation")
        )
        self.dialog._clear_editor(force=True)
        self.assertEqual(choices(), expected)
        self.dialog._undo()
        self.assertEqual(choices(), expected)
        self.assertEqual(self.dialog.document.intervals[0], interval)

        def edit_definitions(modal):
            editor = modal.findChild(BehaviorDefinitionEditor)
            self.assertEqual(
                [editor.table.item(i, 0).text() for i in range(editor.table.rowCount())],
                ["Rearing", "Freezing"],
            )
            editor.table.item(1, 1).setText("No visible movement")
            editor._submit()
            return QDialog.DialogCode.Accepted

        with patch.object(QDialog, "exec", edit_definitions):
            self.dialog._manage_behaviors()
        self.assertEqual(choices(), expected)
        self.assertEqual(load_behavior_catalog(self.root)[0][1].definition, "No visible movement")
        self.dialog._undo()
        self.assertEqual(load_behavior_catalog(self.root)[0][1].definition, "")
        self.assertEqual(choices(), expected)
        self.dialog._redo()
        self.assertEqual(load_behavior_catalog(self.root)[0][1].definition, "No visible movement")
        self.assertEqual(choices(), expected)
        self.assertEqual(load_document(self.root, old.video.id).intervals[0], interval)
        self.dialog._video_ready(index)
        self.assertEqual(choices(), expected)
        self.assertEqual(self.dialog.document.intervals[0], interval)

    def test_legacy_review_labels_survive_but_are_not_new_label_choices(self):
        path = Path(self.root, "video.mkv")
        make_video(path)
        index = index_video(str(path))
        old = BehaviorDocument.new(index.source)
        other = sample_interval(
            old,
            id="other-interval",
            behavior_id=old.behaviors[0].id,
            start_frame=0,
            end_frame_exclusive=1,
            start_pts=index.boundary(0),
            end_pts=index.boundary(1),
        )
        unscorable = sample_interval(
            old,
            id="unscorable-interval",
            behavior_id=None,
            kind="unscorable",
            start_frame=1,
            end_frame_exclusive=2,
            start_pts=index.boundary(1),
            end_pts=index.boundary(2),
        )
        old.put(other)
        old.put(unscorable)
        save_behavior_catalog(self.root, old.behaviors)
        save_document(self.root, old)
        reopened = BehaviorLabelerDialog(self.root)
        self.extra_dialogs.append(reopened)
        self.assertTrue(reopened.setup_complete)
        reopened._video_ready(index)
        expected = ["1. Rearing", "2. Turning", "3. Drinking"]

        def choices():
            return [reopened.behavior.itemText(i) for i in range(reopened.behavior.count())]

        self.assertEqual(choices(), expected)
        self.assertEqual(reopened.document.intervals, [other, unscorable])
        for interval, name in ((other, "Other"), (unscorable, "Unscorable")):
            reopened._select_interval(interval.id)
            self.assertEqual(reopened.behavior.currentText(), f"{name} (saved label)")
            reopened.notes.setText("Reviewed")
            reopened._apply_interval()
            saved = next(i for i in reopened.document.intervals if i.id == interval.id)
            self.assertEqual(saved, replace(interval, notes="Reviewed"))
            reopened._clear_editor(force=True)
            self.assertEqual(choices(), expected)
        reopened.seek(4)
        reopened._clear_editor(force=True)
        reopened._apply_interval()
        saved = load_document(self.root, old.video.id)
        self.assertEqual(len(saved.intervals), 3)
        self.assertEqual(saved.intervals[-1].behavior_id, reopened.project_behaviors[0].id)
        self.assertEqual(saved.intervals[-1].kind, "behavior")

    def test_first_open_defines_behaviors_before_loading_any_video(self):
        self.assertIs(self.dialog.stack.currentWidget(), self.dialog.setup_page)
        self.assertEqual(self.dialog.setup_editor.table.rowCount(), 1)
        self.assertEqual(self.dialog.setup_editor.table.item(0, 0).text(), "")
        self.assertFalse(Path(catalog_path(self.root)).exists())
        with patch("squeakpose.ui.behavior_labeler.VideoIndexJob") as worker:
            self.dialog.open_video("not-yet-selected.mp4")
            worker.assert_not_called()
        self.define_rearing()
        self.assertTrue(self.dialog.setup_complete)
        self.assertIs(self.dialog.stack.currentWidget(), self.dialog.workspace)
        definitions, configured = load_behavior_catalog(self.root)
        self.assertTrue(configured)
        self.assertEqual([b.name for b in definitions], ["Rearing"])
        self.assertIn("front paws", definitions[0].definition)
        self.assertEqual(
            list(Path(self.root, "behavior labels").iterdir()), [Path(catalog_path(self.root))]
        )
        source = sample_document().video
        self.assertEqual(new_document(self.root, source).behaviors, definitions)
        reopened = BehaviorLabelerDialog(self.root)
        self.extra_dialogs.append(reopened)
        self.assertIs(reopened.stack.currentWidget(), reopened.workspace)

    def test_invalid_empty_duplicate_and_reserved_names_do_not_save(self):
        editor = self.dialog.setup_editor
        editor._submit()
        self.assertIn("at least one", editor.error.text())
        editor.table.item(0, 0).setText("Rearing")
        editor.add_row()
        editor.table.item(1, 0).setText(" rearing ")
        editor._submit()
        self.assertIn("unique", editor.error.text())
        editor.table.item(1, 0).setText("Unscorable")
        editor._submit()
        self.assertIn("review category", editor.error.text())
        self.assertFalse(Path(catalog_path(self.root)).exists())
        self.assertFalse(self.dialog.setup_complete)

    def test_cancel_does_not_create_catalog_or_labels(self):
        self.dialog.setup_editor.table.item(0, 0).setText("Drinking")
        self.dialog.setup_editor.canceled.emit()
        self.assertFalse(Path(self.root, "behavior labels").exists())

    def test_failed_save_keeps_setup_and_all_typed_values(self):
        with patch(
            "squeakpose.services.behavior_labels.atomic_write_text",
            side_effect=OSError("disk full"),
        ):
            self.define_rearing()
        self.assertIs(self.dialog.stack.currentWidget(), self.dialog.setup_page)
        self.assertIn("disk full", self.dialog.setup_editor.error.text())
        self.assertEqual(self.dialog.setup_editor.table.item(0, 0).text(), "Rearing")
        self.assertFalse(self.dialog.setup_complete)
        self.dialog.setup_editor._submit()
        self.assertTrue(self.dialog.setup_complete)

    def test_legacy_catalog_gets_one_time_review_without_rewriting_annotations(self):
        old = sample_document()
        old.put(sample_interval(old))
        saved = Path(save_document(self.root, old))
        old_bytes = saved.read_bytes()
        reopened = BehaviorLabelerDialog(self.root)
        self.extra_dialogs.append(reopened)
        self.assertIs(reopened.stack.currentWidget(), reopened.setup_page)
        self.assertEqual(reopened.setup_editor.table.rowCount(), 3)
        reopened.setup_editor.table.item(0, 1).setText("Reviewed definition")
        reopened.setup_editor._submit()
        self.assertTrue(load_behavior_catalog(self.root)[1])
        self.assertEqual(saved.read_bytes(), old_bytes)
        self.assertEqual(load_document(self.root, old.video.id).behaviors, old.behaviors)

    def test_malformed_catalog_is_visible_and_never_replaced(self):
        path = Path(catalog_path(self.root))
        path.parent.mkdir()
        path.write_text('{"schema_version":999}')
        reopened = BehaviorLabelerDialog(self.root)
        self.extra_dialogs.append(reopened)
        self.assertIs(reopened.stack.currentWidget(), reopened.setup_page)
        self.assertFalse(reopened.setup_editor.save_button.isEnabled())
        self.assertIn("Unsupported", reopened.setup_editor.error.text())
        self.assertEqual(json.loads(path.read_text()), {"schema_version": 999})

    def test_setup_flag_survives_definition_edits_and_undo_style_save(self):
        doc = sample_document()
        save_behavior_catalog(self.root, doc.behaviors)
        save_document(self.root, doc)
        previous = doc.behaviors.copy()
        doc.behaviors[1] = replace(doc.behaviors[1], definition="A precise definition")
        save_document(self.root, doc, previous_behaviors=previous)
        self.assertTrue(load_behavior_catalog(self.root)[1])
        changed = doc.behaviors.copy()
        doc.behaviors = previous
        save_document(self.root, doc, previous_behaviors=changed)
        self.assertTrue(load_behavior_catalog(self.root)[1])

    def test_protected_behavior_cannot_be_removed_and_id_is_preserved(self):
        doc = sample_document()
        editor = BehaviorDefinitionEditor(doc.behaviors, protected_ids={doc.behaviors[1].id})
        editor.table.setCurrentCell(0, 0)
        self.assertFalse(editor.remove_button.isEnabled())
        editor.remove_selected()
        self.assertEqual(editor.table.rowCount(), 3)
        editor.table.item(0, 0).setText("Upright rearing")
        self.assertEqual(editor.behaviors()[0].id, doc.behaviors[1].id)
        editor.table.setCurrentCell(1, 0)
        self.assertTrue(editor.remove_button.isEnabled())
        editor.remove_selected()
        self.assertEqual(editor.table.rowCount(), 2)
        editor.deleteLater()

    def test_save_button_commits_cell_being_edited(self):
        editor = self.dialog.setup_editor
        self.dialog.show()
        self.app.processEvents()
        editor.table.setCurrentCell(0, 0)
        editor.table.editItem(editor.table.item(0, 0))
        self.app.processEvents()
        QTest.keyClicks(self.app.focusWidget(), "Drinking")
        QTest.mouseClick(editor.save_button, Qt.MouseButton.LeftButton)
        self.assertTrue(self.dialog.setup_complete)
        self.assertEqual(load_behavior_catalog(self.root)[0][0].name, "Drinking")

    def test_manage_definitions_before_and_after_opening_video(self):
        self.define_rearing()

        def edit_project(modal):
            editor = modal.findChild(BehaviorDefinitionEditor)
            editor.table.item(0, 1).setText("Updated onset and offset criteria")
            editor._submit()
            return QDialog.DialogCode.Accepted

        with patch.object(QDialog, "exec", edit_project):
            self.dialog._manage_behaviors()
        path = Path(self.root, "video.mkv")
        make_video(path)
        self.dialog._video_ready(index_video(str(path)))
        self.assertEqual(
            self.dialog.document.behaviors[0].definition, "Updated onset and offset criteria"
        )
        self.assertEqual(self.dialog.behavior.currentData(), self.dialog.document.behaviors[0].id)
        self.dialog._mark_start()
        self.dialog.seek(1)
        self.dialog._mark_end()
        self.dialog._apply_interval()
        identity = self.dialog.document.intervals[0].id

        def edit_video(modal):
            editor = modal.findChild(BehaviorDefinitionEditor)
            editor.table.setCurrentCell(0, 0)
            self.assertFalse(editor.remove_button.isEnabled())
            editor.table.item(0, 1).setText("Revised current-video definition")
            editor._submit()
            return QDialog.DialogCode.Accepted

        with (
            patch.object(QDialog, "exec", edit_video),
            patch.object(QMessageBox, "warning") as warning,
        ):
            self.dialog._manage_behaviors()
        self.assertFalse(warning.called, warning.call_args_list)
        self.assertEqual(self.dialog.document.intervals[0].id, identity)
        self.assertEqual(
            self.dialog.document.behaviors[0].definition, "Revised current-video definition"
        )
        self.assertTrue(load_behavior_catalog(self.root)[1])
