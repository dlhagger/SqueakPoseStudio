import os
import unittest
from tempfile import gettempdir
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from squeakpose.ui.class_manager import AddClassDialog, ClassManagerDialog


class ClassManagerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["class-manager-test"])

    def test_invalid_new_class_stays_open_and_preserves_edits(self):
        dialog = AddClassDialog(["nose"], existing_classes=["mouse"])
        dialog.show()
        dialog.keypoints_edit.setPlainText("nose\ntail")
        try:
            for name in ("", "mouse"):
                dialog.name_edit.setText(name)
                with patch.object(QMessageBox, "warning") as warning:
                    dialog.accept()
                warning.assert_called_once()
                self.assertTrue(dialog.isVisible())
                self.assertEqual(dialog.keypoints_edit.toPlainText(), "nose\ntail")
            dialog.name_edit.setText("rat")
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertFalse(dialog.isVisible())
        finally:
            dialog.close()

    def test_remove_selected_does_not_overwrite_next_class_keypoints(self):
        dialog = ClassManagerDialog(
            ["mouse", "rat", "object"],
            {"mouse": ["nose", "tail"], "rat": ["head"], "object": ["center"]},
            ["nose", "tail"],
        )
        try:
            dialog.class_list.setCurrentRow(0)
            # Edited keypoints for the class that is about to be removed.
            dialog.keypoints_edit.setPlainText("nose\ntail\near")

            with patch.object(QMessageBox, "information", return_value=None):
                dialog._remove_selected()

            self.assertEqual(dialog._classes, ["rat", "object"])
            self.assertEqual(dialog._kp_map, {"rat": ["head"], "object": ["center"]})
            self.assertEqual(dialog.class_list.count(), 2)
            self.assertEqual(dialog.class_list.currentRow(), 0)
            self.assertEqual(dialog._current_row, 0)
            self.assertEqual(dialog.keypoints_edit.toPlainText(), "head")

            # The surviving classes round-trip unchanged through the accept path.
            dialog._save_current_keypoints()
            self.assertEqual(dialog._kp_map["rat"], ["head"])
        finally:
            dialog.close()

    def test_remove_last_class_clears_editor(self):
        dialog = ClassManagerDialog(["mouse"], {"mouse": ["nose"]}, ["nose"])
        try:
            dialog.class_list.setCurrentRow(0)
            with patch.object(QMessageBox, "information", return_value=None):
                dialog._remove_selected()

            self.assertEqual(dialog._classes, [])
            self.assertEqual(dialog._kp_map, {})
            self.assertEqual(dialog.class_list.count(), 0)
            self.assertEqual(dialog._current_row, -1)
            self.assertEqual(dialog.keypoints_edit.toPlainText(), "")
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
