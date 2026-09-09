import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(gettempdir(), "squeakpose-mpl-tests"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(gettempdir(), "squeakpose-cache-tests"))

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from analysis_dialog import AnalysisDialog
from squeakpose.ui.class_manager import AddClassDialog, ClassManagerDialog
from squeakpose.ui.distillation_dialog import DistillationDialog
from squeakpose.ui.inference_review_dialog import InferenceReviewDialog
from squeakpose.ui.inference_video_dialog import InferenceVideoDialog
from squeakpose.ui.main_window import CongratsPopup
from squeakpose.ui.project_launcher import ProjectLauncherDialog
from squeakpose.ui.project_models_dialog import ProjectModelsDialog
from squeakpose.ui.project_video_picker import ProjectVideoPickerDialog
from squeakpose.ui.training_dialog import TrainDialog
from squeakpose.ui.video_library_dialog import VideoLibraryDialog
from squeakpose.ui.video_reviewer import VideoReviewDialog


class DialogResizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["dialog-resizing-test"])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_all_custom_dialogs_have_a_size_grip_and_can_grow(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos_dir = root / "videos"
            videos_dir.mkdir()
            parent = QWidget()
            parent.project_root = str(root)
            parent.app_base_dir = str(root)

            factories = {
                "add class": lambda: AddClassDialog(["nose"], parent),
                "analysis": lambda: AnalysisDialog(
                    parent,
                    project_root=str(root),
                    app_base_dir=str(root),
                    layer_id="keypoints",
                ),
                "class manager": lambda: ClassManagerDialog(
                    ["mouse"], {"mouse": ["nose"]}, ["nose"], parent
                ),
                "completion": CongratsPopup,
                "distillation": lambda: DistillationDialog(parent),
                "inference picker": lambda: InferenceVideoDialog(str(root), parent=parent),
                "inference review": lambda: InferenceReviewDialog(
                    str(root), parent, auto_scan=False
                ),
                "project launcher": lambda: ProjectLauncherDialog(str(root), "", parent),
                "project models": lambda: ProjectModelsDialog(parent, {}),
                "project video picker": lambda: ProjectVideoPickerDialog(
                    str(videos_dir), parent=parent
                ),
                "training": lambda: TrainDialog(
                    parent,
                    str(root / "datasets"),
                    default_task="pose",
                    layer_id="keypoints",
                ),
                "video library": lambda: VideoLibraryDialog(str(videos_dir), parent),
                "video reviewer": lambda: VideoReviewDialog(
                    parent,
                    "cpu",
                    ["nose"],
                    ["mouse"],
                    layer_id="keypoints",
                ),
            }

            dialogs: list[QDialog] = []
            try:
                for name, factory in factories.items():
                    with self.subTest(dialog=name):
                        dialog = factory()
                        dialogs.append(dialog)
                        self.assertTrue(dialog.isSizeGripEnabled())
                        initial = dialog.size()
                        larger = QSize(initial.width() + 80, initial.height() + 60)
                        dialog.resize(larger)
                        self.assertEqual(dialog.size(), larger)
            finally:
                for dialog in dialogs:
                    dialog.close()
                    dialog.deleteLater()
                parent.close()
                parent.deleteLater()


if __name__ == "__main__":
    unittest.main()
