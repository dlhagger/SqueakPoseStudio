import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from squeakpose.app import (
    _save_last_project_best_effort,
    load_bundled_fonts,
    make_console_output_lossless,
    set_windows_app_id,
)


class AppStartupTests(unittest.TestCase):
    def test_windows_taskbar_identity_is_set(self):
        with (
            patch("sys.platform", "win32"),
            patch("ctypes.OleDLL", create=True) as dll,
            patch("ctypes.HRESULT", create=True),
        ):
            set_windows_app_id()

        dll.assert_called_once_with("shell32")
        dll.return_value.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with(
            "SqueakPoseStudio.Desktop"
        )

    def test_taskbar_identity_is_windows_only(self):
        for platform in ("linux", "darwin"):
            with (
                self.subTest(platform=platform),
                patch("sys.platform", platform),
                patch("ctypes.OleDLL", create=True) as dll,
            ):
                set_windows_app_id()
                dll.assert_not_called()

    def test_taskbar_identity_failure_does_not_block_startup(self):
        with (
            patch("sys.platform", "win32"),
            patch("ctypes.OleDLL", create=True, side_effect=OSError("shell failure")),
            self.assertLogs("squeakpose.app", level="WARNING"),
        ):
            set_windows_app_id()

    def test_last_project_persistence_failure_does_not_escape(self):
        project_root = "/valid/project"

        with (
            patch(
                "squeakpose.app.save_last_project",
                side_effect=PermissionError("read-only state directory"),
            ),
            self.assertLogs("squeakpose.app", level="WARNING") as logs,
        ):
            saved = _save_last_project_best_effort(project_root)

        self.assertFalse(saved)
        self.assertIn("continuing to open /valid/project", "\n".join(logs.output))

    def test_last_project_persistence_success_is_reported(self):
        with patch("squeakpose.app.save_last_project") as save:
            saved = _save_last_project_best_effort("/valid/project")

        self.assertTrue(saved)
        save.assert_called_once_with("/valid/project")

    def test_console_output_escapes_characters_the_encoding_cannot_represent(self):
        # Simulates a redirected Windows console using the cp1252 ANSI code page.
        raw = io.BytesIO()
        stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        with patch("sys.stdout", stdout), patch("sys.stderr", stdout):
            make_console_output_lossless()
            print("\U0001f9e0 Inference device: cuda")
            stdout.flush()
        self.assertIn(b"\\U0001f9e0 Inference device: cuda", raw.getvalue())

    def test_bundled_fonts_register_every_weight_and_select_fira_sans(self):
        class FakeFontDatabase:
            def __init__(self):
                self.loaded = []

            def addApplicationFont(self, path):
                self.loaded.append(os.path.basename(path))
                return len(self.loaded)

            def applicationFontFamilies(self, _font_id):
                return ["Fira Sans"]

        with TemporaryDirectory() as fonts:
            for name in ("FiraSans-Bold.ttf", "FiraSans-Regular.ttf", "OFL.txt"):
                Path(fonts, name).write_bytes(b"font")
            database = FakeFontDatabase()

            family = load_bundled_fonts(fonts, font_database=database)

        self.assertEqual(family, "Fira Sans")
        self.assertEqual(database.loaded, ["FiraSans-Bold.ttf", "FiraSans-Regular.ttf"])
        self.assertEqual(load_bundled_fonts(os.path.join(fonts, "missing")), "")

    def test_repository_bundles_the_interface_weights_and_license(self):
        fonts = Path(__file__).resolve().parents[1] / "fonts"
        for name in (
            "FiraSans-Regular.ttf",
            "FiraSans-SemiBold.ttf",
            "FiraSans-Bold.ttf",
            "FiraSans-ExtraBold.ttf",
            "OFL.txt",
        ):
            self.assertTrue((fonts / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
