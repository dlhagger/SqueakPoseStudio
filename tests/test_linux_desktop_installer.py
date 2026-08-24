import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class LinuxDesktopInstallerTests(unittest.TestCase):
    def test_install_and_uninstall_use_discovered_checkout_paths(self):
        repository = Path(__file__).resolve().parents[1]
        installer = repository / "scripts/install_linux_desktop.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_home, bin_home = root / "data", root / "bin"
            fake_uv = root / "tools/uv"
            fake_uv.parent.mkdir()
            fake_uv.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_uv.chmod(0o755)
            common = ["--data-home", str(data_home), "--bin-home", str(bin_home), "--no-refresh"]
            installed = subprocess.run(
                [
                    sys.executable,
                    str(installer),
                    "--repository",
                    str(repository),
                    "--uv",
                    str(fake_uv),
                    *common,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)
            desktop = data_home / "applications/squeakpose-studio.desktop"
            launcher = bin_home / "squeakpose-studio"
            icon = data_home / "icons/hicolor/1024x1024/apps/squeakpose-studio.png"
            self.assertTrue(desktop.is_file())
            self.assertTrue(os.access(launcher, os.X_OK))
            self.assertEqual(
                icon.read_bytes(), (repository / "squeakpose_studio_logo.png").read_bytes()
            )
            self.assertIn(f"Exec={launcher}", desktop.read_text(encoding="utf-8"))
            self.assertIn(str(repository), launcher.read_text(encoding="utf-8"))

            removed = subprocess.run(
                [sys.executable, str(installer), "--uninstall", *common],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(removed.returncode, 0, removed.stderr)
            self.assertFalse(desktop.exists())
            self.assertFalse(launcher.exists())
            self.assertFalse(icon.exists())


if __name__ == "__main__":
    unittest.main()
