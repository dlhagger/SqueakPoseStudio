#!/usr/bin/env python3
"""Install or remove SqueakPose Studio integration for a Linux desktop."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP_ID = "squeakpose-studio"


def _atomic_write(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def install(
    *,
    repository: Path,
    uv_path: Path,
    data_home: Path,
    bin_home: Path,
    refresh_cache: bool = True,
) -> tuple[Path, Path, Path]:
    repository = repository.resolve()
    entrypoint = repository / "squeakpose_studio.py"
    source_icon = repository / "squeakpose_studio_logo.png"
    if not entrypoint.is_file() or not source_icon.is_file():
        raise ValueError(f"Not a SqueakPose Studio checkout: {repository}")
    if not uv_path.is_file():
        raise ValueError(f"uv executable not found: {uv_path}")

    launcher = bin_home / APP_ID
    icon = data_home / "icons/hicolor/1024x1024/apps" / f"{APP_ID}.png"
    desktop = data_home / "applications" / f"{APP_ID}.desktop"
    launcher_text = (
        "#!/bin/sh\nset -eu\n"
        f"cd {shlex.quote(str(repository))}\n"
        f"exec {shlex.quote(str(uv_path.resolve()))} run "
        f'{shlex.quote(str(entrypoint))} "$@"\n'
    )
    _atomic_write(launcher, launcher_text, 0o755)
    icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_icon, icon)
    os.chmod(icon, 0o644)
    desktop_text = f"""[Desktop Entry]
Type=Application
Version=1.0
Name=SqueakPose Studio
Comment=Small-animal pose estimation, tracking, and behavioral analysis
Exec={launcher}
Icon={APP_ID}
Terminal=false
StartupNotify=true
StartupWMClass={APP_ID}
Categories=Science;
"""
    _atomic_write(desktop, desktop_text, 0o644)
    if refresh_cache and (updater := shutil.which("update-desktop-database")):
        subprocess.run([updater, str(desktop.parent)], check=False)
    return desktop, launcher, icon


def uninstall(*, data_home: Path, bin_home: Path, refresh_cache: bool = True) -> None:
    for target in (
        data_home / "applications" / f"{APP_ID}.desktop",
        data_home / "icons/hicolor/1024x1024/apps" / f"{APP_ID}.png",
        bin_home / APP_ID,
    ):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    applications = data_home / "applications"
    if (
        refresh_cache
        and applications.is_dir()
        and (updater := shutil.which("update-desktop-database"))
    ):
        subprocess.run([updater, str(applications)], check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--data-home", type=Path)
    parser.add_argument("--bin-home", type=Path)
    parser.add_argument("--no-refresh", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Linux desktop integration is only available on Linux")

    home = Path.home()
    data_home = args.data_home or Path(os.environ.get("XDG_DATA_HOME", home / ".local/share"))
    bin_home = args.bin_home or home / ".local/bin"
    if args.uninstall:
        uninstall(data_home=data_home, bin_home=bin_home, refresh_cache=not args.no_refresh)
        print("Removed SqueakPose Studio desktop integration.")
        return 0
    uv = args.uv or (Path(found) if (found := shutil.which("uv")) else None)
    if uv is None:
        parser.error("uv was not found on PATH; pass its location with --uv")
    desktop, launcher, icon = install(
        repository=args.repository,
        uv_path=uv,
        data_home=data_home,
        bin_home=bin_home,
        refresh_cache=not args.no_refresh,
    )
    print(f"Installed desktop entry: {desktop}")
    print(f"Installed launcher: {launcher}")
    print(f"Installed icon: {icon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
