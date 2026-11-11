"""Run every project check in one command.

    uv run tests.py          # lint, format check, type check, unit tests
    uv run tests.py --fix    # apply Ruff fixes and formatting first, then check

Each check uses the same locked dependency groups as CI. All checks run even if an
earlier one fails, and the exit code is non-zero when any of them failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def _uv(group: str, *args: str) -> list[str]:
    return ["uv", "run", "--locked", "--only-group", group, *args]


CHECKS: tuple[tuple[str, list[str]], ...] = (
    ("Ruff lint", _uv("dev", "ruff", "check", ".")),
    ("Ruff format", _uv("dev", "ruff", "format", "--check", ".")),
    ("mypy", _uv("dev", "mypy")),
    ("Unit tests", _uv("test", "python", "-m", "unittest", "discover", "-s", "tests", "-q")),
)

FIXES: tuple[tuple[str, list[str]], ...] = (
    ("Ruff lint fixes", _uv("dev", "ruff", "check", "--fix", ".")),
    ("Ruff formatting", _uv("dev", "ruff", "format", ".")),
)


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    # The tests construct Qt windows without a display.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Offscreen Qt on Windows has no system font access and PyQt6 no longer bundles
    # fonts; use the app's bundled Fira Sans so text renders identically on every
    # machine. Platforms whose font database is fontconfig/CoreText ignore this.
    env.setdefault("QT_QPA_FONTDIR", os.path.join(ROOT, "fonts"))
    return env


def _run(name: str, command: list[str], env: dict[str, str]) -> tuple[str, bool, float]:
    print(f"\n=== {name} ===", flush=True)
    started = time.monotonic()
    returncode = subprocess.call(command, cwd=ROOT, env=env)
    return name, returncode == 0, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SqueakPose Studio checks.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply Ruff lint fixes and formatting before running the checks",
    )
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        print("uv was not found on PATH. Install it from https://docs.astral.sh/uv/.")
        return 2

    env = _environment()
    if args.fix:
        for name, command in FIXES:
            _run(name, command, env)

    results = [_run(name, command, env) for name, command in CHECKS]

    print("\n=== Summary ===")
    for name, passed, seconds in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}  {name}  ({seconds:.1f}s)")
    failed = [name for name, passed, _seconds in results if not passed]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        if "Ruff format" in failed or "Ruff lint" in failed:
            print("Tip: `uv run tests.py --fix` applies Ruff's fixes and formatting.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
