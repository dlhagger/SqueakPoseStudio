"""Child-process entry point for SqueakPose inference analysis."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
from typing import Any, Callable, Optional

from analysis_ops import AnalysisConfig, AnalysisError, run_analysis_workflow


_CANCEL_REQUESTED = False


def _handle_cancel_signal(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise SystemExit(130)


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _emit_event(event_writer: Callable[[dict[str, Any]], None], payload: dict[str, Any]) -> None:
    event_writer(payload)


def run_analysis_worker(
    config: dict[str, Any],
    *,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    """Run one analysis job and emit JSON-compatible event payloads."""
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = False

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "squeakpose-mpl"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "squeakpose-numba-cache"))

    try:
        analysis_config = AnalysisConfig.from_dict(config)
    except Exception as exc:
        _emit_event(event_writer, {"event": "error", "error_message": f"Invalid analysis config: {exc}"})
        return 1

    _emit_event(
        event_writer,
        {
            "event": "started",
            "detections_csv": analysis_config.detections_csv,
            "output_dir": analysis_config.output_dir,
        },
    )

    def progress(step: int, total: int, message: str) -> None:
        _emit_event(event_writer, {"event": "progress", "step": step, "total": total, "message": message})

    try:
        result = run_analysis_workflow(analysis_config, progress_callback=progress)
    except AnalysisError as exc:
        _emit_event(event_writer, {"event": "error", "error_message": str(exc)})
        return 1
    except Exception as exc:
        _emit_event(event_writer, {"event": "error", "error_message": f"Analysis failed: {exc}"})
        return 1

    _emit_event(event_writer, {"event": "result", **result})
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _handle_cancel_signal)
    signal.signal(signal.SIGINT, _handle_cancel_signal)

    parser = argparse.ArgumentParser(description="Run SqueakPose inference analysis in a child process.")
    parser.add_argument("--config", required=True, help="Path to JSON analysis config.")
    args = parser.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    return run_analysis_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
