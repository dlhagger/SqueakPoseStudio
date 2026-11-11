"""Child-process entry point for SqueakPose inference analysis."""

from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from analysis_ops import (
    AnalysisConfig,
    AnalysisError,
    run_analysis_workflow,
)
from squeakpose.project.layers import LAYER_KEYPOINTS, LAYER_SEGMENTATION
from squeakpose.services.analysis import AnalysisConfigError, validate_analysis_output_dir
from squeakpose.services.cancellation import OperationCanceled, cancel_check_scope
from squeakpose.workers.protocol import read_config, start_cancel_listener, write_event
from unified_analysis_ops import run_unified_analysis_workflow

_CANCEL_REQUESTED = False


def _request_cancel() -> None:
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True


def _handle_cancel_signal(_signum, _frame):
    # Only set the flag. Analysis checks it at every progress step and every
    # rendered video frame, then unwinds through the output transaction so the
    # previous published analysis is left untouched and a canceled ``result``
    # is emitted. The owner's kill escalation still covers a stuck worker.
    _request_cancel()


def _cancel_requested() -> bool:
    return _CANCEL_REQUESTED


def _canceled_result(output_dir: str, **extra: Any) -> dict[str, Any]:
    return {
        "event": "result",
        "canceled": True,
        "had_error": False,
        "error_message": "",
        "output_dir": output_dir,
        **extra,
    }


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    write_event(payload)


def _emit_event(event_writer: Callable[[dict[str, Any]], None], payload: dict[str, Any]) -> None:
    event_writer(payload)


def run_analysis_worker(
    config: dict[str, Any],
    *,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
    reset_cancel: bool = True,
) -> int:
    """Run one analysis job and emit JSON-compatible event payloads."""
    global _CANCEL_REQUESTED
    if reset_cancel:
        _CANCEL_REQUESTED = False
    with cancel_check_scope(_cancel_requested):
        return _run_analysis_worker(config, event_writer=event_writer)


def _run_analysis_worker(
    config: dict[str, Any],
    *,
    event_writer: Callable[[dict[str, Any]], None],
) -> int:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "squeakpose-mpl"))
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "squeakpose-numba-cache")
    )

    analysis_inputs = config.get("analysis_inputs")
    selected_layers = config.get("selected_layers")
    if isinstance(analysis_inputs, dict) and isinstance(selected_layers, list):
        return _run_analysis_job(config, event_writer=event_writer)

    try:
        analysis_config = AnalysisConfig.from_dict(config)
    except Exception as exc:
        _emit_event(
            event_writer, {"event": "error", "error_message": f"Invalid analysis config: {exc}"}
        )
        return 1

    _emit_event(
        event_writer,
        {
            "event": "started",
            "layer_id": analysis_config.layer_id,
            "detections_csv": analysis_config.detections_csv,
            "output_dir": analysis_config.output_dir,
        },
    )

    def progress(step: int, total: int, message: str) -> None:
        _emit_event(
            event_writer, {"event": "progress", "step": step, "total": total, "message": message}
        )

    try:
        result = run_analysis_workflow(analysis_config, progress_callback=progress)
    except OperationCanceled:
        _emit_event(event_writer, _canceled_result(analysis_config.output_dir))
        return 0
    except AnalysisError as exc:
        _emit_event(event_writer, {"event": "error", "error_message": str(exc)})
        return 1
    except Exception as exc:
        _emit_event(event_writer, {"event": "error", "error_message": f"Analysis failed: {exc}"})
        return 1

    _emit_event(event_writer, {"event": "result", **result})
    return 0


def _run_analysis_job(
    config: dict[str, Any],
    *,
    event_writer: Callable[[dict[str, Any]], None],
) -> int:
    """Run selected layer workflows and optionally construct their combined frame table."""
    inputs = {
        str(layer): str(path) for layer, path in dict(config.get("analysis_inputs") or {}).items()
    }
    selected_layers = [
        str(layer)
        for layer in list(config.get("selected_layers") or [])
        if str(layer) in {LAYER_KEYPOINTS, LAYER_SEGMENTATION}
    ]
    if not selected_layers:
        _emit_event(
            event_writer,
            {"event": "error", "error_message": "No analysis layers were selected."},
        )
        return 1

    try:
        output_root = validate_analysis_output_dir(
            str(config.get("output_dir") or ""),
            project_root=str(config.get("project_root") or ""),
        )
    except AnalysisConfigError as exc:
        _emit_event(
            event_writer,
            {"event": "error", "error_message": exc.message},
        )
        return 1
    os.makedirs(output_root, exist_ok=True)

    combined_requested = len(selected_layers) == 2
    if combined_requested:
        total_steps = 9
        _emit_event(
            event_writer,
            {
                "event": "started",
                "analysis_mode": "both",
                "analysis_inputs": inputs,
                "output_dir": output_root,
                "total": total_steps,
            },
        )

        def unified_progress(step: int, total: int, message: str) -> None:
            _emit_event(
                event_writer,
                {
                    "event": "progress",
                    "step": step,
                    "total": total,
                    "layer_id": "combined",
                    "message": message,
                },
            )

        try:
            unified_config = AnalysisConfig.from_dict(
                {
                    **config,
                    "layer_id": "",
                    "detections_csv": inputs.get(LAYER_KEYPOINTS, ""),
                    "output_dir": output_root,
                }
            )
            result = run_unified_analysis_workflow(
                unified_config,
                pose_csv=inputs.get(LAYER_KEYPOINTS, ""),
                segmentation_csv=inputs.get(LAYER_SEGMENTATION, ""),
                progress_callback=unified_progress,
            )
        except OperationCanceled:
            _emit_event(event_writer, _canceled_result(output_root))
            return 0
        except AnalysisError as exc:
            _emit_event(event_writer, {"event": "error", "error_message": str(exc)})
            return 1
        except Exception as exc:
            _emit_event(
                event_writer,
                {"event": "error", "error_message": f"Unified analysis failed: {exc}"},
            )
            return 1
        _emit_event(event_writer, {"event": "result", **result})
        return 0

    total_steps = 8 * len(selected_layers)
    _emit_event(
        event_writer,
        {
            "event": "started",
            "analysis_mode": config.get("analysis_mode"),
            "analysis_inputs": inputs,
            "output_dir": output_root,
            "total": total_steps,
        },
    )

    results_by_layer: dict[str, dict[str, Any]] = {}
    errors_by_layer: dict[str, str] = {}
    completed_steps = 0
    for layer in selected_layers:
        layer_config_raw = dict(config)
        layer_config_raw["layer_id"] = layer
        layer_config_raw["detections_csv"] = inputs.get(layer, "")
        layer_config_raw["output_dir"] = output_root
        try:
            layer_config = AnalysisConfig.from_dict(layer_config_raw)

            def progress(step: int, _total: int, message: str, *, _layer=layer) -> None:
                _emit_event(
                    event_writer,
                    {
                        "event": "progress",
                        "step": completed_steps + int(step),
                        "total": total_steps,
                        "layer_id": _layer,
                        "message": f"{_layer}: {message}",
                    },
                )

            results_by_layer[layer] = run_analysis_workflow(
                layer_config, progress_callback=progress
            )
        except OperationCanceled:
            # Each layer publishes independently. Layers finished before the
            # cancel keep their new output; the canceled layer keeps its
            # previous output. Report which is which.
            _emit_event(
                event_writer,
                _canceled_result(
                    output_root,
                    analysis_mode=config.get("analysis_mode"),
                    completed_layers=list(results_by_layer),
                    results_by_layer=results_by_layer,
                ),
            )
            return 0
        except Exception as exc:
            errors_by_layer[layer] = str(exc)
            _emit_event(
                event_writer,
                {
                    "event": "progress",
                    "step": completed_steps + 8,
                    "total": total_steps,
                    "layer_id": layer,
                    "message": f"{layer}: failed — {exc}",
                },
            )
        completed_steps += 8

    manifest_path = os.path.join(output_root, "analysis_manifest.json")
    manifest = {
        "schema_version": 1,
        "analysis_mode": config.get("analysis_mode"),
        "video_path": str(config.get("video_path") or ""),
        "analysis_inputs": inputs,
        "selected_layers": selected_layers,
        "results_by_layer": results_by_layer,
        "errors_by_layer": errors_by_layer,
        "rois": config.get("rois") or [],
        "pixel_distance": config.get("pixel_distance"),
        "real_world_distance_mm": config.get("real_world_distance_mm"),
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not results_by_layer:
        detail = "; ".join(f"{layer}: {error}" for layer, error in errors_by_layer.items())
        _emit_event(
            event_writer,
            {"event": "error", "error_message": detail or "Analysis failed."},
        )
        return 1

    primary_result = results_by_layer[selected_layers[0]]
    _emit_event(
        event_writer,
        {
            "event": "result",
            **primary_result,
            "analysis_mode": config.get("analysis_mode"),
            "output_dir": output_root,
            "results_by_layer": results_by_layer,
            "errors_by_layer": errors_by_layer,
            "manifest_path": manifest_path,
        },
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    signal.signal(signal.SIGTERM, _handle_cancel_signal)
    signal.signal(signal.SIGINT, _handle_cancel_signal)
    start_cancel_listener(_request_cancel)

    parser = argparse.ArgumentParser(
        description="Run SqueakPose inference analysis in a child process."
    )
    parser.add_argument("--config", required=True, help="Path to JSON analysis config.")
    args = parser.parse_args(argv)

    try:
        config = read_config(args.config)
    except Exception as exc:
        _stdout_event_writer({"event": "error", "error_message": f"Could not read config: {exc}"})
        return 1
    # Keep a cancel that arrived while the config was loading.
    return run_analysis_worker(config, reset_cancel=False)


if __name__ == "__main__":
    raise SystemExit(main())
