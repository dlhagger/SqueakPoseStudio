#!/usr/bin/env python3
"""Persistent child-process entry point for interactive SAM segmentation."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from squeakpose.annotation.segmentation_assistant import inspect_sam_contour
from squeakpose.services.sam_assistant import serialize_sam_selection
from squeakpose.workers.protocol import write_event


def _stdout_event_writer(payload: dict[str, Any]) -> None:
    write_event(payload)


def _load_sam_factory(
    model_factory: Callable[[str], Any] | None,
) -> tuple[Callable[[str], Any] | None, str]:
    if model_factory is not None:
        return model_factory, ""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from ultralytics import SAM
    except Exception as exc:
        return None, f"Could not import ultralytics SAM: {exc}"
    return SAM, ""


def run_sam_server(
    request_lines: Iterable[str],
    *,
    model_factory: Callable[[str], Any] | None = None,
    event_writer: Callable[[dict[str, Any]], None] = _stdout_event_writer,
) -> int:
    """Serve prompt requests while retaining at most one SAM model instance."""
    factory, dependency_error = _load_sam_factory(model_factory)
    if factory is None:
        event_writer(
            {
                "event": "error",
                "request_id": None,
                "error_message": dependency_error,
            }
        )
        return 1

    cached_model: Any = None
    cached_model_path = ""
    event_writer({"event": "ready"})

    for raw_line in request_lines:
        line = str(raw_line).strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("request must be a JSON object")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            event_writer(
                {
                    "event": "error",
                    "request_id": None,
                    "error_message": f"Invalid request: {exc}",
                }
            )
            continue

        request_id = request.get("request_id")
        command = str(request.get("command") or "predict").lower()
        if command == "shutdown":
            event_writer({"event": "stopped", "request_id": request_id})
            return 0
        if command not in {"load", "predict"}:
            event_writer(
                {
                    "event": "error",
                    "request_id": request_id,
                    "error_message": f"Unsupported command: {command}",
                }
            )
            continue

        model_path = str(request.get("model_path") or "")
        if not model_path:
            event_writer(
                {
                    "event": "error",
                    "request_id": request_id,
                    "error_message": "model_path is required",
                }
            )
            continue

        try:
            if cached_model is None or cached_model_path != model_path:
                event_writer(
                    {
                        "event": "loading",
                        "request_id": request_id,
                        "model_path": model_path,
                    }
                )
                with contextlib.redirect_stdout(sys.stderr):
                    cached_model = factory(model_path)
                cached_model_path = model_path
            if command == "load":
                event_writer(
                    {
                        "event": "loaded",
                        "request_id": request_id,
                        "model_path": model_path,
                    }
                )
                continue

            image_path = str(request.get("image_path") or "")
            if not image_path:
                raise ValueError("image_path is required")
            points, labels = _validated_prompts(
                request.get("points"),
                request.get("labels"),
            )
            event_writer(
                {
                    "event": "started",
                    "request_id": request_id,
                    "image_path": image_path,
                }
            )
            predict_kwargs: dict[str, Any] = {
                "source": image_path,
                "points": points,
                "labels": labels,
                "verbose": False,
            }
            device = str(request.get("device") or "")
            if device:
                predict_kwargs["device"] = device
            with contextlib.redirect_stdout(sys.stderr):
                raw_results = cached_model.predict(**predict_kwargs)
            results = list(raw_results) if raw_results is not None else []
            selection = inspect_sam_contour(results)
            event_writer(
                {
                    "event": "result",
                    "request_id": request_id,
                    "image_path": image_path,
                    "canceled": False,
                    "had_error": False,
                    "error_message": "",
                    "prediction": serialize_sam_selection(selection),
                }
            )
        except Exception as exc:
            if command == "load":
                event_writer(
                    {
                        "event": "error",
                        "request_id": request_id,
                        "error_message": str(exc),
                    }
                )
            else:
                event_writer(
                    {
                        "event": "result",
                        "request_id": request_id,
                        "canceled": False,
                        "had_error": True,
                        "error_message": str(exc),
                        "prediction": None,
                    }
                )
    return 0


def _validated_prompts(points: Any, labels: Any) -> tuple[list[list[float]], list[int]]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValueError("points must be a list")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a list")
    if not points or len(points) != len(labels):
        raise ValueError("points and labels must be non-empty and have equal length")
    normalized_points: list[list[float]] = []
    normalized_labels: list[int] = []
    for point, label in zip(points, labels, strict=True):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            raise ValueError("each point requires x and y")
        numeric_label = int(label)
        if numeric_label not in {0, 1}:
            raise ValueError("prompt labels must be 0 or 1")
        normalized_points.append([float(point[0]), float(point[1])])
        normalized_labels.append(numeric_label)
    return normalized_points, normalized_labels


def _handle_stop(_signum: int, _frame: Any) -> None:
    raise SystemExit(130)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    parser = argparse.ArgumentParser(
        description="Run the SqueakPose SAM assistant in a child process."
    )
    parser.add_argument(
        "--server",
        action="store_true",
        required=True,
        help="Serve newline-delimited requests on stdin.",
    )
    parser.parse_args(argv)
    return run_sam_server(sys.stdin)


if __name__ == "__main__":
    raise SystemExit(main())
