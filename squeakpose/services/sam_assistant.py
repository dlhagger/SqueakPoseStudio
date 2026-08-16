"""JSON protocol contracts for the isolated persistent SAM assistant."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from squeakpose.annotation.segmentation import Point, PromptPoint
from squeakpose.annotation.segmentation_assistant import (
    SamContourResult,
    SamContourSelection,
    SamPromptRequest,
)


class SamAssistantValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SamWorkerRequest:
    command: Literal["load", "predict"]
    request_id: Any
    model_path: str
    device: str = ""
    image_path: str = ""
    prompts: tuple[PromptPoint, ...] = ()

    def as_worker_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": self.command,
            "request_id": self.request_id,
            "model_path": self.model_path,
            "device": self.device,
        }
        if self.command == "predict":
            payload.update(
                {
                    "image_path": self.image_path,
                    "points": [[x, y] for x, y, _label in self.prompts],
                    "labels": [label for _x, _y, label in self.prompts],
                }
            )
        return payload


SamAssistantAction = Literal[
    "ignore",
    "background_error",
    "cancel",
    "error",
    "discard",
    "apply",
]


@dataclass(frozen=True, slots=True)
class SamAssistantDecision:
    action: SamAssistantAction
    request_id: Any = None
    result: SamContourResult | None = None
    failure: Literal["", "no_masks", "no_polygon"] = ""
    error_message: str = ""


def build_sam_load_request(
    *,
    request_id: Any,
    model_path: str,
    device: str = "",
) -> SamWorkerRequest:
    return SamWorkerRequest(
        command="load",
        request_id=request_id,
        model_path=_required_text(model_path, "model_path"),
        device=str(device or ""),
    )


def build_sam_prediction_request(
    *,
    request_id: Any,
    model_path: str,
    prompt: SamPromptRequest,
    device: str = "",
) -> SamWorkerRequest:
    if not isinstance(prompt, SamPromptRequest):
        raise SamAssistantValidationError("prompt must be a SamPromptRequest")
    image_path = _required_text(prompt.source, "image_path")
    prompts = tuple(_normalize_prompt(value) for value in prompt.prompts)
    if not prompts:
        raise SamAssistantValidationError("at least one SAM prompt is required")
    return SamWorkerRequest(
        command="predict",
        request_id=request_id,
        model_path=_required_text(model_path, "model_path"),
        device=str(device or ""),
        image_path=image_path,
        prompts=prompts,
    )


def serialize_sam_selection(selection: SamContourSelection) -> dict[str, Any]:
    result = selection.result
    return {
        "points": [[float(x), float(y)] for x, y in (result.points if result else ())],
        "score": float(result.score if result else 0.0),
        "failure": selection.failure,
    }


def deserialize_sam_selection(payload: Mapping[str, Any]) -> SamContourSelection:
    if not isinstance(payload, Mapping):
        raise SamAssistantValidationError("SAM prediction must be a JSON object")
    raw_failure = str(payload.get("failure") or "")
    if raw_failure not in {"", "no_masks", "no_polygon"}:
        raise SamAssistantValidationError(f"unsupported SAM failure: {raw_failure}")
    failure = cast(Literal["", "no_masks", "no_polygon"], raw_failure)
    raw_points = payload.get("points") or []
    if not isinstance(raw_points, list):
        raise SamAssistantValidationError("SAM contour points must be a list")
    points: list[Point] = []
    try:
        for point in raw_points:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
                raise TypeError
            if len(point) < 2:
                raise ValueError
            points.append((float(point[0]), float(point[1])))
        score = float(payload.get("score") or 0.0)
    except (TypeError, ValueError) as exc:
        raise SamAssistantValidationError("SAM contour payload is invalid") from exc
    if points and len(points) < 3:
        raise SamAssistantValidationError("SAM contour requires at least three points")
    if not points:
        return SamContourSelection(failure=failure or "no_polygon")
    return SamContourSelection(
        result=SamContourResult(points=tuple(points), score=score),
        failure="",
    )


def correlate_sam_event(
    event: Mapping[str, Any],
    *,
    current_request_id: Any,
    requested_image_path: str,
    displayed_image_path: str,
) -> SamAssistantDecision:
    event_type = str(event.get("event") or "")
    request_id = event.get("request_id")
    if event_type == "error":
        message = str(event.get("error_message") or "SAM worker error")
        action: SamAssistantAction = (
            "error"
            if current_request_id is not None
            and (request_id is None or request_id == current_request_id)
            else "background_error"
        )
        return SamAssistantDecision(action, request_id=request_id, error_message=message)
    if event_type != "result" or request_id != current_request_id:
        return SamAssistantDecision("ignore", request_id=request_id)
    if bool(event.get("canceled")):
        return SamAssistantDecision("cancel", request_id=request_id)
    if bool(event.get("had_error")):
        return SamAssistantDecision(
            "error",
            request_id=request_id,
            error_message=str(event.get("error_message") or "Unknown SAM worker error"),
        )
    if _normalized_path(requested_image_path) != _normalized_path(displayed_image_path):
        return SamAssistantDecision("discard", request_id=request_id)
    prediction = event.get("prediction")
    if not isinstance(prediction, Mapping):
        return SamAssistantDecision(
            "error",
            request_id=request_id,
            error_message="SAM worker returned no prediction payload.",
        )
    try:
        selection = deserialize_sam_selection(prediction)
    except SamAssistantValidationError as exc:
        return SamAssistantDecision("error", request_id=request_id, error_message=str(exc))
    return SamAssistantDecision(
        "apply",
        request_id=request_id,
        result=selection.result,
        failure=selection.failure,
    )


def _required_text(value: Any, name: str) -> str:
    normalized = str(value or "")
    if not normalized:
        raise SamAssistantValidationError(f"{name} is required")
    return normalized


def _normalize_prompt(value: Sequence[Any]) -> PromptPoint:
    if len(value) < 3:
        raise SamAssistantValidationError("each SAM prompt requires x, y, and label")
    try:
        x = float(value[0])
        y = float(value[1])
        label = int(value[2])
    except (TypeError, ValueError) as exc:
        raise SamAssistantValidationError("SAM prompt values are invalid") from exc
    if label not in {0, 1}:
        raise SamAssistantValidationError("SAM prompt labels must be 0 or 1")
    return x, y, label


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "")))


__all__ = [
    "SamAssistantDecision",
    "SamAssistantValidationError",
    "SamWorkerRequest",
    "build_sam_load_request",
    "build_sam_prediction_request",
    "correlate_sam_event",
    "deserialize_sam_selection",
    "serialize_sam_selection",
]
