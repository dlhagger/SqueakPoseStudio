"""OpenAI Responses contracts for editable pose-label proposals."""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

RESPONSES_URL = "https://api.openai.com/v1/responses"
MODELS_URL = "https://api.openai.com/v1/models"
COORDINATE_RANGE = 1000.0


class OpenAILabelingError(RuntimeError):
    """A safe, user-presentable model request or response error."""


@dataclass(frozen=True, slots=True)
class OpenAIModelChoice:
    slug: str
    display_name: str


@dataclass(frozen=True, slots=True)
class ProposedKeypoint:
    name: str
    x: float
    y: float
    visibility: int
    confidence: float


@dataclass(frozen=True, slots=True)
class OpenAIUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class PoseProposal:
    source_image: str
    model: str
    class_name: str
    box: tuple[float, float, float, float]
    keypoints: tuple[ProposedKeypoint, ...]
    usage: OpenAIUsage = field(default_factory=OpenAIUsage)


def list_models(
    access_token: str, *, client: httpx.Client | None = None
) -> tuple[OpenAIModelChoice, ...]:
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        response = http.get(
            MODELS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise OpenAILabelingError("Could not load models for this ChatGPT account.") from exc
    finally:
        if owns_client:
            http.close()
    raw_models = payload.get("models") if isinstance(payload, Mapping) else None
    plan_catalog = raw_models is not None
    if raw_models is None and isinstance(payload, Mapping):
        raw_models = payload.get("data")
    choices: list[OpenAIModelChoice] = []
    for value in raw_models or ():
        if not isinstance(value, Mapping):
            continue
        if plan_catalog and value.get("visibility") != "list":
            continue
        slug = str(value.get("slug") or value.get("id") or "")
        if not slug:
            continue
        choices.append(OpenAIModelChoice(slug, str(value.get("display_name") or slug)))
    return tuple(choices)


def build_pose_request(
    *,
    model: str,
    image_path: str,
    class_name: str,
    keypoint_names: Sequence[str],
) -> dict[str, Any]:
    names = tuple(str(name) for name in keypoint_names if str(name))
    if not names:
        raise OpenAILabelingError("The selected class has no keypoints to label.")
    image_url = _image_data_url(image_path)
    name_list = ", ".join(names)
    prompt = (
        f"Locate exactly one {class_name} and label these anatomical keypoints in this exact "
        f"order: {name_list}. Use the animal's anatomical left and right, not screen left and "
        "right. Coordinates are relative to the complete input image on a 0 to 1000 scale. "
        "Use visibility 2 when visible, 1 when occluded but anatomically inferable, and 0 only "
        "when the point cannot be identified; use x=0 and y=0 for visibility 0. The bounding "
        "box should enclose the full visible animal including its visible tail. Return no extra "
        "keypoints and do not omit a requested keypoint. Confidence is your estimated confidence "
        "from 0 to 1, not a calibrated probability."
    )
    return {
        "model": str(model),
        "instructions": (
            "You create pose-annotation proposals for scientific review. Follow the supplied "
            "coordinate system and schema exactly. Never invent an additional animal."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": "original"},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "squeakpose_pose_proposal",
                "strict": True,
                "schema": _pose_schema(names),
            }
        },
        "reasoning": {"effort": "low"},
        "store": False,
        "stream": True,
    }


def request_pose_proposal(
    *,
    access_token: str,
    model: str,
    image_path: str,
    image_width: float,
    image_height: float,
    class_name: str,
    keypoint_names: Sequence[str],
    client: httpx.Client | None = None,
) -> PoseProposal:
    payload = build_pose_request(
        model=model,
        image_path=image_path,
        class_name=class_name,
        keypoint_names=keypoint_names,
    )
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0))
    completed = False
    usage = OpenAIUsage()
    output_parts: list[str] = []
    try:
        with http.stream(
            "POST",
            RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            for event in _iter_sse_events(response.iter_lines()):
                event_type = str(event.get("type") or "")
                if event_type == "response.output_text.delta":
                    output_parts.append(str(event.get("delta") or ""))
                elif event_type == "response.failed":
                    error = event.get("response", {}).get("error", {})
                    message = str(error.get("message") or error.get("code") or "unknown error")
                    raise OpenAILabelingError(f"OpenAI auto-labeling failed: {message}")
                elif event_type == "response.incomplete":
                    raise OpenAILabelingError("OpenAI returned an incomplete labeling response.")
                elif event_type == "response.completed":
                    completed = True
                    completed_response = event.get("response")
                    if isinstance(completed_response, Mapping):
                        usage = _parse_usage(completed_response.get("usage"))
                    if not output_parts:
                        text = _completed_output_text(completed_response)
                        if text:
                            output_parts.append(text)
    except OpenAILabelingError:
        raise
    except httpx.HTTPStatusError as exc:
        detail = _safe_http_error(exc.response)
        raise OpenAILabelingError(f"OpenAI rejected the labeling request: {detail}") from exc
    except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
        raise OpenAILabelingError("Could not complete the OpenAI labeling request.") from exc
    finally:
        if owns_client:
            http.close()
    if not completed:
        raise OpenAILabelingError("The OpenAI response stream ended before completion.")
    try:
        result = json.loads("".join(output_parts))
    except (json.JSONDecodeError, TypeError) as exc:
        raise OpenAILabelingError("OpenAI returned an unreadable pose proposal.") from exc
    return parse_pose_proposal(
        result,
        source_image=image_path,
        model=model,
        class_name=class_name,
        keypoint_names=keypoint_names,
        image_width=image_width,
        image_height=image_height,
        usage=usage,
    )


def parse_pose_proposal(
    value: Mapping[str, Any],
    *,
    source_image: str,
    model: str,
    class_name: str,
    keypoint_names: Sequence[str],
    image_width: float,
    image_height: float,
    usage: OpenAIUsage | None = None,
) -> PoseProposal:
    if image_width <= 0 or image_height <= 0:
        raise OpenAILabelingError("The displayed image dimensions are invalid.")
    requested = tuple(str(name) for name in keypoint_names)
    raw_points = value.get("keypoints") if isinstance(value, Mapping) else None
    if not isinstance(raw_points, list):
        raise OpenAILabelingError("OpenAI returned no keypoint list.")
    points_by_name: dict[str, Mapping[str, Any]] = {}
    for point in raw_points:
        if isinstance(point, Mapping):
            points_by_name[_normalized_name(point.get("name"))] = point
    points: list[ProposedKeypoint] = []
    for name in requested:
        raw = points_by_name.get(_normalized_name(name))
        if raw is None:
            raise OpenAILabelingError(f"OpenAI omitted the '{name}' keypoint.")
        visibility = _bounded_int(raw.get("visibility"), minimum=0, maximum=2)
        normalized_x = _bounded_float(raw.get("x"), minimum=0.0, maximum=COORDINATE_RANGE)
        normalized_y = _bounded_float(raw.get("y"), minimum=0.0, maximum=COORDINATE_RANGE)
        points.append(
            ProposedKeypoint(
                name=name,
                x=0.0 if visibility == 0 else normalized_x / COORDINATE_RANGE * image_width,
                y=0.0 if visibility == 0 else normalized_y / COORDINATE_RANGE * image_height,
                visibility=visibility,
                confidence=_bounded_float(raw.get("confidence"), minimum=0.0, maximum=1.0),
            )
        )
    raw_box = value.get("bounding_box")
    if not isinstance(raw_box, Mapping):
        raise OpenAILabelingError("OpenAI returned no bounding box.")
    x1 = _bounded_float(raw_box.get("x_min"), minimum=0.0, maximum=COORDINATE_RANGE)
    y1 = _bounded_float(raw_box.get("y_min"), minimum=0.0, maximum=COORDINATE_RANGE)
    x2 = _bounded_float(raw_box.get("x_max"), minimum=0.0, maximum=COORDINATE_RANGE)
    y2 = _bounded_float(raw_box.get("y_max"), minimum=0.0, maximum=COORDINATE_RANGE)
    if x2 <= x1 or y2 <= y1:
        raise OpenAILabelingError("OpenAI returned an invalid bounding box.")
    return PoseProposal(
        source_image=str(Path(source_image).absolute()),
        model=str(model),
        class_name=str(class_name),
        box=(
            x1 / COORDINATE_RANGE * image_width,
            y1 / COORDINATE_RANGE * image_height,
            (x2 - x1) / COORDINATE_RANGE * image_width,
            (y2 - y1) / COORDINATE_RANGE * image_height,
        ),
        keypoints=tuple(points),
        usage=usage or OpenAIUsage(),
    )


def _parse_usage(value: Any) -> OpenAIUsage:
    if not isinstance(value, Mapping):
        return OpenAIUsage()
    output_details = value.get("output_tokens_details")
    reasoning_tokens = (
        _nonnegative_int(output_details.get("reasoning_tokens"))
        if isinstance(output_details, Mapping)
        else 0
    )
    return OpenAIUsage(
        input_tokens=_nonnegative_int(value.get("input_tokens")),
        output_tokens=_nonnegative_int(value.get("output_tokens")),
        reasoning_tokens=reasoning_tokens,
        total_tokens=_nonnegative_int(value.get("total_tokens")),
    )


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _pose_schema(keypoint_names: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "mouse_id": {"type": "integer"},
            "coordinate_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "left_right_convention": {
                "type": "string",
                "enum": ["mouse_anatomical"],
            },
            "keypoints": {
                "type": "array",
                "minItems": len(keypoint_names),
                "maxItems": len(keypoint_names),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(keypoint_names)},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "visibility": {"type": "integer", "enum": [0, 1, 2]},
                        "confidence": {"type": "number"},
                    },
                    "required": ["name", "x", "y", "visibility", "confidence"],
                    "additionalProperties": False,
                },
            },
            "bounding_box": {
                "type": "object",
                "properties": {
                    "x_min": {"type": "number"},
                    "y_min": {"type": "number"},
                    "x_max": {"type": "number"},
                    "y_max": {"type": "number"},
                },
                "required": ["x_min", "y_min", "x_max", "y_max"],
                "additionalProperties": False,
            },
        },
        "required": [
            "mouse_id",
            "coordinate_range",
            "left_right_convention",
            "keypoints",
            "bounding_box",
        ],
        "additionalProperties": False,
    }


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OpenAILabelingError("Could not read the current image for auto-labeling.") from exc
    if not data:
        raise OpenAILabelingError("The current image is empty.")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise OpenAILabelingError("OpenAI does not support this image format.")
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _iter_sse_events(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


def _completed_output_text(response: Any) -> str:
    if not isinstance(response, Mapping):
        return ""
    parts: list[str] = []
    for output in response.get("output", ()):
        if not isinstance(output, Mapping) or output.get("type") != "message":
            continue
        for content in output.get("content", ()):
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _safe_http_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", {}) if isinstance(payload, Mapping) else {}
        return str(error.get("message") or error.get("code") or response.status_code)
    except (ValueError, TypeError):
        return str(response.status_code)


def _normalized_name(value: Any) -> str:
    # Model output commonly varies only in separators (for example tailbase,
    # tail_base, or "tail base"). Those forms identify the same configured point.
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OpenAILabelingError("OpenAI returned a non-numeric coordinate.") from exc
    if not minimum <= number <= maximum:
        raise OpenAILabelingError("OpenAI returned a coordinate outside the requested range.")
    return number


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise OpenAILabelingError("OpenAI returned an invalid visibility value.") from exc
    if not minimum <= number <= maximum:
        raise OpenAILabelingError("OpenAI returned an invalid visibility value.")
    return number


__all__ = [
    "OpenAILabelingError",
    "OpenAIModelChoice",
    "OpenAIUsage",
    "PoseProposal",
    "ProposedKeypoint",
    "build_pose_request",
    "list_models",
    "parse_pose_proposal",
    "request_pose_proposal",
]
