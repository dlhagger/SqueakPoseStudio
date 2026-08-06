"""Qt-free helpers for validating and persisting dense depth predictions."""

from __future__ import annotations

import json
import os
from typing import Any


class DepthMapError(ValueError):
    """Raised when an Ultralytics result has no usable dense depth map."""


def sample_depth_map(
    depth_map: Any,
    *,
    x: float,
    y: float,
    numpy_module: Any,
) -> dict[str, Any]:
    """Sample one aligned image pixel from a raw ``(H, W)`` depth map."""
    array = numpy_module.asarray(depth_map)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise DepthMapError("Depth map must be a non-empty (H, W) array.")
    px = int(float(x))
    py = int(float(y))
    height, width = array.shape
    if px < 0 or py < 0 or px >= width or py >= height:
        raise DepthMapError(
            f"Pixel ({px}, {py}) is outside depth map {width}×{height}."
        )
    value = float(array[py, px])
    valid = bool(numpy_module.isfinite(value) and value > 0)
    return {
        "x": px,
        "y": py,
        "depth": value if valid else None,
        "valid": valid,
    }


def keypoint_depth_label(
    name: str,
    depth_map: Any,
    *,
    x: float,
    y: float,
    numpy_module: Any,
) -> str:
    """Return a display-only keypoint label with its aligned depth value."""
    sample = sample_depth_map(
        depth_map,
        x=x,
        y=y,
        numpy_module=numpy_module,
    )
    value = sample.get("depth")
    value_text = f"{float(value):.3f} m" if value is not None else "invalid"
    return f"{str(name)} · {value_text}"


def depth_array_from_result(result: Any, *, numpy_module: Any) -> Any:
    """Return a clean float32 ``(H, W)`` depth array from a result object."""
    depth = getattr(result, "depth", None)
    data = getattr(depth, "data", None)
    if data is None:
        raise DepthMapError("Depth model returned no depth map.")
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()

    array = numpy_module.asarray(data, dtype=numpy_module.float32)
    array = numpy_module.squeeze(array)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise DepthMapError(
            f"Depth map must have shape (H, W); received {tuple(array.shape)}."
        )

    valid = numpy_module.isfinite(array) & (array > 0)
    if not bool(numpy_module.any(valid)):
        raise DepthMapError("Depth map contains no finite positive pixels.")
    return numpy_module.where(valid, array, 0.0).astype(
        numpy_module.float32, copy=False
    )


def depth_map_summary(depth_map: Any, *, numpy_module: Any) -> dict[str, Any]:
    valid = depth_map[numpy_module.isfinite(depth_map) & (depth_map > 0)]
    if valid.size == 0:
        raise DepthMapError("Depth map contains no finite positive pixels.")
    return {
        "height": int(depth_map.shape[0]),
        "width": int(depth_map.shape[1]),
        "valid_pixels": int(valid.size),
        "min_depth": float(valid.min()),
        "max_depth": float(valid.max()),
        "median_depth": float(numpy_module.median(valid)),
        "p02_depth": float(numpy_module.percentile(valid, 2.0)),
        "p98_depth": float(numpy_module.percentile(valid, 98.0)),
    }


def colorize_depth_map(
    depth_map: Any,
    *,
    numpy_module: Any,
    mode: str = "disparity",
) -> Any:
    """Create an RGB uint8 preview with a stable, app-owned color ramp."""
    valid_mask = numpy_module.isfinite(depth_map) & (depth_map > 0)
    valid_depth = depth_map[valid_mask]
    if valid_depth.size == 0:
        raise DepthMapError("Depth map contains no finite positive pixels.")

    values = numpy_module.zeros_like(depth_map, dtype=numpy_module.float32)
    if str(mode).lower() == "metric":
        values[valid_mask] = depth_map[valid_mask]
    else:
        values[valid_mask] = 1.0 / numpy_module.maximum(
            depth_map[valid_mask], numpy_module.finfo(numpy_module.float32).eps
        )

    valid_values = values[valid_mask]
    low = float(numpy_module.percentile(valid_values, 2.0))
    high = float(numpy_module.percentile(valid_values, 98.0))
    if high <= low:
        high = low + 1.0
    normalized = numpy_module.clip((values - low) / (high - low), 0.0, 1.0)

    # Compact inferno-like RGB ramp. Invalid pixels remain black.
    anchors = numpy_module.asarray(
        [
            [0, 0, 4],
            [66, 10, 104],
            [147, 38, 103],
            [221, 81, 58],
            [252, 165, 10],
            [252, 255, 164],
        ],
        dtype=numpy_module.float32,
    )
    scaled = normalized * float(len(anchors) - 1)
    lower = numpy_module.floor(scaled).astype(numpy_module.int32)
    upper = numpy_module.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    rgb = numpy_module.clip(rgb, 0, 255).astype(numpy_module.uint8)
    rgb[~valid_mask] = 0
    return rgb


def write_depth_prediction_files(
    depth_map: Any,
    *,
    map_path: str,
    preview_path: str,
    metadata_path: str,
    model_path: str,
    image_path: str,
    numpy_module: Any,
    cv2_module: Any,
    source_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Write one staged raw map, color preview, and compact metadata file."""
    if cv2_module is None:
        raise DepthMapError("OpenCV is required to save a depth preview.")
    for path in (map_path, preview_path, metadata_path):
        if not path:
            raise DepthMapError("Depth prediction output paths are required.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    summary = depth_map_summary(depth_map, numpy_module=numpy_module)
    if source_shape is not None:
        normalized_source_shape = tuple(int(value) for value in source_shape[:2])
        if tuple(depth_map.shape) != normalized_source_shape:
            raise DepthMapError(
                "Depth map is not aligned to the source image: "
                f"map {tuple(depth_map.shape)}, source {normalized_source_shape}."
            )
    preview_rgb = colorize_depth_map(
        depth_map,
        numpy_module=numpy_module,
        mode="disparity",
    )
    with open(map_path, "wb") as handle:
        numpy_module.save(handle, depth_map, allow_pickle=False)
    preview_bgr = preview_rgb[..., ::-1]
    if not cv2_module.imwrite(preview_path, preview_bgr):
        raise OSError(f"Could not write depth preview: {preview_path}")

    metadata = {
        **summary,
        "image_path": os.path.abspath(image_path),
        "model_path": str(model_path),
        "units": "estimated_meters",
        "scale_status": "model_default",
        "display_mode": "disparity",
        "aligned_to_source": source_shape is not None,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def serialize_depth_prediction_result(
    result: Any,
    *,
    map_path: str,
    preview_path: str,
    metadata_path: str,
    model_path: str,
    image_path: str,
    numpy_module: Any,
    cv2_module: Any,
) -> dict[str, Any]:
    """Persist a dense result and return a JSON-safe worker payload."""
    if numpy_module is None:
        raise DepthMapError("NumPy is required for depth prediction.")
    depth_map = depth_array_from_result(result, numpy_module=numpy_module)
    raw_source_shape = getattr(result, "orig_shape", None)
    source_shape = None
    try:
        if raw_source_shape is not None and len(raw_source_shape) >= 2:
            source_shape = (int(raw_source_shape[0]), int(raw_source_shape[1]))
    except (TypeError, ValueError):
        source_shape = None
    metadata = write_depth_prediction_files(
        depth_map,
        map_path=map_path,
        preview_path=preview_path,
        metadata_path=metadata_path,
        model_path=model_path,
        image_path=image_path,
        numpy_module=numpy_module,
        cv2_module=cv2_module,
        source_shape=source_shape,
    )
    return {
        "ok": True,
        "layer_id": "depth",
        "workflow": "depth",
        "depth_map_path": map_path,
        "depth_preview_path": preview_path,
        "depth_metadata_path": metadata_path,
        "depth_metadata": metadata,
    }
