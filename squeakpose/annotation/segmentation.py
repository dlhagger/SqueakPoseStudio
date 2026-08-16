"""Qt-free edit state and geometry helpers for segmentation annotations."""

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]
PromptPoint = tuple[float, float, int]
SegmentationEntry = dict[str, object]


@dataclass(frozen=True, slots=True)
class SegmentationBrushResult:
    """Detached geometry result from one in-place raster brush stroke."""

    points: list[Point]
    erased: bool
    pixel_count: int


@dataclass
class SegmentationEditSnapshot:
    """Detached copy of segmentation edit state suitable for undo history."""

    prompt_points: list[PromptPoint]
    preview_points: list[Point]
    preview_score: float
    accepted_masks: dict[int, SegmentationEntry]
    selected_target: int | None


@dataclass
class SegmentationEditState:
    """Qt-free prompt, preview, and accepted-mask editing state."""

    prompt_points: list[PromptPoint] = field(default_factory=list)
    preview_points: list[Point] = field(default_factory=list)
    preview_score: float = 0.0
    accepted_masks: dict[int, SegmentationEntry] = field(default_factory=dict)
    selected_target: int | None = None
    _undo_snapshots: list[SegmentationEditSnapshot] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def has_preview(self) -> bool:
        return len(self.preview_points) >= 3

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_snapshots)

    def select_target(self, class_id: int | None) -> None:
        self.selected_target = None if class_id is None else int(class_id)

    def add_prompt(self, x: float, y: float, *, positive: bool = True) -> PromptPoint:
        prompt = (float(x), float(y), 1 if positive else 0)
        self.prompt_points.append(prompt)
        return prompt

    def remove_last_prompt(self) -> PromptPoint | None:
        if not self.prompt_points:
            return None
        return self.prompt_points.pop()

    def set_preview(self, points: Sequence[Point], score: float = 0.0) -> None:
        self.preview_points = [(float(x), float(y)) for x, y in points]
        self.preview_score = float(score)

    def clear_preview(self) -> None:
        self.preview_points = []
        self.preview_score = 0.0

    def clear_prompt_state(self) -> None:
        self.prompt_points.clear()
        self.clear_preview()

    def accept_preview(self) -> SegmentationEntry | None:
        """Accept the current preview for the selected class, if usable."""
        if self.selected_target is None or not self.has_preview:
            return None
        class_id = int(self.selected_target)
        entry: SegmentationEntry = {
            "class_id": class_id,
            "segments": [(float(x), float(y)) for x, y in self.preview_points],
            "score": float(self.preview_score),
        }
        self.accepted_masks[class_id] = entry
        self.clear_prompt_state()
        return deepcopy(entry)

    def set_accepted_entry(
        self,
        class_id: int,
        entry: SegmentationEntry,
    ) -> SegmentationEntry:
        normalized_id = int(class_id)
        stored = deepcopy(entry)
        stored["class_id"] = normalized_id
        self.accepted_masks[normalized_id] = stored
        return deepcopy(stored)

    def replace_accepted_masks(self, entries: Mapping[int, SegmentationEntry]) -> None:
        self.accepted_masks = {
            int(class_id): self._normalized_entry(class_id, entry)
            for class_id, entry in entries.items()
        }

    @staticmethod
    def _normalized_entry(class_id: int, entry: SegmentationEntry) -> SegmentationEntry:
        stored = deepcopy(entry)
        stored["class_id"] = int(class_id)
        return stored

    def clear_accepted_mask(self, class_id: int | None = None) -> bool:
        target = self.selected_target if class_id is None else int(class_id)
        if target is None or target not in self.accepted_masks:
            return False
        del self.accepted_masks[target]
        return True

    def reset(
        self,
        *,
        accepted_masks: Mapping[int, SegmentationEntry] | None = None,
        selected_target: int | None = None,
    ) -> None:
        self.prompt_points = []
        self.preview_points = []
        self.preview_score = 0.0
        self.replace_accepted_masks(accepted_masks or {})
        self.selected_target = None if selected_target is None else int(selected_target)
        self._undo_snapshots.clear()

    def snapshot(self) -> SegmentationEditSnapshot:
        return SegmentationEditSnapshot(
            prompt_points=deepcopy(self.prompt_points),
            preview_points=deepcopy(self.preview_points),
            preview_score=float(self.preview_score),
            accepted_masks=deepcopy(self.accepted_masks),
            selected_target=self.selected_target,
        )

    def restore(self, snapshot: SegmentationEditSnapshot) -> None:
        self.prompt_points = deepcopy(snapshot.prompt_points)
        self.preview_points = deepcopy(snapshot.preview_points)
        self.preview_score = float(snapshot.preview_score)
        self.accepted_masks = deepcopy(snapshot.accepted_masks)
        self.selected_target = snapshot.selected_target

    def push_undo_snapshot(self) -> SegmentationEditSnapshot:
        snapshot = self.snapshot()
        self._undo_snapshots.append(snapshot)
        return snapshot

    def undo(self) -> bool:
        if not self._undo_snapshots:
            return False
        self.restore(self._undo_snapshots.pop())
        return True


def clamp_point_to_image(
    x: float,
    y: float,
    image_width: float,
    image_height: float,
) -> tuple[int, int]:
    """Round and clamp a point to the image coordinate bounds."""
    max_x = max(1, int(round(float(image_width))) - 1)
    max_y = max(1, int(round(float(image_height))) - 1)
    clamped_x = int(round(float(x)))
    clamped_y = int(round(float(y)))
    if clamped_x < 0:
        clamped_x = 0
    elif clamped_x > max_x:
        clamped_x = max_x
    if clamped_y < 0:
        clamped_y = 0
    elif clamped_y > max_y:
        clamped_y = max_y
    return clamped_x, clamped_y


def normalize_polygon_points(points: Iterable[object]) -> list[Point]:
    """Convert indexable point pairs to floats while preserving source order.

    Malformed entries are ignored, matching the existing scene-item extraction
    behavior. Polygon closure is implicit and is therefore not duplicated here.
    """
    normalized: list[Point] = []
    for pair in points:
        try:
            normalized.append((float(pair[0]), float(pair[1])))  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return normalized


def polygon_bounds(points: Iterable[object]) -> tuple[float, float, float, float] | None:
    """Return the tight ``(x, y, width, height)`` bounds of a usable polygon."""
    normalized = normalize_polygon_points(points)
    if len(normalized) < 3:
        return None
    xs = [point[0] for point in normalized]
    ys = [point[1] for point in normalized]
    min_x = min(xs)
    min_y = min(ys)
    width = max(xs) - min_x
    height = max(ys) - min_y
    if width <= 0 or height <= 0:
        return None
    return min_x, min_y, width, height


def segmentation_mask_shape(image_width: float, image_height: float) -> tuple[int, int]:
    """Return the existing OpenCV mask shape convention, ``(height, width)``."""
    return (
        max(1, int(round(float(image_height)))),
        max(1, int(round(float(image_width)))),
    )


def polygon_to_mask(
    points: Iterable[object],
    *,
    image_width: float,
    image_height: float,
    numpy_module: Any,
    cv2_module: Any,
) -> Any | None:
    """Rasterize one polygon to the uint8 mask used by brush editing."""
    normalized = normalize_polygon_points(points)
    if numpy_module is None or cv2_module is None or len(normalized) < 3:
        return None
    mask = numpy_module.zeros(
        segmentation_mask_shape(image_width, image_height),
        dtype=numpy_module.uint8,
    )
    polygon = numpy_module.array(normalized, dtype=numpy_module.int32).reshape((-1, 1, 2))
    cv2_module.fillPoly(mask, [polygon], 255)
    return mask


def mask_to_polygon(
    mask: Any,
    *,
    cv2_module: Any,
    anchor_points: Iterable[object] | None = None,
    max_points: int = 1200,
) -> list[Point]:
    """Extract the anchored or largest external contour from a raster mask."""
    if cv2_module is None or mask is None:
        return []
    contours_info = cv2_module.findContours(
        mask,
        cv2_module.RETR_EXTERNAL,
        cv2_module.CHAIN_APPROX_NONE,
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return []

    anchor = normalize_polygon_points(() if anchor_points is None else anchor_points)
    contour = None
    if anchor:
        anchor_x, anchor_y = anchor[0]
        anchored = []
        for candidate in contours:
            try:
                inside = cv2_module.pointPolygonTest(
                    candidate,
                    (float(anchor_x), float(anchor_y)),
                    False,
                )
            except Exception:  # noqa: BLE001 - tolerate backend-specific contour errors
                inside = -1
            if inside >= 0:
                anchored.append(candidate)
        if anchored:
            contour = max(anchored, key=cv2_module.contourArea)
    if contour is None:
        contour = max(contours, key=cv2_module.contourArea)
    if contour is None or len(contour) < 3:
        return []

    points: list[Point] = []
    for node in contour:
        try:
            points.append((float(node[0][0]), float(node[0][1])))
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    if len(points) < 3:
        return []
    points = downsample_polygon_points(points, max_points=max_points)

    if len(anchor) >= 3 and len(points) >= 3:
        if polygon_signed_area(anchor) * polygon_signed_area(points) < 0:
            points.reverse()
        points = rotate_polygon_to_anchor(points, anchor[0])
    return points


def apply_brush_stroke(
    mask: Any,
    *,
    end: Point,
    start: Point | None = None,
    radius: float = 8,
    add: bool,
    image_width: float,
    image_height: float,
    cv2_module: Any,
    anchor_points: Iterable[object] | None = None,
    max_points: int = 1200,
) -> SegmentationBrushResult | None:
    """Apply one clipped circular/linear brush stroke to ``mask`` in place."""
    if cv2_module is None or mask is None or image_width <= 0 or image_height <= 0:
        return None
    end_x, end_y = clamp_point_to_image(end[0], end[1], image_width, image_height)
    if start is None:
        start_x, start_y = end_x, end_y
    else:
        start_x, start_y = clamp_point_to_image(
            start[0],
            start[1],
            image_width,
            image_height,
        )

    normalized_radius = max(2, int(round(float(radius))))
    value = 255 if add else 0
    thickness = max(2, normalized_radius * 2)
    cv2_module.circle(mask, (end_x, end_y), normalized_radius, value, thickness=-1)
    if start_x != end_x or start_y != end_y:
        cv2_module.line(
            mask,
            (start_x, start_y),
            (end_x, end_y),
            value,
            thickness=thickness,
        )

    pixel_count = int(cv2_module.countNonZero(mask))
    if pixel_count == 0:
        return SegmentationBrushResult(points=[], erased=True, pixel_count=0)
    return SegmentationBrushResult(
        points=mask_to_polygon(
            mask,
            cv2_module=cv2_module,
            anchor_points=anchor_points,
            max_points=max_points,
        ),
        erased=False,
        pixel_count=pixel_count,
    )


def downsample_polygon_points(points: list[Point], max_points: int = 1200) -> list[Point]:
    """Reduce a polygon by taking evenly spaced source vertices."""
    if len(points) <= max_points:
        return points
    step = max(1, (len(points) + max_points - 1) // max_points)
    reduced = points[::step]
    if len(reduced) < 3:
        return points[:3]
    return reduced


def polygon_signed_area(points: Sequence[Point]) -> float:
    """Return a polygon's signed area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += (float(x1) * float(y2)) - (float(x2) * float(y1))
    return 0.5 * total


def rotate_polygon_to_anchor(points: list[Point], anchor: Point) -> list[Point]:
    """Rotate vertices so the point nearest to ``anchor`` is first."""
    if not points:
        return points
    anchor_x, anchor_y = float(anchor[0]), float(anchor[1])
    best_index = 0
    best_distance = float("inf")
    for index, (x, y) in enumerate(points):
        distance = ((float(x) - anchor_x) ** 2) + ((float(y) - anchor_y) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    if best_index == 0:
        return points
    return points[best_index:] + points[:best_index]
