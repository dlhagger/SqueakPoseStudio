"""Detached pose-instance creation and box-only replacement from saved masks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any

from squeakpose.annotation.documents import PoseAnnotationDocument
from squeakpose.services.frame_annotations import (
    SegmentationBoxTransferPlan,
    SegmentationBoxUnavailableError,
)


def _target_class(plans: Sequence[SegmentationBoxTransferPlan]) -> int:
    if not plans or len({plan.pose_class_id for plan in plans}) != 1:
        raise SegmentationBoxUnavailableError("Choose saved masks from one class.")
    return plans[0].pose_class_id


def create_pose_instances(
    document: PoseAnnotationDocument, plans: Sequence[SegmentationBoxTransferPlan]
) -> PoseAnnotationDocument:
    """Replace this class with mask boxes and unplaced keypoints; keep other classes."""
    cid = _target_class(plans)
    entries = [entry for entry in document.all_entries() if entry["class_id"] != cid]
    for plan in plans:
        box = plan.box
        entries.append(
            {
                "class_id": cid,
                "bbox": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
                "keypoints": [],
            }
        )
    result = PoseAnnotationDocument()
    result.replace_instance_entries(entries)
    result.select_instance(cid, result.instance_ids(cid)[0])
    return result


@dataclass(frozen=True, slots=True)
class BoxAssignmentPlan:
    assignments: tuple[int, ...]
    requires_review: bool


def _contains_point(polygon: Sequence[tuple[float, float]], x: float, y: float) -> bool:
    """Ray crossing with boundary points included in the mask evidence."""
    inside = False
    for index, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(index + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (
            abs(cross) <= 1e-6
            and min(ax, bx) - 1e-6 <= x <= max(ax, bx) + 1e-6
            and min(ay, by) - 1e-6 <= y <= max(ay, by) + 1e-6
        ):
            return True
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _maximum_assignment(scores: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Maximum-weight one-to-one assignment using augmenting paths, O(n cubed)."""
    count = len(scores)
    row_potential = [0.0] * (count + 1)
    column_potential = [0.0] * (count + 1)
    matched_row = [0] * (count + 1)
    previous_column = [0] * (count + 1)
    for row in range(1, count + 1):
        matched_row[0] = row
        column = 0
        minimum = [float("inf")] * (count + 1)
        visited = [False] * (count + 1)
        while True:
            visited[column] = True
            active_row = matched_row[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, count + 1):
                if visited[candidate]:
                    continue
                cost = (
                    -scores[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if cost < minimum[candidate]:
                    minimum[candidate] = cost
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(count + 1):
                if visited[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while column:
            prior = previous_column[column]
            matched_row[column] = matched_row[prior]
            column = prior
    result = [0] * count
    for column in range(1, count + 1):
        result[matched_row[column] - 1] = column - 1
    return tuple(result)


def _assignment_margin(scores: Sequence[Sequence[float]], assignments: Sequence[int]) -> float:
    """Score lost by the best alternative pairing, via shortest alternating cycles."""
    count = len(scores)
    costs = [
        [scores[row][assignments[row]] - scores[row][assignments[other]] for other in range(count)]
        for row in range(count)
    ]
    distances = [row[:] for row in costs]
    for middle in range(count):
        for row in range(count):
            for column in range(count):
                distances[row][column] = min(
                    distances[row][column], distances[row][middle] + distances[middle][column]
                )
    return min(
        (
            costs[row][other] + distances[other][row]
            for row in range(count)
            for other in range(count)
            if row != other
        ),
        default=float("inf"),
    )


def plan_box_assignments(
    entries: Sequence[Mapping[str, Any]], plans: Sequence[SegmentationBoxTransferPlan]
) -> BoxAssignmentPlan:
    """Associate same-image geometry, independent of instance IDs or labeling order.

    Keypoints inside the actual polygon are the strongest evidence. Visible points
    have full weight, occluded/inferred points half weight, unlocated points none.
    Box overlap and proximity provide fallback evidence for drafts without points.
    A global assignment prevents reusing a mask or a greedy local pairing.
    """
    if len(entries) != len(plans) or not entries:
        raise SegmentationBoxUnavailableError(
            f"There are {len(entries)} keypoint instances and {len(plans)} saved masks. "
            "Replacing all boxes requires equal counts. Add/delete instances first, or create instances from masks."
        )
    scores: list[list[float]] = []
    supported: list[list[bool]] = []
    for entry in entries:
        box = entry.get("bbox", {})
        x, y, w, h = (float(box.get(key, 0)) for key in ("x", "y", "w", "h"))
        points = [
            (float(point["x"]), float(point["y"]), 1.0 if int(point.get("vis", 2)) == 2 else 0.5)
            for point in entry.get("keypoints", [])
            if int(point.get("vis", 2)) > 0
        ]
        total_weight = sum(weight for _x, _y, weight in points)
        row_scores, row_support = [], []
        for plan in plans:
            target = plan.box
            polygon = plan.polygon or (
                (target.x, target.y),
                (target.x + target.w, target.y),
                (target.x + target.w, target.y + target.h),
                (target.x, target.y + target.h),
            )
            coverage = (
                sum(weight for px, py, weight in points if _contains_point(polygon, px, py))
                / total_weight
                if total_weight
                else 0.0
            )
            intersection = max(0.0, min(x + w, target.x + target.w) - max(x, target.x)) * max(
                0.0, min(y + h, target.y + target.h) - max(y, target.y)
            )
            union = w * h + target.w * target.h - intersection
            iou = intersection / union if union > 0 else 0.0
            scale = max(1.0, hypot(target.w, target.h))
            center_x, center_y = target.x + target.w / 2, target.y + target.h / 2
            proximity = (
                sum(
                    weight / (1.0 + hypot(px - center_x, py - center_y) / scale)
                    for px, py, weight in points
                )
                / total_weight
                if total_weight
                else 0.0
            )
            box_proximity = 1.0 / (1.0 + hypot(x + w / 2 - center_x, y + h / 2 - center_y) / scale)
            row_scores.append(10.0 * coverage + 2.0 * proximity + iou + 0.1 * box_proximity)
            row_support.append(coverage > 0.0 or iou > 0.05)
        scores.append(row_scores)
        supported.append(row_support)
    assignments = _maximum_assignment(scores)
    # Judge the complete one-to-one solution, not individual row ties: one clearly
    # located object can resolve the remaining object's assignment automatically.
    requires_review = (
        any(not supported[row][column] for row, column in enumerate(assignments))
        or _assignment_margin(scores, assignments) < 0.25
    )
    return BoxAssignmentPlan(assignments, requires_review)


def suggest_box_assignments(
    entries: Sequence[Mapping[str, Any]], plans: Sequence[SegmentationBoxTransferPlan]
) -> tuple[int, ...]:
    """Compatibility accessor for the geometry-based assignment."""
    return plan_box_assignments(entries, plans).assignments


def replace_pose_boxes(
    document: PoseAnnotationDocument,
    plans: Sequence[SegmentationBoxTransferPlan],
    assignments: Sequence[int],
) -> PoseAnnotationDocument:
    """Apply spatially associated mask boxes, preserving all keypoint data and editor IDs."""
    cid = _target_class(plans)
    entries = document.all_entries()
    targets = [entry for entry in entries if entry["class_id"] == cid]
    if (
        len(targets) != len(plans)
        or len(assignments) != len(targets)
        or sorted(assignments) != list(range(len(plans)))
    ):
        raise SegmentationBoxUnavailableError(
            "Assign each saved mask to exactly one keypoint instance."
        )
    for entry, source_index in zip(targets, assignments):
        box = plans[source_index].box
        entry["bbox"] = {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
    result = PoseAnnotationDocument()
    result.replace_instance_entries(entries)
    for class_id in document:
        result.select_instance(class_id, document.selected_instance_id(class_id))
    return result


def incomplete_pose_instances(
    document: PoseAnnotationDocument, keypoints_by_class: Sequence[Sequence[str]]
) -> list[str]:
    """Find drafts that must be completed before saving a training label file."""
    incomplete = []
    for entry in document.all_entries():
        cid = int(entry["class_id"])
        box = entry.get("bbox", {})
        available = {point["name"] for point in entry.get("keypoints", [])}
        if (
            float(box.get("w", 0)) <= 0
            or float(box.get("h", 0)) <= 0
            or not set(keypoints_by_class[cid]).issubset(available)
        ):
            incomplete.append(document.entry_id(entry))
    return incomplete
