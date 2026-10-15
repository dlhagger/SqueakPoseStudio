"""Class-isolated Ultralytics trackers with frame-local detection index remapping."""

from __future__ import annotations

from typing import Any


class ClassIsolatedTracker:
    """Run each class through an independent tracker, sharing an output ID namespace."""

    def __init__(self, template: Any, class_ids: list[int]) -> None:
        # Construct every tracker before processing frames: Ultralytics constructors
        # reset the shared BaseTrack counter.
        self.trackers = {class_id: type(template)(args=template.args) for class_id in class_ids}
        self.ids: dict[tuple[int, int], int] = {}

    @property
    def tracked_stracks(self) -> list[Any]:
        return [track for tracker in self.trackers.values() for track in tracker.tracked_stracks]

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()
        self.ids.clear()

    def update(self, detections: Any, image: Any = None, **kwargs: Any) -> Any:
        import numpy as np

        unknown = set(int(value) for value in detections.cls) - self.trackers.keys()
        if unknown:
            raise ValueError(f"Unexpected model class IDs: {sorted(unknown)}")
        output = []
        for class_id, tracker in self.trackers.items():
            indices = np.flatnonzero(detections.cls == class_id)
            extras = dict(kwargs)
            if extras.get("feats") is not None:
                extras["feats"] = extras["feats"][indices]
            tracks = tracker.update(detections[indices], image, **extras)
            for track in tracks:
                row = track.copy()
                key = (class_id, int(row[4]))
                row[4] = self.ids.setdefault(key, len(self.ids) + 1)
                row[-1] = indices[int(row[-1])]
                output.append(row)
        if not output:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(sorted(output, key=lambda row: row[-1]), dtype=np.float32)


def install_class_isolated_tracking(model: Any) -> None:
    """Wrap the model's trackers after Ultralytics initializes them; no global patches."""
    if not hasattr(model, "add_callback"):
        return  # Lightweight injected runtime adapters may own their own tracking.
    if getattr(model, "_squeakpose_class_tracking", False):
        return
    from ultralytics.trackers import register_tracker

    register_tracker(model, persist=True)
    names = model.names
    class_ids = list(names) if isinstance(names, dict) else list(range(len(names)))

    def isolate(predictor: Any) -> None:
        predictor.trackers = [
            tracker
            if isinstance(tracker, ClassIsolatedTracker)
            else ClassIsolatedTracker(tracker, class_ids)
            for tracker in predictor.trackers
        ]

    model.add_callback("on_predict_start", isolate)
    model._squeakpose_class_tracking = True
