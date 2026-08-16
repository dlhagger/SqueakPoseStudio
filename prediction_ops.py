"""Compatibility imports for prediction serialization helpers.

Package code should import :mod:`squeakpose.services.prediction_serialization`.
"""

from squeakpose.services.prediction_serialization import (
    best_predictions_by_class_from_payload,
    prediction_confidences_by_class,
    rank_prediction_frames,
    serialize_prediction_result,
    top_prediction_from_payload,
)

__all__ = [
    "best_predictions_by_class_from_payload",
    "prediction_confidences_by_class",
    "rank_prediction_frames",
    "serialize_prediction_result",
    "top_prediction_from_payload",
]
