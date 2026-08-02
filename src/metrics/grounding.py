from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


Box = Sequence[float]


def validate_bbox(box: Box, *, name: str = "bbox") -> np.ndarray:
    """Return a validated normalized [x1, y1, x2, y2] box."""
    try:
        value = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain four numeric values") from exc

    if value.shape != (4,):
        raise ValueError(f"{name} must contain exactly four values, got shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinity")

    x1, y1, x2, y2 = value.tolist()
    if not (0.0 <= x1 < x2 <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= x1 < x2 <= 1, got {value.tolist()}")
    if not (0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= y1 < y2 <= 1, got {value.tolist()}")
    return value


def box_iou(prediction: Box, target: Box) -> float:
    """Calculate IoU for two normalized xyxy boxes."""
    pred = validate_bbox(prediction, name="prediction")
    truth = validate_bbox(target, name="target")

    intersection_width = max(0.0, min(pred[2], truth[2]) - max(pred[0], truth[0]))
    intersection_height = max(0.0, min(pred[3], truth[3]) - max(pred[1], truth[1]))
    intersection = intersection_width * intersection_height

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    union = pred_area + truth_area - intersection
    return float(intersection / union)


def acc_at_iou(
    predictions: Sequence[Box],
    targets: Sequence[Box],
    *,
    threshold: float = 0.5,
) -> float:
    """Return the fraction of samples whose IoU is at least threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if len(predictions) != len(targets):
        raise ValueError(
            f"prediction and target counts differ: {len(predictions)} != {len(targets)}"
        )
    if not predictions:
        raise ValueError("at least one sample is required")

    ious = np.asarray(
        [box_iou(prediction, target) for prediction, target in zip(predictions, targets)],
        dtype=np.float64,
    )
    return float(np.mean(ious >= threshold))


def evaluate_predictions(
    predictions: Sequence[Box],
    targets: Sequence[Box],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return competition metric plus diagnostic IoU statistics."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"prediction and target counts differ: {len(predictions)} != {len(targets)}"
        )
    if not predictions:
        raise ValueError("at least one sample is required")

    ious = np.asarray(
        [box_iou(prediction, target) for prediction, target in zip(predictions, targets)],
        dtype=np.float64,
    )
    passed = ious >= threshold
    return {
        "sample_count": int(ious.size),
        "correct_count": int(passed.sum()),
        "threshold": float(threshold),
        "acc_at_iou": float(passed.mean()),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "ious": ious.tolist(),
    }
