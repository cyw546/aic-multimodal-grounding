from __future__ import annotations

import math

import pytest

from src.metrics import acc_at_iou, box_iou, evaluate_predictions, validate_bbox


def test_identical_boxes_have_iou_one() -> None:
    box = [0.1, 0.2, 0.5, 0.8]
    assert box_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero() -> None:
    assert box_iou([0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]) == 0.0


def test_threshold_is_inclusive() -> None:
    target = [0.0, 0.0, 1.0, 1.0]
    prediction = [0.0, 0.0, 0.5, 1.0]
    assert box_iou(prediction, target) == pytest.approx(0.5)
    assert acc_at_iou([prediction], [target], threshold=0.5) == 1.0


def test_acc_and_diagnostics() -> None:
    targets = [[0.0, 0.0, 1.0, 1.0]] * 2
    predictions = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.25, 1.0]]
    report = evaluate_predictions(predictions, targets)
    assert report["sample_count"] == 2
    assert report["correct_count"] == 1
    assert report["acc_at_iou"] == pytest.approx(0.5)
    assert report["mean_iou"] == pytest.approx(0.625)


@pytest.mark.parametrize(
    "box",
    [
        [0.1, 0.2, 0.3],
        [0.8, 0.2, 0.3, 0.9],
        [-0.1, 0.2, 0.3, 0.9],
        [0.1, 0.2, 1.1, 0.9],
        [0.1, 0.2, math.nan, 0.9],
        [0.1, 0.2, math.inf, 0.9],
    ],
)
def test_invalid_boxes_are_rejected(box: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_bbox(box)
