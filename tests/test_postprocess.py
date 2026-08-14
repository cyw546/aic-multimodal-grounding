import torch
import pytest

from baseline.postprocess import (
    box_cxcywh_to_xyxy,
    process_prediction,
)


def test_box_conversion():
    box = torch.tensor([0.5, 0.5, 0.4, 0.2])

    result = box_cxcywh_to_xyxy(box)

    assert result == pytest.approx(
        [0.3, 0.4, 0.7, 0.6]
    )


def test_box_clip():
    box = torch.tensor([0.0, 0.5, 0.4, 0.4])

    result = box_cxcywh_to_xyxy(box)

    assert result == pytest.approx(
        [0.0, 0.3, 0.2, 0.7]
    )


def test_invalid_box():
    box = torch.tensor([0.5, 0.5, 0.0, 0.2])

    result = box_cxcywh_to_xyxy(box)

    assert result is None


def test_process_prediction():
    boxes = torch.tensor([
        [0.5, 0.5, 0.4, 0.2],
        [0.4, 0.4, 0.2, 0.2],
    ])

    scores = torch.tensor([0.8, 0.3])
    labels = ["target", "other"]

    result = process_prediction(
        boxes,
        scores,
        labels
    )

    assert result["bbox"] == pytest.approx(
        [0.3, 0.4, 0.7, 0.6]
    )

    assert result["score"] == pytest.approx(0.8)