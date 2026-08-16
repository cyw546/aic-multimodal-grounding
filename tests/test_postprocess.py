import json

import pytest
import torch

from baseline.postprocess import box_cxcywh_to_xyxy, process_prediction


def test_box_conversion():
    result = box_cxcywh_to_xyxy(torch.tensor([0.5, 0.5, 0.4, 0.2]))
    assert result == pytest.approx([0.3, 0.4, 0.7, 0.6])
    assert all(type(value) is float for value in result)


def test_box_clip():
    result = box_cxcywh_to_xyxy(torch.tensor([0.0, 0.5, 0.4, 0.4]))
    assert result == pytest.approx([0.0, 0.3, 0.2, 0.7])


def test_invalid_box():
    assert box_cxcywh_to_xyxy(torch.tensor([0.5, 0.5, 0.0, 0.2])) is None


def test_process_prediction_is_json_safe():
    result = process_prediction(
        torch.tensor([[0.5, 0.5, 0.4, 0.2], [0.4, 0.4, 0.2, 0.2]]),
        torch.tensor([0.8, 0.3]),
        ["target", "other"],
    )
    assert result["bbox"] == pytest.approx([0.3, 0.4, 0.7, 0.6])
    assert result["score"] == pytest.approx(0.8)
    assert result["label"] == "target"
    assert all(type(value) is float for value in result["bbox"])
    json.dumps(result)


def test_empty_prediction():
    assert process_prediction(torch.empty((0, 4)), torch.empty(0), []) is None


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        process_prediction(torch.zeros((2, 4)), torch.ones(1), ["target"])


@pytest.mark.parametrize("box", [
    [0.5, 0.5, float("nan"), 0.2],
    [0.5, 0.5, float("inf"), 0.2],
    [0.5, 0.2],
])
def test_invalid_box_values(box):
    with pytest.raises(ValueError):
        box_cxcywh_to_xyxy(box)
