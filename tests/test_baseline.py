import json

import numpy as np
import pytest
import torch

from baseline.inference import Baseline


class FakeModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, image, query, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def raw(boxes, scores, labels):
    return {"boxes": boxes, "scores": scores, "labels": labels}


def test_predict_contract_is_json_safe():
    model = FakeModel([raw(
        torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        torch.tensor([0.8]),
        ["target"],
    )])
    result = Baseline(model=model).predict(
        np.zeros((8, 10, 3), dtype=np.uint8), "target"
    )
    assert set(result) == {"bbox", "score"}
    assert result["bbox"] == pytest.approx([0.3, 0.4, 0.7, 0.6])
    assert all(type(value) is float for value in result["bbox"])
    assert type(result["score"]) is float
    json.dumps(result)


def test_predict_retries_and_returns_default():
    empty = raw(torch.empty((0, 4)), torch.empty(0), [])
    found = raw(torch.tensor([[0.5, 0.5, 0.2, 0.2]]), torch.tensor([0.4]), ["x"])
    model = FakeModel([empty, found])
    result = Baseline(model=model).predict(np.zeros((4, 4, 3), dtype=np.uint8), "x")
    assert result["bbox"] == pytest.approx([0.4, 0.4, 0.6, 0.6])
    assert model.calls[1] == {"box_threshold": 0.05, "text_threshold": 0.05}

    model = FakeModel([empty, empty])
    result = Baseline(model=model).predict(np.zeros((4, 4, 3), dtype=np.uint8), "x")
    assert result == {"bbox": [0.25, 0.25, 0.75, 0.75], "score": 0.0}
    json.dumps(result)


def test_paths_are_required_without_model():
    with pytest.raises(ValueError, match="config_path and weight_path"):
        Baseline()


def test_yaml_configuration(tmp_path, monkeypatch):
    config = tmp_path / "baseline.yaml"
    config.write_text(
        "model:\n  box_threshold: 0.4\n  text_threshold: 0.3\n"
        "inference:\n  device: auto\n",
        encoding="utf-8",
    )
    captured = {}

    class StubModel:
        def __init__(self, config_path, weight_path, device=None):
            captured.update(config_path=config_path, weight_path=weight_path, device=device)

    monkeypatch.setattr("baseline.inference.GroundingModel", StubModel)
    predictor = Baseline.from_yaml(
        config, model_config_path="model.py", weight_path="model.pth"
    )
    assert captured == {
        "config_path": "model.py", "weight_path": "model.pth", "device": None
    }
    assert predictor.box_threshold == pytest.approx(0.4)
    assert predictor.text_threshold == pytest.approx(0.3)
