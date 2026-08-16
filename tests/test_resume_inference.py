from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from scripts.run_official_inference import run_inference


class FixedPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, *, image, query):
        self.calls += 1
        return {"bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.7}


class FailingPredictor:
    def predict(self, *, image, query):
        raise RuntimeError("synthetic failure")


def make_dataset(tmp_path):
    data_root = tmp_path / "official"
    for modality in ("visible", "infrared", "depth"):
        (data_root / "Images" / modality).mkdir(parents=True)
    cv2.imwrite(
        str(data_root / "Images" / "visible" / "000001.png"),
        np.zeros((12, 16, 3), dtype=np.uint8),
    )
    cv2.imwrite(
        str(data_root / "Images" / "infrared" / "000001.png"),
        np.zeros((12, 16), dtype=np.uint8),
    )
    cv2.imwrite(
        str(data_root / "Images" / "depth" / "000001.png"),
        np.zeros((12, 16), dtype=np.uint16),
    )
    (data_root / "queries").mkdir()
    original = {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "the target",
        }
    }
    (data_root / "queries" / "queries.json").write_text(
        json.dumps(original), encoding="utf-8"
    )
    return data_root, original


def kwargs(tmp_path, predictor):
    data_root, _ = make_dataset(tmp_path)
    return {
        "data_root": data_root,
        "pipeline_config": tmp_path / "unused.yaml",
        "model_config": tmp_path / "unused.py",
        "weights": tmp_path / "unused.pth",
        "output": tmp_path / "predictions.json",
        "checkpoint_every": 1,
        "progress_every": 1,
        "predictor": predictor,
    }


def test_failure_saves_checkpoint_and_error_log(tmp_path):
    options = kwargs(tmp_path, FailingPredictor())
    with pytest.raises(RuntimeError, match="checkpoint saved"):
        run_inference(**options)
    assert (tmp_path / "predictions.partial.json").is_file()
    error = json.loads(
        (tmp_path / "predictions.errors.jsonl").read_text(encoding="utf-8")
    )
    assert error["query_id"] == "000001_001"


def test_resume_skips_completed_prediction(tmp_path):
    predictor = FixedPredictor()
    options = kwargs(tmp_path, predictor)
    report = run_inference(**options)
    assert report["status"] == "PASS"
    assert not (tmp_path / "predictions.partial.json").exists()

    original = json.loads(
        (options["data_root"] / "queries" / "queries.json").read_text()
    )
    original["000001_001"]["bbox"] = [0.1, 0.2, 0.8, 0.9]
    (tmp_path / "predictions.partial.json").write_text(
        json.dumps(original), encoding="utf-8"
    )
    second = FixedPredictor()
    options["predictor"] = second
    options["resume"] = True
    report = run_inference(**options)
    assert report["status"] == "PASS"
    assert second.calls == 0


def test_continue_on_error_returns_incomplete(tmp_path):
    options = kwargs(tmp_path, FailingPredictor())
    options["continue_on_error"] = True
    report = run_inference(**options)
    assert report["status"] == "INCOMPLETE"
    assert report["failure_count"] == 1
