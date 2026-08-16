from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from scripts.run_official_inference import run_inference
from src.inference import GroundingDINOPredictor


class FakeBaseline:
    def predict(self, *, image, query):
        assert image.shape == (12, 16, 3)
        assert query == "the target"
        return {"bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.7}


def test_adapter_implements_keyword_pipeline_contract():
    predictor = GroundingDINOPredictor(FakeBaseline())
    result = predictor.predict(
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        query="the target",
    )
    assert result == {"bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.7}


def test_adapter_rejects_missing_model_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="model weights"):
        GroundingDINOPredictor.from_paths(
            pipeline_config_path=tmp_path / "pipeline.yaml",
            model_config_path=tmp_path / "model.py",
            weight_path=tmp_path / "model.pth",
        )


def test_official_runner_with_fake_predictor(tmp_path):
    data_root = tmp_path / "official"
    image_dir = data_root / "Images"
    for modality in ("visible", "infrared", "depth"):
        (image_dir / modality).mkdir(parents=True)

    visible = np.zeros((12, 16, 3), dtype=np.uint8)
    infrared = np.zeros((12, 16), dtype=np.uint8)
    depth = np.zeros((12, 16), dtype=np.uint16)
    assert cv2.imwrite(str(image_dir / "visible" / "000001.png"), visible)
    assert cv2.imwrite(str(image_dir / "infrared" / "000001.png"), infrared)
    assert cv2.imwrite(str(image_dir / "depth" / "000001.png"), depth)

    query_dir = data_root / "queries"
    query_dir.mkdir()
    original = {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "the target",
        }
    }
    query_path = query_dir / "queries.json"
    query_path.write_text(json.dumps(original), encoding="utf-8")
    output = tmp_path / "predictions.json"

    report = run_inference(
        data_root=data_root,
        pipeline_config=tmp_path / "unused.yaml",
        model_config=tmp_path / "unused.py",
        weights=tmp_path / "unused.pth",
        output=output,
        predictor=GroundingDINOPredictor(FakeBaseline()),
    )

    assert report["status"] == "PASS"
    assert report["processed_count"] == 1
    assert report["complete"] is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["000001_001"]["bbox"] == [0.1, 0.2, 0.8, 0.9]
