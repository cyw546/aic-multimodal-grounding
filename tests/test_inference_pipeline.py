from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from src.inference import (
    generate_predictions,
    load_json_object,
    write_prediction_json,
)


class FakeDataset:
    def __init__(self) -> None:
        self.samples = [
            {
                "query_id": "000001_001",
                "query": "the target",
                "visible": np.zeros(
                    (32, 48, 3),
                    dtype=np.uint8,
                ),
            },
            {
                "query_id": "000001_002",
                "query": "another target",
                "visible": np.zeros(
                    (32, 48, 3),
                    dtype=np.uint8,
                ),
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class DummyPredictor:
    def predict(
        self,
        *,
        image: np.ndarray,
        query: str,
    ) -> dict[str, object]:
        assert image.shape == (32, 48, 3)
        assert query

        return {
            "bbox": [0.25, 0.25, 0.75, 0.75],
            "score": 0.0,
        }


class InvalidPredictor:
    def predict(
        self,
        *,
        image: np.ndarray,
        query: str,
    ) -> dict[str, object]:
        return {
            "bbox": [0.8, 0.2, 0.3, 0.9],
        }


def original_payload() -> dict[str, object]:
    return {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "the target",
        },
        "000001_002": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "another target",
        },
    }


def test_generate_predictions_preserves_official_fields() -> None:
    original = original_payload()

    prediction = generate_predictions(
        FakeDataset(),
        original,
        DummyPredictor(),
    )

    assert list(prediction) == [
        "000001_001",
        "000001_002",
    ]

    assert prediction["000001_001"]["bbox"] == [
        0.25,
        0.25,
        0.75,
        0.75,
    ]

    for query_id, source_record in original.items():
        for field, value in source_record.items():
            assert prediction[query_id][field] == value

    assert "bbox" not in original["000001_001"]


def test_generate_predictions_supports_limit() -> None:
    prediction = generate_predictions(
        FakeDataset(),
        original_payload(),
        DummyPredictor(),
        limit=1,
    )

    assert list(prediction) == ["000001_001"]


def test_generate_predictions_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError):
        generate_predictions(
            FakeDataset(),
            original_payload(),
            InvalidPredictor(),
        )


def test_write_and_reload_prediction_json(
    tmp_path,
) -> None:
    prediction = generate_predictions(
        FakeDataset(),
        original_payload(),
        DummyPredictor(),
    )

    output_path = (
        tmp_path
        / "baseline_v001"
        / "prediction.json"
    )

    written_path = write_prediction_json(
        prediction,
        output_path,
    )

    assert written_path == output_path
    assert output_path.is_file()
    assert not output_path.with_suffix(
        ".json.tmp"
    ).exists()

    loaded = load_json_object(output_path)
    assert loaded == json.loads(
        json.dumps(prediction)
    )
