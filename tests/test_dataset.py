from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import MultimodalGroundingDataset, _validate_bbox


def test_bbox_validation() -> None:
    bbox = _validate_bbox([0.1, 0.2, 0.8, 0.9], "query")
    np.testing.assert_allclose(bbox, [0.1, 0.2, 0.8, 0.9])


@pytest.mark.parametrize(
    "bbox",
    [
        [0.1, 0.2, 0.3],
        [0.8, 0.2, 0.3, 0.9],
        [-0.1, 0.2, 0.3, 0.9],
        [0.1, 0.2, float("nan"), 0.9],
    ],
)
def test_invalid_bbox_is_rejected(bbox: list[float]) -> None:
    with pytest.raises(ValueError):
        _validate_bbox(bbox, "query")


def test_official_sample() -> None:
    root = Path(os.environ.get("AIC_SAMPLE_ROOT", "data/sample"))
    json_path = root / "sample.json"
    if not json_path.is_file():
        pytest.skip(f"Sample data not found at {root}")

    dataset = MultimodalGroundingDataset(json_path, require_bbox=True)
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["query_id"] == "000108_001"
    assert sample["visible"].shape == (1080, 1920, 3)
    assert sample["visible"].dtype == np.uint8
    assert sample["infrared"].shape[:2] == (1080, 1920)
    assert sample["depth"].shape == (1080, 1920)
    assert sample["depth"].dtype == np.uint16
    assert sample["depth_valid_mask"].dtype == np.bool_
    assert sample["image_size"] == (1080, 1920)
