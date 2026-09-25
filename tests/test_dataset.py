from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.dataset import MultimodalGroundingDataset, _read_image, _validate_bbox


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


def test_depth_jpeg_is_converted_to_single_channel(tmp_path) -> None:
    path = tmp_path / "depth.jpg"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 20
    image[:, :, 1] = 80
    image[:, :, 2] = 140
    assert cv2.imwrite(str(path), image)

    depth = _read_image(path, "depth")

    assert depth.shape == (12, 16)
    assert depth.dtype == np.uint8
    assert depth.flags.c_contiguous


def test_official_dataset() -> None:
    root_value = os.environ.get("AIC_OFFICIAL_ROOT")
    if not root_value:
        pytest.skip(
            "AIC_OFFICIAL_ROOT is not configured; "
            "skipping official dataset integration test"
        )

    root = Path(root_value)
    json_path = root / "queries" / "queries.json"

    try:
        json_exists = json_path.is_file()
    except OSError as exc:
        pytest.skip(f"Official data is not accessible: {exc}")

    if not json_exists:
        pytest.skip(f"Official dataset not found at {root}")

    dataset = MultimodalGroundingDataset(
        json_path=json_path,
        data_root=root,
        require_bbox=False,
        validate_files=True,
    )

    assert len(dataset) > 0
    expected_count = os.environ.get("AIC_EXPECTED_QUERY_COUNT")
    if expected_count:
        assert len(dataset) == int(expected_count)

    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in indices:
        sample = dataset[index]

        assert isinstance(sample["query_id"], str)
        assert sample["query_id"]
        assert isinstance(sample["query"], str)
        assert sample["query"]

        assert sample["visible"].ndim == 3
        assert sample["visible"].shape[2] == 3
        assert sample["visible"].dtype == np.uint8

        assert sample["infrared"].ndim in (2, 3)
        assert sample["depth"].ndim == 2
        assert sample["depth"].dtype in (np.uint8, np.uint16)
        assert sample["depth_valid_mask"].dtype == np.bool_

        assert (
            sample["visible"].shape[:2]
            == sample["infrared"].shape[:2]
            == sample["depth"].shape[:2]
            == sample["image_size"]
        )
        assert "bbox" not in sample

        for path in sample["paths"].values():
            assert Path(path).is_file()