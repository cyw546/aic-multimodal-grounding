from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.submission.validator import (
    validate_submission_files,
    validate_submission_zip,
)


def _record(bbox: list[float]) -> dict[str, object]:
    return {
        "visible": "Images/visible/000001.png",
        "infrared": "Images/infrared/000001.png",
        "depth": "Images/depth/000001.png",
        "query": "the target",
        "bbox": bbox,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_prediction_json(tmp_path: Path) -> None:
    original = {"000001_001": _record([0.1, 0.1, 0.5, 0.5])}
    prediction = {"000001_001": _record([0.2, 0.2, 0.6, 0.6])}
    original_path = tmp_path / "original.json"
    prediction_path = tmp_path / "prediction.json"
    _write_json(original_path, original)
    _write_json(prediction_path, prediction)

    report = validate_submission_files(original_path, prediction_path)
    assert report == {"sample_count": 1, "valid_bbox_count": 1}


def test_non_bbox_field_cannot_change(tmp_path: Path) -> None:
    original = {"000001_001": _record([0.1, 0.1, 0.5, 0.5])}
    prediction = {"000001_001": _record([0.2, 0.2, 0.6, 0.6])}
    prediction["000001_001"]["query"] = "modified"
    original_path = tmp_path / "original.json"
    prediction_path = tmp_path / "prediction.json"
    _write_json(original_path, original)
    _write_json(prediction_path, prediction)

    with pytest.raises(ValueError, match="was modified"):
        validate_submission_files(original_path, prediction_path)


@pytest.mark.parametrize(
    "bbox",
    [
        [0.8, 0.2, 0.3, 0.9],
        [-0.1, 0.2, 0.3, 0.9],
        [0.1, 0.2, 1.2, 0.9],
        [0.1, 0.2, 0.3],
        [0.1, None, 0.5, 0.9],
    ],
)
def test_invalid_prediction_bbox(tmp_path: Path, bbox: list[object]) -> None:
    original = {"000001_001": _record([0.1, 0.1, 0.5, 0.5])}
    prediction = {"000001_001": _record(bbox)}
    original_path = tmp_path / "original.json"
    prediction_path = tmp_path / "prediction.json"
    _write_json(original_path, original)
    _write_json(prediction_path, prediction)

    with pytest.raises(ValueError):
        validate_submission_files(original_path, prediction_path)


def test_missing_query_id_is_rejected(tmp_path: Path) -> None:
    original_path = tmp_path / "original.json"
    prediction_path = tmp_path / "prediction.json"
    _write_json(original_path, {"000001_001": _record([0.1, 0.1, 0.5, 0.5])})
    _write_json(prediction_path, {})

    with pytest.raises(ValueError, match="Query ID mismatch"):
        validate_submission_files(original_path, prediction_path)


def test_valid_submission_zip(tmp_path: Path) -> None:
    original = {"000001_001": _record([0.1, 0.1, 0.5, 0.5])}
    prediction = {"000001_001": _record([0.2, 0.2, 0.6, 0.6])}
    original_path = tmp_path / "original.json"
    prediction_path = tmp_path / "prediction.json"
    archive_path = tmp_path / "submission.zip"
    _write_json(original_path, original)
    _write_json(prediction_path, prediction)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(prediction_path, arcname="prediction.json")

    report = validate_submission_zip(original_path, archive_path)
    assert report["sample_count"] == 1


def test_zip_with_extra_file_is_rejected(tmp_path: Path) -> None:
    original_path = tmp_path / "original.json"
    archive_path = tmp_path / "submission.zip"
    _write_json(original_path, {"000001_001": _record([0.1, 0.1, 0.5, 0.5])})
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("prediction.json", "{}")
        archive.writestr("notes.txt", "not allowed")

    with pytest.raises(ValueError, match="exactly one file"):
        validate_submission_zip(original_path, archive_path)
