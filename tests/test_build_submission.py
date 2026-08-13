from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.build_submission import build_submission


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_build_submission(tmp_path: Path) -> None:
    original = {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "the target",
        }
    }

    prediction = {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "the target",
            "bbox": [0.1, 0.2, 0.7, 0.9],
        }
    }

    original_path = tmp_path / "queries.json"
    prediction_path = tmp_path / "prediction.json"
    output_path = tmp_path / "submission.zip"

    write_json(original_path, original)
    write_json(prediction_path, prediction)

    report = build_submission(
        original=original_path,
        prediction=prediction_path,
        output=output_path,
    )

    assert report["status"] == "PASS"
    assert report["sample_count"] == 1
    assert report["valid_bbox_count"] == 1
    assert output_path.is_file()

    with zipfile.ZipFile(output_path) as archive:
        assert archive.namelist() == [
            "prediction.json"
        ]
