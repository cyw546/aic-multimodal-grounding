from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_local_validation import build_local_validation


def _record(index: int) -> dict[str, object]:
    name = f"image_{index}.jpg"
    return {
        "sample_id": f"refcoco_val_{index:08d}",
        "source": "refcoco",
        "split": "val",
        "query_id": f"refcoco_val_{index:08d}",
        "query": f"target {index}",
        "visible_path": name,
        "infrared_path": None,
        "depth_path": None,
        "bbox": [0.1, 0.2, 0.8, 0.9],
        "bbox_format": "xyxy_norm",
        "width": 16,
        "height": 12,
        "image_id": index,
    }


def _dataset(tmp_path: Path, count: int = 6) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    records = [_record(index) for index in range(count)]
    for record in records:
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_root / record["visible_path"]), image)
    source = tmp_path / "val.jsonl"
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return source, image_root


def test_builds_deterministic_validated_subset(tmp_path: Path) -> None:
    source, image_root = _dataset(tmp_path)
    output = tmp_path / "local_val_4.jsonl"
    visualization_dir = tmp_path / "visualizations"

    first = build_local_validation(
        source,
        image_root,
        output,
        visualization_dir,
        sample_count=4,
        visualization_count=2,
        seed=42,
    )
    first_content = output.read_text(encoding="utf-8")
    second = build_local_validation(
        source,
        image_root,
        output,
        visualization_dir,
        sample_count=4,
        visualization_count=2,
        seed=42,
    )

    assert output.read_text(encoding="utf-8") == first_content
    assert first["status"] == second["status"] == "PASS"
    assert first["source_count"] == 6
    assert first["sample_count"] == 4
    assert first["visualization_count"] == 2
    assert len(list(visualization_dir.glob("*.jpg"))) == 2


def test_rejects_request_larger_than_dataset(tmp_path: Path) -> None:
    source, image_root = _dataset(tmp_path, count=2)
    with pytest.raises(ValueError, match="dataset contains 2"):
        build_local_validation(
            source,
            image_root,
            tmp_path / "output.jsonl",
            tmp_path / "visualizations",
            sample_count=3,
            visualization_count=1,
        )


def test_rejects_invalid_bbox(tmp_path: Path) -> None:
    source, image_root = _dataset(tmp_path, count=1)
    record = _record(0)
    record["bbox"] = [0.8, 0.2, 0.1, 0.9]
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid normalized bbox"):
        build_local_validation(
            source,
            image_root,
            tmp_path / "output.jsonl",
            tmp_path / "visualizations",
            sample_count=1,
            visualization_count=1,
        )
