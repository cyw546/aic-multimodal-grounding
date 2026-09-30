from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.export_grounding_dino_candidates import export_candidates


class FakePredictor:
    def __init__(self) -> None:
        self.calls = []

    def predict_candidates(self, *, image, query, top_k):
        self.calls.append((image.shape, query, top_k))
        return {
            "bbox": [0.1, 0.2, 0.8, 0.9],
            "score": 0.9,
            "candidates": [
                {"bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.9, "label": "target"},
                {"bbox": [0.2, 0.1, 0.9, 0.8], "score": 0.8, "label": "other"},
            ][:top_k],
            "stage": "primary",
            "primary_candidate_count": 2,
            "fallback_candidate_count": 0,
        }


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((12, 16, 3), dtype=np.uint8),
    )
    assert success
    (image_root / "image.jpg").write_bytes(encoded.tobytes())
    record = {
        "sample_id": "refcoco_val_00000001",
        "source": "refcoco",
        "split": "val",
        "query_id": "refcoco_val_00000001",
        "query": "the target",
        "visible_path": "image.jpg",
        "bbox": [0.1, 0.2, 0.8, 0.9],
        "bbox_format": "xyxy_norm",
        "width": 16,
        "height": 12,
    }
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return validation, image_root


def test_export_candidates_writes_unified_jsonl(tmp_path: Path) -> None:
    validation, image_root = _dataset(tmp_path)
    ids = tmp_path / "ids.txt"
    ids.write_text("refcoco_val_00000001\n", encoding="utf-8")
    output = tmp_path / "candidates.jsonl"
    predictor = FakePredictor()

    report = export_candidates(
        validation_path=validation,
        image_root=image_root,
        output_path=output,
        pipeline_config=tmp_path / "unused.yaml",
        model_config=tmp_path / "unused.py",
        weights=tmp_path / "unused.pth",
        ids_path=ids,
        top_k=2,
        predictor=predictor,
    )

    assert report["sample_count"] == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["query_id"] == "refcoco_val_00000001"
    assert len(record["candidates"]) == 2
    assert record["model_name"] == "grounding_dino_swinb"
    assert predictor.calls == [((12, 16, 3), "the target", 2)]
