from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation import (
    evaluate_prediction_records,
    freeze_query_ids,
    load_candidate_records,
    load_prediction_records,
    load_query_ids,
    load_validation_records,
    recommend_priority,
    select_validation_records,
    visualize_failures,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _validation(path: Path) -> Path:
    records = [
        {
            "query_id": "a",
            "query": "target a",
            "visible_path": "a.jpg",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "bbox_format": "xyxy_norm",
            "difficulty_categories": ["small_or_distant_target"],
        },
        {
            "query_id": "b",
            "query": "target b",
            "visible_path": "b.jpg",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "bbox_format": "xyxy_norm",
            "difficulty_categories": ["occlusion"],
        },
        {
            "query_id": "c",
            "query": "target c",
            "visible_path": "c.jpg",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "bbox_format": "xyxy_norm",
            "difficulty_categories": ["similar_targets"],
        },
    ]
    return _write_jsonl(path, records)


def test_evaluation_counts_missing_invalid_and_candidate_coverage(tmp_path: Path) -> None:
    validation = load_validation_records(_validation(tmp_path / "validation.jsonl"))
    prediction_path = _write_jsonl(tmp_path / "predictions.jsonl", [
        {"query_id": "a", "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.9},
        {"query_id": "b", "bbox": [0.8, 0.8, 0.7, 0.9], "score": 0.8},
    ])
    candidate_path = _write_jsonl(tmp_path / "candidates.jsonl", [
        {"query_id": "a", "candidates": [{"bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.9}]},
        {"query_id": "b", "candidates": [{"bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.7}]},
        {"query_id": "c", "candidates": [{"bbox": [0.0, 0.0, 0.4, 1.0], "score": 0.4}]},
    ])

    result = evaluate_prediction_records(
        validation,
        load_prediction_records(prediction_path),
        load_candidate_records(candidate_path),
    )
    summary = result["summary"]
    rows = {row["query_id"]: row for row in result["rows"]}

    assert summary["acc_at_0_5"] == pytest.approx(1 / 3)
    assert summary["mean_iou"] == pytest.approx(1 / 3)
    assert summary["valid_bbox_count"] == 1
    assert summary["invalid_bbox_count"] == 1
    assert summary["missing_prediction_count"] == 1
    assert summary["coverage_at_1"] == pytest.approx(2 / 3)
    assert rows["b"]["failure_category"] == "candidate_available_but_final_missing_or_invalid"
    assert rows["c"]["failure_category"] == "target_not_in_candidates"
    assert "small_target" in rows["a"]["difficulty_categories"]


def test_ranking_error_is_detected_at_top_five(tmp_path: Path) -> None:
    validation = load_validation_records(_validation(tmp_path / "validation.jsonl"))[:1]
    predictions = {
        "a": {
            "query_id": "a",
            "bbox": [0.0, 0.0, 0.3, 1.0],
            "bbox_error": None,
            "score": 0.8,
            "model_name": None,
            "model_version": None,
        }
    }
    candidates = {
        "a": [
            {"rank": 1, "bbox": [0.0, 0.0, 0.3, 1.0], "bbox_error": None, "score": 0.8, "label": None},
            {"rank": 2, "bbox": [0.0, 0.0, 1.0, 1.0], "bbox_error": None, "score": 0.7, "label": None},
        ]
    }

    result = evaluate_prediction_records(validation, predictions, candidates)
    row = result["rows"][0]

    assert row["coverage_at_1"] is False
    assert row["coverage_at_5"] is True
    assert row["failure_category"] == "candidate_ranking_error"
    assert result["summary"]["coverage_at_5"] == 1.0


def test_fixed_manifest_selection_and_freeze(tmp_path: Path) -> None:
    validation = load_validation_records(_validation(tmp_path / "validation.jsonl"))
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("c\na\n", encoding="utf-8")
    selected = select_validation_records(validation, load_query_ids(ids_path))
    output = freeze_query_ids([record["query_id"] for record in selected], tmp_path / "fixed.txt")

    assert [record["query_id"] for record in selected] == ["c", "a"]
    assert output.read_text(encoding="utf-8") == "c\na\n"


def test_failure_visualization_draws_ground_truth_prediction_and_candidates(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    (image_root / "a.jpg").write_bytes(encoded.tobytes())

    validation = load_validation_records(_validation(tmp_path / "validation.jsonl"))[:1]
    predictions = {
        "a": {
            "query_id": "a",
            "bbox": [0.0, 0.0, 0.3, 1.0],
            "bbox_error": None,
            "score": None,
            "model_name": None,
            "model_version": None,
        }
    }
    candidates = {"a": [
        {"rank": 1, "bbox": [0.0, 0.0, 0.3, 1.0], "bbox_error": None, "score": 0.8, "label": None},
        {"rank": 2, "bbox": [0.0, 0.0, 1.0, 1.0], "bbox_error": None, "score": 0.7, "label": None},
    ]}
    result = evaluate_prediction_records(validation, predictions, candidates)
    outputs = visualize_failures(
        result["rows"],
        validation,
        candidates,
        image_root,
        tmp_path / "visualizations",
        limit=1,
    )

    assert len(outputs) == 1
    assert outputs[0].is_file()


def test_recommendation_prefers_candidate_generation_when_coverage_is_low() -> None:
    recommendation = recommend_priority({
        "acc_at_0_5": 0.54,
        "candidate_coverage_sample_rate": 1.0,
        "coverage_at_1": 0.45,
        "coverage_at_10": 0.70,
    })
    assert recommendation["priority"] == "candidate_generation"


def test_cli_runner_writes_traceable_outputs(tmp_path: Path) -> None:
    from scripts.evaluate_predictions import run_evaluation

    validation_path = _validation(tmp_path / "validation.jsonl")
    prediction_path = _write_jsonl(tmp_path / "predictions.jsonl", [
        {"query_id": "a", "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.9},
        {"query_id": "b", "bbox": [0.0, 0.0, 0.3, 1.0], "score": 0.8},
        {"query_id": "c", "bbox": [0.0, 0.0, 0.3, 1.0], "score": 0.7},
    ])
    candidate_path = _write_jsonl(tmp_path / "candidates.jsonl", [
        {"query_id": "a", "candidates": [{"bbox": [0.0, 0.0, 1.0, 1.0]}]},
        {"query_id": "b", "candidates": [{"bbox": [0.0, 0.0, 0.3, 1.0]}]},
        {"query_id": "c", "candidates": [{"bbox": [0.0, 0.0, 0.3, 1.0]}]},
    ])
    image_root = tmp_path / "images"
    image_root.mkdir()
    success, encoded = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))
    assert success
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (image_root / name).write_bytes(encoded.tobytes())

    output_dir = tmp_path / "report"
    report = run_evaluation(
        validation_path=validation_path,
        output_dir=output_dir,
        prediction_paths={"baseline": str(prediction_path)},
        candidate_paths={"baseline": str(candidate_path)},
        image_root=image_root,
        failure_limit=2,
    )

    assert report["status"] == "PASS"
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "models" / "baseline" / "per_sample.csv").is_file()
    expected_visualizations = output_dir / "models" / "baseline" / "failure_visualizations"
    assert len(list(expected_visualizations.glob("*.jpg"))) == 2
    assert (output_dir / "fixed_query_ids.txt").read_text(encoding="utf-8") == "a\nb\nc\n"


def test_cli_runner_can_freeze_ids_without_predictions(tmp_path: Path) -> None:
    from scripts.evaluate_predictions import run_evaluation

    validation_path = _validation(tmp_path / "validation.jsonl")
    fixed_path = tmp_path / "fixed_ids.txt"
    report = run_evaluation(
        validation_path=validation_path,
        output_dir=tmp_path / "freeze_report",
        prediction_paths={},
        candidate_paths={},
        freeze_ids_path=fixed_path,
        failure_limit=0,
    )

    assert report["model_count"] == 0
    assert fixed_path.read_text(encoding="utf-8") == "a\nb\nc\n"