from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from torch.utils.data import Dataset

from src.metrics import validate_bbox


Sample = dict[str, Any]


class Predictor(Protocol):
    """Common interface shared by dummy and real grounding models."""

    def predict(
        self,
        *,
        image: np.ndarray,
        query: str,
    ) -> Mapping[str, Any]:
        """Return at least one normalized xyxy bbox."""


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object keyed by Query ID."""
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Top-level JSON value must be an object: {path}"
        )

    return payload


def generate_prediction(
    sample: Sample,
    original: Mapping[str, Any],
    predictor: Predictor,
    *,
    index: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Predict and validate one sample while preserving official fields."""
    location = f"Sample {index}" if index is not None else "Sample"
    query_id = sample.get("query_id")
    query = sample.get("query")
    image = sample.get("visible")

    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"{location}: invalid query_id {query_id!r}")
    if query_id not in original:
        raise ValueError(
            f"{location}: query_id {query_id!r} is missing from original JSON"
        )
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{query_id}: query must be a non-empty string")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{query_id}: visible image must be a NumPy array")

    result = predictor.predict(image=image, query=query)
    if not isinstance(result, Mapping):
        raise ValueError(f"{query_id}: predictor output must be a mapping")
    if "bbox" not in result:
        raise ValueError(f"{query_id}: predictor output is missing bbox")

    bbox = validate_bbox(
        result["bbox"],
        name=f"{query_id}.bbox",
    ).tolist()
    source_record = original[query_id]
    if not isinstance(source_record, dict):
        raise ValueError(f"{query_id}: original record must be an object")

    output_record = copy.deepcopy(source_record)
    output_record["bbox"] = bbox
    return query_id, output_record


def generate_predictions(
    dataset: Dataset[Sample],
    original: Mapping[str, Any],
    predictor: Predictor,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run inference while preserving every official record field."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    sample_count = len(dataset)
    run_count = sample_count if limit is None else min(limit, sample_count)
    predictions: dict[str, Any] = {}

    for index in range(run_count):
        query_id, output_record = generate_prediction(
            dataset[index],
            original,
            predictor,
            index=index,
        )
        predictions[query_id] = output_record

    return predictions


def write_prediction_json(
    predictions: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write predictions atomically to avoid incomplete JSON files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            predictions,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")

    temporary_path.replace(output_path)
    return output_path
