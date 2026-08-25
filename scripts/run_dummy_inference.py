from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.dataset import MultimodalGroundingDataset
from src.inference import (
    generate_predictions,
    load_json_object,
    write_prediction_json,
)
from src.submission.validator import validate_submission_payloads


class DummyPredictor:
    """Fixed-box predictor for pipeline testing only."""

    def predict(
        self,
        *,
        image: np.ndarray,
        query: str,
    ) -> dict[str, object]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image, got {image.shape}"
            )

        if not query.strip():
            raise ValueError("Query must not be empty")

        return {
            "bbox": [0.25, 0.25, 0.75, 0.75],
            "score": 0.0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed dummy predictions to test the official "
            "inference pipeline. Do not submit these predictions."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/aic_grounding/data/"
            "official/preliminary"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of records to test, for example 1 or 10.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    json_path = (
        args.data_root
        / "queries"
        / "queries.json"
    )

    original = load_json_object(json_path)

    dataset = MultimodalGroundingDataset(
        json_path=json_path,
        data_root=args.data_root,
        require_bbox=False,
        validate_files=True,
    )

    predictions = generate_predictions(
        dataset=dataset,
        original=original,
        predictor=DummyPredictor(),
        limit=args.limit,
    )

    limited_original = {
        query_id: original[query_id]
        for query_id in predictions
    }

    report = validate_submission_payloads(
        limited_original,
        predictions,
    )

    output_path = write_prediction_json(
        predictions,
        args.output,
    )

    summary = {
        "status": "PASS",
        "warning": "DUMMY OUTPUT - DO NOT SUBMIT",
        "dataset_size": len(dataset),
        "processed_count": len(predictions),
        "output": str(output_path),
        **report,
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
