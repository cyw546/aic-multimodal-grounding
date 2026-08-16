from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import MultimodalGroundingDataset
from src.inference import (
    GroundingDINOPredictor,
    generate_predictions,
    load_json_object,
    write_prediction_json,
)
from src.submission.validator import validate_submission_payloads


DEFAULT_DATA_ROOT = Path(
    "/root/autodl-tmp/aic_grounding/data/official/preliminary"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO on the official preliminary dataset."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records for a smoke test.",
    )
    return parser.parse_args()


def run_inference(
    *,
    data_root: Path,
    pipeline_config: Path,
    model_config: Path,
    weights: Path,
    output: Path,
    limit: int | None = None,
    predictor: object | None = None,
) -> dict[str, object]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    json_path = data_root / "queries" / "queries.json"
    original = load_json_object(json_path)
    dataset = MultimodalGroundingDataset(
        json_path=json_path,
        data_root=data_root,
        require_bbox=False,
        validate_files=True,
    )

    if predictor is None:
        predictor = GroundingDINOPredictor.from_paths(
            pipeline_config_path=pipeline_config,
            model_config_path=model_config,
            weight_path=weights,
        )

    predictions = generate_predictions(
        dataset=dataset,
        original=original,
        predictor=predictor,
        limit=limit,
    )
    selected_original = {
        query_id: original[query_id]
        for query_id in predictions
    }
    report = validate_submission_payloads(
        selected_original,
        predictions,
    )
    output_path = write_prediction_json(predictions, output)
    return {
        "status": "PASS",
        "dataset_size": len(dataset),
        "processed_count": len(predictions),
        "complete": len(predictions) == len(dataset),
        "output": str(output_path),
        **report,
    }


def main() -> None:
    args = parse_args()
    report = run_inference(
        data_root=args.data_root,
        pipeline_config=args.pipeline_config,
        model_config=args.model_config,
        weights=args.weights,
        output=args.output,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
