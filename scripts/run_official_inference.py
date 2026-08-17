from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.data.dataset import MultimodalGroundingDataset
from src.inference import (
    GroundingDINOPredictor,
    generate_prediction,
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
        "--pipeline-config", type=Path, default=Path("configs/baseline.yaml")
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Atomically save partial predictions every N new records.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed records.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output stem>.partial.json when it exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed Query IDs and continue instead of stopping.",
    )
    return parser.parse_args()


def _partial_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.partial{output.suffix}")


def _error_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.errors.jsonl")


def _append_error(path: Path, query_id: str, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "query_id": query_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False) + "\n")


def _validate_resumed_predictions(
    predictions: dict[str, Any],
    original: dict[str, Any],
    target_ids: list[str],
) -> None:
    unexpected = sorted(set(predictions) - set(target_ids))
    if unexpected:
        raise ValueError(
            f"Checkpoint contains Query IDs outside this run: {unexpected[:10]}"
        )
    selected_original = {query_id: original[query_id] for query_id in predictions}
    validate_submission_payloads(selected_original, predictions)


def run_inference(
    *,
    data_root: Path,
    pipeline_config: Path,
    model_config: Path,
    weights: Path,
    output: Path,
    limit: int | None = None,
    checkpoint_every: int = 25,
    progress_every: int = 10,
    resume: bool = False,
    continue_on_error: bool = False,
    predictor: object | None = None,
) -> dict[str, object]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if checkpoint_every <= 0 or progress_every <= 0:
        raise ValueError("checkpoint_every and progress_every must be positive")

    json_path = data_root / "queries" / "queries.json"
    original = load_json_object(json_path)
    dataset = MultimodalGroundingDataset(
        json_path=json_path,
        data_root=data_root,
        require_bbox=False,
        validate_files=True,
    )
    target_count = len(dataset) if limit is None else min(limit, len(dataset))
    target_ids = [query_id for query_id, _ in dataset.records[:target_count]]
    partial_path = _partial_path(output)
    error_path = _error_path(output)

    predictions: dict[str, Any] = {}
    if resume and partial_path.is_file():
        predictions = load_json_object(partial_path)
        _validate_resumed_predictions(predictions, original, target_ids)
        print(f"Resuming with {len(predictions)}/{target_count} completed", flush=True)
    elif not resume and error_path.exists():
        error_path.unlink()

    if predictor is None:
        predictor = GroundingDINOPredictor.from_paths(
            pipeline_config_path=pipeline_config,
            model_config_path=model_config,
            weight_path=weights,
        )

    started = time.monotonic()
    new_count = 0
    failures = 0
    for index in range(target_count):
        query_id = target_ids[index]
        if query_id in predictions:
            continue
        try:
            predicted_id, output_record = generate_prediction(
                dataset[index], original, predictor, index=index
            )
            predictions[predicted_id] = output_record
            new_count += 1
        except Exception as exc:
            failures += 1
            _append_error(error_path, query_id, exc)
            write_prediction_json(predictions, partial_path)
            if not continue_on_error:
                raise RuntimeError(
                    f"Inference failed at {query_id}; checkpoint saved to {partial_path}"
                ) from exc

        completed = len(predictions) + failures
        if new_count and new_count % checkpoint_every == 0:
            write_prediction_json(predictions, partial_path)
        if completed % progress_every == 0 or completed == target_count:
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = new_count / elapsed
            remaining = target_count - completed
            eta_seconds = remaining / rate if rate > 0 else None
            eta_text = "unknown" if eta_seconds is None else f"{eta_seconds / 60:.1f} min"
            print(
                f"Progress {completed}/{target_count} "
                f"({completed / target_count:.1%}), failures={failures}, ETA={eta_text}",
                flush=True,
            )

    write_prediction_json(predictions, partial_path)
    if failures:
        return {
            "status": "INCOMPLETE",
            "dataset_size": len(dataset),
            "target_count": target_count,
            "processed_count": len(predictions),
            "failure_count": failures,
            "checkpoint": str(partial_path),
            "errors": str(error_path),
        }

    selected_original = {query_id: original[query_id] for query_id in target_ids}
    report = validate_submission_payloads(selected_original, predictions)
    output_path = write_prediction_json(predictions, output)
    partial_path.unlink(missing_ok=True)
    error_path.unlink(missing_ok=True)
    return {
        "status": "PASS",
        "dataset_size": len(dataset),
        "target_count": target_count,
        "processed_count": len(predictions),
        "complete": target_count == len(dataset),
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
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
