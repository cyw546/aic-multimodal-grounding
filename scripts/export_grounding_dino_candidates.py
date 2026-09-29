"""Export Grounding DINO top-K candidates for the fixed public validation set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from baseline.inference import Baseline
from src.data.public_dataset import PublicGroundingDataset
from src.evaluation import load_query_ids


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict) or not isinstance(value.get("query_id"), str):
                raise ValueError(f"{path}:{line_number}: invalid candidate record")
            records.append(value)
    return records


def export_candidates(
    *,
    validation_path: Path,
    image_root: Path,
    output_path: Path,
    pipeline_config: Path,
    model_config: Path,
    weights: Path,
    ids_path: Path | None = None,
    top_k: int = 10,
    limit: int | None = None,
    checkpoint_every: int = 10,
    resume: bool = False,
    predictor: Any | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer")

    dataset = PublicGroundingDataset(validation_path, image_root, validate_files=True)
    index_by_id = {
        str(record["query_id"]): index
        for index, record in enumerate(dataset.records)
    }
    if ids_path is None:
        target_ids = [
            str(record["query_id"])
            for record in dataset.records
        ]
    else:
        target_ids = load_query_ids(ids_path)
        missing = [query_id for query_id in target_ids if query_id not in index_by_id]
        if missing:
            raise ValueError(f"validation dataset is missing fixed IDs: {missing[:10]}")
    if limit is not None:
        target_ids = target_ids[:limit]

    existing = load_jsonl(output_path) if resume else []
    records_by_id = {str(record["query_id"]): record for record in existing}
    unexpected = sorted(set(records_by_id) - set(target_ids))
    if unexpected:
        raise ValueError(f"checkpoint contains IDs outside this run: {unexpected[:10]}")

    if predictor is None:
        predictor = Baseline.from_yaml(
            pipeline_config,
            model_config_path=str(model_config),
            weight_path=str(weights),
        )

    started = time.monotonic()
    completed_new = 0
    for query_id in target_ids:
        if query_id in records_by_id:
            continue
        sample = dataset[index_by_id[query_id]]
        result = predictor.predict_candidates(
            image=sample["visible"],
            query=sample["query"],
            top_k=top_k,
        )
        records_by_id[query_id] = {
            "query_id": query_id,
            "bbox": result.get("bbox"),
            "score": result.get("score"),
            "candidates": result.get("candidates", []),
            "stage": result.get("stage"),
            "primary_candidate_count": result.get("primary_candidate_count"),
            "fallback_candidate_count": result.get("fallback_candidate_count"),
            "model_name": "grounding_dino_swinb",
            "model_version": weights.name,
            "config": pipeline_config.name,
        }
        completed_new += 1
        completed = len(records_by_id)
        if completed_new % checkpoint_every == 0:
            write_jsonl(output_path, [records_by_id[key] for key in target_ids if key in records_by_id])
        if completed % checkpoint_every == 0 or completed == len(target_ids):
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"Progress {completed}/{len(target_ids)} "
                f"({completed / len(target_ids):.1%}), rate={completed_new / elapsed:.2f}/s",
                flush=True,
            )

    ordered = [records_by_id[query_id] for query_id in target_ids]
    output_path = write_jsonl(output_path, ordered)
    return {
        "status": "PASS",
        "sample_count": len(ordered),
        "top_k": top_k,
        "output": str(output_path),
        "resumed_count": len(existing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Grounding DINO top-K candidates for model evaluation."
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_candidates(
        validation_path=args.validation,
        image_root=args.image_root,
        output_path=args.output,
        pipeline_config=args.pipeline_config,
        model_config=args.model_config,
        weights=args.weights,
        ids_path=args.ids,
        top_k=args.top_k,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
