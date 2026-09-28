"""Run the fixed repechage RGB threshold experiment with resumable outputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from baseline.grounding_model import GroundingModel
from baseline.inference import Baseline
from src.data.dataset import MultimodalGroundingDataset
from src.metrics import validate_bbox
from src.submission.validator import validate_submission_payloads


THRESHOLD_CONFIGS = (
    {"name": "box035_text025", "box_threshold": 0.35, "text_threshold": 0.25},
    {"name": "box025_text025", "box_threshold": 0.25, "text_threshold": 0.25},
    {"name": "box035_text020", "box_threshold": 0.35, "text_threshold": 0.20},
    {"name": "box030_text020", "box_threshold": 0.30, "text_threshold": 0.20},
)
FALLBACK_BOX_THRESHOLD = 0.05
FALLBACK_TEXT_THRESHOLD = 0.05


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def groundingdino_commit(root: Path) -> str | None:
    commit = git_value(root, "rev-parse", "HEAD")
    if commit is not None:
        return commit
    marker = root / ".codex_pinned_commit"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or None
    return None


def summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        raise ValueError("cannot summarize an empty experiment")
    stages = [record["stage"] for record in details]
    scores = [float(record["score"]) for record in details]
    runtimes = [float(record["runtime_seconds"]) for record in details]
    return {
        "sample_count": len(details),
        "valid_output_count": len(details),
        "primary_success_count": stages.count("primary"),
        "primary_empty_count": sum(stage != "primary" for stage in stages),
        "fallback_success_count": stages.count("fallback_threshold"),
        "default_box_count": stages.count("default_box"),
        "mean_score": float(statistics.fmean(scores)),
        "median_score": float(statistics.median(scores)),
        "min_score": float(min(scores)),
        "max_score": float(max(scores)),
        "total_runtime_seconds": float(sum(runtimes)),
        "mean_runtime_seconds": float(statistics.fmean(runtimes)),
        "median_runtime_seconds": float(statistics.median(runtimes)),
    }


def select_visualization_ids(
    query_ids: list[str],
    details_by_config: dict[str, list[dict[str, Any]]],
    count: int,
) -> list[str]:
    """Prefer defaults, fallbacks, score disagreement and bbox disagreement."""
    if count <= 0:
        return []
    indexed = {
        name: {record["query_id"]: record for record in records}
        for name, records in details_by_config.items()
    }
    ranked: list[tuple[tuple[float, ...], str]] = []
    for order, query_id in enumerate(query_ids):
        records = [mapping[query_id] for mapping in indexed.values()]
        stages = [record["stage"] for record in records]
        scores = [float(record["score"]) for record in records]
        boxes = [record["bbox"] for record in records]
        bbox_spread = max(
            max(float(box[coordinate]) for box in boxes)
            - min(float(box[coordinate]) for box in boxes)
            for coordinate in range(4)
        )
        rank = (
            float("default_box" in stages),
            float("fallback_threshold" in stages),
            max(scores) - min(scores),
            bbox_spread,
            -float(order),
        )
        ranked.append((rank, query_id))
    ranked.sort(reverse=True)
    return [query_id for _, query_id in ranked[: min(count, len(ranked))]]


def _load_or_create_state(
    checkpoint: Path,
    config: dict[str, Any],
    query_ids: list[str],
    resume: bool,
) -> dict[str, Any]:
    if resume and checkpoint.is_file():
        with checkpoint.open("r", encoding="utf-8") as stream:
            state = json.load(stream)
        if state.get("config") != config or state.get("query_ids") != query_ids:
            raise ValueError(f"checkpoint does not match this run: {checkpoint}")
        if not isinstance(state.get("predictions"), dict) or not isinstance(
            state.get("details"), list
        ):
            raise ValueError(f"invalid checkpoint structure: {checkpoint}")
        return state
    return {
        "config": config,
        "query_ids": query_ids,
        "predictions": {},
        "details": [],
    }


def _run_until(
    *,
    dataset: MultimodalGroundingDataset,
    original: dict[str, Any],
    predictor: Baseline,
    state: dict[str, Any],
    target_count: int,
    checkpoint: Path,
) -> None:
    completed = set(state["predictions"])
    for index in range(target_count):
        query_id = state["query_ids"][index]
        if query_id in completed:
            continue
        sample = dataset[index]
        started = time.perf_counter()
        result = predictor.predict_detailed(sample["visible"], sample["query"])
        runtime_seconds = time.perf_counter() - started
        bbox = validate_bbox(result["bbox"], name=f"{query_id}.bbox").tolist()

        source_record = original[query_id]
        if not isinstance(source_record, dict):
            raise ValueError(f"{query_id}: original record must be an object")
        output_record = copy.deepcopy(source_record)
        output_record["bbox"] = bbox
        state["predictions"][query_id] = output_record
        state["details"].append(
            {
                "query_id": query_id,
                "bbox": bbox,
                "score": float(result["score"]),
                "stage": result["stage"],
                "primary_candidate_count": int(result["primary_candidate_count"]),
                "fallback_candidate_count": int(result["fallback_candidate_count"]),
                "runtime_seconds": float(runtime_seconds),
            }
        )
        completed.add(query_id)
        write_json(checkpoint, state)
        print(
            f"[{state['config']['name']}] {len(completed)}/{target_count} "
            f"stage={result['stage']} score={float(result['score']):.4f} "
            f"time={runtime_seconds:.3f}s",
            flush=True,
        )


def _validate_prefix(
    original: dict[str, Any],
    query_ids: list[str],
    predictions: dict[str, Any],
    count: int,
) -> dict[str, int]:
    selected_ids = query_ids[:count]
    selected_original = {query_id: original[query_id] for query_id in selected_ids}
    selected_predictions = {query_id: predictions[query_id] for query_id in selected_ids}
    return validate_submission_payloads(selected_original, selected_predictions)


def _finish_config(
    *,
    config_dir: Path,
    config: dict[str, Any],
    original: dict[str, Any],
    query_ids: list[str],
    state: dict[str, Any],
) -> dict[str, Any]:
    validation = _validate_prefix(
        original, query_ids, state["predictions"], len(query_ids)
    )
    ordered_predictions = {
        query_id: state["predictions"][query_id] for query_id in query_ids
    }
    details_by_id = {record["query_id"]: record for record in state["details"]}
    ordered_details = [details_by_id[query_id] for query_id in query_ids]
    summary = {
        "name": config["name"],
        "box_threshold": config["box_threshold"],
        "text_threshold": config["text_threshold"],
        "fallback_box_threshold": FALLBACK_BOX_THRESHOLD,
        "fallback_text_threshold": FALLBACK_TEXT_THRESHOLD,
        **summarize_details(ordered_details),
        "validation": validation,
    }
    write_json(config_dir / "prediction.json", ordered_predictions)
    write_jsonl(config_dir / "details.jsonl", ordered_details)
    write_json(config_dir / "summary.json", summary)
    return {"summary": summary, "details": ordered_details}


def _draw_panel(
    rgb: np.ndarray,
    query_id: str,
    query: str,
    config_name: str,
    detail: dict[str, Any],
) -> np.ndarray:
    max_width = 720
    scale = min(1.0, max_width / rgb.shape[1])
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    height, width = image.shape[:2]
    x1, y1, x2, y2 = detail["bbox"]
    color = {
        "primary": (0, 220, 0),
        "fallback_threshold": (0, 190, 255),
        "default_box": (0, 0, 255),
    }[detail["stage"]]
    cv2.rectangle(
        image,
        (round(x1 * width), round(y1 * height)),
        (round(x2 * width), round(y2 * height)),
        color,
        3,
    )
    header = np.zeros((76, width, 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{query_id} | {config_name} | {detail['stage']} | score={detail['score']:.4f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        query[:105],
        (8, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def create_visualizations(
    *,
    dataset: MultimodalGroundingDataset,
    query_ids: list[str],
    details_by_config: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    count: int,
) -> list[dict[str, Any]]:
    selected = select_visualization_ids(query_ids, details_by_config, count)
    detail_maps = {
        name: {record["query_id"]: record for record in records}
        for name, records in details_by_config.items()
    }
    query_to_index = {query_id: index for index, query_id in enumerate(query_ids)}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for query_id in selected:
        sample = dataset[query_to_index[query_id]]
        panels = [
            _draw_panel(
                sample["visible"],
                query_id,
                sample["query"],
                config["name"],
                detail_maps[config["name"]][query_id],
            )
            for config in THRESHOLD_CONFIGS
        ]
        target_width = max(panel.shape[1] for panel in panels)
        padded = [
            cv2.copyMakeBorder(
                panel,
                0,
                0,
                0,
                target_width - panel.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            for panel in panels
        ]
        canvas = np.vstack(padded)
        output_path = output_dir / f"{query_id}.jpg"
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f"failed to write visualization: {output_path}")
        manifest.append(
            {
                "query_id": query_id,
                "query": sample["query"],
                "path": str(output_path),
            }
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--groundingdino-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--visualization-count", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 100:
        raise ValueError("--max-samples must be at least 100 for the required experiment")
    if args.visualization_count < 20:
        raise ValueError("--visualization-count must be at least 20")

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = data_root / "queries" / "queries.json"
    with json_path.open("r", encoding="utf-8") as stream:
        original = json.load(stream)
    if not isinstance(original, dict):
        raise ValueError("official queries JSON must be an object")

    dataset = MultimodalGroundingDataset(
        json_path,
        data_root=data_root,
        require_bbox=False,
        validate_files=False,
    )
    if len(dataset) < args.max_samples:
        raise ValueError(
            f"dataset has {len(dataset)} records, fewer than {args.max_samples}"
        )
    query_ids = [query_id for query_id, _ in dataset.records[: args.max_samples]]
    query_list_path = output_root / f"query_ids_{args.max_samples}.json"
    if args.resume and query_list_path.is_file():
        with query_list_path.open("r", encoding="utf-8") as stream:
            saved_query_ids = json.load(stream)
        if saved_query_ids != query_ids:
            raise ValueError("saved query list differs from current official JSON order")
    else:
        write_json(query_list_path, query_ids)

    started_at = datetime.now(timezone.utc)
    manifest = {
        "started_at": started_at.isoformat(),
        "data_root": str(data_root),
        "dataset_count": len(dataset),
        "selected_count": len(query_ids),
        "model_config": str(args.model_config.resolve()),
        "weights": str(args.weights.resolve()),
        "weight_sha256": sha256_file(args.weights),
        "groundingdino_root": str(args.groundingdino_root.resolve()),
        "groundingdino_commit": groundingdino_commit(
            args.groundingdino_root.resolve()
        ),
        "thresholds": list(THRESHOLD_CONFIGS),
        "fallback_box_threshold": FALLBACK_BOX_THRESHOLD,
        "fallback_text_threshold": FALLBACK_TEXT_THRESHOLD,
        "device": args.device,
    }
    write_json(output_root / "run_manifest.json", manifest)

    model = GroundingModel(
        str(args.model_config.resolve()),
        str(args.weights.resolve()),
        device=None if args.device == "auto" else args.device,
    )

    summaries: list[dict[str, Any]] = []
    details_by_config: dict[str, list[dict[str, Any]]] = {}
    for config_index, config in enumerate(THRESHOLD_CONFIGS):
        config = dict(config)
        config_dir = output_root / config["name"]
        config_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = config_dir / "checkpoint.json"
        state = _load_or_create_state(checkpoint, config, query_ids, args.resume)
        predictor = Baseline(
            model=model,
            box_threshold=config["box_threshold"],
            text_threshold=config["text_threshold"],
            fallback_box_threshold=FALLBACK_BOX_THRESHOLD,
            fallback_text_threshold=FALLBACK_TEXT_THRESHOLD,
        )

        if config_index == 0:
            _run_until(
                dataset=dataset,
                original=original,
                predictor=predictor,
                state=state,
                target_count=1,
                checkpoint=checkpoint,
            )
            smoke_one = _validate_prefix(
                original, query_ids, state["predictions"], 1
            )
            write_json(
                output_root / "smoke_1.json",
                {"status": "PASS", "query_id": query_ids[0], **smoke_one},
            )
            _run_until(
                dataset=dataset,
                original=original,
                predictor=predictor,
                state=state,
                target_count=10,
                checkpoint=checkpoint,
            )
            smoke_ten = _validate_prefix(
                original, query_ids, state["predictions"], 10
            )
            write_json(
                output_root / "smoke_10.json",
                {"status": "PASS", "query_ids": query_ids[:10], **smoke_ten},
            )

        _run_until(
            dataset=dataset,
            original=original,
            predictor=predictor,
            state=state,
            target_count=len(query_ids),
            checkpoint=checkpoint,
        )
        finished = _finish_config(
            config_dir=config_dir,
            config=config,
            original=original,
            query_ids=query_ids,
            state=state,
        )
        summaries.append(finished["summary"])
        details_by_config[config["name"]] = finished["details"]
        checkpoint.unlink(missing_ok=True)

    visualizations = create_visualizations(
        dataset=dataset,
        query_ids=query_ids,
        details_by_config=details_by_config,
        output_dir=output_root / "visualizations",
        count=args.visualization_count,
    )
    write_json(output_root / "visualization_manifest.json", visualizations)
    completed_at = datetime.now(timezone.utc)
    final_report = {
        "status": "PASS",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_runtime_seconds": (completed_at - started_at).total_seconds(),
        "dataset_count": len(dataset),
        "selected_count": len(query_ids),
        "visualization_count": len(visualizations),
        "summaries": summaries,
    }
    write_json(output_root / "experiment_summary.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
