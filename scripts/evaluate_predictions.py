"""Evaluate one or more grounding prediction files with candidate diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation import (
    DEFAULT_COVERAGE_KS,
    evaluate_prediction_records,
    freeze_query_ids,
    load_candidate_records,
    load_prediction_records,
    load_query_ids,
    load_validation_records,
    recommend_priority,
    select_validation_records,
    sha256_file,
    visualize_failures,
)


def _parse_assignment(value: str, *, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{label} must use NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"{label} must use non-empty NAME=PATH")
    return name, path


def _assignment_map(values: list[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, path = _parse_assignment(value, label=label)
        if name in result:
            raise ValueError(f"duplicate {label} name: {name}")
        result[name] = path
    return result


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    temporary.replace(path)
    return path


def _safe_name(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in name)


def _prediction_from_candidates(
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for query_id, values in candidates.items():
        first_valid = next((value for value in values if value.get("bbox") is not None), None)
        predictions[query_id] = {
            "query_id": query_id,
            "bbox": first_valid.get("bbox") if first_valid else None,
            "bbox_error": None if first_valid else "no valid candidate",
            "score": first_valid.get("score") if first_valid else None,
            "model_name": None,
            "model_version": None,
        }
    return predictions


def _comparison_row(
    name: str,
    summary: dict[str, Any],
    *,
    prediction_source: str,
    prediction_path: str | None,
    candidate_path: str | None,
    model_version: str | None,
    config: str | None,
) -> dict[str, Any]:
    recommendation = recommend_priority(summary)
    return {
        "model": name,
        "prediction_source": prediction_source,
        "prediction_file": prediction_path,
        "candidate_file": candidate_path,
        "model_version": model_version,
        "config": config,
        "sample_count": summary["sample_count"],
        "acc_at_0_5": summary["acc_at_0_5"],
        "ACC@0.5": summary["ACC@0.5"],
        "mean_iou": summary["mean_iou"],
        "mean_IoU": summary["mean_IoU"],
        "valid_bbox_rate": summary["valid_bbox_rate"],
        "valid_bbox_count": summary["valid_bbox_count"],
        "invalid_bbox_count": summary["invalid_bbox_count"],
        "missing_prediction_count": summary["missing_prediction_count"],
        "coverage_at_1": summary.get("coverage_at_1"),
        "coverage_at_5": summary.get("coverage_at_5"),
        "coverage_at_10": summary.get("coverage_at_10"),
        "coverage@1": summary.get("coverage@1"),
        "coverage@5": summary.get("coverage@5"),
        "coverage@10": summary.get("coverage@10"),
        "failure_count": summary["failure_count"],
        "top_failure_categories": json.dumps(
            summary["failure_category_counts"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "recommended_priority": recommendation["priority"],
        "recommendation_reason": recommendation["reason"],
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> Path:
    columns = [
        ("模型", "model"),
        ("ACC@0.5", "acc_at_0_5"),
        ("平均IoU", "mean_iou"),
        ("有效框率", "valid_bbox_rate"),
        ("非法框", "invalid_bbox_count"),
        ("缺失", "missing_prediction_count"),
        ("C@1", "coverage_at_1"),
        ("C@5", "coverage_at_5"),
        ("C@10", "coverage_at_10"),
        ("优先方向", "recommended_priority"),
    ]
    lines = [
        "# 模型评估与候选框诊断",
        "",
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## 推荐下一步",
        "",
    ])
    for row in rows:
        lines.append(
            f"- **{row['model']}**：`{row['recommended_priority']}`，"
            f"{row['recommendation_reason']}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _parse_coverage_ks(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--coverage-k must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("--coverage-k must contain positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RGB baseline, API and reranked predictions with one fixed "
            "validation manifest. Prediction files may be JSON or JSONL."
        )
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--model-version", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--config", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--coverage-k", type=_parse_coverage_ks, default=DEFAULT_COVERAGE_KS)
    parser.add_argument("--failure-limit", type=int, default=20)
    parser.add_argument("--visualization-top-k", type=int, default=5)
    parser.add_argument("--freeze-ids", type=Path, default=None)
    return parser.parse_args()


def run_evaluation(
    *,
    validation_path: Path,
    output_dir: Path,
    prediction_paths: dict[str, str],
    candidate_paths: dict[str, str],
    ids_path: Path | None = None,
    image_root: Path | None = None,
    model_versions: dict[str, str] | None = None,
    configs: dict[str, str] | None = None,
    threshold: float = 0.5,
    coverage_ks: tuple[int, ...] = DEFAULT_COVERAGE_KS,
    failure_limit: int = 20,
    visualization_top_k: int = 5,
    freeze_ids_path: Path | None = None,
) -> dict[str, Any]:
    if not prediction_paths and not candidate_paths and freeze_ids_path is None:
        raise ValueError("at least one --prediction or --candidate is required")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if failure_limit < 0:
        raise ValueError("failure_limit must be non-negative")
    if visualization_top_k <= 0:
        raise ValueError("visualization_top_k must be positive")

    model_versions = model_versions or {}
    configs = configs or {}
    validation = load_validation_records(validation_path)
    query_ids = load_query_ids(ids_path) if ids_path is not None else None
    validation = select_validation_records(validation, query_ids)
    fixed_ids = [str(record["query_id"]) for record in validation]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = freeze_query_ids(
        fixed_ids,
        freeze_ids_path or (output_dir / "fixed_query_ids.txt"),
    )

    if not prediction_paths and not candidate_paths:
        report = {
            "status": "PASS",
            "output_dir": str(output_dir),
            "fixed_query_ids": str(frozen_path),
            "model_count": 0,
            "comparison": [],
            "recommendations": {},
        }
        _write_json(output_dir / "run_manifest.json", {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "coverage_ks": list(coverage_ks),
            "fixed_sample_count": len(fixed_ids),
            "fixed_ids": str(frozen_path),
            "validation": {
                "path": str(validation_path.expanduser().resolve()),
                "sha256": sha256_file(validation_path),
            },
            "predictions": {},
            "candidates": {},
            "models": {},
        })
        return report

    prediction_cache = {
        name: load_prediction_records(path)
        for name, path in prediction_paths.items()
    }
    candidate_cache = {
        name: load_candidate_records(path)
        for name, path in candidate_paths.items()
    }
    names = sorted(set(prediction_cache) | set(candidate_cache))
    comparison_rows: list[dict[str, Any]] = []
    recommendations: dict[str, Any] = {}
    run_models: dict[str, Any] = {}

    for name in names:
        predictions = prediction_cache.get(name)
        candidates = candidate_cache.get(name)
        if predictions is None:
            predictions = _prediction_from_candidates(candidates or {})
            prediction_source = "candidate_top1"
        else:
            prediction_source = "prediction_file"

        result = evaluate_prediction_records(
            validation,
            predictions,
            candidates,
            threshold=threshold,
            coverage_ks=coverage_ks,
        )
        model_dir = output_dir / "models" / _safe_name(name)
        _write_json(model_dir / "summary.json", result["summary"])
        _write_jsonl(model_dir / "per_sample.jsonl", result["rows"])
        _write_csv(model_dir / "per_sample.csv", result["rows"])
        failure_rows = [row for row in result["rows"] if not bool(row["hit_at_0_5"])]
        _write_csv(model_dir / "failure_samples.csv", failure_rows)
        _write_json(model_dir / "failure_summary.json", {
            "failure_count": len(failure_rows),
            "primary_category_counts": result["summary"]["failure_category_counts"],
            "all_category_counts": result["summary"]["failure_category_counts_all"],
            "difficulty_failure_counts": result["summary"]["difficulty_failure_counts"],
        })

        visualization_paths: list[str] = []
        if failure_limit and failure_rows:
            if image_root is None:
                if validation and validation[0].get("visible_path") and Path(str(validation[0]["visible_path"])).is_absolute():
                    resolved_root = None
                else:
                    raise ValueError("--image-root is required for failure visualizations")
            else:
                resolved_root = image_root
            visualization_paths = [
                str(path)
                for path in visualize_failures(
                    result["rows"],
                    validation,
                    candidates,
                    resolved_root,
                    model_dir / "failure_visualizations",
                    limit=failure_limit,
                    top_k=visualization_top_k,
                )
            ]

        recommendation = recommend_priority(result["summary"])
        recommendations[name] = recommendation
        row = _comparison_row(
            name,
            result["summary"],
            prediction_source=prediction_source,
            prediction_path=prediction_paths.get(name),
            candidate_path=candidate_paths.get(name),
            model_version=model_versions.get(name),
            config=configs.get(name),
        )
        comparison_rows.append(row)
        run_models[name] = {
            "summary": result["summary"],
            "recommendation": recommendation,
            "prediction_source": prediction_source,
            "prediction_file": prediction_paths.get(name),
            "candidate_file": candidate_paths.get(name),
            "visualization_count": len(visualization_paths),
            "visualization_files": visualization_paths,
        }

    _write_csv(output_dir / "comparison.csv", comparison_rows)
    _write_json(output_dir / "comparison.json", comparison_rows)
    _write_markdown(output_dir / "comparison.md", comparison_rows)
    _write_json(output_dir / "recommendations.json", recommendations)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "coverage_ks": list(coverage_ks),
        "fixed_sample_count": len(fixed_ids),
        "fixed_ids": str(frozen_path),
        "validation": {
            "path": str(validation_path.expanduser().resolve()),
            "sha256": sha256_file(validation_path),
        },
        "predictions": {
            name: {"path": str(Path(path).expanduser().resolve()), "sha256": sha256_file(path)}
            for name, path in prediction_paths.items()
        },
        "candidates": {
            name: {"path": str(Path(path).expanduser().resolve()), "sha256": sha256_file(path)}
            for name, path in candidate_paths.items()
        },
        "models": run_models,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "fixed_query_ids": str(frozen_path),
        "model_count": len(names),
        "comparison": comparison_rows,
        "recommendations": recommendations,
    }


def main() -> None:
    args = parse_args()
    prediction_paths = _assignment_map(args.prediction, label="--prediction")
    candidate_paths = _assignment_map(args.candidate, label="--candidate")
    model_versions = _assignment_map(args.model_version, label="--model-version")
    configs = _assignment_map(args.config, label="--config")
    report = run_evaluation(
        validation_path=args.validation,
        output_dir=args.output_dir,
        prediction_paths=prediction_paths,
        candidate_paths=candidate_paths,
        ids_path=args.ids,
        image_root=args.image_root,
        model_versions=model_versions,
        configs=configs,
        threshold=args.threshold,
        coverage_ks=args.coverage_k,
        failure_limit=args.failure_limit,
        visualization_top_k=args.visualization_top_k,
        freeze_ids_path=args.freeze_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
