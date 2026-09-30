"""Unified evaluation for grounding predictions and candidate proposals.

The module deliberately keeps parsing, scoring and visualization independent
from any specific model. A model only needs to be converted once to a small
prediction JSON/JSONL interface before it can participate in comparisons.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.metrics import box_iou, validate_bbox


DEFAULT_COVERAGE_KS = (1, 5, 10)
_TAG_ALIASES = {
    "small": "small_target",
    "small_target": "small_target",
    "small_or_distant_target": "small_target",
    "occlusion": "occlusion",
    "occluded": "occlusion",
    "similar": "similar_targets",
    "similar_target": "similar_targets",
    "similar_targets": "similar_targets",
    "night": "night_or_low_light",
    "low_light": "night_or_low_light",
    "night_or_low_light": "night_or_low_light",
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def _iter_payload_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_payload_records(item)
        return

    if not isinstance(payload, dict):
        raise ValueError("evaluation inputs must contain JSON objects")

    record_keys = {
        "query_id", "sample_id", "id", "bbox", "gt_bbox", "target_bbox",
        "prediction", "candidates", "boxes",
    }
    if record_keys.intersection(payload):
        yield payload
        return

    for container_key in ("records", "predictions", "results", "items", "data"):
        if container_key in payload:
            yield from _iter_payload_records(payload[container_key])
            return

    for key, value in payload.items():
        if isinstance(value, dict):
            record = dict(value)
            record.setdefault("query_id", key)
            yield record
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            yield {"query_id": key, "bbox": value}
        else:
            raise ValueError(f"invalid keyed evaluation record for {key!r}")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON object, JSON array or JSONL file as records."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        payload = _read_jsonl(path)
    else:
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    return list(_iter_payload_records(payload))


def _query_id(record: Mapping[str, Any]) -> str:
    for key in ("query_id", "sample_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("record is missing a non-empty query_id/sample_id/id")


def _safe_bbox(value: Any, *, name: str) -> tuple[list[float] | None, str | None]:
    if value is None:
        return None, "missing bbox"
    try:
        return validate_bbox(value, name=name).tolist(), None
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _model_metadata(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    model_name = record.get("model_name", record.get("model"))
    model_version = record.get("model_version", record.get("version"))
    model_name = str(model_name) if model_name is not None else None
    model_version = str(model_version) if model_version is not None else None
    return model_name, model_version


def load_prediction_records(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the unified prediction interface keyed by Query ID."""

    records: dict[str, dict[str, Any]] = {}
    for raw in read_records(path):
        query_id = _query_id(raw)
        if query_id in records:
            raise ValueError(f"duplicate prediction query_id: {query_id}")

        bbox_value = raw.get("bbox")
        if bbox_value is None and isinstance(raw.get("prediction"), Mapping):
            bbox_value = raw["prediction"].get("bbox")
        bbox, bbox_error = _safe_bbox(bbox_value, name=f"{query_id}.bbox")
        score_value = raw.get("score")
        if score_value is None and isinstance(raw.get("prediction"), Mapping):
            score_value = raw["prediction"].get("score")
        model_name, model_version = _model_metadata(raw)
        records[query_id] = {
            "query_id": query_id,
            "bbox": bbox,
            "bbox_error": bbox_error,
            "score": _finite_score(score_value),
            "model_name": model_name,
            "model_version": model_version,
        }
    return records


def _candidate_values(record: Mapping[str, Any]) -> list[Any]:
    if "candidates" in record:
        value = record["candidates"]
        if isinstance(value, Mapping):
            return list(value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        raise ValueError("candidates must be a list or object")

    boxes = record.get("boxes")
    scores = record.get("scores")
    if boxes is not None:
        if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
            raise ValueError("boxes must be a list")
        if scores is None:
            scores = [None] * len(boxes)
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            raise ValueError("scores must be a list")
        if len(boxes) != len(scores):
            raise ValueError("boxes and scores must have equal lengths")
        return [
            {"bbox": bbox, "score": score}
            for bbox, score in zip(boxes, scores)
        ]

    if "bbox" in record:
        return [{"bbox": record["bbox"], "score": record.get("score")}]
    return []


def load_candidate_records(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load ranked candidate boxes, preserving the supplied rank order."""

    records: dict[str, list[dict[str, Any]]] = {}
    for raw in read_records(path):
        query_id = _query_id(raw)
        if query_id in records:
            raise ValueError(f"duplicate candidate query_id: {query_id}")
        candidates: list[dict[str, Any]] = []
        for rank, item in enumerate(_candidate_values(raw), start=1):
            if isinstance(item, Mapping):
                bbox_value = item.get("bbox", item.get("box"))
                score_value = item.get("score", item.get("logit"))
                label = item.get("label", item.get("phrase"))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                bbox_value = item
                score_value = None
                label = None
            else:
                bbox_value = None
                score_value = None
                label = None
            bbox, bbox_error = _safe_bbox(
                bbox_value,
                name=f"{query_id}.candidates[{rank}].bbox",
            )
            candidates.append({
                "rank": rank,
                "bbox": bbox,
                "bbox_error": bbox_error,
                "score": _finite_score(score_value),
                "label": str(label) if label is not None else None,
            })
        records[query_id] = candidates
    return records


def _difficulty_tags(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("difficulty_categories", "difficulty_tags", "tags", "categories"):
        value = record.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(str(item) for item in value)
    normalized: list[str] = []
    for value in values:
        tag = _TAG_ALIASES.get(value.strip().lower(), value.strip().lower())
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _image_value(record: Mapping[str, Any]) -> str | None:
    for key in ("visible_path", "image_path", "visible", "image"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_validation_records(path: str | Path) -> list[dict[str, Any]]:
    """Load the public validation manifest with ground-truth boxes."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in read_records(path):
        query_id = _query_id(raw)
        if query_id in seen:
            raise ValueError(f"duplicate validation query_id: {query_id}")
        seen.add(query_id)
        bbox_format = raw.get("bbox_format", "xyxy_norm")
        if bbox_format != "xyxy_norm":
            raise ValueError(f"{query_id}: bbox_format must be xyxy_norm")
        bbox_value = raw.get("bbox", raw.get("gt_bbox", raw.get("target_bbox")))
        bbox, bbox_error = _safe_bbox(bbox_value, name=f"{query_id}.bbox")
        if bbox_error is not None:
            raise ValueError(f"{query_id}: {bbox_error}")
        records.append({
            "query_id": query_id,
            "query": str(raw.get("query", "")),
            "bbox": bbox,
            "visible_path": _image_value(raw),
            "difficulty_categories": _difficulty_tags(raw),
        })
    if not records:
        raise ValueError(f"validation file contains no records: {path}")
    return records


def load_query_ids(path: str | Path) -> list[str]:
    """Read a text or JSON/JSONL manifest containing Query IDs."""

    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        result: list[str] = []
        for raw in read_records(path):
            result.append(_query_id(raw))
    else:
        result = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                value = line.strip()
                if value and not value.startswith("#"):
                    result.append(value)
    if not result:
        raise ValueError(f"query ID manifest is empty: {path}")
    if len(result) != len(set(result)):
        raise ValueError(f"query ID manifest contains duplicates: {path}")
    return result


def freeze_query_ids(query_ids: Sequence[str], output_path: str | Path) -> Path:
    """Atomically write the fixed evaluation order as one ID per line."""

    if not query_ids:
        raise ValueError("cannot freeze an empty query ID list")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for query_id in query_ids:
                stream.write(query_id.strip() + "\n")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def select_validation_records(
    records: Sequence[Mapping[str, Any]],
    query_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a fixed manifest order and reject missing or duplicate IDs."""

    mapping = {str(record["query_id"]): dict(record) for record in records}
    if query_ids is None:
        return [dict(record) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("fixed query ID list contains duplicates")
    missing = [query_id for query_id in query_ids if query_id not in mapping]
    if missing:
        raise ValueError(f"validation file is missing fixed IDs: {missing[:10]}")
    return [mapping[query_id] for query_id in query_ids]


def _geometry_failure(prediction: Sequence[float], target: Sequence[float]) -> str:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    target_area = max(0.0, truth[2] - truth[0]) * max(0.0, truth[3] - truth[1])
    area_ratio = pred_area / target_area if target_area > 0 else float("inf")
    pred_center = np.asarray([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
    target_center = np.asarray([(truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2])
    center_distance = float(np.linalg.norm(pred_center - target_center))
    if area_ratio >= 1.5:
        return "oversized_box"
    if area_ratio <= 0.6666666667:
        return "undersized_box"
    if center_distance >= 0.15:
        return "localization_shift"
    return "partial_overlap"


def _classify_failure(
    *,
    prediction_valid: bool,
    candidates_available: bool,
    best_candidate_rank: int | None,
    difficulty_tags: Sequence[str],
    prediction: Sequence[float] | None,
    target: Sequence[float],
) -> list[str]:
    categories: list[str] = []
    if candidates_available:
        if best_candidate_rank is None:
            categories.append("target_not_in_candidates")
        elif best_candidate_rank > 1:
            categories.append("candidate_ranking_error")
        elif not prediction_valid:
            categories.append("candidate_available_but_final_missing_or_invalid")
        else:
            categories.append("candidate_selection_or_regression")
    elif not prediction_valid:
        categories.append("missing_or_invalid_prediction")

    for tag in difficulty_tags:
        if tag in {"small_target", "occlusion", "similar_targets"}:
            categories.append(tag)

    if prediction_valid and prediction is not None:
        geometry = _geometry_failure(prediction, target)
        if geometry not in categories:
            categories.append(geometry)
    if not categories:
        categories.append("other")
    return categories


def _coverage_value(
    candidates: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    k: int,
    threshold: float,
) -> bool:
    for candidate in candidates[:k]:
        bbox = candidate.get("bbox")
        if bbox is None:
            continue
        if box_iou(bbox, target) >= threshold:
            return True
    return False


def evaluate_prediction_records(
    validation: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    threshold: float = 0.5,
    coverage_ks: Sequence[int] = DEFAULT_COVERAGE_KS,
) -> dict[str, Any]:
    """Evaluate one model and return a summary plus per-sample diagnostics."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    normalized_ks = tuple(sorted({int(k) for k in coverage_ks}))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("coverage_ks must contain positive integers")
    if not validation:
        raise ValueError("validation records must not be empty")

    candidate_mode = candidates is not None
    candidates = candidates or {}
    rows: list[dict[str, Any]] = []

    for record in validation:
        query_id = str(record["query_id"])
        target = list(record["bbox"])
        prediction_record = predictions.get(query_id)
        prediction_valid = bool(
            prediction_record
            and prediction_record.get("bbox") is not None
            and prediction_record.get("bbox_error") is None
        )
        prediction = (
            list(prediction_record["bbox"])
            if prediction_valid and prediction_record is not None
            else None
        )
        prediction_error = (
            prediction_record.get("bbox_error")
            if prediction_record is not None and not prediction_valid
            else None
        )
        if prediction_record is None:
            prediction_status = "missing"
        elif not prediction_valid:
            prediction_status = "invalid"
        else:
            prediction_status = "valid"

        iou = box_iou(prediction, target) if prediction is not None else None
        hit = bool(iou is not None and iou >= threshold)

        query_candidates = list(candidates.get(query_id, []))
        candidate_available = candidate_mode and query_id in candidates
        candidate_ious: list[tuple[int, float]] = []
        invalid_candidate_count = 0
        for candidate in query_candidates:
            bbox = candidate.get("bbox")
            if bbox is None:
                invalid_candidate_count += 1
                continue
            candidate_ious.append((int(candidate["rank"]), box_iou(bbox, target)))
        best_candidate_iou = (
            max(iou_value for _, iou_value in candidate_ious)
            if candidate_ious
            else None
        )
        best_candidate_rank = (
            min(
                rank
                for rank, iou_value in candidate_ious
                if best_candidate_iou is not None and iou_value == best_candidate_iou
            )
            if candidate_ious
            else None
        )
        correct_candidate_ranks = [
            rank
            for rank, iou_value in candidate_ious
            if iou_value >= threshold
        ]
        correct_candidate_rank = min(correct_candidate_ranks) if correct_candidate_ranks else None
        coverage = {
            k: _coverage_value(
                query_candidates,
                target,
                k=k,
                threshold=threshold,
            )
            for k in normalized_ks
        }
        difficulty_tags = list(record.get("difficulty_categories", []))
        failure_categories = (
            []
            if hit
            else _classify_failure(
                prediction_valid=prediction_valid,
                candidates_available=candidate_available,
                best_candidate_rank=correct_candidate_rank,
                difficulty_tags=difficulty_tags,
                prediction=prediction,
                target=target,
            )
        )

        row: dict[str, Any] = {
            "query_id": query_id,
            "query": record.get("query", ""),
            "ground_truth_bbox": target,
            "prediction_bbox": prediction,
            "prediction_status": prediction_status,
            "prediction_error": prediction_error,
            "iou": iou,
            "hit_at_0_5": hit,
            "valid_bbox": prediction_valid,
            "invalid_bbox": prediction_status == "invalid",
            "missing_prediction": prediction_status == "missing",
            "score": prediction_record.get("score") if prediction_record else None,
            "model_name": prediction_record.get("model_name") if prediction_record else None,
            "model_version": prediction_record.get("model_version") if prediction_record else None,
            "candidate_count": len(query_candidates),
            "invalid_candidate_count": invalid_candidate_count,
            "best_candidate_iou": best_candidate_iou,
            "best_candidate_rank": best_candidate_rank,
            "correct_candidate_rank": correct_candidate_rank,
            "failure_category": failure_categories[0] if failure_categories else None,
            "failure_categories": failure_categories,
            "difficulty_categories": difficulty_tags,
            "visible_path": record.get("visible_path"),
        }
        for k in normalized_ks:
            row[f"coverage_at_{k}"] = coverage[k]
        rows.append(row)

    summary = summarize_rows(rows, threshold=threshold, coverage_ks=normalized_ks)
    summary["candidate_diagnosis_enabled"] = candidate_mode
    if not candidate_mode:
        for k in normalized_ks:
            summary[f"coverage_at_{k}"] = None
    return {"summary": summary, "rows": rows}


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    coverage_ks: Sequence[int] = DEFAULT_COVERAGE_KS,
) -> dict[str, Any]:
    """Aggregate per-sample records without pandas or model dependencies."""

    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    total = len(rows)
    ious = [float(row["iou"]) if row["iou"] is not None else 0.0 for row in rows]
    valid_ious = [float(row["iou"]) for row in rows if row["iou"] is not None]
    correct = sum(bool(row["hit_at_0_5"]) for row in rows)
    status_counts = Counter(str(row["prediction_status"]) for row in rows)
    category_counts = Counter(
        str(row["failure_category"])
        for row in rows
        if row.get("failure_category") is not None
    )
    all_category_counts: Counter[str] = Counter()
    difficulty_failure_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("hit_at_0_5"):
            continue
        all_category_counts.update(str(value) for value in row.get("failure_categories", []))
        difficulty_failure_counts.update(str(value) for value in row.get("difficulty_categories", []))

    summary: dict[str, Any] = {
        "sample_count": total,
        "correct_count": int(correct),
        "threshold": float(threshold),
        "acc_at_iou": float(correct / total),
        "acc_at_0_5": float(correct / total),
        "ACC@0.5": float(correct / total),
        "mean_iou": float(np.mean(ious)),
        "mean_IoU": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_iou_valid_only": float(np.mean(valid_ious)) if valid_ious else None,
        "valid_bbox_count": int(status_counts.get("valid", 0)),
        "valid_bbox_rate": float(status_counts.get("valid", 0) / total),
        "invalid_bbox_count": int(status_counts.get("invalid", 0)),
        "missing_prediction_count": int(status_counts.get("missing", 0)),
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "failure_count": int(total - correct),
        "failure_category_counts": dict(sorted(category_counts.items())),
        "failure_category_counts_all": dict(sorted(all_category_counts.items())),
        "difficulty_failure_counts": dict(sorted(difficulty_failure_counts.items())),
    }
    for k in sorted({int(value) for value in coverage_ks}):
        coverage_value = float(
            np.mean([bool(row.get(f"coverage_at_{k}", False)) for row in rows])
        )
        summary[f"coverage_at_{k}"] = coverage_value
        summary[f"coverage@{k}"] = coverage_value
    candidate_rows = [row for row in rows if int(row.get("candidate_count", 0)) > 0]
    summary["candidate_sample_count"] = len(candidate_rows)
    summary["candidate_coverage_sample_rate"] = float(len(candidate_rows) / total)
    return summary


def recommend_priority(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Recommend the next optimization target from candidate-coverage gaps."""

    acc = float(summary.get("acc_at_0_5", 0.0))
    candidate_rate = float(summary.get("candidate_coverage_sample_rate", 0.0))
    if candidate_rate <= 0.0:
        return {
            "priority": "candidate_generation",
            "reason": "没有候选框诊断结果，先补齐 DINO top-5/top-10 覆盖率。",
            "coverage_at_1": None,
            "coverage_at_10": None,
            "selection_gap": None,
        }

    coverage_1 = float(summary.get("coverage_at_1", 0.0))
    coverage_10 = float(summary.get("coverage_at_10", 0.0))
    selection_gap = max(0.0, coverage_10 - acc)
    ranking_gap = max(0.0, coverage_10 - coverage_1)
    if coverage_10 < 0.80:
        priority = "candidate_generation"
        reason = "top-10 覆盖率低于 0.80，主要问题是正确目标尚未稳定进入候选集。"
    elif ranking_gap >= 0.10 or selection_gap >= 0.15:
        priority = "candidate_selection_or_ranking"
        reason = "候选集已包含较多正确目标，但 top-1/最终排序与候选覆盖之间存在明显差距。"
    elif coverage_1 < acc + 0.05:
        priority = "box_regression"
        reason = "正确候选基本存在且排序接近最终预测，下一步应优化框精修或坐标回归。"
    else:
        priority = "candidate_selection_or_ranking"
        reason = "候选生成已基本达标，优先验证语言模型重排与多候选选择。"
    return {
        "priority": priority,
        "reason": reason,
        "coverage_at_1": coverage_1,
        "coverage_at_10": coverage_10,
        "selection_gap": selection_gap,
        "ranking_gap": ranking_gap,
    }


def _resolve_image_path(
    value: str | None,
    image_root: str | Path | None,
) -> Path | None:
    if not value:
        return None
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    return Path(image_root) / path if image_root is not None else path


def _draw_box(
    image: np.ndarray,
    bbox: Sequence[float],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    point1 = (round(x1 * width), round(y1 * height))
    point2 = (round(x2 * width), round(y2 * height))
    cv2.rectangle(image, point1, point2, color, thickness)
    text_origin = (point1[0], max(16, point1[1] - 5))
    cv2.putText(
        image,
        label[:120],
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def visualize_failures(
    rows: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    image_root: str | Path | None,
    output_dir: str | Path,
    *,
    limit: int = 20,
    top_k: int = 5,
) -> list[Path]:
    """Render failed samples with GT, final prediction and ranked candidates."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_by_id = {str(record["query_id"]): record for record in validation}
    candidates = candidates or {}
    failures = [
        row for row in rows
        if not bool(row.get("hit_at_0_5"))
    ]
    failures.sort(key=lambda row: (
        float(row["iou"]) if row.get("iou") is not None else -1.0,
        str(row["query_id"]),
    ))
    output_paths: list[Path] = []
    for row in failures[:limit]:
        query_id = str(row["query_id"])
        source = validation_by_id[query_id]
        image_path = _resolve_image_path(source.get("visible_path"), image_root)
        if image_path is None:
            raise ValueError(f"{query_id}: validation record has no visible image path")
        try:
            encoded_image = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        except OSError as exc:
            raise FileNotFoundError(f"failed to read visualization image: {image_path}") from exc
        if image is None:
            raise FileNotFoundError(f"failed to read visualization image: {image_path}")

        _draw_box(
            image,
            source["bbox"],
            (0, 200, 0),
            f"GT {query_id}",
        )
        prediction = row.get("prediction_bbox")
        if prediction is not None:
            _draw_box(
                image,
                prediction,
                (0, 0, 255),
                f"PRED IoU={float(row['iou']):.3f}",
                thickness=2,
            )
        for candidate in candidates.get(query_id, [])[:top_k]:
            bbox = candidate.get("bbox")
            if bbox is None:
                continue
            candidate_iou = box_iou(bbox, source["bbox"])
            color = (255, 180, 0) if candidate_iou < 0.5 else (255, 0, 255)
            score = candidate.get("score")
            score_text = "" if score is None else f" score={float(score):.3f}"
            _draw_box(
                image,
                bbox,
                color,
                f"C{candidate['rank']} IoU={candidate_iou:.3f}{score_text}",
                thickness=1,
            )

        query_text = str(source.get("query", ""))
        category = str(row.get("failure_category") or "unknown")
        cv2.putText(
            image,
            f"{category} | {query_text}"[:150],
            (8, image.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        safe_id = query_id.replace("/", "_").replace("\\", "_")
        output_path = output_dir / f"{safe_id}.jpg"
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise OSError(f"failed to encode visualization: {output_path}")
        try:
            output_path.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise OSError(f"failed to write visualization: {output_path}") from exc
        output_paths.append(output_path)
    return output_paths
