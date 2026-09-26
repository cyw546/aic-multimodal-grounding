#!/usr/bin/env python3
"""Validate official multimodal data without reading duplicate image groups."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.official import (
    load_official_queries,
    resolve_relative_path,
)


MODALITIES = ("visible", "infrared", "depth")
DEPTH_MIN_VALID_MM = 300
DEPTH_MAX_VALID_MM = 19999


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _numeric_summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def _input_path(data_root: str | Path | None) -> Path:
    value = data_root if data_root is not None else os.environ.get("AIC_OFFICIAL_ROOT")
    if value is None or not str(value).strip():
        raise ValueError("Pass --data-root or set AIC_OFFICIAL_ROOT")
    return Path(value).expanduser().resolve()


def _new_modality_stats() -> dict[str, Any]:
    return {
        "files_checked": 0,
        "read_failures": [],
        "dtypes": Counter(),
        "channels": Counter(),
        "shape_counts": Counter(),
        "global_min": None,
        "global_max": None,
        "image_mean_sum": 0.0,
        "image_min_sum": 0.0,
        "image_max_sum": 0.0,
        "infrared_channel_checks": 0,
        "infrared_channel_equal": 0,
        "depth_zero_pixels": 0,
        "depth_total_pixels": 0,
        "depth_valid_pixels": 0,
        "depth_valid_sum": 0,
        "depth_valid_min": None,
        "depth_valid_max": None,
    }


def _finalize_modality_stats(stats: dict[str, Any]) -> dict[str, Any]:
    checked = stats["files_checked"]
    report = {
        "files_checked": checked,
        "read_failure_count": len(stats["read_failures"]),
        "read_failure_examples": stats["read_failures"],
        "dtype_counts": _counter_dict(stats["dtypes"]),
        "channel_counts": _counter_dict(stats["channels"]),
        "shape_counts": _counter_dict(stats["shape_counts"]),
        "value_range": {
            "min": stats["global_min"],
            "max": stats["global_max"],
            "mean_of_image_means": (
                stats["image_mean_sum"] / checked if checked else None
            ),
            "mean_of_image_mins": (
                stats["image_min_sum"] / checked if checked else None
            ),
            "mean_of_image_maxs": (
                stats["image_max_sum"] / checked if checked else None
            ),
        },
    }

    channel_checks = stats["infrared_channel_checks"]
    if channel_checks:
        report["channel_consistent_ratio"] = (
            stats["infrared_channel_equal"] / channel_checks
        )

    total_depth_pixels = stats["depth_total_pixels"]
    if total_depth_pixels:
        valid_pixels = stats["depth_valid_pixels"]
        report["depth"] = {
            "zero_pixel_ratio": stats["depth_zero_pixels"] / total_depth_pixels,
            "valid_300_19999_pixel_ratio": valid_pixels / total_depth_pixels,
            "valid_pixel_count": valid_pixels,
            "valid_min_mm": stats["depth_valid_min"],
            "valid_max_mm": stats["depth_valid_max"],
            "valid_mean_mm": (
                stats["depth_valid_sum"] / valid_pixels if valid_pixels else None
            ),
        }
    return report


def _observe_image(stats: dict[str, Any], modality: str, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    image_min = int(image.min())
    image_max = int(image.max())
    image_mean = float(image.mean())

    stats["files_checked"] += 1
    stats["dtypes"][str(image.dtype)] += 1
    stats["channels"][channels] += 1
    stats["shape_counts"][f"{height}x{width}"] += 1
    stats["global_min"] = image_min if stats["global_min"] is None else min(stats["global_min"], image_min)
    stats["global_max"] = image_max if stats["global_max"] is None else max(stats["global_max"], image_max)
    stats["image_mean_sum"] += image_mean
    stats["image_min_sum"] += image_min
    stats["image_max_sum"] += image_max

    if modality == "infrared" and image.ndim == 3 and image.shape[2] == 3:
        stats["infrared_channel_checks"] += 1
        if (
            np.array_equal(image[:, :, 0], image[:, :, 1])
            and np.array_equal(image[:, :, 1], image[:, :, 2])
        ):
            stats["infrared_channel_equal"] += 1

    if modality == "depth":
        stats["depth_total_pixels"] += int(image.size)
        stats["depth_zero_pixels"] += int(np.count_nonzero(image == 0))
        valid = image[
            (image >= DEPTH_MIN_VALID_MM) & (image <= DEPTH_MAX_VALID_MM)
        ]
        if valid.size:
            valid_min = int(valid.min())
            valid_max = int(valid.max())
            stats["depth_valid_pixels"] += int(valid.size)
            stats["depth_valid_sum"] += int(valid.sum(dtype=np.int64))
            stats["depth_valid_min"] = (
                valid_min
                if stats["depth_valid_min"] is None
                else min(stats["depth_valid_min"], valid_min)
            )
            stats["depth_valid_max"] = (
                valid_max
                if stats["depth_valid_max"] is None
                else max(stats["depth_valid_max"], valid_max)
            )


def _query_count_distribution(group_query_counts: list[int]) -> dict[str, Any]:
    histogram = Counter()
    for count in group_query_counts:
        if count == 1:
            histogram["1"] += 1
        elif count == 2:
            histogram["2"] += 1
        elif count <= 5:
            histogram["3-5"] += 1
        elif count <= 10:
            histogram["6-10"] += 1
        else:
            histogram[">10"] += 1
    return {
        **_numeric_summary(group_query_counts),
        "histogram": dict(histogram),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _format_counter(counter: dict[str, int]) -> str:
    return ", ".join(f"`{key}`: {value}" for key, value in counter.items()) or "-"


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Official Repechage Data Quality Report",
        "",
        f"- Status: **{report['status']}**",
        f"- Data root: `{report['data_root']}`",
        f"- Queries JSON: `{report['queries_json']}`",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Query and image groups",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Query count | {summary['query_count']} |",
        f"| Expected Query count | {summary['expected_query_count']} |",
        f"| Unique image groups referenced by Queries | {summary['unique_image_group_count']} |",
        f"| Expected image group count | {summary['expected_image_group_count']} |",
        f"| Unique visible image IDs referenced by Queries | {summary['unique_visible_image_id_count']} |",
        f"| Image group count matches expectation | {summary['image_group_count_matches']} |",
        "",
        "## Image files and references",
        "",
        "| Modality | Files in directory | Referenced references | Unique referenced files | Unreferenced files | Missing references |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for modality in MODALITIES:
        counts = report["path_checks"][modality]
        lines.append(
            f"| {modality} | {counts['directory_file_count']} | "
            f"{counts['query_reference_count']} | {counts['unique_referenced_file_count']} | "
            f"{counts['unreferenced_file_count']} | {counts['missing_reference_count']} |"
        )

    lines.extend([
        "",
        "## Modality diagnostics",
        "",
        "| Modality | Shapes | dtype | Channels | Pixel range | Read failures |",
        "| --- | --- | --- | --- | --- | ---: |",
    ])
    for modality in MODALITIES:
        stats = report["modalities"][modality]
        pixel_range = stats["value_range"]
        lines.append(
            f"| {modality} | {_format_counter(stats['shape_counts'])} | "
            f"{_format_counter(stats['dtype_counts'])} | {_format_counter(stats['channel_counts'])} | "
            f"{pixel_range['min']} to {pixel_range['max']} | {stats['read_failure_count']} |"
        )

    depth = report["modalities"]["depth"].get("depth")
    if depth:
        lines.extend([
            "",
            "## Depth statistics",
            "",
            f"- Zero pixel ratio: {depth['zero_pixel_ratio']:.6f}",
            f"- Valid 300-19999 mm pixel ratio: {depth['valid_300_19999_pixel_ratio']:.6f}",
            f"- Valid value range: {depth['valid_min_mm']} to {depth['valid_max_mm']} mm",
            f"- Mean of valid depth values: {depth['valid_mean_mm']:.3f} mm",
        ])

    distribution = report["query_count_per_image_group"]
    lines.extend([
        "",
        "## Queries per image group",
        "",
        f"- Minimum: {distribution['min']}",
        f"- Maximum: {distribution['max']}",
        f"- Mean: {distribution['mean']:.3f}",
        f"- Median: {distribution['median']:.3f}",
        f"- Histogram: {json.dumps(distribution['histogram'], ensure_ascii=False)}",
        "",
        "## Warnings and errors",
        "",
    ])
    if report["warnings"]:
        lines.extend(f"- WARNING: {warning}" for warning in report["warnings"])
    if report["errors"]:
        lines.extend(f"- ERROR: {error}" for error in report["errors"])
    if not report["warnings"] and not report["errors"]:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def check_official_dataset(
    data_root: str | Path | None = None,
    *,
    queries_path: str | Path | None = None,
    expected_query_count: int | None = 5690,
    expected_image_group_count: int | None = 2000,
    examples: int = 20,
    json_report: str | Path | None = None,
    markdown_report: str | Path | None = None,
) -> dict[str, Any]:
    """Validate official data and optionally write JSON and Markdown reports."""

    if examples < 0:
        raise ValueError("examples must be non-negative")

    root, resolved_queries_path, records = load_official_queries(
        data_root=data_root,
        queries_path=queries_path,
    )

    errors: list[str] = []
    warnings: list[str] = []
    missing_references: list[dict[str, str]] = []
    referenced_files = {modality: set() for modality in MODALITIES}
    query_reference_counts = {modality: 0 for modality in MODALITIES}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}

    for query_id, record in records.items():
        relative_paths: dict[str, str] = {}
        resolved_paths: dict[str, Path] = {}
        record_valid = True
        for modality in MODALITIES:
            try:
                resolved = resolve_relative_path(
                    root,
                    record[modality],
                    field=modality,
                    query_id=query_id,
                )
            except ValueError as exc:
                errors.append(str(exc))
                record_valid = False
                continue

            relative = Path(str(record[modality]).strip().replace("\\", "/")).as_posix()
            relative_paths[modality] = relative
            resolved_paths[modality] = resolved
            referenced_files[modality].add(relative)
            query_reference_counts[modality] += 1
            if not resolved.is_file():
                missing_references.append({
                    "query_id": query_id,
                    "modality": modality,
                    "path": relative,
                })

        if not record_valid:
            continue

        group_key = tuple(relative_paths[modality] for modality in MODALITIES)
        group = groups.setdefault(
            group_key,
            {
                "query_ids": [],
                "resolved_paths": resolved_paths,
                "relative_paths": relative_paths,
            },
        )
        group["query_ids"].append(query_id)

    modality_stats = {
        modality: _new_modality_stats() for modality in MODALITIES
    }
    processed_files: set[tuple[str, str]] = set()

    for group_index, group in enumerate(groups.values(), start=1):
        for modality in MODALITIES:
            relative = group["relative_paths"][modality]
            file_key = (modality, relative)
            if file_key in processed_files:
                continue
            processed_files.add(file_key)

            path = group["resolved_paths"][modality]
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                modality_stats[modality]["read_failures"].append(relative)
                continue
            _observe_image(modality_stats[modality], modality, image)

        if group_index % 100 == 0:
            print(f"Checked {group_index}/{len(groups)} image groups", flush=True)

    path_checks: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        image_dir = root / "Images" / modality
        actual_files = (
            sorted(
                path.relative_to(root).as_posix()
                for path in image_dir.iterdir()
                if path.is_file()
            )
            if image_dir.is_dir()
            else []
        )
        actual_set = set(actual_files)
        unreferenced = sorted(actual_set - referenced_files[modality])
        missing = [
            item for item in missing_references if item["modality"] == modality
        ]
        path_checks[modality] = {
            "directory": str(image_dir),
            "directory_file_count": len(actual_files),
            "query_reference_count": query_reference_counts[modality],
            "unique_referenced_file_count": len(referenced_files[modality]),
            "missing_reference_count": len(missing),
            "missing_reference_examples": missing[:examples],
            "unreferenced_file_count": len(unreferenced),
            "unreferenced_file_examples": unreferenced[:examples],
        }

    query_lengths = [len(record["query"]) for record in records.values()]
    group_query_counts = [len(group["query_ids"]) for group in groups.values()]
    unique_visible_image_ids = {
        Path(group["relative_paths"]["visible"]).stem for group in groups.values()
    }

    count_matches = {
        "query_count": (
            expected_query_count is None or len(records) == expected_query_count
        ),
        "image_group_count": (
            expected_image_group_count is None
            or len(groups) == expected_image_group_count
        ),
    }

    for modality in MODALITIES:
        missing_count = path_checks[modality]["missing_reference_count"]
        if missing_count:
            errors.append(f"{modality}: {missing_count} referenced files are missing")
        if modality_stats[modality]["read_failures"]:
            errors.append(
                f"{modality}: {len(modality_stats[modality]['read_failures'])} files could not be decoded"
            )
        if path_checks[modality]["unreferenced_file_count"]:
            warnings.append(
                f"{modality}: {path_checks[modality]['unreferenced_file_count']} files are not referenced by queries.json"
            )

    if not count_matches["query_count"]:
        errors.append(
            f"Query count mismatch: expected {expected_query_count}, got {len(records)}"
        )
    if not count_matches["image_group_count"]:
        errors.append(
            "Image group count mismatch: "
            f"expected {expected_image_group_count}, got {len(groups)}"
        )

    status = "PASS" if not errors else "FAIL"
    report: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "queries_json": str(resolved_queries_path),
        "summary": {
            "query_count": len(records),
            "expected_query_count": expected_query_count,
            "query_count_matches": count_matches["query_count"],
            "unique_image_group_count": len(groups),
            "expected_image_group_count": expected_image_group_count,
            "image_group_count_matches": count_matches["image_group_count"],
            "unique_visible_image_id_count": len(unique_visible_image_ids),
            "query_text_length": _numeric_summary(query_lengths),
        },
        "path_checks": path_checks,
        "query_count_per_image_group": _query_count_distribution(group_query_counts),
        "modalities": {
            modality: _finalize_modality_stats(modality_stats[modality])
            for modality in MODALITIES
        },
        "warnings": warnings,
        "errors": errors,
    }

    if json_report is not None:
        _write_json(Path(json_report).expanduser().resolve(), report)
    if markdown_report is not None:
        _write_text(
            Path(markdown_report).expanduser().resolve(),
            _markdown_report(report),
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate official visible, infrared, and depth datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Official dataset root. Defaults to AIC_OFFICIAL_ROOT.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Optional queries JSON path. Defaults to <data-root>/queries/queries.json.",
    )
    parser.add_argument("--expected-query-count", type=int, default=5690)
    parser.add_argument("--expected-image-group-count", type=int, default=2000)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("outputs/inspection/repechage_quality.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("outputs/inspection/repechage_quality.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_official_dataset(
        data_root=args.data_root,
        queries_path=args.queries,
        expected_query_count=args.expected_query_count,
        expected_image_group_count=args.expected_image_group_count,
        examples=args.examples,
        json_report=args.json_report,
        markdown_report=args.markdown_report,
    )
    print(json.dumps({
        "status": report["status"],
        "query_count": report["summary"]["query_count"],
        "unique_image_group_count": report["summary"]["unique_image_group_count"],
        "json_report": str(args.json_report),
        "markdown_report": str(args.markdown_report),
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())