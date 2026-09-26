#!/usr/bin/env python3
"""Inspect candidate competition JSON files without changing server data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODALITY_KEYS = ("visible", "infrared", "depth")


def safe_relative_path(root: Path, value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "not a non-empty string"
    relative = Path(value.strip().replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None, "unsafe or absolute path"
    return root / relative, None


def image_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not result["exists"]:
        return result

    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            result["read_error"] = "cv2.imread returned None"
        else:
            result.update(
                {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                }
            )
    except Exception as exc:  # Diagnostic only: preserve audit progress.
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_record(data_root: Path, key: str, record: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "first_record_id": key,
        "first_record_type": type(record).__name__,
    }
    if not isinstance(record, dict):
        return result

    result["first_record_keys"] = sorted(str(item) for item in record)
    query = record.get("query")
    result["query_present"] = isinstance(query, str) and bool(query.strip())
    result["query_length"] = len(query) if isinstance(query, str) else None

    modalities: dict[str, Any] = {}
    for modality in MODALITY_KEYS:
        if modality not in record:
            continue
        resolved, error = safe_relative_path(data_root, record[modality])
        if error is not None:
            modalities[modality] = {"value_error": error}
        else:
            assert resolved is not None
            modalities[modality] = image_metadata(resolved)
    result["modalities"] = modalities
    return result


def inspect_json(path: Path, data_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["top_level_type"] = type(payload).__name__
    if isinstance(payload, dict):
        result["record_count"] = len(payload)
        if payload:
            first_key = next(iter(payload))
            result.update(inspect_record(data_root, str(first_key), payload[first_key]))
    elif isinstance(payload, list):
        result["record_count"] = len(payload)
        if payload:
            result.update(inspect_record(data_root, "0", payload[0]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("json_files", nargs="*")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    inspected: list[dict[str, Any]] = []
    for value in args.json_files[:30]:
        path = Path(value).expanduser().resolve()
        inspected.append(inspect_json(path, data_root))

    output = {
        "data_root": str(data_root),
        "data_root_exists": data_root.is_dir(),
        "candidate_count": len(inspected),
        "candidates": inspected,
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

