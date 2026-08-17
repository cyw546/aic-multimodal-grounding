from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from src.metrics import validate_bbox


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")
    return payload


def validate_submission_payloads(
    original: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, int]:
    """Validate IDs, immutable fields and predicted normalized boxes."""
    original_ids = set(original)
    prediction_ids = set(prediction)
    missing = sorted(original_ids - prediction_ids)
    extra = sorted(prediction_ids - original_ids)
    if missing or extra:
        raise ValueError(
            f"Query ID mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    for query_id in original:
        source = original[query_id]
        result = prediction[query_id]
        if not isinstance(source, dict) or not isinstance(result, dict):
            raise ValueError(f"{query_id}: each record must be a JSON object")

        source_fields = set(source)
        result_fields = set(result)
        expected_result_fields = source_fields | {"bbox"}

        if result_fields != expected_result_fields:
            raise ValueError(
                f"{query_id}: field mismatch; "
                f"expected={sorted(expected_result_fields)}, "
                f"got={sorted(result_fields)}"
            )

        for field, source_value in source.items():
            if field != "bbox" and result[field] != source_value:
                raise ValueError(f"{query_id}: field {field!r} was modified")

        validate_bbox(result["bbox"], name=f"{query_id}.bbox")

    return {"sample_count": len(original), "valid_bbox_count": len(original)}


def validate_submission_files(
    original_json: str | Path,
    prediction_json: str | Path,
) -> dict[str, int]:
    return validate_submission_payloads(
        _load_json(original_json),
        _load_json(prediction_json),
    )


def validate_submission_zip(
    original_json: str | Path,
    submission_zip: str | Path,
) -> dict[str, int]:
    """Validate a ZIP containing exactly one JSON file and no unsafe paths."""
    submission_zip = Path(submission_zip)
    if not zipfile.is_zipfile(submission_zip):
        raise ValueError(f"Not a valid ZIP archive: {submission_zip}")

    with zipfile.ZipFile(submission_zip) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(files) != 1:
            raise ValueError(f"Submission ZIP must contain exactly one file, got {len(files)}")
        entry = files[0]
        entry_path = Path(entry.filename)
        if entry_path.is_absolute() or ".." in entry_path.parts:
            raise ValueError(f"Unsafe path in ZIP: {entry.filename}")
        if entry_path.suffix.lower() != ".json":
            raise ValueError(f"Submission file must be JSON, got {entry.filename}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            extracted = Path(temporary_directory) / "prediction.json"
            extracted.write_bytes(archive.read(entry))
            return validate_submission_files(original_json, extracted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an AIC prediction JSON or ZIP")
    parser.add_argument("--original", required=True, help="Original official JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prediction", help="Prediction JSON to validate")
    group.add_argument("--zip", dest="zip_path", help="Submission ZIP to validate")
    args = parser.parse_args()

    if args.prediction:
        report = validate_submission_files(args.original, args.prediction)
    else:
        report = validate_submission_zip(args.original, args.zip_path)
    print(json.dumps({"status": "PASS", **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
