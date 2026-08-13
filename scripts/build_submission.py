from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from src.submission.validator import (
    validate_submission_files,
    validate_submission_zip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate prediction JSON, build a submission ZIP, "
            "then validate the ZIP again."
        )
    )

    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Official queries.json path.",
    )

    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Prediction JSON containing bbox fields.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ZIP path.",
    )

    return parser.parse_args()


def build_submission(
    original: Path,
    prediction: Path,
    output: Path,
) -> dict[str, object]:
    prediction_report = validate_submission_files(
        original,
        prediction,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = output.with_suffix(
        output.suffix + ".tmp"
    )

    if temporary_output.exists():
        temporary_output.unlink()

    with zipfile.ZipFile(
        temporary_output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(
            prediction,
            arcname="prediction.json",
        )

    temporary_output.replace(output)

    zip_report = validate_submission_zip(
        original,
        output,
    )

    if prediction_report != zip_report:
        raise RuntimeError(
            "Prediction and ZIP validation reports differ"
        )

    return {
        "status": "PASS",
        "prediction": str(prediction),
        "submission_zip": str(output),
        **zip_report,
    }


def main() -> None:
    args = parse_args()

    report = build_submission(
        original=args.original,
        prediction=args.prediction,
        output=args.output,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
