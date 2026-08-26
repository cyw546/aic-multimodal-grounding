"""Build a deterministic, validated RefCOCO subset for local evaluation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data import PublicGroundingDataset


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _image_path(image_root: Path, value: str) -> Path:
    relative = Path(value.strip().replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe visible_path: {value!r}")
    return image_root / relative


def _visualize(
    records: list[dict[str, Any]],
    image_root: Path,
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in records:
        image_path = _image_path(image_root, record["visible_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read RGB image: {image_path}")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = record["bbox"]
        point1 = (round(x1 * width), round(y1 * height))
        point2 = (round(x2 * width), round(y2 * height))
        cv2.rectangle(image, point1, point2, (0, 255, 0), 2)
        label = f"{record['sample_id']} | {record['query']}"
        cv2.putText(
            image,
            label[:100],
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        output_path = output_dir / f"{record['sample_id']}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"failed to write visualization: {output_path}")
        count += 1
    return count


def build_local_validation(
    input_jsonl: str | Path,
    image_root: str | Path,
    output_jsonl: str | Path,
    visualization_dir: str | Path,
    *,
    sample_count: int = 400,
    visualization_count: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0 <= visualization_count <= sample_count:
        raise ValueError("visualization_count must be between 0 and sample_count")

    input_jsonl = Path(input_jsonl).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output_jsonl = Path(output_jsonl).expanduser().resolve()
    visualization_dir = Path(visualization_dir).expanduser().resolve()

    dataset = PublicGroundingDataset(
        input_jsonl,
        image_root,
        validate_files=True,
    )
    if sample_count > len(dataset):
        raise ValueError(
            f"requested {sample_count} samples, but dataset contains {len(dataset)}"
        )

    random_generator = random.Random(seed)
    selected_indices = sorted(
        random_generator.sample(range(len(dataset)), sample_count)
    )
    selected = [dataset.records[index] for index in selected_indices]
    _write_jsonl(selected, output_jsonl)

    # Re-open the generated subset through the production reader. This checks
    # schema, duplicate IDs, normalized boxes and every selected image path.
    validated = PublicGroundingDataset(
        output_jsonl,
        image_root,
        validate_files=True,
    )
    visual_records = random_generator.sample(
        validated.records,
        visualization_count,
    )
    visualized = _visualize(visual_records, image_root, visualization_dir)

    widths = np.asarray(
        [record["bbox"][2] - record["bbox"][0] for record in selected],
        dtype=np.float64,
    )
    heights = np.asarray(
        [record["bbox"][3] - record["bbox"][1] for record in selected],
        dtype=np.float64,
    )
    return {
        "status": "PASS",
        "source_count": len(dataset),
        "sample_count": len(validated),
        "visualization_count": visualized,
        "seed": seed,
        "output": str(output_jsonl),
        "visualization_dir": str(visualization_dir),
        "bbox_width": {
            "min": float(widths.min()),
            "mean": float(widths.mean()),
            "max": float(widths.max()),
        },
        "bbox_height": {
            "min": float(heights.min()),
            "mean": float(heights.mean()),
            "max": float(heights.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic local RefCOCO validation subset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=400)
    parser.add_argument("--visualization-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_local_validation(
        args.input,
        args.image_root,
        args.output,
        args.visualization_dir,
        sample_count=args.sample_count,
        visualization_count=args.visualization_count,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
