from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/root/autodl-tmp/aic_grounding/data/sample")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def main() -> None:
    json_path = ROOT / "sample.json"
    with json_path.open("r", encoding="utf-8") as stream:
        samples = json.load(stream)

    print(f"sample_count={len(samples)}")
    for query_id, sample in samples.items():
        print(f"query_id={query_id}")
        print(f"query={sample['query']}")
        print(f"bbox={sample['bbox']}")

        shapes: dict[str, tuple[int, ...]] = {}
        for modality in ("visible", "infrared", "depth"):
            path = ROOT / sample[modality]
            image = read_image(path)
            shapes[modality] = image.shape[:2]
            print(
                f"{modality}: path={path.relative_to(ROOT)}, "
                f"shape={image.shape}, dtype={image.dtype}, "
                f"min={image.min()}, max={image.max()}, "
                f"mean={image.mean():.3f}"
            )
            if modality == "depth":
                print(f"depth_zero_ratio={np.mean(image == 0):.6f}")

        if len(set(shapes.values())) != 1:
            raise ValueError(f"Spatial shapes are not aligned: {shapes}")

    print("sample_inspection=PASS")


if __name__ == "__main__":
    main()
