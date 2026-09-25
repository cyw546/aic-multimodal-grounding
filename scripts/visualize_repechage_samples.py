#!/usr/bin/env python3
"""Create deterministic three-modality visualizations for official data."""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.official import load_official_queries, resolve_relative_path


DIFFICULTY_PATTERNS = {
    "night_or_low_light": re.compile(
        r"\b(night|dark|low[- ]?light|shadow|dim|evening|dusk)\b",
        re.IGNORECASE,
    ),
    "occlusion": re.compile(
        r"\b(occlud|hidden|behind|partially|covered|blocked|obscured)\b",
        re.IGNORECASE,
    ),
    "small_or_distant_target": re.compile(
        r"\b(small|tiny|distant|far|faraway|background)\b",
        re.IGNORECASE,
    ),
    "similar_targets": re.compile(
        r"\b(similar|another|same|identical|left one|right one)\b",
        re.IGNORECASE,
    ),
    "complex_spatial_description": re.compile(
        r"\b(left|right|above|below|between|near|next to|front|behind|"
        r"corner|center|middle|top|bottom)\b",
        re.IGNORECASE,
    ),
}


def _safe_id(query_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", query_id).strip("._")
    return value or "query"


def _classify_query(query: str) -> list[str]:
    return [
        category
        for category, pattern in DIFFICULTY_PATTERNS.items()
        if pattern.search(query)
    ]


def _read_image(path: Path, modality: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read {modality} image: {path}")
    return image


def _display_infrared(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Infrared image must be 2D or 3-channel, got {image.shape}")


def _display_depth(image: np.ndarray) -> tuple[np.ma.MaskedArray, float | None, float | None]:
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Depth image must be 2D, got {image.shape}")

    valid = (image >= 300) & (image <= 19999)
    if not np.any(valid):
        valid = image > 0
    if not np.any(valid):
        return np.ma.masked_all(image.shape, dtype=np.float32), None, None

    valid_values = image[valid].astype(np.float32)
    lower, upper = np.percentile(valid_values, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(valid_values.min())
        upper = float(valid_values.max())
    if upper <= lower:
        upper = lower + 1.0
    masked = np.ma.array(image.astype(np.float32), mask=~valid)
    return masked, float(lower), float(upper)


def _choose_query_ids(
    records: dict[str, dict[str, Any]],
    *,
    count: int,
    query_ids: list[str] | None,
    seed: int,
) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive")

    selected: list[str] = []
    if query_ids:
        for query_id in query_ids:
            if query_id not in records:
                raise KeyError(f"Query ID not found: {query_id}")
            if query_id not in selected:
                selected.append(query_id)
        if len(selected) > count:
            raise ValueError(
                f"received {len(selected)} Query IDs but count is only {count}"
            )

    remaining = [query_id for query_id in records if query_id not in selected]
    random_generator = random.Random(seed)
    needed = count - len(selected)
    if needed > len(remaining):
        raise ValueError(
            f"requested {count} samples, but only {len(records)} are available"
        )
    selected.extend(random_generator.sample(remaining, needed))
    return selected


def _save_figure(
    *,
    query_id: str,
    query: str,
    visible: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    output_path: Path,
) -> float | None:
    infrared_display = _display_infrared(infrared)
    depth_display, depth_vmin, depth_vmax = _display_depth(depth)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    axes[0].imshow(cv2.cvtColor(visible, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Visible (RGB)")

    axes[1].imshow(infrared_display, cmap="inferno")
    axes[1].set_title("Infrared")

    depth_image = axes[2].imshow(
        depth_display,
        cmap="turbo",
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    axes[2].set_title("Depth (mm)")
    figure.colorbar(depth_image, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.axis("off")

    wrapped_query = textwrap.fill(query, width=110)
    figure.suptitle(f"{query_id}\n{wrapped_query}", fontsize=13, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return depth_vmax


def visualize_repechage_samples(
    data_root: str | Path | None = None,
    *,
    queries_path: str | Path | None = None,
    output_dir: str | Path = "outputs/inspection/repechage_samples",
    count: int = 30,
    query_ids: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate three-panel images and a metadata manifest."""

    root, resolved_queries_path, records = load_official_queries(
        data_root=data_root,
        queries_path=queries_path,
    )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    selected_ids = _choose_query_ids(
        records,
        count=count,
        query_ids=query_ids,
        seed=seed,
    )
    manifest: dict[str, Any] = {
        "data_root": str(root),
        "queries_json": str(resolved_queries_path),
        "output_dir": str(output_path),
        "seed": seed,
        "requested_count": count,
        "generated_count": 0,
        "samples": [],
        "difficulty_categories": {
            category: [] for category in DIFFICULTY_PATTERNS
        },
    }

    for index, query_id in enumerate(selected_ids, start=1):
        record = records[query_id]
        resolved = {
            modality: resolve_relative_path(
                root,
                record[modality],
                field=modality,
                query_id=query_id,
            )
            for modality in ("visible", "infrared", "depth")
        }
        visible = _read_image(resolved["visible"], "visible")
        infrared = _read_image(resolved["infrared"], "infrared")
        depth = _read_image(resolved["depth"], "depth")

        if visible.ndim != 3 or visible.shape[2] != 3:
            raise ValueError(f"{query_id}: visible image is not HxWx3")
        if visible.shape[:2] != infrared.shape[:2] or visible.shape[:2] != depth.shape[:2]:
            raise ValueError(f"{query_id}: modalities are not spatially aligned")

        image_name = f"{index:03d}_{_safe_id(query_id)}.jpg"
        image_path = output_path / image_name
        depth_vmax = _save_figure(
            query_id=query_id,
            query=record["query"],
            visible=visible,
            infrared=infrared,
            depth=depth,
            output_path=image_path,
        )

        categories = _classify_query(record["query"])
        sample = {
            "query_id": query_id,
            "image": image_name,
            "visible": str(resolved["visible"]),
            "infrared": str(resolved["infrared"]),
            "depth": str(resolved["depth"]),
            "query_length": len(record["query"]),
            "difficulty_categories": categories,
            "depth_display_max_mm": depth_vmax,
        }
        manifest["samples"].append(sample)
        for category in categories:
            manifest["difficulty_categories"][category].append(query_id)

        print(f"Generated {index}/{len(selected_ids)}: {query_id}", flush=True)

    manifest["generated_count"] = len(manifest["samples"])
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize official visible, infrared, depth, and Query text."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Official dataset root. Defaults to AIC_OFFICIAL_ROOT.",
    )
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inspection/repechage_samples"),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of samples. Defaults to 30 for random selection.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Specific Query ID. May be provided more than once.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = args.count
    if count is None:
        count = len(args.query_id) if args.query_id else 30
    report = visualize_repechage_samples(
        data_root=args.data_root,
        queries_path=args.queries,
        output_dir=args.output_dir,
        count=count,
        query_ids=args.query_id,
        seed=args.seed,
    )
    print(json.dumps({
        "status": "PASS",
        "generated_count": report["generated_count"],
        "output_dir": report["output_dir"],
        "manifest": str(Path(report["output_dir"]) / "manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())