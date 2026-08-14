from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from .multimodal_preprocess import multimodal_early_fusion_from_raw


Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]


def _validate_bbox(value: Sequence[float], query_id: str) -> np.ndarray:
    bbox = np.asarray(value, dtype=np.float32)
    if bbox.shape != (4,):
        raise ValueError(f"{query_id}: bbox must contain four values, got {value!r}")
    if not np.all(np.isfinite(bbox)):
        raise ValueError(f"{query_id}: bbox contains NaN or infinity")
    x1, y1, x2, y2 = bbox.tolist()
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{query_id}: invalid normalized bbox {value!r}")
    return bbox


def _read_image(path: Path, modality: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read {modality} image: {path}")

    if modality == "visible":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Visible image must be HxWx3, got {image.shape}: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif modality == "infrared":
        if image.ndim not in (2, 3):
            raise ValueError(f"Infrared image must be 2D or 3D, got {image.shape}: {path}")
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif modality == "depth":
        if image.ndim != 2:
            raise ValueError(f"Depth image must be single‑channel, got {image.shape}: {path}")
        if image.dtype != np.uint16:
            raise ValueError(f"Depth image must be uint16, got {image.dtype}: {path}")

    return np.ascontiguousarray(image)


class MultimodalGroundingDataset(Dataset[Sample]):
    """Read aligned visible, infrared, depth and query samples from official JSON.

    Images are returned in their original resolution and dtype. Visible images are
    converted from OpenCV BGR order to RGB. Model‑specific resizing and
    normalization should be supplied through ``transform``.
    """

    REQUIRED_FIELDS = ("visible", "infrared", "depth", "query")

    def __init__(
        self,
        json_path: str | Path,
        data_root: str | Path | None = None,
        transform: Transform | None = None,
        require_bbox: bool = False,
        validate_files: bool = True,
    ) -> None:
        self.json_path = Path(json_path).expanduser().resolve()
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else self.json_path.parent
        )
        self.transform = transform
        self.require_bbox = require_bbox

        with self.json_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Top‑level JSON value must be an object keyed by Query ID")

        self.records: list[tuple[str, dict[str, Any]]] = []
        for query_id, record in payload.items():
            if not isinstance(query_id, str) or not isinstance(record, dict):
                raise ValueError("Every dataset entry must map a string Query ID to an object")
            missing = [field for field in self.REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{query_id}: missing required fields {missing}")
            if not isinstance(record["query"], str) or not record["query"].strip():
                raise ValueError(f"{query_id}: query must be a non‑empty string")
            if require_bbox and "bbox" not in record:
                raise ValueError(f"{query_id}: bbox is required but missing")
            if "bbox" in record:
                _validate_bbox(record["bbox"], query_id)
            if validate_files:
                for modality in ("visible", "infrared", "depth"):
                    path = self._resolve_path(record[modality])
                    if not path.is_file():
                        raise FileNotFoundError(f"{query_id}: missing {modality} file: {path}")
            self.records.append((query_id, record))

    def _resolve_path(self, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Image path must be a non‑empty string, got {value!r}")
        relative = Path(value.strip().replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path in JSON: {value!r}")
        return self.data_root / relative

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        query_id, record = self.records[index]
        visible = _read_image(self._resolve_path(record["visible"]), "visible")
        infrared = _read_image(self._resolve_path(record["infrared"]), "infrared")
        depth = _read_image(self._resolve_path(record["depth"]), "depth")

        spatial_shapes = {
            "visible": visible.shape[:2],
            "infrared": infrared.shape[:2],
            "depth": depth.shape[:2],
        }
        if len(set(spatial_shapes.values())) != 1:
            raise ValueError(f"{query_id}: modalities are not aligned: {spatial_shapes}")

        fused = multimodal_early_fusion_from_raw(visible, infrared, depth)

        sample: Sample = {
            "query_id": query_id,
            "query": record["query"].strip(),
            "visible": visible,
            "infrared": infrared,
            "depth": depth,
            "depth_valid_mask": depth > 0,
            "image_size": visible.shape[:2],
            "fused_rgb_infra_depth": fused,
            "paths": {
                modality: str(self._resolve_path(record[modality]))
                for modality in ("visible", "infrared", "depth")
            },
        }
        if "bbox" in record:
            sample["bbox"] = _validate_bbox(record["bbox"], query_id)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def grounding_collate_fn(samples: list[Sample]) -> list[Sample]:
    """Keep variable‑resolution samples as a list until a model transform resizes them."""
    return samples