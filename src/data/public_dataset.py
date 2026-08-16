from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from .dataset import Sample, Transform, _validate_bbox


class PublicGroundingDataset(Dataset[Sample]):
    """Read RGB referring-expression samples from unified JSONL."""

    REQUIRED_FIELDS = (
        "sample_id", "source", "split", "query_id", "query",
        "visible_path", "bbox", "bbox_format", "width", "height",
    )

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        transform: Transform | None = None,
        validate_files: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.transform = transform
        self.records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        with self.jsonl_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{self.jsonl_path}:{line_number}: record must be an object")
                missing = [field for field in self.REQUIRED_FIELDS if field not in record]
                if missing:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_number}: missing fields {missing}"
                    )
                sample_id = record["sample_id"]
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{self.jsonl_path}:{line_number}: invalid sample_id")
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                seen_ids.add(sample_id)
                if record["bbox_format"] != "xyxy_norm":
                    raise ValueError(f"{sample_id}: bbox_format must be xyxy_norm")
                _validate_bbox(record["bbox"], sample_id)
                path = self._resolve_image(record["visible_path"])
                if validate_files and not path.is_file():
                    raise FileNotFoundError(f"{sample_id}: missing RGB image: {path}")
                self.records.append(record)

    def _resolve_image(self, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"visible_path must be a non-empty string, got {value!r}")
        relative = Path(value.strip().replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe visible_path: {value!r}")
        return self.image_root / relative

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        path = self._resolve_image(record["visible_path"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read RGB image: {path}")
        image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        expected_size = (record["width"], record["height"])
        if all(isinstance(value, int) for value in expected_size) and (
            image.shape[1], image.shape[0]
        ) != expected_size:
            raise ValueError(f"{record['sample_id']}: image size differs from annotation")
        sample: Sample = {
            "sample_id": record["sample_id"],
            "query_id": str(record["query_id"]),
            "query": record["query"].strip(),
            "visible": image,
            "infrared": None,
            "depth": None,
            "image_size": image.shape[:2],
            "bbox": _validate_bbox(record["bbox"], record["sample_id"]),
            "paths": {"visible": str(path), "infrared": None, "depth": None},
            "source": record["source"],
            "split": record["split"],
        }
        return self.transform(sample) if self.transform is not None else sample
