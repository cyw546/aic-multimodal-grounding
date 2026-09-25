from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def official_dataset_factory(tmp_path):
    counter = 0

    def factory(
        *,
        query_count: int = 3,
        group_count: int = 2,
        extra_visible: int = 0,
    ) -> tuple[Path, dict[str, dict[str, object]]]:
        nonlocal counter
        counter += 1
        root = tmp_path / f"official_{counter}"
        queries_dir = root / "queries"
        queries_dir.mkdir(parents=True)

        for modality in ("visible", "infrared", "depth"):
            (root / "Images" / modality).mkdir(parents=True)

        image_names: list[str] = []
        for index in range(group_count):
            image_name = f"{index + 1:06d}.png"
            image_names.append(image_name)

            height, width = 12, 16
            visible = np.zeros((height, width, 3), dtype=np.uint8)
            visible[:, :, 0] = 20
            visible[:, :, 1] = 80
            visible[:, :, 2] = 140
            assert cv2.imwrite(
                str(root / "Images" / "visible" / image_name),
                visible,
            )

            infrared = np.full((height, width, 3), 70, dtype=np.uint8)
            assert cv2.imwrite(
                str(root / "Images" / "infrared" / image_name),
                infrared,
            )

            depth = np.full((height, width), 1500, dtype=np.uint16)
            depth[0, 0] = 0
            assert cv2.imwrite(
                str(root / "Images" / "depth" / image_name),
                depth,
            )

        for index in range(extra_visible):
            image_name = f"extra_{index + 1:06d}.png"
            visible = np.zeros((12, 16, 3), dtype=np.uint8)
            assert cv2.imwrite(
                str(root / "Images" / "visible" / image_name),
                visible,
            )

        records: dict[str, dict[str, object]] = {}
        for index in range(query_count):
            group_index = index % group_count
            image_name = image_names[group_index]
            query_id = f"{group_index + 1:06d}_{index + 1:03d}"
            records[query_id] = {
                "visible": f"Images/visible/{image_name}",
                "infrared": f"Images/infrared/{image_name}",
                "depth": f"Images/depth/{image_name}",
                "query": "the small target near the left corner"
                if index == 0
                else f"target example {index}",
            }

        (queries_dir / "queries.json").write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )
        return root, records

    return factory