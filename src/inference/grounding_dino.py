from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class GroundingDINOPredictor:
    """Adapt the RGB Grounding DINO baseline to the common pipeline API."""

    def __init__(self, baseline: Any) -> None:
        if not callable(getattr(baseline, "predict", None)):
            raise TypeError("baseline must provide a callable predict method")
        self.baseline = baseline

    @classmethod
    def from_paths(
        cls,
        *,
        pipeline_config_path: str | Path,
        model_config_path: str | Path,
        weight_path: str | Path,
    ) -> "GroundingDINOPredictor":
        paths = {
            "pipeline config": Path(pipeline_config_path),
            "model config": Path(model_config_path),
            "model weights": Path(weight_path),
        }
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing Grounding DINO files: " + "; ".join(missing))

        # Import lazily so dataset and dummy-pipeline tests work without the
        # optional Grounding DINO package or a CUDA device.
        from baseline.inference import Baseline

        baseline = Baseline.from_yaml(
            paths["pipeline config"],
            model_config_path=str(paths["model config"]),
            weight_path=str(paths["model weights"]),
        )
        return cls(baseline)

    def predict(
        self,
        *,
        image: np.ndarray,
        query: str,
    ) -> dict[str, object]:
        result = self.baseline.predict(image=image, query=query)
        if not isinstance(result, dict):
            raise TypeError("Grounding DINO baseline output must be a dict")
        if "bbox" not in result:
            raise ValueError("Grounding DINO baseline output is missing bbox")
        return result
