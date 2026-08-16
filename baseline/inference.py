from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from baseline.grounding_model import GroundingModel
from baseline.postprocess import process_prediction


class Baseline:
    def __init__(
        self,
        config_path: str | None = None,
        weight_path: str | None = None,
        *,
        device: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        fallback_box_threshold: float = 0.05,
        fallback_text_threshold: float = 0.05,
        model: Any | None = None,
    ) -> None:
        thresholds = (
            box_threshold, text_threshold,
            fallback_box_threshold, fallback_text_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("all thresholds must be in [0, 1]")
        if model is None:
            if config_path is None or weight_path is None:
                raise ValueError("config_path and weight_path are required")
            model = GroundingModel(config_path, weight_path, device=device)
        self.model = model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.fallback_box_threshold = fallback_box_threshold
        self.fallback_text_threshold = fallback_text_threshold

    @classmethod
    def from_yaml(
        cls,
        pipeline_config_path: str | Path,
        *,
        model_config_path: str,
        weight_path: str,
    ) -> "Baseline":
        with Path(pipeline_config_path).open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        model_cfg = payload.get("model", {})
        inference_cfg = payload.get("inference", {})
        device = inference_cfg.get("device", "auto")
        return cls(
            model_config_path,
            weight_path,
            device=None if device == "auto" else str(device),
            box_threshold=float(model_cfg.get("box_threshold", 0.35)),
            text_threshold=float(model_cfg.get("text_threshold", 0.25)),
            fallback_box_threshold=float(model_cfg.get("fallback_box_threshold", 0.05)),
            fallback_text_threshold=float(model_cfg.get("fallback_text_threshold", 0.05)),
        )

    @staticmethod
    def _public_result(result: dict) -> dict:
        return {
            "bbox": [float(value) for value in result["bbox"]],
            "score": float(result["score"]),
        }

    def predict(self, image, query) -> dict:
        raw = self.model.predict(
            image, query,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        result = process_prediction(raw["boxes"], raw["scores"], raw["labels"])
        if result is not None:
            return self._public_result(result)
        raw = self.model.predict(
            image, query,
            box_threshold=self.fallback_box_threshold,
            text_threshold=self.fallback_text_threshold,
        )
        result = process_prediction(raw["boxes"], raw["scores"], raw["labels"])
        if result is not None:
            return self._public_result(result)
        return {"bbox": [0.25, 0.25, 0.75, 0.75], "score": 0.0}
