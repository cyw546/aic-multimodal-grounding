from __future__ import annotations

import numpy as np
import torch
from PIL import Image

try:
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import load_model, predict as dino_predict
except ModuleNotFoundError as exc:
    T = load_model = dino_predict = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_groundingdino() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "GroundingDINO is not installed; follow the pinned README instructions"
        ) from _IMPORT_ERROR


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    _require_groundingdino()
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transformed_image, _ = transform(Image.fromarray(image), None)
    return transformed_image


class GroundingModel:
    def __init__(
        self,
        config_path: str,
        weight_path: str,
        device: str | None = None,
    ) -> None:
        _require_groundingdino()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model = load_model(config_path, weight_path, device=self.device)
        self.model.to(self.device)

    def predict(
        self,
        image: np.ndarray,
        query: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> dict:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be numpy.ndarray, got {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise TypeError(f"image dtype must be uint8, got {image.dtype}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not 0.0 <= box_threshold <= 1.0:
            raise ValueError("box_threshold must be in [0, 1]")
        if not 0.0 <= text_threshold <= 1.0:
            raise ValueError("text_threshold must be in [0, 1]")

        boxes, logits, phrases = dino_predict(
            model=self.model,
            image=preprocess_image(image),
            caption=query.strip(),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        return {
            "boxes": boxes,
            "scores": logits,
            "labels": phrases,
            "image_size": image.shape[:2],
        }
