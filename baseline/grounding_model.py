import torch
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import (
    load_model,
    predict
)

def preprocess_image(image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    pil_image = Image.fromarray(image)

    transformed_image, _ = transform(pil_image, None)

    return transformed_image


class GroundingModel:

    def __init__(
        self,
        config_path,
        weight_path
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = load_model(
            config_path,
            weight_path
        )

        self.model.to(self.device)


    def predict(
        self,
        image: np.ndarray, query: str,
        box_threshold=0.35,
        text_threshold=0.25
    ):
        # 1. 检查 image 类型
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be numpy.ndarray, got {type(image)}"
            )

        # 2. 检查图片形状
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape (H, W, 3), got {image.shape}"
            )

        # 3. 检查数据类型
        if image.dtype != np.uint8:
            raise TypeError(
                f"image dtype must be uint8, got {image.dtype}"
            )

        # 4. 检查 query
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string"
            )

        transformed_image = preprocess_image(image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=transformed_image,
            caption=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        result = {
            "boxes": boxes,
            "scores": logits,
            "labels": phrases,
            "image_size": image.shape[:2]
        }

        return result