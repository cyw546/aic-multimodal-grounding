import torch

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict
)


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


    def inference(
        self,
        image_path,
        text_prompt,
        box_threshold=0.35,
        text_threshold=0.25
    ):

        image_source, image = load_image(
            image_path
        )


        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )


        result = {
            "boxes": boxes,
            "scores": logits,
            "labels": phrases,
            "image_size": image_source.shape[:2]
        }


        return result
