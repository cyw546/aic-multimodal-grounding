from baseline.grounding_model import GroundingModel
from baseline.postprocess import process_prediction

class Baseline:


    def __init__(
        self,
        config_path,
        weight_path
    ):

        self.model = GroundingModel(
            config_path,
            weight_path
        )


    def predict(
        self,
        image,
        query
    ):

        raw_result = self.model.predict(
            image,
            query
        )


        result = process_prediction(
            raw_result["boxes"],
            raw_result["scores"],
            raw_result["labels"]
        )

        if result is not None:
            return result

        # 降低阈值重新推理
        raw_result = self.model.predict(
            image,
            query,
            box_threshold=0.05,
            text_threshold=0.05
        )

        result = process_prediction(
            raw_result["boxes"],
            raw_result["scores"],
            raw_result["labels"]
        )

        if result is not None:
            return result

        # 最终合法回退
        return {
            "bbox": [0.25, 0.25, 0.75, 0.75],
            "score": 0.0,
            "label": ""
        }