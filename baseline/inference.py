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

        raw_result = self.model.inference(
            image,
            query
        )


        result = process_prediction(
            raw_result["boxes"],
            raw_result["scores"],
            raw_result["labels"]
        )


        return result
