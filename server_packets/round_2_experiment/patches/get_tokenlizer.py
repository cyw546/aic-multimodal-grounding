import os

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    RobertaModel,
)


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(
                    type(text_encoder_type)
                )
            )
    print("final text_encoder_type: {}".format(text_encoder_type))
    return AutoTokenizer.from_pretrained(
        text_encoder_type,
        local_files_only=text_encoder_type == "bert-base-uncased",
    )


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        print(
            "Initializing BERT architecture offline; official GroundingDINO "
            "checkpoint supplies the trained BERT parameters."
        )
        return BertModel(BertConfig())
    if os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
