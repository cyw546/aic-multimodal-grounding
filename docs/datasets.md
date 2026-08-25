# Dataset preparation

Competition preliminary data is an unlabeled test set, not supervised training data.
Training data and generated files stay outside this repository.

## RefCOCO baseline

Sources:
- Original REFER project: https://github.com/lichengunc/refer
- COCO 2014 images: https://cocodataset.org/
- Processed JSONL mirror: https://huggingface.co/datasets/PaDT-MLLM/RefCOCO

The mirror stores one referring expression per line with a normalized `xyxy` box.
Keeping every expression avoids losing valid queries for the same object.

Server data lives under `/root/autodl-tmp/aic_grounding/data/public/`:
COCO images in `coco/train2014`, source annotations in `refcoco/source`, and
converted files in `refcoco/processed`.

Never commit images, annotations, archives, generated JSONL, or model weights.
Record source URLs, sizes and SHA-256 hashes for every downloaded copy.
