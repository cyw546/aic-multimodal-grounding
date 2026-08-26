# RefCOCO local validation set

The local validation set is a fixed RefCOCO subset with ground-truth boxes. It
is used for repeatable IoU and ACC@0.5 experiments and is never used for model
training.

## Server locations

- Converted validation JSONL:
  `/root/autodl-tmp/aic_grounding/data/public/refcoco/processed/val.jsonl`
- COCO 2014 images:
  `/root/autodl-tmp/aic_grounding/data/public/coco/train2014`
- Generated subsets and visualizations belong under `output_refcoco/`, which is
  ignored by Git and must not be committed.

## Build and validate 400 samples

Run from the repository root:

```bash
python scripts/build_local_validation.py \
  --input /root/autodl-tmp/aic_grounding/data/public/refcoco/processed/val.jsonl \
  --image-root /root/autodl-tmp/aic_grounding/data/public/coco/train2014 \
  --output output_refcoco/local_val/local_val_400.jsonl \
  --visualization-dir output_refcoco/local_val/visualizations \
  --sample-count 400 \
  --visualization-count 20 \
  --seed 42
```

The command validates the complete source JSONL, verifies every image selected
for the subset, writes JSONL atomically, opens the generated subset again with
the production dataset reader, and creates 20 images containing the annotated
box and referring expression.

The random seed and source file must remain unchanged when comparing model
configurations. A different seed creates a different benchmark and makes scores
incomparable.

## Output record

```json
{
  "sample_id": "refcoco_val_00000001",
  "source": "refcoco",
  "split": "val",
  "query_id": "refcoco_val_00000001",
  "query": "person furthest to the right",
  "visible_path": "COCO_train2014_000000376848.jpg",
  "infrared_path": null,
  "depth_path": null,
  "bbox": [0.57539063, 0.05517312, 1.0, 0.98985743],
  "bbox_format": "xyxy_norm",
  "width": null,
  "height": null,
  "image_id": 376848
}
```

Never commit the generated JSONL, COCO images, visualizations, archives or model
weights. Commit only scripts, tests, configurations and documentation.
