# Model evaluation and candidate diagnosis

This workflow evaluates RGB baselines, API/VLM predictions and candidate-reranking
experiments on exactly the same public validation samples. It never uses the
official test set for training, tuning or manual box correction.

## Frozen validation manifest

The frozen list is committed as
`configs/validation/local_val_400_ids.txt`. It contains 400 IDs selected from
10,834 public PaDT-MLLM RefCOCO validation records with `seed=42`, using the
same selection algorithm as `scripts/build_local_validation.py`.

`configs/validation/local_val_400_manifest.json` records the source hash,
converted-file hash, selection parameters and ID-list hash. The generated JSONL
and COCO images remain under `output_refcoco/` and are not committed.

Every evaluation run must pass `--ids` so the sample order and denominator
cannot change. Do not regenerate the manifest when comparing models.

## Unified prediction interface

Prediction files can be JSONL, a JSON array, or a JSON object keyed by Query ID.
Each prediction record contains at least:

```json
{"query_id": "refcoco_val_00000001", "bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.73, "model_name": "grounding_dino_swinb"}
```

Allowed optional fields are `model_name`, `model_version` and `config`. Bounding
boxes must be normalized `[x1, y1, x2, y2]`. Missing predictions and invalid
boxes are recorded explicitly rather than silently dropped.

Candidate files use the same Query ID key and retain ranking order:

```json
{"query_id": "refcoco_val_00000001", "candidates": [{"bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.73, "label": "person"}, {"bbox": [0.2, 0.1, 0.9, 0.8], "score": 0.61, "label": "person"}]}
```

`coverage@K` is true when at least one of the first K candidates has IoU >= 0.5
with the ground-truth box. The original candidate order is treated as the model
ranking; conversion scripts must sort candidates before writing this file.

## Evaluation command

```bash
python scripts/evaluate_predictions.py \
  --validation output_refcoco/local_val/local_val_400.jsonl \
  --ids configs/validation/local_val_400_ids.txt \
  --image-root /root/autodl-tmp/aic_grounding/data/public/coco/train2014 \
  --prediction dino=/path/to/dino_prediction.jsonl \
  --candidate dino=/path/to/dino_candidates.jsonl \
  --prediction qwen_api=/path/to/qwen_prediction.jsonl \
  --prediction dino_qwen_rerank=/path/to/rerank_prediction.jsonl \
  --model-version dino=groundingdino_swinb_cogcoor \
  --config dino=box035_text025 \
  --output-dir outputs/evaluation/model_evaluator \
  --failure-limit 20
```

If a candidate file is provided without a matching prediction file, the top-1
candidate is evaluated as the model prediction. This is useful for measuring a
candidate generator directly, but the comparison table identifies it as
`candidate_top1`.

## Reported metrics

- `ACC@0.5`: correct samples divided by all fixed validation samples. Missing
  and invalid predictions count as incorrect.
- `mean_iou`: all fixed samples count; missing and invalid predictions contribute
  IoU zero.
- `mean_iou_valid_only`: mean IoU for valid boxes, useful for box-regression
  debugging but not a replacement for `mean_iou`.
- `valid_bbox_rate`, `invalid_bbox_count`, `missing_prediction_count`: interface
  and coordinate quality.
- `coverage@1`, `coverage@5`, `coverage@10`: candidate recall at IoU >= 0.5.
- Per-sample records: Query ID, ground truth, prediction, IoU, hit status,
  candidate rank/IoU, coverage flags and failure categories.

## Failure diagnosis

Failure categories are not mutually exclusive in the per-sample record. The
primary category follows this order:

1. `target_not_in_candidates`: no ranked candidate reaches IoU >= 0.5.
2. `candidate_ranking_error`: a correct candidate exists, but ranks below top-1.
3. `candidate_available_but_final_missing_or_invalid`: a correct top candidate
   exists while the final prediction is absent or illegal.
4. `candidate_selection_or_regression`: candidates are available but final box
   quality remains insufficient.
5. `missing_or_invalid_prediction`: the prediction interface itself failed.
6. `small_target`, `occlusion`, `similar_targets`: validation metadata explains
   a difficult case.
7. `oversized_box`, `undersized_box`, `localization_shift`, `partial_overlap`:
   geometric error when no earlier category explains the failure.

The report separately aggregates primary categories, all applicable categories
and difficulty labels.

## Decision rule

- If `coverage@10 < 0.80`, improve candidate generation first (Grounding DINO
  prompts, thresholds, query decomposition or multimodal fusion).
- If `coverage@10 - coverage@1 >= 0.10` or `coverage@10 - ACC@0.5 >= 0.15`,
  improve candidate selection/ranking (for example Qwen3-VL reranking or score
  calibration).
- If candidates cover the target but top-1 and final ACC plateau near
  `coverage@1`, prioritize box refinement/regression.
- A model comparison without candidate coverage cannot distinguish a detector
  miss from a bad selector; do not choose the next model from ACC alone.

## Outputs

```text
outputs/evaluation/model_evaluator/
  fixed_query_ids.txt
  comparison.csv
  comparison.json
  comparison.md
  recommendations.json
  run_manifest.json
  models/<model_name>/
    summary.json
    per_sample.csv
    per_sample.jsonl
    failure_samples.csv
    failure_summary.json
    failure_visualizations/*.jpg
```

Failure images show the RGB image, green ground-truth box, red final prediction
and the first five candidates. Cyan candidates reach IoU >= 0.5; orange
candidates do not. At least 20 failures are emitted when that many exist.

Input paths, SHA-256 hashes, the fixed manifest, threshold and experiment labels
are written to `run_manifest.json` so that every reported number is traceable.
Generated data, predictions, images and weights remain outside Git.
