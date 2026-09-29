# Validation manifests

`local_val_400_ids.txt` is the committed fixed evaluation sample list. It was
generated from the public PaDT-MLLM RefCOCO validation JSONL with 10,834 source
records, `sample_count=400`, `seed=42`, using the same selection algorithm as
`scripts/build_local_validation.py`.

`local_val_400_manifest.json` records the source and converted-file SHA-256
values, source record count, first/last Query ID and the SHA-256 of the ID list.
The generated JSONL and COCO images remain outside Git.

Do not regenerate or reorder this manifest when comparing models. Use it with
`--ids configs/validation/local_val_400_ids.txt`.
