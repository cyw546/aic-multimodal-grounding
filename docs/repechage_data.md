# Official Repechage Data

## Server locations

```text
Data root: /srv/nfs/data/aic-repechage/official/repechage
Queries:   /srv/nfs/data/aic-repechage/official/repechage/queries/queries.json
Visible:   /srv/nfs/data/aic-repechage/official/repechage/Images/visible
Infrared:  /srv/nfs/data/aic-repechage/official/repechage/Images/infrared
Depth:     /srv/nfs/data/aic-repechage/official/repechage/Images/depth
```

The persistent Conda environment is:

```text
/srv/nfs/home/njnu_lhf/aic-repechage/env
```

The `/dev/shm/njnu_lhf_aic_repechage` emergency environment is temporary and is
lost after a server restart.

## Data structure

`queries.json` is an object keyed by Query ID. Every value contains four fields:

```json
{
  "visible": "Images/visible/000001.png",
  "infrared": "Images/infrared/000001.png",
  "depth": "Images/depth/000001.png",
  "query": "English referring expression"
}
```

Image paths are relative to the official data root. Multiple Query IDs can
refer to one image group. The repeat data has no ground-truth `bbox` field, so
the dataset reader must use `require_bbox=false`.

The shared reader supports sample, preliminary, and repechage datasets. Set the
root explicitly when switching official datasets:

```bash
export AIC_OFFICIAL_ROOT=/srv/nfs/data/aic-repechage/official/repechage
```

## Observed characteristics

The values below are intentionally committed as aggregate statistics so that
model and data issues can be diagnosed without committing competition data.

| Item | Observed value |
| --- | ---: |
| Query count | 5690 |
| Queries per referenced image group | min 1, max 14, mean 4.39, median 3 |
| Image files in `Images/visible` | 2005 |
| Image files in `Images/infrared` | 2000 |
| Image files in `Images/depth` | 2000 |
| Unique image groups referenced by Queries | 1295 |
| Unreferenced visible / infrared / depth files | 710 / 705 / 705 |
| Assignment-stated image group count | 2000 |

The assignment states that there are 2000 image groups, but the current
`queries.json` only references 1295 unique image groups. The remaining files are
present in the image directories but are not referenced by any Query. The
quality checker reports this discrepancy as a failure instead of silently
changing the expected count.

The full decode found two resolution families:

| Modality | `1080x1920` | `360x640` | dtype | channels |
| --- | ---: | ---: | --- | --- |
| Visible | 1198 | 97 | `uint8` | 3 |
| Infrared | 1198 | 97 | `uint8` | 3 |
| Depth | 1198 | 97 | `uint16` / `uint8` | 1 / 3 |

Only about 6.02 percent of the three-channel infrared images have exactly equal
channels. The 97 low-resolution depth images are 3-channel `uint8`, which does
not match the expected single-channel `uint16` depth format.

Depth statistics over referenced image groups:

- Zero pixel ratio: `0.296187`
- Valid 300-19999 mm pixel ratio: `0.698158`
- Valid value range: `300-19999` mm
- Mean valid depth: `7719.286` mm

Invalid depth values use `0`; visualizations mask them before percentile
normalization.

## Full data quality check

Run from the repository root:

```bash
AIC_OFFICIAL_ROOT=/srv/nfs/data/aic-repechage/official/repechage \
python scripts/check_official_dataset.py
```

The checker:

- Validates every Query reference without printing Query text.
- Deduplicates image groups before decoding images.
- Reports file counts, missing paths, resolutions, dtype, channels, ranges,
  depth validity, and Queries-per-image-group distribution.
- Reports image files that are not referenced by `queries.json`.
- Writes machine-readable JSON and a human-readable Markdown report.

Reports are written under `outputs/`, which is ignored by Git.

## Visualization

Generate 30 deterministic random samples:

```bash
AIC_OFFICIAL_ROOT=/srv/nfs/data/aic-repechage/official/repechage \
python scripts/visualize_repechage_samples.py
```

Generate one or more specified Query IDs:

```bash
AIC_OFFICIAL_ROOT=/srv/nfs/data/aic-repechage/official/repechage \
python scripts/visualize_repechage_samples.py \
  --output-dir outputs/inspection/repechage_samples_specific \
  --query-id 000001_001
```

Each output image contains visible RGB, infrared, depth with a valid-value
mask, and the Query text. `manifest.json` records the Query IDs, image paths,
and difficulty-category tags without duplicating the Query text.

## Difficulty categories

The visualization manifest tags samples using these categories:

- `night_or_low_light`: night, dark, low-light, shadow, dim, evening, dusk
- `occlusion`: occluded, hidden, behind, partially covered, blocked
- `small_or_distant_target`: small, tiny, distant, far, faraway, background
- `similar_targets`: similar, another, same, identical, left/right one
- `complex_spatial_description`: left, right, above, below, between, near,
  next to, front, behind, corner, center, middle, top, bottom

## Recorded hard cases

The first 30-sample inspection and targeted Query lookup recorded at least one
representative Query ID for each required difficulty category:

| Category | Query ID | Generated visualization |
| --- | --- | --- |
| Night or low light | `001289_008` | `outputs/inspection/repechage_samples/008_001289_008.jpg` |
| Occlusion | `014617_006` | `outputs/inspection/repechage_samples/025_014617_006.jpg` |
| Small or distant target | `000449_002` | `outputs/inspection/repechage_samples/003_000449_002.jpg` |
| Similar targets | `000085_001` | `outputs/inspection/repechage_similar/001_000085_001.jpg` |
| Complex spatial description | `016283_003` | `outputs/inspection/repechage_samples/001_016283_003.jpg` |

These IDs identify samples for manual review. The generated images remain
outside Git under `outputs/`.

Tags are query-language signals, not ground-truth scene annotations. Manual
inspection is still required before treating a sample as a confirmed hard case.

## Tests

```bash
AIC_OFFICIAL_ROOT=/srv/nfs/data/aic-repechage/official/repechage \
python -m pytest -q
python scripts/check_repo_safety.py
```

The integration dataset test runs only when `AIC_OFFICIAL_ROOT` is configured.
Unit tests use generated images under temporary directories.

Official images, Query text, generated visualizations, reports, caches, model
weights, and ZIP files must not be committed.