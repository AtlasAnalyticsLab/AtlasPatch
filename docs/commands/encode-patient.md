# encode-patient

- [Usage Guide](#usage-guide)
  - [CSV format](#csv-format)
  - [One case with multiple slides](#one-case-with-multiple-slides)
  - [Many cases in one run](#many-cases-in-one-run)
- [Arguments](#arguments)
- [Outputs](#outputs)

## Usage Guide

`atlaspatch encode-patient` reads a CSV file listing slides for each patient, runs or reuses the per-slide patch pipeline for every referenced slide, ensures the required upstream patch features exist, and then writes one patient embedding per case.

Patient encoders operate on the patch features stored in each slide H5. They do not consume `slide_features/<encoder>`.

### CSV format

The input is a CSV with one row per slide. All rows with the same `case_id` are grouped into one patient case.

| Column | Required | Description |
|--------|----------|-------------|
| `case_id` | yes | Case identifier. AtlasPatch uses this as the output filename stem under `patient_features/<encoder>/`. |
| `slide_path` | yes | Path to one slide belonging to that case. Relative paths are resolved relative to the CSV file directory. |
| `mpp` | no | Optional per-slide MPP override. If omitted, AtlasPatch falls back to the slide metadata. |

Additional validation:

- duplicate slide paths within the same case are rejected
- invalid `case_id` values are rejected before pipeline work starts
- duplicate slide stems that would collide in `patches/<stem>.h5` are rejected

Example CSV:

```csv
case_id,slide_path,mpp
case_001,/data/case_001_slide_a.svs,0.25
case_001,/data/case_001_slide_b.svs,0.25
case_002,/data/case_002_slide_a.svs,
```

### One case with multiple slides

In `v1.1.0`, AtlasPatch ships one built-in patient encoder: `moozy`. If a case has multiple slides, AtlasPatch builds or reuses one H5 file per slide, then aggregates all of that case's slide-level patch-feature inputs into one patient embedding per encoder.

```bash
atlaspatch encode-patient cases.csv \
  --output ./output \
  --patient-encoders moozy \
  --patch-size 224 \
  --target-mag 20 \
  --device cuda
```

### Many cases in one run

One CSV file can contain many cases. AtlasPatch groups rows by `case_id` and writes one output H5 per case under `patient_features/<encoder>/`.

```bash
atlaspatch encode-patient cases.csv \
  --output ./output \
  --patient-encoders moozy \
  --patch-size 224 \
  --target-mag 20
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `MANIFEST_PATH` | path | yes | - | Path to the CSV file that maps slides to patient cases. Each row names one slide, and rows are grouped by `case_id` during patient encoding. |
| `--output`, `-o` | path | yes | - | Output root for the per-slide H5 files, optional overlays or patch images, and final patient embedding files. |
| `--patient-encoders` | text | yes | - | One or more patient encoders, separated by spaces or commas. In `v1.1.0`, the built-in choice is `moozy`. Each encoder writes one file under `patient_features/<encoder>/`. |
| `--patch-size` | int | yes | - | Patch size, in pixels, at the requested target magnification. This must match the geometry required by the selected patient encoder set. |
| `--step-size` | int | no | same as `--patch-size` | Stride, in pixels, between adjacent patches at the target magnification when AtlasPatch needs to build or refresh per-slide H5 files. |
| `--target-mag` | int | yes | - | Target magnification used when extracting or validating the per-slide H5 files referenced by the patient cases. |
| `--feature-device` | text | no | same as `--device` | Device used for any upstream patch feature extraction required by the selected patient encoders. |
| `--feature-batch-size` | int | no | `32` | Batch size used while computing any missing upstream patch features. |
| `--feature-num-workers` | int | no | `4` | DataLoader worker count for upstream patch feature extraction. |
| `--feature-precision` | choice | no | `float16` | Computation precision for any missing upstream patch feature extraction. Supported values are `float32`, `float16`, and `bfloat16`. |
| `--feature-plugin` | path | no | - | Path to a Python module that registers custom patch feature extractors. This matters only if a selected patient encoder depends on a custom upstream patch encoder. |
| `--device` | text | no | `cuda` | Device used for tissue segmentation and patient encoder inference. AtlasPatch accepts values such as `cuda`, `cuda:0`, and `cpu`. |
| `--tissue-thresh` | float | no | `0.0` | Minimum tissue area fraction required for a patch to be kept while building or refreshing per-slide H5 files. |
| `--white-thresh` | int | no | `15` | Saturation threshold used by the optional white-filtering stage in `--no-fast-mode`. |
| `--black-thresh` | int | no | `50` | RGB threshold used by the optional black-filtering stage in `--no-fast-mode`. |
| `--seg-batch-size` | int | no | `1` | Batch size for thumbnail-level tissue segmentation. |
| `--write-batch` | int | no | `8192` | Number of coordinate rows buffered before writing to H5 while building or refreshing per-slide H5 files. |
| `--patch-workers` | int | no | CPU count | Number of worker threads used during patch extraction and optional patch PNG export. |
| `--max-open-slides` | int | no | `200` | Upper bound on how many slides AtlasPatch keeps open across segmentation and extraction. |
| `--fast-mode / --no-fast-mode` | flag | no | `--fast-mode` | `--fast-mode` skips per-patch black and white filtering after segmentation. Use `--no-fast-mode` if you want that extra filtering pass. |
| `--save-images` | flag | no | off | Save extracted patches as PNGs under `images/<stem>/` while building or refreshing per-slide H5 files. |
| `--visualize-grids` | flag | no | off | Save patch-grid overlays under `visualization/`. |
| `--visualize-mask` | flag | no | off | Save tissue-mask overlays under `visualization/`. |
| `--visualize-contours` | flag | no | off | Save contour overlays under `visualization/`. |
| `--skip-existing / --force` | flag | no | `--skip-existing` | Reuse existing per-slide H5 files and existing patient embedding files when their saved metadata still matches the current source H5 files. Use `--force` to rebuild and overwrite them. |
| `--verbose`, `-v` | flag | no | off | Enable debug logging. |

## Outputs

`atlaspatch encode-patient` writes or reuses per-slide H5 files under:

- `<output>/patches/<stem>.h5`

Patient embeddings are written as separate files under:

- `<output>/patient_features/<encoder>/<case_id>.h5`

Important constraints:

- Patient encoders consume patch features from the per-slide H5 files, not slide embeddings.
- AtlasPatch resolves required upstream patch encoders automatically. You do not pass `--feature-extractors` directly.
- The built-in MOOZY path uses the upstream public Python API.
- MOOZY's public API cannot force CPU when CUDA is visible. On a GPU-visible host, use `--device cuda` or run in a CPU-only environment if you need CPU inference.

More detail: [../../README.md#available-patient-encoders](../../README.md#available-patient-encoders)
