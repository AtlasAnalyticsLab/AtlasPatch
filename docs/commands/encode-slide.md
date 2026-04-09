# encode-slide

- [Usage Guide](#usage-guide)
  - [One slide](#one-slide)
  - [Directory of slides](#directory-of-slides)
  - [Multiple slide encoders in one run](#multiple-slide-encoders-in-one-run)
- [Arguments](#arguments)
- [Outputs](#outputs)

## Usage Guide

`atlaspatch encode-slide` runs the full WSI pipeline for each requested slide, reuses existing per-slide H5 files when possible, ensures the required upstream patch features exist, and then appends slide embeddings into the same H5.

This command is WSI-only. You pass slide paths, not precomputed H5 files.

### One slide

Use this when you want a slide embedding for one WSI and AtlasPatch should manage the upstream patch pipeline for you.

```bash
atlaspatch encode-slide /path/to/slide.svs \
  --output ./output \
  --slide-encoders titan \
  --patch-size 512 \
  --target-mag 20 \
  --device cuda
```

### Directory of slides

Point `WSI_PATH` at a directory to encode many slides in one run. Add `--recursive` if slides are nested in subdirectories.

```bash
atlaspatch encode-slide /path/to/slides \
  --output ./output \
  --slide-encoders prism \
  --patch-size 224 \
  --target-mag 20 \
  --recursive
```

### Multiple slide encoders in one run

AtlasPatch can run multiple slide encoders in one pass as long as they depend on the same upstream patch geometry. For example, `prism` and `moozy` can share a run because both use 224-pixel patches, while `titan` must run separately because it expects 512-pixel patches.

```bash
atlaspatch encode-slide /path/to/slides \
  --output ./output \
  --slide-encoders prism,moozy \
  --patch-size 224 \
  --target-mag 20
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `WSI_PATH` | path | yes | - | Path to one slide file or a directory of slides. When a directory is provided, AtlasPatch scans for supported WSI extensions and uses `--recursive` to control whether subdirectories are included. |
| `--output`, `-o` | path | yes | - | Output root for the per-slide H5 files and any optional overlays or patch images generated while building or refreshing those H5 files. |
| `--slide-encoders` | text | yes | - | One or more slide encoders, separated by spaces or commas. Each encoder writes one dataset under `slide_features/<encoder>` inside the per-slide H5. |
| `--patch-size` | int | yes | - | Patch size, in pixels, at the requested target magnification. This must match the geometry required by the selected slide encoder set. |
| `--step-size` | int | no | same as `--patch-size` | Stride, in pixels, between adjacent patches at the target magnification. Use a smaller value than `--patch-size` if you want overlapping patches in the per-slide H5. |
| `--target-mag` | int | yes | - | Target magnification used when extracting or validating the per-slide H5. AtlasPatch records this in the H5 metadata and uses it to determine whether existing H5 files are reusable. |
| `--feature-device` | text | no | same as `--device` | Device used for any upstream patch feature extraction required by the selected slide encoders. |
| `--feature-batch-size` | int | no | `32` | Batch size used while computing any missing upstream patch features. |
| `--feature-num-workers` | int | no | `4` | DataLoader worker count for upstream patch feature extraction. |
| `--feature-precision` | choice | no | `float16` | Computation precision for any missing upstream patch feature extraction. Supported values are `float32`, `float16`, and `bfloat16`. |
| `--feature-plugin` | path | no | - | Path to a Python module that registers custom patch feature extractors. This matters only if a selected slide encoder depends on a custom upstream patch encoder. |
| `--device` | text | no | `cuda` | Device used for tissue segmentation and slide encoder inference. AtlasPatch accepts values such as `cuda`, `cuda:0`, and `cpu`. |
| `--tissue-thresh` | float | no | `0.0` | Minimum tissue area fraction required for a patch to be kept while building or refreshing the per-slide H5. |
| `--white-thresh` | int | no | `15` | Saturation threshold used by the optional white-filtering stage in `--no-fast-mode`. |
| `--black-thresh` | int | no | `50` | RGB threshold used by the optional black-filtering stage in `--no-fast-mode`. |
| `--seg-batch-size` | int | no | `1` | Batch size for thumbnail-level tissue segmentation. |
| `--write-batch` | int | no | `8192` | Number of coordinate rows buffered before writing to H5 while building or refreshing the per-slide H5. |
| `--patch-workers` | int | no | CPU count | Number of worker threads used during patch extraction and optional patch PNG export. |
| `--max-open-slides` | int | no | `200` | Upper bound on how many slides AtlasPatch keeps open across segmentation and extraction. |
| `--fast-mode / --no-fast-mode` | flag | no | `--fast-mode` | `--fast-mode` skips per-patch black and white filtering after segmentation. Use `--no-fast-mode` if you want that extra filtering pass. |
| `--save-images` | flag | no | off | Save extracted patches as PNGs under `images/<stem>/` while building or refreshing the per-slide H5. |
| `--visualize-grids` | flag | no | off | Save patch-grid overlays under `visualization/`. |
| `--visualize-mask` | flag | no | off | Save tissue-mask overlays under `visualization/`. |
| `--visualize-contours` | flag | no | off | Save contour overlays under `visualization/`. |
| `--skip-existing / --force` | flag | no | `--skip-existing` | Reuse existing H5 files and existing slide embeddings when their saved metadata still matches the current H5 file. Use `--force` to rebuild and overwrite them. |
| `--recursive` | flag | no | off | Recurse into subdirectories when `WSI_PATH` is a directory. Ignored when `WSI_PATH` is a single slide file. |
| `--mpp-csv` | path | no | - | CSV file with columns `wsi,mpp` that overrides the slide microns-per-pixel metadata for selected slides. Slides are matched by stem. |
| `--verbose`, `-v` | flag | no | off | Enable debug logging. |

## Outputs

`atlaspatch encode-slide` writes or reuses the per-slide H5:

- `<output>/patches/<stem>.h5`

Slide embeddings are appended inside that H5 under:

- `slide_features/<encoder>`

Optional outputs:

- patch PNGs under `<output>/images/<stem>/`
- overlays under `<output>/visualization/`

Important constraints:

- `encode-slide` resolves required upstream patch encoders automatically. You do not pass `--feature-extractors` directly.
- Existing slide embeddings are reused only if their saved metadata still matches the current H5 file.
- Slide encoders depend on patch features in the per-slide H5, not on raw patch pixels after extraction time.

More detail: [../../README.md#available-slide-encoders](../../README.md#available-slide-encoders)
