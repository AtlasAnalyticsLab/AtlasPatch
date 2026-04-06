# process

- [Usage Guide](#usage-guide)
  - [One slide](#one-slide)
  - [Directory of slides](#directory-of-slides)
  - [Separate segmentation and feature devices](#separate-segmentation-and-feature-devices)
- [Arguments](#arguments)
- [Outputs](#outputs)

## Usage Guide

`atlaspatch process` runs the full per-slide patch pipeline: tissue segmentation, patch coordinate extraction, patch feature extraction, and any optional image or overlay exports you enable.

### One slide

Use this when you want one H5 file for one slide.

```bash
atlaspatch process /path/to/slide.svs \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --feature-extractors uni_v2 \
  --device cuda
```

### Directory of slides

Point `WSI_PATH` at a directory to process many slides in one run. Add `--recursive` if slides are nested in subdirectories.

```bash
atlaspatch process /path/to/slides \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --feature-extractors resnet50,uni_v2 \
  --recursive
```

### Separate segmentation and feature devices

Segmentation and feature extraction can run on different devices. This is useful when one GPU is reserved for SAM2 and another for patch encoders.

```bash
atlaspatch process /path/to/slides \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --feature-extractors virchow_v1 \
  --device cuda:0 \
  --feature-device cuda:1
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `WSI_PATH` | path | yes | - | Path to one slide file or a directory of slides. When a directory is provided, AtlasPatch scans that directory for supported WSI extensions and uses `--recursive` to control whether subdirectories are included. |
| `--output`, `-o` | path | yes | - | Output root for all generated outputs. H5 files are written under `patches/`, optional patch PNGs under `images/`, and optional overlays under `visualization/`. |
| `--feature-extractors` | text | yes | - | One or more patch feature extractors, separated by spaces or commas. Each extractor writes one dataset under `features/<encoder>` inside the per-slide H5. |
| `--patch-size` | int | yes | - | Patch size, in pixels, at the requested target magnification. This controls both the patch grid and the expected geometry for any downstream encoders that reuse the generated H5. |
| `--step-size` | int | no | same as `--patch-size` | Stride, in pixels, between adjacent patch origins at the target magnification. Use a smaller value than `--patch-size` if you want overlapping patches. |
| `--target-mag` | int | yes | - | Target magnification used when reading patches from the slide pyramid. AtlasPatch records this in the H5 metadata and uses it later to validate reuse of the artifact. |
| `--feature-device` | text | no | same as `--device` | Device used for patch feature extraction. This can differ from `--device` when segmentation and embedding should run on different devices. |
| `--feature-batch-size` | int | no | `32` | Batch size used when embedding extracted patches. Larger values improve throughput but increase memory use. |
| `--feature-num-workers` | int | no | `4` | DataLoader worker count for patch feature extraction. Increase this if patch embedding becomes input-bound rather than model-bound. |
| `--feature-precision` | choice | no | `float16` | Computation precision for patch feature extraction. Supported values are `float32`, `float16`, and `bfloat16`. |
| `--feature-plugin` | path | no | - | Path to a Python module that registers custom patch feature extractors. Use this when extending AtlasPatch beyond the built-in registry. |
| `--device` | text | no | `cuda` | Device used for tissue segmentation. AtlasPatch accepts values such as `cuda`, `cuda:0`, and `cpu`. |
| `--tissue-thresh` | float | no | `0.0` | Minimum tissue area fraction required for a patch to be kept after segmentation. Increase this to drop patches with too little tissue. |
| `--white-thresh` | int | no | `15` | Saturation threshold used by the optional white-filtering stage in `--no-fast-mode`. Higher values mark more bright background as discardable. |
| `--black-thresh` | int | no | `50` | RGB threshold used by the optional black-filtering stage in `--no-fast-mode`. Lower values mark darker regions as discardable. |
| `--seg-batch-size` | int | no | `1` | Batch size for thumbnail-level tissue segmentation. This affects the SAM2 inference stage, not patch embedding. |
| `--write-batch` | int | no | `8192` | Number of coordinate rows buffered before writing to H5. Larger values reduce write frequency but increase transient memory use. |
| `--patch-workers` | int | no | CPU count | Number of worker threads used while extracting patch coordinates and, if enabled, saving patch PNGs. |
| `--max-open-slides` | int | no | `200` | Upper bound on how many slides AtlasPatch keeps open across segmentation and extraction. Reduce this if the host has strict file-handle limits. |
| `--fast-mode / --no-fast-mode` | flag | no | `--fast-mode` | `--fast-mode` skips per-patch black and white filtering after segmentation. Use `--no-fast-mode` if you want the additional filtering pass. |
| `--save-images` | flag | no | off | Save each extracted patch as a PNG under `images/<stem>/`. This is optional and is not required for H5-based downstream workflows. |
| `--visualize-grids` | flag | no | off | Save a patch-grid overlay for each processed slide under `visualization/`. |
| `--visualize-mask` | flag | no | off | Save the predicted tissue mask overlay for each processed slide under `visualization/`. |
| `--visualize-contours` | flag | no | off | Save the contour overlay used during patch extraction under `visualization/`. |
| `--skip-existing / --force` | flag | no | `--skip-existing` | Reuse existing H5 outputs by default. Use `--force` to rebuild them even when the output files already exist. |
| `--recursive` | flag | no | off | Recurse into subdirectories when `WSI_PATH` is a directory. Ignored when `WSI_PATH` is a single slide file. |
| `--mpp-csv` | path | no | - | CSV file with columns `wsi,mpp` that overrides the slide microns-per-pixel metadata for selected slides. Slides are matched by stem. |
| `--verbose`, `-v` | flag | no | off | Enable debug logging. |

## Outputs

`atlaspatch process` writes one H5 file per slide:

- `<output>/patches/<stem>.h5`

That H5 contains:

- `coords`
- `features/<patch_encoder>`

Optional outputs:

- patch PNGs under `<output>/images/<stem>/`
- overlays under `<output>/visualization/`

More detail:

- [../../README.md#using-extracted-data](../../README.md#using-extracted-data)
- [../../README.md#available-patch-feature-extractors](../../README.md#available-patch-feature-extractors)
