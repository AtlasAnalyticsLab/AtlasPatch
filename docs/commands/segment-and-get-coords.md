# segment-and-get-coords

- [Usage Guide](#usage-guide)
  - [One slide](#one-slide)
  - [Directory of slides](#directory-of-slides)
  - [Coordinates plus overlays](#coordinates-plus-overlays)
- [Arguments](#arguments)
- [Outputs](#outputs)

## Usage Guide

`atlaspatch segment-and-get-coords` runs tissue segmentation and patch coordinate extraction, then writes the per-slide H5 without patch feature matrices.

Use this command when you want AtlasPatch's coordinates and metadata now, but you want to defer feature extraction to a later `process`, `encode-slide`, or `encode-patient` run.

### One slide

```bash
atlaspatch segment-and-get-coords /path/to/slide.svs \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --device cuda
```

### Directory of slides

Point `WSI_PATH` at a directory to patchify many slides in one run. Add `--recursive` if slides are nested in subdirectories.

```bash
atlaspatch segment-and-get-coords /path/to/slides \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --recursive
```

### Coordinates plus overlays

Add visualization flags when you want to inspect the extracted grid and segmentation outputs alongside the H5 coordinates.

```bash
atlaspatch segment-and-get-coords /path/to/slide.svs \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --visualize-grids \
  --visualize-mask \
  --visualize-contours
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `WSI_PATH` | path | yes | - | Path to one slide file or a directory of slides. When a directory is provided, AtlasPatch scans for supported WSI extensions and uses `--recursive` to control whether subdirectories are included. |
| `--output`, `-o` | path | yes | - | Output root for the H5 files and any optional overlays or patch images generated during patch extraction. |
| `--patch-size` | int | yes | - | Patch size, in pixels, at the requested target magnification. This controls the extracted grid written into the H5 file. |
| `--step-size` | int | no | same as `--patch-size` | Stride, in pixels, between adjacent patches at the target magnification. Use a smaller value than `--patch-size` if you want overlapping coordinates. |
| `--target-mag` | int | yes | - | Target magnification used when reading patches from the slide pyramid. AtlasPatch records this in the H5 metadata. |
| `--device` | text | no | `cuda` | Device used for tissue segmentation. AtlasPatch accepts values such as `cuda`, `cuda:0`, and `cpu`. |
| `--tissue-thresh` | float | no | `0.0` | Minimum tissue area fraction required for a patch to be kept after segmentation. |
| `--white-thresh` | int | no | `15` | Saturation threshold used by the optional white-filtering stage in `--no-fast-mode`. |
| `--black-thresh` | int | no | `50` | RGB threshold used by the optional black-filtering stage in `--no-fast-mode`. |
| `--seg-batch-size` | int | no | `1` | Batch size for thumbnail-level tissue segmentation. |
| `--write-batch` | int | no | `8192` | Number of coordinate rows buffered before writing to H5. Larger values reduce write frequency but increase transient memory use. |
| `--patch-workers` | int | no | CPU count | Number of worker threads used during patch extraction and optional patch PNG export. |
| `--max-open-slides` | int | no | `200` | Upper bound on how many slides AtlasPatch keeps open across segmentation and extraction. |
| `--fast-mode / --no-fast-mode` | flag | no | `--fast-mode` | `--fast-mode` skips per-patch black and white filtering after segmentation. Use `--no-fast-mode` if you want that extra filtering pass. |
| `--save-images` | flag | no | off | Save extracted patches as PNGs under `images/<stem>/`. This is optional and is not required for later H5-based feature extraction. |
| `--visualize-grids` | flag | no | off | Save patch-grid overlays under `visualization/`. |
| `--visualize-mask` | flag | no | off | Save tissue-mask overlays under `visualization/`. |
| `--visualize-contours` | flag | no | off | Save contour overlays under `visualization/`. |
| `--skip-existing / --force` | flag | no | `--skip-existing` | Reuse existing H5 outputs by default. Use `--force` to rebuild them even when the output files already exist. |
| `--recursive` | flag | no | off | Recurse into subdirectories when `WSI_PATH` is a directory. Ignored when `WSI_PATH` is a single slide file. |
| `--mpp-csv` | path | no | - | CSV file with columns `wsi,mpp` that overrides the slide microns-per-pixel metadata for selected slides. Slides are matched by stem. |
| `--verbose`, `-v` | flag | no | off | Enable debug logging. |

## Outputs

`atlaspatch segment-and-get-coords` writes one H5 file per slide:

- `<output>/patches/<stem>.h5`

That H5 contains slide metadata and:

- `coords`

It does not contain `features/<patch_encoder>` datasets unless you later run `process`, `encode-slide`, or `encode-patient`.

Optional outputs:

- patch PNGs under `<output>/images/<stem>/`
- overlays under `<output>/visualization/`
