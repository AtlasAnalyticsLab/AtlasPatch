# SlideProcessor

A Python package for processing and handling whole slide images (WSI).

## Table of Contents
- [Installation](#installation)
- [Development Setup](#development-setup)
- [CLI Usage](#cli-usage)
  - [Quick Start](#quick-start)
  - [Commands](#commands)
  - [Usage Examples](#usage-examples)
    - [Basic Single File Processing](#basic-single-file-processing)
    - [Batch Processing Multiple Files](#batch-processing-multiple-files)
    - [Custom Patch Extraction Parameters](#custom-patch-extraction-parameters)
    - [Export Individual Patch Images](#export-individual-patch-images)
    - [CPU Inference](#cpu-inference)
    - [Custom MPP Values via CSV](#custom-mpp-values-via-csv)
    - [Custom Filtering Thresholds](#custom-filtering-thresholds)
    - [Verbose Output](#verbose-output)
    - [Generate Visualizations](#generate-visualizations)
    - [`slideproc info`](#slideproc-info)
  - [Parameter Guide](#parameter-guide)
- [HDF5 Output Structure](#hdf5-output-structure)
  - [Output](#output)
- [Using TRIDENT for feature extraction](#using-trident-for-feature-extraction)
  - [Setup TRIDENT](#setup-trident)
  - [Extract patch features from SlideProcessor outputs](#extract-patch-features-from-slideprocessor-outputs)
- [SLURM job scripts](#slurm-job-scripts)
- [Feedback](#feedback)
- [License](#license)

## Installation

### Using Conda (Recommended)

1. Create a conda environment:
```bash
conda create -n slide_processor python=3.10
conda activate slide_processor
```

2. Install the OpenSlide system library (required for WSI processing):
```bash
conda install -c conda-forge openslide
```

3. Install SAM2 (Segment Anything Model 2.0):
```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Using venv

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the OpenSlide system library:
   - **Ubuntu/Debian**: `sudo apt-get install openslide-tools`
   - **macOS**: `brew install openslide`
   - **Other systems**: Visit [OpenSlide Documentation](https://openslide.org/)

3. Install SAM2 (Segment Anything Model 2.0):
```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Development Setup

To set up the development environment with linting and pre-commit hooks:

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

## CLI Usage

SlideProcessor provides an intuitive command-line interface for processing whole slide images. The CLI supports both single file and batch processing with flexible configuration options.

### Quick Start

```bash
# Process a single WSI file (uses built-in Tiny SAM2 config)
slideproc process sample.svs --checkpoint model.pt \
    --patch-size 256 --target-mag 20

# Process all WSI files in a directory
slideproc process ./wsi_folder/ --checkpoint model.pt \
    --patch-size 256 --target-mag 20 --output ./results

# With custom patch settings and visualization
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 512 --step-size 256 --target-mag 20 \
    --output ./output --save-images --visualize-grids
```

### Commands

#### `slideproc process`

Main command for processing whole slide images with tissue segmentation and patch extraction.

**Required Arguments:**
- `WSI_PATH` **(required)**: Path to a single WSI file or directory containing multiple WSI files

**Required Options:**
- `--checkpoint/-c` **(required)**: Path to SAM2 model checkpoint file (.pt)
- `--patch-size` **(required)**: Target size of extracted patches in pixels (final patch dimensions)
- `--target-mag` **(required)**: Target magnification for extraction (e.g., 10, 20, 40)

**Optional Parameters:**

| Option | Type | Default | Required? | Description |
|--------|------|---------|-----------|-------------|
| `--step-size` | int | patch-size | No | Stride between patches at target magnification; defaults to patch size |
| `--output/-o` | Path | `./output` | No | Output directory root for results (contains `patches/`, `visualization/`, and `images/`) |
| `--device` | choice | `cuda` | No | Device for inference: `cuda` or `cpu` |
| `--tissue-thresh` | float | `0.01` | No | Minimum tissue area threshold as fraction of image (0–1) |
| `--white-thresh` | int | `15` | No | Saturation threshold for filtering white patches |
| `--black-thresh` | int | `50` | No | RGB threshold for filtering black patches |
| `--seg-batch-size` | int | 1 | No | Batch size for SAM2 thumbnail segmentation |
| `--patch-workers` | int | CPU count | No | Parallel threads for per-slide patch extraction/H5 writing |
| `--max-open-slides` | int | 200 | No | Cap on simultaneous open WSIs across segmentation + extraction |
| `--write-batch` | int | 8192 | No | Rows per HDF5 flush when writing coordinates |
| `--save-images` | flag | False | No | Export individual patch images as PNG files under `images/<stem>/` |
| `--fast-mode/--no-fast-mode` | flag | True | No | `--fast-mode` skips per-patch content filtering (default); use `--no-fast-mode` to enable filtering |
| `--visualize-grids` | flag | False | No | Generate patch grid overlay on WSI thumbnail |
| `--visualize-mask` | flag | False | No | Generate predicted tissue mask overlay visualization on thumbnail |
| `--visualize-contours` | flag | False | No | Generate tissue contour overlay visualization on thumbnail |
| `--recursive` | flag | False | No | Recursively search directories for WSI files |
| `--mpp-csv` | Path | None | No | CSV with custom MPP values (`wsi,mpp`) |
| `--skip-existing/--force` | flag | Skip existing | No | Skip existing outputs by default; pass `--force` to reprocess |
| `--verbose/-v` | flag | False | No | Enable verbose logging output (disables progress bar) |

**Supported WSI Formats:**
- **OpenSlide formats**: .svs, .tif, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .dcm
- **Image formats**: .png, .jpg, .jpeg, .bmp, .webp, .gif

### Usage Examples

#### Basic Single File Processing

```bash
slideproc process sample.svs --checkpoint model.pt \
    --patch-size 256 --target-mag 20
```

#### Batch Processing Multiple Files

```bash
# Process all .svs files in a directory
slideproc process ./slides/ \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --output ./processed_slides
# Batch thumbnails for segmentation
slideproc process ./slides/ \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --seg-batch-size 8 \
    --patch-workers 4 \
    --max-open-slides 12 \
    --output ./processed_slides
```

#### Custom Patch Extraction Parameters

```bash
# Extract larger patches with different stride and magnification
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 512 \
    --step-size 256 \
    --target-mag 20 \
    --output ./results
```

#### Export Individual Patch Images

```bash
# Generate individual PNG files for each patch
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --save-images \
    --output ./output
# Creates: output/images/<stem>/<stem>_x<x>_y<y>.png
```


#### CPU Inference

```bash
# Use CPU instead of GPU (slower but no GPU required)
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --device cpu
```

#### Custom MPP Values via CSV

When WSI files don't have MPP (microns per pixel) metadata or the metadata is incorrect, you can provide custom MPP values via a CSV file.

**CSV Format (required columns: `wsi` and `mpp`):**

```csv
wsi,mpp
slide1.svs,0.5
slide2.svs,0.25
sample.png,0.4
```

**Usage:**

```bash
slideproc process ./wsi_folder/ \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --mpp-csv mpp_values.csv
```

**CSV Specifications:**
- **Required columns**: `wsi` (filename with or without path) and `mpp` (float value)
- **WSI names**: Can use just the filename (e.g., `slide1.svs`) or full path; only the stem (filename without extension) is matched

#### Custom Filtering Thresholds

```bash
# Adjust thresholds for different tissue characteristics
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --white-thresh 20 \
    --black-thresh 40 \
    --tissue-thresh 0.05
```

#### Verbose Output

```bash
# Enable detailed logging for debugging
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --verbose
```

- Default (no --verbose): quiet mode with a progress bar and minimal output.

#### Generate Visualizations

```bash
# Generate visualizations on thumbnail
slideproc process sample.svs \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --visualize-grids --visualize-mask --visualize-contours

```

The visualization flags create the following images under `output/<mag>x_<patch>px_<overlap>px_overlap/visualization/`:
- `<wsi_stem>.png`: patch grid overlay (`--visualize-grids`)
- `<wsi_stem>_mask.png`: mask overlay (`--visualize-mask`)
- `<wsi_stem>_contours.png`: contour overlay (`--visualize-contours`)

#### `slideproc info`

Display information about supported formats and features.

```bash
slideproc info
```

### Parameter Guide

**Patch Extraction Parameters:**

- **`--patch-size`**: Target size of extracted patches (e.g., 256 = 256x256 pixels) at the chosen magnification (`--target-mag`). Coordinates in the H5 are always at level 0.
  - Larger values reduce number of patches but capture more context
  - Common values: 256, 512

- **`--step-size`**: Sliding window stride during patch extraction, defined in target-magnification pixels. Internally converted to a level-0 stride.
  - If equal to patch-size: non-overlapping patches
  - If smaller: overlapping patches (more patches)
  - Default behavior: uses patch-size (non-overlapping)

- **`--target-mag`**: Target magnification for extraction (40, 20, 10, etc.). Must be less than or equal to the WSI's native magnification. If a higher magnification is requested than available, the CLI exits with an error.

**Filtering Parameters:**

- **`--white-thresh`**: Saturation threshold for white patches
  - Lower values = more aggressively filter white regions (HSV saturation)
  - Filtering uses majority rule: a patch is considered white if ≥70% of pixels have
    saturation below this threshold AND brightness/value ≥ 200.

- **`--black-thresh`**: RGB threshold for black patches
  - Lower values = filter darker regions
  - Filtering uses majority rule: a patch is considered black if ≥70% of grayscale
    pixels are below this threshold.

- **`--tissue-thresh`**: Minimum tissue area as fraction of input image which is of size 1024x1024
  - Filters out very small tissue regions
  - Range: 0.0–1.0
  - Unit: fraction (0–1)

## HDF5 Output Structure

Each processed slide produces a single HDF5 file under `<output>/<mag>x_<patch>px_<overlap>px_overlap/patches/<stem>_patches.h5`. Each file adopts a structure similar to [TRIDENT](https://github.com/mahmoodlab/TRIDENT) to maintain compatability

- Datasets
  - `coords`: int32 shape `(N, 2)` containing `(x, y)` at level 0
  - `coords_ext`: int32 shape `(N, 5)` containing `(x, y, w, h, level)` for reliable re-reading
- File attributes
  - `patch_size`: int (target patch size)
  - `wsi_path`: original WSI path
  - `num_patches`: total number of patches
  - `level0_magnification`: magnification of the highest-resolution level (if known)
  - `target_magnification`: magnification used for extraction
  - `patch_size_level0`: size of the patch footprint at level 0 in pixels. Used by some slide encoders which uses it for positional encoding module (e.g., [ALiBi](https://arxiv.org/pdf/2108.12409) in [TITAN](https://arxiv.org/abs/2411.19666))


### Output

Results are written under a run-specific subdirectory named `<mag>x_<patch>px_<overlap>px_overlap` (where `overlap = patch_size - step_size`). Inside this directory:

- `patches/` contains the HDF5 outputs (`<stem>_patches.h5`).
- `images/` contains optional per-patch PNGs when `--save-images` is set.
- `visualization/` contains optional overlays for grids, masks, and contours.

**HDF5 Files** (per input WSI):
```
<output>/<mag>x_<patch>px_<overlap>px_overlap/patches/<wsi_stem>_patches.h5
```

Contains:
- `coords`: Shape (N, 2), dtype int32 - (x, y) coordinates at level 0
- `coords_ext`: Shape (N, 5), dtype int32 - (x, y, w, h, level)
- File attributes: `patch_size`, `wsi_path`, `num_patches`, `level0_magnification`, `target_magnification`, `patch_size_level0`

**Optional PNG Images** (if `--save-images` is used):
```
<output>/<mag>x_<patch>px_<overlap>px_overlap/images/<wsi_stem>/<wsi_stem>_x<x>_y<y>.png
```

Each file represents a single extracted patch with its coordinates in the filename.

## Using TRIDENT for feature extraction

SlideProcessor writes HDF5 patch coordinate files in the same structure that [TRIDENT](https://github.com/mahmoodlab/TRIDENT) consumes (`coords`/`coords_ext` plus patch metadata), so you can run TRIDENT feature extraction directly on our outputs without conversion.

### Setup TRIDENT

```bash
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT
conda create -n trident python=3.10  # or use your preferred env
conda activate trident
pip install -e .
```

### Extract patch features from SlideProcessor outputs

Point `--job_dir` to the `--output` you used with `slideproc process` (TRIDENT will pick up the run folder such as `20x_256px_0px_overlap/patches/*.h5`):

```bash
python run_batch_of_slides.py \
    --task feat \
    --wsi_dir <source directory for whole slide image> \
    --job_dir <output directory> \
    --batch_size 64 \
    --patch_encoder uni_v1 \
    --mag 20 \
    --patch_size 256
```

- `--patch_encoder` supports the full TRIDENT patch model list (examples: `uni_v1`, `uni_v2`, `conch_v15`, `virchow`, `phikon`, `gigapath`, `hoptimus0/1`, `musk`, `midnight12k`, `kaiko-*`, `lunit-*`, `dino_vit_small_p8/p16`, `hibou_l`, `ctranspath`, `resnet50`, etc.). Check [TRIDENT’s README](https://github.com/mahmoodlab/TRIDENT) for the complete table and any extra dependencies for specific encoders.
- For slide-level embeddings instead of patch-only, use `--slide_encoder` (e.g., `titan`, `prism`, `gigapath`, `chief`, `madeleine`, `feather`) and TRIDENT will perform patch encoding + slide pooling automatically.

## SLURM job scripts

We prepared ready-to-run SLURM templates are under `jobs/`:

- Patch extraction (SAM2 + H5/PNG): `jobs/slideproc_patch.slurm.sh`. Edits to make:
  - Set `WSI_ROOT`, `OUTPUT_ROOT`, `SAM_CHECKPOINT`, `PATCH_SIZE`, `TARGET_MAG`, `SEG_BATCH`.
  - Ensure `--cpus-per-task` matches the CPU you want; the script passes `--patch-workers ${SLURM_CPUS_PER_TASK}` and caps `--max-open-slides` at 200.
  - `--fast-mode` is on by default; append `--no-fast-mode` to enable content filtering.
  - Submit with `sbatch jobs/slideproc_patch.slurm.sh`.

- TRIDENT feature extraction: `jobs/trident_features.slurm.sh`.
  - Set `WSI_ROOT`, `PATCH_OUTPUT` (same as SlideProcessor `--output`), `PATCH_ENCODER`, `MAG`, `PATCH_SIZE`, `BATCH_SIZE`.
  - Activate your TRIDENT env and set `PYTHONPATH` to the TRIDENT repo if needed.
  - Submit with `sbatch jobs/trident_features.slurm.sh`.

## Feedback

- Report problems via the [bug report template](https://github.com/AtlasAnalyticsLab/SlideProcessor/issues/new?template=bug_report.md) so we can reproduce and fix them quickly.
- Suggest enhancements through the [feature request template](https://github.com/AtlasAnalyticsLab/SlideProcessor/issues/new?template=feature_request.md) with your use case and proposal.
- When opening a PR, fill out the [pull request template](.github/pull_request_template.md) and run the listed checks (lint, format, type-check, tests).

## License

SlideProcessor is licensed under the **PolyForm Noncommercial License 1.0.0**, which strictly prohibits commercial use of this software or any derivative works. This applies to all forms of commercialization, including selling the software, offering it as a commercial service, using it in commercial products, or creating forked versions for commercial purposes. However, the license explicitly permits use for research, experimentation, and non-commercial purposes. Personal use for research, hobby projects, and educational purposes is allowed, as is use by academic institutions, educational organizations, public research organizations, and non-profit entities regardless of their funding sources. If you wish to use SlideProcessor commercially, you must obtain a separate commercial license from the authors. For the complete license text and detailed terms, see the [LICENSE](./LICENSE) file in this repository.
