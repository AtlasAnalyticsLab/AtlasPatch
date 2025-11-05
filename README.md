# SlideProcessor

A Python package for processing and handling whole slide images (WSI).

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
- `--target-mag` **(required)**: Target magnification for extraction: one of 1, 2, 4, 5, 10, 20, 40, 60, 80

**Optional Parameters:**

| Option | Type | Default | Required? | Description |
|--------|------|---------|-----------|-------------|
| `--patch-size` | int | — | Yes | Target size of extracted patches in pixels (final patch dimensions) |
| `--target-mag` | choice | — | Yes | Target magnification for extraction: one of 5, 10, 20, 40, 60, 80 |
| `--step-size` | int | patch-size | No | Step size for patch extraction (stride) at the target magnification. Defaults to patch-size if not set |
| `--output/-o` | Path | `./output` | No | Output directory root for results (contains `patches/`, `visualization/`, and `images/`) |
| `--device` | choice | `cuda` | No | Device for inference: `cuda` or `cpu` |
| `--tissue-thresh` | float | `0.01` | No | Minimum tissue area threshold as fraction of image (0–1) |
| `--white-thresh` | int | `15` | No | Saturation threshold for filtering white patches |
| `--black-thresh` | int | `50` | No | RGB threshold for filtering black patches |
| `--save-images` | flag | False | No | Export individual patch images as PNG files under `images/<stem>/` |
| `--fast-mode` | flag | False | No | Skip per-patch content filtering for faster extraction (may include background patches) |
| `--visualize-grids` | flag | False | No | Generate patch grid overlay on WSI thumbnail |
| `--visualize-mask` | flag | False | No | Generate predicted tissue mask overlay visualization on thumbnail |
| `--visualize-contours` | flag | False | No | Generate tissue contour overlay visualization on thumbnail |
| `--recursive` | flag | False | No | Recursively search directories for WSI files |
| `--verbose/-v` | flag | False | No | Enable verbose logging output |
| `--seg-batch-size` | int | 1 | No | Batch size for SAM2 thumbnail segmentation when processing a folder; set >1 to enable batched inference |
| `--workers` | int | 1 | No | CPU workers for processing multiple WSIs in parallel (per-WSI) |
| `--pipeline` | flag | False | No | Concurrent GPU segmentation with CPU patchification for multi-file processing |

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

# Concurrent GPU segmentation with CPU patchification
slideproc process ./slides/ \
    --checkpoint model.pt \
    --patch-size 256 --target-mag 20 \
    --workers 4 --seg-batch-size 8 --pipeline \
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

The visualization flags create the following images under `output/visualization/`:
- `<wsi_stem>.png`: patch grid overlay (`--visualize-grids`)
- `<wsi_stem>_mask.png`: mask overlay (`--visualize-mask`)
- `<wsi_stem>_contours.png`: contour overlay (`--visualize-contours`)

#### `slideproc info`

Display information about supported formats and features.

```bash
slideproc info
```

## HDF5 Output Structure

Each processed slide produces a single HDF5 file under `<output>/patches/<stem>.h5`. Each file adopt similar structure to [TRIDENT](https://github.com/mahmoodlab/TRIDENT) to maintain compatability

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

The CLI generates the following outputs:

**HDF5 Files** (per input WSI):
```
output/patches/<wsi_stem>.h5
```

Contains:
- `coords`: Shape (N, 2), dtype int32 - (x, y) coordinates at level 0
- `coords_ext`: Shape (N, 5), dtype int32 - (x, y, w, h, level)
- File attributes: `patch_size`, `wsi_path`, `num_patches`, `level0_magnification`, `target_magnification`, `patch_size_level0`

**Optional PNG Images** (if `--save-images` is used):
```
output/images/<wsi_stem>/<wsi_stem>_x<x>_y<y>.png
```

Each file represents a single extracted patch with its coordinates in the filename.

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

**Parallelism:**

- **`--workers`**: Number of CPU workers for WSI parallelism (aka how many WSI are patchified at a time)
