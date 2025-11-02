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
# Process a single WSI file (YAML config path)
slideproc process sample.svs --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20

# Process all WSI files in a directory
slideproc process ./wsi_folder/ --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 --output ./results

# With custom patch settings and visualization
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 512 --step-size 256 --target-mag 20 \
    --output ./output --save-images --visualize
```

### Commands

#### `slideproc process`

Main command for processing whole slide images with tissue segmentation and patch extraction.

**Required Arguments:**
- `WSI_PATH` **(required)**: Path to a single WSI file or directory containing multiple WSI files

**Required Options:**
- `--checkpoint/-c` **(required)**: Path to SAM2 model checkpoint file (.pt)
- `--config` **(required)**: Path to a SAM2 YAML config file. Only filesystem paths are accepted (e.g., `slide_processor/configs/sam2.1_hiera_b+.yaml`).
- `--patch-size` **(required)**: Target size of extracted patches in pixels (final patch dimensions)
- `--target-mag` **(required)**: Target magnification for extraction: one of 1, 2, 4, 5, 10, 20, 40, 60, 80

**Optional Parameters:**

| Option | Type | Default | Required? | Description |
|--------|------|---------|-----------|-------------|
| `--output/-o` | Path | `./output` | No | Output directory for HDF5 files and patches |
| `--patch-size` | int | — | Yes | Target size of extracted patches in pixels (final patch dimensions) |
| `--step-size` | int | patch-size | No | Step size for patch extraction (stride) at the target magnification. Defaults to patch-size if not set |
| `--target-mag` | choice | — | Yes | Target magnification for extraction: one of 5, 10, 20, 40, 60, 80 |
| `--device` | choice | `cuda` | No | Device for inference: `cuda` or `cpu` |
| `--thumbnail-size` | int | `1024` | No | Size of thumbnail for segmentation (max dimension) |
| `--tissue-thresh` | float | `0.01` | No | Minimum tissue area threshold as percentage of image |
| `--white-thresh` | int | `15` | No | Saturation threshold for filtering white patches |
| `--black-thresh` | int | `50` | No | RGB threshold for filtering black patches |
| `--require-all-points` | flag | False | No | Require all 4 corner points inside tissue (strict mode) |
| `--use-padding` | flag | True | No | Allow patches at image boundaries with padding |
| `--save-images` | flag | False | No | Export individual patch images as PNG files in `/images` folder |
| `--fast-mode` | flag | False | No | Skip per-patch content filtering for faster extraction (may include background patches) |
| `--visualize` | flag | False | No | Generate visualization of patches overlaid on WSI thumbnail with processing info |
| `--verbose/-v` | flag | False | No | Enable verbose logging output |

**Available SAM2 Configs (YAML paths):**

- `sam2.1_hiera_t` — Tiny model, fastest and lowest memory; lowest accuracy.
- `sam2.1_hiera_s` — Small model, balanced speed/VRAM/accuracy.
- `sam2.1_hiera_b+` — Base+ model, recommended default; higher accuracy; more VRAM.
- `sam2.1_hiera_l` — Large model, highest accuracy; slowest; high VRAM.

Configs are provided under `slide_processor/configs/`. Always pass the YAML path, for example:

- `--config slide_processor/configs/sam2.1_hiera_t.yaml`
- `--config slide_processor/configs/sam2.1_hiera_s.yaml`
- `--config slide_processor/configs/sam2.1_hiera_b+.yaml`
- `--config slide_processor/configs/sam2.1_hiera_l.yaml`

**Supported WSI Formats:**
- **OpenSlide formats**: .svs, .tif, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .dcm
- **Image formats**: .png, .jpg, .jpeg, .bmp, .webp, .gif

### Usage Examples

#### Basic Single File Processing

```bash
slideproc process sample.svs --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20
```

#### Batch Processing Multiple Files

```bash
# Process all .svs files in a directory
slideproc process ./slides/ \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --output ./processed_slides
```

#### Custom Patch Extraction Parameters

```bash
# Extract larger patches with different stride and magnification
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 512 \
    --step-size 256 \
    --target-mag 20 \
    --output ./results
```

#### Export Individual Patch Images

```bash
# Generate individual PNG files for each patch
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --save-images \
    --output ./output
# Creates: output/<stem>/images/<stem>_x<x>_y<y>.png
```

#### Strict Tissue Requirement

```bash
# Require all 4 patch corners to be within tissue (stricter filtering)
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --require-all-points
```

#### CPU Inference

```bash
# Use CPU instead of GPU (slower but no GPU required)
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --device cpu
```

#### Custom Filtering Thresholds

```bash
# Adjust thresholds for different tissue characteristics
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --white-thresh 20 \
    --black-thresh 40 \
    --tissue-thresh 0.05
```

#### Verbose Output

```bash
# Enable detailed logging for debugging
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --verbose
```

#### Generate Visualizations

```bash
# Generate patch overlay visualization on thumbnail
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 256 --target-mag 20 \
    --visualize

```

The `--visualize` flag creates a visualization showing:
- WSI thumbnail with patch locations overlaid as green rectangles
- Information panel with extraction statistics and parameters used

The visualization is saved in the output directory: `output/<wsi_stem>/patches_on_thumbnail.png`.

#### `slideproc info`

Display information about supported formats and features.

```bash
slideproc info
```

## HDF5 Output Structure

Each processed slide produces a single HDF5 file under `<output>/<stem>/<stem>.h5`.

- Datasets
  - `coords`: int32 shape `(N, 2)` containing `(x, y)` at level 0
  - `coords_ext`: int32 shape `(N, 5)` containing `(x, y, w, h, level)` for reliable re-reading
- File attributes
  - `patch_size`: int (target patch size)
  - `wsi_path`: original WSI path
  - `num_patches`: total number of patches
  - `level0_magnification`: magnification of the highest-resolution level (if known)
  - `target_magnification`: magnification used for extraction
  - `patch_size_level0`: size of the patch footprint at level 0 in pixels. Used by visualizations and downstream tooling.


### Performance Notes

- The SAM2 predictor is now initialized once and reused across files to reduce per-slide overhead.
- Enable `--fast-mode` to skip per-patch white/black filtering. This can substantially reduce I/O.

Shows:
- Supported WSI formats
- Supported image formats
- Output format details

### Output

The CLI generates the following outputs:

**HDF5 Files** (per input WSI):
```
output/<wsi_stem>/<wsi_stem>.h5
```

Contains:
- `coords`: Shape (N, 2), dtype int32 - (x, y) coordinates at level 0
- `coords_ext`: Shape (N, 5), dtype int32 - (x, y, w, h, level)
- File attributes: `patch_size`, `wsi_path`, `num_patches`, `level0_magnification`, `target_magnification`, `patch_size_level0`

**Optional PNG Images** (if `--save-images` is used):
```
output/<wsi_stem>/images/<wsi_stem>_x<x>_y<y>.png
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
  - Lower values = more aggressively filter white regions
  - Useful for filtering background and faint areas

- **`--black-thresh`**: RGB threshold for black patches
  - Lower values = filter darker regions
  - Useful for filtering shadows and staining artifacts

- **`--tissue-thresh`**: Minimum tissue area as percentage of image
  - Filters out very small tissue regions
  - Typical range: 0.01-1.0
  - Unit: percentage (not proportion)

**Segmentation Parameters:**

- **`--thumbnail-size`**: Size of thumbnail for SAM2 inference
  - Larger values = more accurate segmentation but slower
  - Common values: 1024, 2048
  - Limited by GPU memory

**Geometry Parameters:**

- **`--require-all-points`**: Toggle for patch validation strictness
  - False (default, lenient): Any of 4 corners inside tissue = valid
  - True (strict): All 4 corners must be inside tissue

- **`--use-padding`**: Allow patches extending beyond image boundaries
  - True (default): Patches at edges can extend beyond bounds
  - False: Patches must be fully contained within image

## Tasks
- [x] Extract patches at different magnification level
- [ ] Add visualization module to visualize the patches
- [ ] Decide on the final arguments and parameter we are going to expose / de-expose
- [ ] Check if file exist in the output directory and if so then skip it
- [ ] Support parallelization for a single GPU
- [ ] Support paralelization for multipl GPUs
