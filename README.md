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
slideproc process sample.svs --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml

# Process all WSI files in a directory
slideproc process ./wsi_folder/ --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml --output ./results

# With custom patch settings
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 512 --step-size 256 \
    --output ./output --save-images
```

### Commands

#### `slideproc process`

Main command for processing whole slide images with tissue segmentation and patch extraction.

**Required Arguments:**
- `WSI_PATH` **(required)**: Path to a single WSI file or directory containing multiple WSI files

**Required Options:**
- `--checkpoint/-c` **(required)**: Path to SAM2 model checkpoint file (.pt)
- `--config` **(required)**: Path to a SAM2 YAML config file. Only filesystem paths are accepted (e.g., `slide_processor/configs/sam2.1_hiera_b+.yaml`).

**Optional Parameters:**

| Option | Type | Default | Required? | Description |
|--------|------|---------|-----------|-------------|
| `--output/-o` | Path | `./output` | No | Output directory for HDF5 files and patches |
| `--patch-size` | int | `256` | No | Size of extracted patches in pixels |
| `--step-size` | int | patch-size | No | Step size for patch extraction (stride). Defaults to patch-size if not set |
| `--device` | choice | `cuda` | No | Device for inference: `cuda` or `cpu` |
| `--thumbnail-size` | int | `1024` | No | Size of thumbnail for segmentation (max dimension) |
| `--tissue-thresh` | float | `0.01` | No | Minimum tissue area threshold as percentage of image |
| `--white-thresh` | int | `15` | No | Saturation threshold for filtering white patches |
| `--black-thresh` | int | `50` | No | RGB threshold for filtering black patches |
| `--require-all-points` | flag | False | No | Require all 4 corner points inside tissue (strict mode) |
| `--use-padding` | flag | True | No | Allow patches at image boundaries with padding |
| `--save-images` | flag | False | No | Export individual patch images as PNG files |
| `--h5-images/--no-h5-images` | flag | `--h5-images` | No | Store image arrays in the HDF5 file (`imgs` dataset). Disable to save only coordinates + metadata |
| `--fast-mode` | flag | False | No | Skip per-patch content filtering for faster extraction (may include background patches) |
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
slideproc process sample.svs --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml
```

#### Batch Processing Multiple Files

```bash
# Process all .svs files in a directory
slideproc process ./slides/ \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --output ./processed_slides
```

#### Custom Patch Extraction Parameters

```bash
# Extract larger patches with different stride
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --patch-size 512 \
    --step-size 256 \
    --output ./results
```

#### Export Individual Patch Images

```bash
# Generate individual PNG files for each patch
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --save-images \
    --output ./output
# Creates: output/<stem>/images/<stem>_x<x>_y<y>.png
```

#### Strict Tissue Requirement

```bash
# Require all 4 patch corners to be within tissue (stricter filtering)
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --require-all-points
```

#### CPU Inference

```bash
# Use CPU instead of GPU (slower but no GPU required)
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --device cpu
```

#### Custom Filtering Thresholds

```bash
# Adjust thresholds for different tissue characteristics
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --white-thresh 20 \
    --black-thresh 40 \
    --tissue-thresh 0.05
```

#### Verbose Output

```bash
# Enable detailed logging for debugging
slideproc process sample.svs \
    --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \
    --verbose
```

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
  - `imgs`: uint8 shape `(N, H, W, 3)` RGB patches (optional; present only when `--h5-images` is enabled)
- File attributes
  - `patch_size`: int
  - `wsi_path`: original WSI path
  - `num_patches`: total number of patches

When images are not stored in the HDF5 (`--no-h5-images`), you can reconstruct any patch by re-reading from the original WSI using `(x, y, w, h, level)` from `coords_ext`.

### Performance Notes

- The SAM2 predictor is now initialized once and reused across files to reduce per-slide overhead.
- Enable `--fast-mode` to skip per-patch white/black filtering. This can substantially reduce I/O, especially together with `--no-h5-images`.

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
- `imgs`: Shape (N, H, W, 3), dtype uint8 - RGB patch images
- `coords`: Shape (N, 2), dtype int32 - (x, y) coordinates at level 0
- File attributes: `patch_size`, `wsi_path`, `num_patches`

**Optional PNG Images** (if `--save-images` is used):
```
output/<wsi_stem>/images/<wsi_stem>_x<x>_y<y>.png
```

Each file represents a single extracted patch with its coordinates in the filename.

### Parameter Guide

**Patch Extraction Parameters:**

- **`--patch-size`**: Size of extracted patches (e.g., 256 = 256x256 pixels)
  - Larger values reduce number of patches but capture more context
  - Common values: 256, 512

- **`--step-size`**: Sliding window stride during patch extraction
  - If equal to patch-size: non-overlapping patches
  - If smaller: overlapping patches (more patches)
  - Default behavior: uses patch-size (non-overlapping)

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
