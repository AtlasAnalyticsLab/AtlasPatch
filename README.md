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
    - [Full Processing with Feature Extraction](#full-processing-with-feature-extraction)
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
- [Available Feature Extractors](#available-feature-extractors)
- [HDF5 Output Structure](#hdf5-output-structure)
  - [Output](#output)
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
slideproc segment-and-get-coords sample.svs --patch-size 256 --target-mag 20

# Segment, extract patches, and embed features (stores features in the patch H5)
slideproc process sample.svs \
    --patch-size 256 --target-mag 20 \
    --feature-extractors resnet18,resnet50

# Process all WSI files in a directory
slideproc segment-and-get-coords ./wsi_folder/ \
    --patch-size 256 --target-mag 20 --output ./results

# With custom patch settings and visualization
slideproc segment-and-get-coords sample.svs \
    --patch-size 512 --step-size 256 --target-mag 20 \
    --output ./output --save-images --visualize-grids
```

### Commands

#### `slideproc segment-and-get-coords`

Main command for processing whole slide images with tissue segmentation and patch extraction.

**Required Arguments:**
- `WSI_PATH` **(required)**: Path to a single WSI file or directory containing multiple WSI files

**Required Options:**
- `--patch-size` **(required)**: Target size of extracted patches in pixels (final patch dimensions)
- `--target-mag` **(required)**: Target magnification for extraction (e.g., 10, 20, 40)

**Optional Parameters:**

| Option | Type | Default | Required? | Description |
|--------|------|---------|-----------|-------------|
| `--step-size` | int | patch-size | No | Stride between patches at target magnification; defaults to patch size |
| `--output/-o` | Path | `./output` | No | Output directory root for results (contains `patches/`, `visualization/`, and `images/`) |
| `--device` | string | `cuda` | No | Segmentation device: `cuda`, `cuda:<idx>`, or `cpu` |
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

#### `slideproc process`

End-to-end command that runs SAM2 segmentation, patch extraction, and feature embedding into a single HDF5. Feature matrices are stored under `features/<extractor_name>` alongside coordinates.

**Required Arguments:**
- `WSI_PATH` **(required)**: Path to a single WSI file or directory containing multiple WSI files

**Required Options:**
- `--patch-size` **(required)**: Target size of extracted patches in pixels
- `--target-mag` **(required)**: Target magnification for extraction (e.g., 10, 20, 40)
- `--feature-extractors` **(required)**: Space/comma separated feature extractors to run. Built-ins: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`, `dinov2_small`, `dinov2_base`, `dinov2_large`, `dinov2_giant`, `dinov3_vits16`, `dinov3_vits16_plus`, `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vitl16_sat`, `dinov3_vith16_plus`, `dinov3_vit7b16`, `dinov3_vit7b16_sat`, `uni_v1`, `uni_v2`, `biomedclip`, `clip_rn50`, `clip_rn101`, `clip_rn50x4`, `clip_rn50x16`, `clip_rn50x64`, `clip_vit_b_32`, `clip_vit_b_16`, `clip_vit_l_14`, `clip_vit_l_14_336`, `plip`, `medsiglip`, `phikon_v1`, `phikon_v2`, `virchow_v1`, `virchow_v2`, `prov_gigapath`, `midnight`, `musk`, `openmidnight`, `pathorchestra`, `h_optimus_0`, `h_optimus_1`, `h0_mini`, `hibou_b`, `hibou_l`, `quilt_b_32`, `quilt_b_16`, `quilt_b_16_pmb`.

**Feature Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--feature-extractors` | comma/space separated | — | Models to embed patches with (built-ins: resnet18/34/50/101/152, convnext_tiny/small/base/large, vit_b_16/b_32/l_16/l_32/h_14, dinov2_small/base/large/giant, dinov3_vits16/vits16_plus/vitb16/vitl16/vitl16_sat/vith16_plus/vit7b16/vit7b16_sat, uni_v1, uni_v2, lunit_resnet50_bt, lunit_resnet50_swav, lunit_resnet50_mocov2, lunit_vit_small_patch16_dino, lunit_vit_small_patch8_dino, biomedclip, clip_rn50, clip_rn101, clip_rn50x4, clip_rn50x16, clip_rn50x64, clip_vit_b_32, clip_vit_b_16, clip_vit_l_14, clip_vit_l_14_336, plip, medsiglip, phikon_v1, phikon_v2, virchow_v1, virchow_v2, prov_gigapath, midnight, musk, openmidnight, pathorchestra, h_optimus_0, h_optimus_1, h0_mini, hibou_b, hibou_l, quilt_b_32, quilt_b_16, quilt_b_16_pmb) |
| `--feature-batch-size` | int | 32 | Batch size for feature forward passes |
| `--feature-device` | choice | inherits `--device` | Device for feature extraction (cpu/cuda/cuda:<idx>) |
| `--feature-num-workers` | int | 4 | DataLoader worker count for feature extraction |
| `--feature-precision` | choice | float32 | Compute precision for feature forward passes (`float32`, `float16`, `bfloat16`) |

Feature embedding runs as a deferred pass over saved coordinates for each slide.

All other parameters mirror `segment-and-get-coords` (stride, thresholds, workers, visualization flags, etc.).
This command respects `--skip-existing`; rerun with `--force` when you need to regenerate features with a different extractor set.

**Supported WSI Formats:**
- **OpenSlide formats**: .svs, .tif, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .dcm
- **Image formats**: .png, .jpg, .jpeg, .bmp, .webp, .gif

### Usage Examples

#### Basic Single File Processing

```bash
slideproc segment-and-get-coords sample.svs \
    --patch-size 256 --target-mag 20
```

#### Full Processing with Feature Extraction

```bash
slideproc process sample.svs \
    --patch-size 256 --target-mag 20 \
    --feature-extractors resnet18,resnet50 \
    --feature-batch-size 32
```

This produces a single H5 file containing coordinates plus two feature matrices under `features/resnet18` and `features/resnet50`.

#### Batch Processing Multiple Files

```bash
# Process all .svs files in a directory
slideproc segment-and-get-coords ./slides/ \
    --patch-size 256 --target-mag 20 \
    --output ./processed_slides
# Batch thumbnails for segmentation
slideproc segment-and-get-coords ./slides/ \
    --patch-size 256 --target-mag 20 \
    --seg-batch-size 8 \
    --patch-workers 4 \
    --max-open-slides 12 \
    --output ./processed_slides
```

#### Custom Patch Extraction Parameters

```bash
# Extract larger patches with different stride and magnification
slideproc segment-and-get-coords sample.svs \
    --patch-size 512 \
    --step-size 256 \
    --target-mag 20 \
    --output ./results
```

#### Export Individual Patch Images

```bash
# Generate individual PNG files for each patch
slideproc segment-and-get-coords sample.svs \
    --patch-size 256 --target-mag 20 \
    --save-images \
    --output ./output
# Creates: output/images/<stem>/<stem>_x<x>_y<y>.png
```


#### CPU Inference

```bash
# Use CPU instead of GPU (slower but no GPU required)
slideproc segment-and-get-coords sample.svs \
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
slideproc segment-and-get-coords ./wsi_folder/ \
    --patch-size 256 --target-mag 20 \
    --mpp-csv mpp_values.csv
```

**CSV Specifications:**
- **Required columns**: `wsi` (filename with or without path) and `mpp` (float value)
- **WSI names**: Can use just the filename (e.g., `slide1.svs`) or full path; only the stem (filename without extension) is matched

#### Custom Filtering Thresholds

```bash
# Adjust thresholds for different tissue characteristics
slideproc segment-and-get-coords sample.svs \
    --patch-size 256 --target-mag 20 \
    --white-thresh 20 \
    --black-thresh 40 \
    --tissue-thresh 0.05
```

#### Verbose Output

```bash
# Enable detailed logging for debugging
slideproc segment-and-get-coords sample.svs \
    --patch-size 256 --target-mag 20 \
    --verbose
```

- Default (no --verbose): quiet mode with a progress bar and minimal output.

#### Generate Visualizations

```bash
# Generate visualizations on thumbnail
slideproc segment-and-get-coords sample.svs \
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

## Available Feature Extractors

### Core vision backbones on Natural Images

| Name | Output Dim |
| --- | --- |
| `resnet18` | 512 |
| `resnet34` | 512 |
| `resnet50` | 2048 |
| `resnet101` | 2048 |
| `resnet152` | 2048 |
| `convnext_tiny` | 768 |
| `convnext_small` | 768 |
| `convnext_base` | 1024 |
| `convnext_large` | 1536 |
| `vit_b_16` | 768 |
| `vit_b_32` | 768 |
| `vit_l_16` | 1024 |
| `vit_l_32` | 1024 |
| `vit_h_14` | 1280 |
| [`dinov2_small`](https://huggingface.co/facebook/dinov2-small) ([DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)) | 384 |
| [`dinov2_base`](https://huggingface.co/facebook/dinov2-base) ([DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)) | 768 |
| [`dinov2_large`](https://huggingface.co/facebook/dinov2-large) ([DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)) | 1024 |
| [`dinov2_giant`](https://huggingface.co/facebook/dinov2-giant) ([DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)) | 1536 |
| [`dinov3_vits16`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 384 |
| [`dinov3_vits16_plus`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 384 |
| [`dinov3_vitb16`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 768 |
| [`dinov3_vitl16`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 1024 |
| [`dinov3_vitl16_sat`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 1024 |
| [`dinov3_vith16_plus`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 1280 |
| [`dinov3_vit7b16`](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 4096 |
| [`dinov3_vit7b16_sat`](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m) ([DINOv3](https://arxiv.org/abs/2508.10104)) | 4096 |

### Medical- and Pathology-Specific Vision Encoders

| Name | Output Dim |
| --- | --- |
| [`uni_v1`](https://huggingface.co/MahmoodLab/UNI) ([Towards a General-Purpose Foundation Model for Computational Pathology](https://www.nature.com/articles/s41591-024-02857-3)) | 1024 |
| [`uni_v2`](https://huggingface.co/MahmoodLab/UNI2-h) ([Towards a General-Purpose Foundation Model for Computational Pathology](https://www.nature.com/articles/s41591-024-02857-3)) | 1536 |
| [`phikon_v1`](https://huggingface.co/owkin/phikon) ([Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1)) | 768 |
| [`phikon_v2`](https://huggingface.co/owkin/phikon-v2) ([Phikon-v2, A large and public feature extractor for biomarker prediction](https://arxiv.org/abs/2409.09173)) | 1024 |
| [`virchow_v1`](https://huggingface.co/paige-ai/Virchow) ([Virchow: A Million-Slide Digital Pathology Foundation Model](https://arxiv.org/abs/2309.07778)) | 2560 |
| [`virchow_v2`](https://huggingface.co/paige-ai/Virchow2) ([Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology](https://arxiv.org/abs/2408.00738)) | 2560 |
| [`prov_gigapath`](https://huggingface.co/prov-gigapath/prov-gigapath) ([A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w)) | 1536 |
| [`midnight`](https://huggingface.co/kaiko-ai/midnight) ([Training state-of-the-art pathology foundation models with orders of magnitude less data](https://arxiv.org/abs/2504.05186)) | 3072 |
| [`musk`](https://github.com/lilab-stanford/MUSK) ([MUSK: A Vision-Language Foundation Model for Precision Oncology](https://www.nature.com/articles/s41586-024-08378-w)) | 1024 |
| [`openmidnight`](https://sophontai.com/blog/openmidnight) ([How to Train a State-of-the-Art Pathology Foundation Model with $1.6k](https://sophontai.com/blog/openmidnight)) | 1536 |
| [`pathorchestra`](https://huggingface.co/AI4Pathology/PathOrchestra) ([PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks](https://arxiv.org/abs/2503.24345)) | 512 |
| [`h_optimus_0`](https://huggingface.co/bioptimus/H-optimus-0) | 1536 |
| [`h_optimus_1`](https://huggingface.co/bioptimus/H-optimus-1) | 1536 |
| [`h0_mini`](https://huggingface.co/bioptimus/H0-mini) ([Distilling foundation models for robust and efficient models in digital pathology](https://doi.org/10.48550/arXiv.2501.16239)) | 1536 |
| [`hibou_b`](https://huggingface.co/histai/hibou-B) ([Hibou: A Family of Foundational Vision Transformers for Pathology](https://arxiv.org/abs/2406.05074)) | 768 |
| [`hibou_l`](https://huggingface.co/histai/hibou-L) ([Hibou: A Family of Foundational Vision Transformers for Pathology](https://arxiv.org/abs/2406.05074)) | 1024 |
| [`lunit_resnet50_bt`](https://huggingface.co/1aurent/resnet50.lunit_bt) ([Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.pdf)) | 2048 |
| [`lunit_resnet50_swav`](https://huggingface.co/1aurent/resnet50.lunit_swav) ([Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.pdf)) | 2048 |
| [`lunit_resnet50_mocov2`](https://huggingface.co/1aurent/resnet50.lunit_mocov2) ([Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.pdf)) | 2048 |
| [`lunit_vit_small_patch16_dino`](https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino) ([Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.pdf)) | 384 |
| [`lunit_vit_small_patch8_dino`](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) ([Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.pdf)) | 384 |

### CLIP-like models

#### Natural Images

| Name | Output Dim |
| --- | --- |
| `clip_rn50` | 1024 |
| `clip_rn101` | 512 |
| `clip_rn50x4` | 640 |
| `clip_rn50x16` | 768 |
| `clip_rn50x64` | 1024 |
| `clip_vit_b_32` | 512 |
| `clip_vit_b_16` | 512 |
| `clip_vit_l_14` | 768 |
| `clip_vit_l_14_336` | 768 |

#### Medical- and Pathology-Specific CLIP

| Name | Output Dim |
| --- | --- |
| [`plip`](https://github.com/PathologyFoundation/plip) ([Pathology Language and Image Pre-Training (PLIP)](https://www.nature.com/articles/s41591-023-02504-3)) | 512 (projection dim) |
| [`medsiglip`](https://huggingface.co/google/medsiglip-448) ([MedGemma Technical Report](https://arxiv.org/abs/2507.05201)) | 1152 |
| [`quilt_b_32`](https://quilt1m.github.io/) ([Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/pdf/2306.11207)) | 512 |
| [`quilt_b_16`](https://quilt1m.github.io/) ([Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/pdf/2306.11207)) | 512 |
| [`quilt_b_16_pmb`](https://quilt1m.github.io/) ([Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/pdf/2306.11207)) | 512 |
| [`biomedclip`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) ([BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs](https://aka.ms/biomedclip-paper)) | 512 |

## HDF5 Output Structure

Each processed slide produces a single HDF5 file under `<output>/<mag>x_<patch>px_<overlap>px_overlap/patches/<stem>.h5`.

- Datasets
  - `coords`: int32 shape `(N, 2)` containing `(x, y)` at level 0
  - `coords_ext`: int32 shape `(N, 5)` containing `(x, y, w, h, level)` for reliable re-reading
  - `features/<extractor>`: float32 shape `(N, D)` feature matrix for each requested extractor (e.g., `features/resnet18`, `features/resnet50`)
- File attributes
  - `patch_size`: int (target patch size)
  - `wsi_path`: original WSI path
  - `num_patches`: total number of patches
  - `level0_magnification`: magnification of the highest-resolution level (if known)
  - `target_magnification`: magnification used for extraction
  - `patch_size_level0`: size of the patch footprint at level 0 in pixels. Used by some slide encoders which uses it for positional encoding module (e.g., [ALiBi](https://arxiv.org/pdf/2108.12409) in [TITAN](https://arxiv.org/abs/2411.19666))
  - `feature_sets`: JSON metadata describing each extractor stored in the file (name, embedding_dim, dataset path)


### Output

Results are written under a run-specific subdirectory named `<mag>x_<patch>px_<overlap>px_overlap` (where `overlap = patch_size - step_size`). Inside this directory:

- `patches/` contains the HDF5 outputs (`<stem>.h5`).
- `images/` contains optional per-patch PNGs when `--save-images` is set.
- `visualization/` contains optional overlays for grids, masks, and contours.
- Feature matrices live inside each slide's H5 under `features/<extractor>` when `slideproc process` is used.

**HDF5 Files** (per input WSI):
```
<output>/<mag>x_<patch>px_<overlap>px_overlap/patches/<wsi_stem>.h5
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

## SLURM job scripts

We prepared ready-to-run SLURM templates under `jobs/`:

- Patch extraction (SAM2 + H5/PNG): `jobs/slideproc_patch.slurm.sh`. Edits to make:
  - Set `WSI_ROOT`, `OUTPUT_ROOT`, `PATCH_SIZE`, `TARGET_MAG`, `SEG_BATCH`.
  - Ensure `--cpus-per-task` matches the CPU you want; the script passes `--patch-workers ${SLURM_CPUS_PER_TASK}` and caps `--max-open-slides` at 200.
  - `--fast-mode` is on by default; append `--no-fast-mode` to enable content filtering.
  - Submit with `sbatch jobs/slideproc_patch.slurm.sh`.

## Feedback

- Report problems via the [bug report template](https://github.com/AtlasAnalyticsLab/SlideProcessor/issues/new?template=bug_report.md) so we can reproduce and fix them quickly.
- Suggest enhancements through the [feature request template](https://github.com/AtlasAnalyticsLab/SlideProcessor/issues/new?template=feature_request.md) with your use case and proposal.
- When opening a PR, fill out the [pull request template](.github/pull_request_template.md) and run the listed checks (lint, format, type-check, tests).

## License

SlideProcessor is licensed under the **PolyForm Noncommercial License 1.0.0**, which strictly prohibits commercial use of this software or any derivative works. This applies to all forms of commercialization, including selling the software, offering it as a commercial service, using it in commercial products, or creating forked versions for commercial purposes. However, the license explicitly permits use for research, experimentation, and non-commercial purposes. Personal use for research, hobby projects, and educational purposes is allowed, as is use by academic institutions, educational organizations, public research organizations, and non-profit entities regardless of their funding sources. If you wish to use SlideProcessor commercially, you must obtain a separate commercial license from the authors. For the complete license text and detailed terms, see the [LICENSE](./LICENSE) file in this repository.

# TODO

## Refactor
- Update name from Slide Processor to `Atlas Patch`

## Patch Encoders
- CONCH v1
- CONCH v1.5
- CHIEF
- Omiclip
- CTransPath
- DINO v3

## Model loading
- Add automatic model loading from Hugging Face

## Saving to H5 file:
- remove the duplication in coordinate attributes and the attributes in general
```
  level0_height: 12288
  level0_magnification: 20
  level0_width: 3584
  num_patches: 79
  overlap: 0
  patch_size: 512
  patch_size_level0: 512
  target_magnification: 20
  wsi_path: /home/mila/k/kotpy/scratch/datasets/PANDA/images/0a0f8e20b1222b69416301444b117678.tiff
coords: shape=(79, 2), dtype=int32
coords attrs:
  description: (x, y) coordinates at level 0
  level0_height: 12288
  level0_magnification: 20
  level0_width: 3584
  name: 0a0f8e20b1222b69416301444b117678
  overlap: 0
  patch_size: 512
  patch_size_level0: 512
  savetodir: /network/scratch/k/kotpy/datasets/PANDA/slide_proc/20x_512px_0px_overlap/patches
  target_magnification: 20
```

- Remove the feature set entry entirely

## Shipping
- Make `pip install atlas_patch`

## Documentation
- Update and decrease the README file size, make it more straight forward
- Add part on how to use those extracted features

## Fancy features
- Support `bring your own encoder` functionality

## Contours
- filter_params  in mask_to_contours in `utils/contours.py`
```
filter_params = {
            "a_t": 100,  # Minimum tissue contour area (in pixels)
            "a_h": 16,  # Minimum hole area
            "max_n_holes": 10,
        }
```
