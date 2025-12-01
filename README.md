# AtlasPatch

A Python package for processing and handling whole slide images (WSI).

## Table of Contents
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Minimal Run](#minimal-run)
  - [Process Command Arguments](#process-command-arguments)
- [Supported Formats](#supported-formats)
- [Using Extracted Data](#using-extracted-data)
  - [Patch Coordinates](#patch-coordinates)
  - [Feature Matrices](#feature-matrices)
- [Available Feature Extractors](#available-feature-extractors)
- [Bring Your Own Encoder](#bring-your-own-encoder)
- [SLURM job scripts](#slurm-job-scripts)
- [Feedback](#feedback)
- [Citation](#citation)
- [License](#license)
- [Future Updates](#future-updates)

## Installation

### Using Conda (Recommended)

1. Create a conda environment:
```bash
conda create -n atlas_patch python=3.10
conda activate atlas_patch
```

2. Install the OpenSlide system library (required for WSI processing):
```bash
conda install -c conda-forge openslide
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Using uv (pip-compatible, faster installs)

1. Install uv if not already available (see [uv docs](https://docs.astral.sh/uv/getting-started/)):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment (UV_VENV defaults to `.venv`):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode with uv:
```bash
uv pip install -e .
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

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage Guide

`atlaspatch process` runs segmentation, patch extraction, and feature embedding into a single HDF5 per slide.

### Minimal Run

```bash
atlaspatch process /path/to/slide.svs \
  --output ./output \
  --patch-size 256 \
  --target-mag 20 \
  --feature-extractors uni_v1,dinov2_small,openmidnight
```

Pass a directory instead of a single file to process multiple WSIs; outputs land in `<output>/patches/<stem>.h5` based on the path you provide to `--output`.

### Process Command Arguments

#### Required
- `WSI_PATH` — file or directory of slides to process.
- `--output/-o` — root directory for results.
- `--patch-size` — final patch size in pixels at target magnification.
- `--target-mag` — magnification to extract at (e.g., 10/20/40).
- `--feature-extractors` — comma/space separated names from [Available Feature Extractors](#available-feature-extractors).

#### Optional
- **Patch layout**: `--step-size` sets stride; omit for non-overlapping grids (stride = patch-size), smaller values create overlaps.
- **Segmentation & extraction performance**: `--device` picks the segmentation device (`cuda`, `cuda:<idx>`, or `cpu`). `--seg-batch-size` controls SAM2 thumbnail batch size. `--patch-workers` threads handle patch extraction/H5 writes (defaults to CPU count). `--max-open-slides` caps simultaneously open WSIs across segmentation/extraction.
- **Feature extraction/embedding**: `--feature-device` defaults to `--device`; set separately if you want different GPU for feature extraction than segmentation. `--feature-batch-size` sets forward-pass batch size for the feature model. `--feature-num-workers` configures DataLoader workers. `--feature-precision` can reduce memory/bandwidth (`float32/float16`/`bfloat16` on GPU when supported).
- **Filtering & quality**: `--fast-mode` (default) skips per-patch black/white filtering; `--no-fast-mode` enables it. `--tissue-thresh` filters tiny tissue regions (fraction of a 1024x1024 mask). When filtering is on (`--no-fast-mode`), `--white-thresh` is the HSV saturation cutoff for white patches (lower = stricter) and `--black-thresh` is the grayscale cutoff for dark patches (lower = stricter).
- **Visualization**: `--visualize-grids`, `--visualize-mask`, `--visualize-contours` render overlays on thumbnails under `<output>/visualization/`.
- **Run control**: `--save-images` exports per-patch PNGs. `--recursive` walks subdirectories. `--mpp-csv` supplies custom `wsi,mpp` overrides when metadata is missing/wrong. `--skip-existing` avoids reprocessing; use `--force` to overwrite. `--verbose/-v` switches to debug logging and disables the progress bar. `--write-batch` controls how many coord rows are buffered before flushing to H5 (tune for RAM vs. I/O).

## Supported Formats

AtlasPatch uses OpenSlide for WSIs and Pillow for standard images:

- WSIs: `.svs`, `.tif`, `.tiff`, `.ndpi`, `.vms`, `.vmu`, `.scn`, `.mrxs`, `.bif`, `.dcm`
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`

## Using Extracted Data

`atlaspatch process` writes one HDF5 per slide under `<output>/patches/<stem>.h5` containing coordinates and feature matrices. Coordinates and features share row order.

### Patch Coordinates

- Dataset: `coords` (int32, shape `(N, 5)`) with columns `(x, y, read_w, read_h, level)`.
- `x` and `y` are level-0 pixel coordinates. `read_w`, `read_h`, and `level` describe how the patch was read from the WSI.
- The level-0 footprint of each patch is stored as the `patch_size_level0` file attribute; some slide encoders use it for positional encoding (e.g., ALiBi in TITAN).

Example:

```python
import h5py
import numpy as np
import openslide
from PIL import Image

h5_path = "output/patches/sample.h5"
with h5py.File(h5_path, "r") as f:
    coords = f["coords"][...]            # (N, 5) int32
    patch_size_lvl0 = int(f.attrs["patch_size_level0"])
    boxes_lvl0 = np.column_stack(
        [coords[:, 0], coords[:, 1], coords[:, 0] + patch_size_lvl0, coords[:, 1] + patch_size_lvl0]
    )  # [x0, y0, x1, y1] at level 0
    levels = coords[:, 4]
    read_wh = coords[:, 2:4]

    # Opening the slide
    slide_path = f.attrs["wsi_path"]
    patch_size = int(f.attrs["patch_size"])
    slide = openslide.OpenSlide(slide_path)
    try:
        # Reading the first patch (same flow used before feature extraction)
        x, y, read_w, read_h, level = coords[0]
        region = slide.read_region((int(x), int(y)), int(level), (int(read_w), int(read_h))).convert("RGB")
        patch = region.resize((patch_size, patch_size), resample=Image.Resampling.BILINEAR)
        patch_np = np.array(patch)  # shape (patch_size, patch_size, 3), uint8
    finally:
        slide.close()
```

### Feature Matrices

- Group: `features/` inside the same HDF5.
- Each extractor is stored as `features/<name>` (float32, shape `(N, D)`), aligned row-for-row with `coords`.
- List available feature sets with `list(f['features'].keys())`.

```python
import h5py

with h5py.File("output/patches/sample.h5", "r") as f:
    feat_names = list(f["features"].keys())
    resnet18_feats = f["features/resnet18"][...]  # numpy array of shape (N, embedding_dim)
```

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
| [`chief-ctranspath`](https://github.com/hms-dbmi/CHIEF?tab=readme-ov-file) ([CHIEF: Clinical Histopathology Imaging Evaluation Foundation Model](https://www.nature.com/articles/s41586-024-07894-z)) | 768 |
| [`midnight`](https://huggingface.co/kaiko-ai/midnight) ([Training state-of-the-art pathology foundation models with orders of magnitude less data](https://arxiv.org/abs/2504.05186)) | 3072 |
| [`musk`](https://github.com/lilab-stanford/MUSK) ([MUSK: A Vision-Language Foundation Model for Precision Oncology](https://www.nature.com/articles/s41586-024-08378-w)) | 1024 |
| [`openmidnight`](https://sophontai.com/blog/openmidnight) ([How to Train a State-of-the-Art Pathology Foundation Model with $1.6k](https://sophontai.com/blog/openmidnight)) | 1536 |
| [`pathorchestra`](https://huggingface.co/AI4Pathology/PathOrchestra) ([PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks](https://arxiv.org/abs/2503.24345)) | 512 |
| [`h_optimus_0`](https://huggingface.co/bioptimus/H-optimus-0) | 1536 |
| [`h_optimus_1`](https://huggingface.co/bioptimus/H-optimus-1) | 1536 |
| [`h0_mini`](https://huggingface.co/bioptimus/H0-mini) ([Distilling foundation models for robust and efficient models in digital pathology](https://doi.org/10.48550/arXiv.2501.16239)) | 1536 |
| [`conch_v1`](https://huggingface.co/MahmoodLab/CONCH) ([A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4)) | 512 |
| [`conch_v15`](https://huggingface.co/MahmoodLab/conchv1_5) - [From TITAN](https://huggingface.co/MahmoodLab/TITAN) ([A multimodal whole-slide foundation model for pathology](https://www.nature.com/articles/s41591-025-03982-3)) | 768 |
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
| `clip_rn50` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 1024 |
| `clip_rn101` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 512 |
| `clip_rn50x4` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 640 |
| `clip_rn50x16` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 768 |
| `clip_rn50x64` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 1024 |
| `clip_vit_b_32` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 512 |
| `clip_vit_b_16` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 512 |
| `clip_vit_l_14` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 768 |
| `clip_vit_l_14_336` ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)) | 768 |

#### Medical- and Pathology-Specific CLIP

| Name | Output Dim |
| --- | --- |
| [`plip`](https://github.com/PathologyFoundation/plip) ([Pathology Language and Image Pre-Training (PLIP)](https://www.nature.com/articles/s41591-023-02504-3)) | 512 |
| [`medsiglip`](https://huggingface.co/google/medsiglip-448) ([MedGemma Technical Report](https://arxiv.org/abs/2507.05201)) | 1152 |
| [`quilt_b_32`](https://quilt1m.github.io/) ([Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/pdf/2306.11207)) | 512 |
| [`quilt_b_16`](https://quilt1m.github.io/) ([Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/pdf/2306.11207)) | 512 |
| [`quilt_b_16_pmb`](https://quilt1m.github.io/) ([Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/pdf/2306.11207)) | 512 |
| [`biomedclip`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) ([BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs](https://aka.ms/biomedclip-paper)) | 512 |
| [`omiclip`](https://huggingface.co/WangGuangyuLab/Loki) ([A visual-omics foundation model to bridge histopathology with spatial transcriptomics](https://www.nature.com/articles/s41592-025-02707-1)) | 768 |

## Bring Your Own Encoder

Add a custom encoder without touching AtlasPatch by writing a small plugin and pointing the CLI at it with `--feature-plugin /path/to/plugin.py`. The plugin must expose a `register_feature_extractors(registry, device, dtype, num_workers)` function; inside that hook call `register_custom_encoder` with a loader that knows how to load the model and run a forward pass.

```python
import torch
from torchvision import transforms
from atlas_patch.models.patch.custom import CustomEncoderComponents, register_custom_encoder


def build_my_encoder(device: torch.device, dtype: torch.dtype) -> CustomEncoderComponents:
    """
    Build the components used by AtlasPatch to embed patches with a custom model.

    Returns:
        CustomEncoderComponents describing the model, preprocess transform, and forward pass.
    """
    model = ...  # your torch.nn.Module
    model = model.to(device=device, dtype=dtype).eval()
    preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    def forward(batch: torch.Tensor) -> torch.Tensor:
        return model(batch)  # must return [batch, embedding_dim]

    return CustomEncoderComponents(model=model, preprocess=preprocess, forward_fn=forward)


def register_feature_extractors(registry, device, dtype, num_workers):
    register_custom_encoder(
        registry=registry,
        name="my_encoder",
        embedding_dim=512,
        loader=build_my_encoder,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
    )
```

Run AtlasPatch with `--feature-plugin /path/to/plugin.py --feature-extractors my_encoder` to benchmark your encoder alongside the built-ins, multiple plugins and extractors can be added at once. Outputs keep the same HDF5 layout—your custom embeddings live under `features/my_encoder` (row-aligned with `coords`) next to other extractors.

## SLURM job scripts

We prepared ready-to-run SLURM templates under `jobs/`:

- Patch extraction (SAM2 + H5/PNG): `jobs/atlaspatch_patch.slurm.sh`. Edits to make:
  - Set `WSI_ROOT`, `OUTPUT_ROOT`, `PATCH_SIZE`, `TARGET_MAG`, `SEG_BATCH`.
  - Ensure `--cpus-per-task` matches the CPU you want; the script passes `--patch-workers ${SLURM_CPUS_PER_TASK}` and caps `--max-open-slides` at 200.
  - `--fast-mode` is on by default; append `--no-fast-mode` to enable content filtering.
  - Submit with `sbatch jobs/atlaspatch_patch.slurm.sh`.
- Feature embedding (adds features into existing H5 files): `jobs/atlaspatch_features.slurm.sh`. Edits to make:
  - Set `WSI_ROOT`, `OUTPUT_ROOT`, `PATCH_SIZE`, and `TARGET_MAG`.
  - Configure `FEATURES` (comma/space list, multiple extractors are supported), `FEATURE_DEVICE`, `FEATURE_BATCH`, `FEATURE_WORKERS`, and `FEATURE_PRECISION`.
  - This script is intended for feature extraction; use the patch script when you need segmentation + coordinates, and run the feature script to embed one or more models into those H5 files.
  - Submit with `sbatch jobs/atlaspatch_features.slurm.sh`.
- Running multiple jobs: you can submit several jobs in a loop (e.g., 50 job using `for i in {1..50}; do sbatch jobs/atlaspatch_features.slurm.sh; done`). AtlasPatch uses per-slide lock files to avoid overlapping work on the same slide.

## Feedback

- Report problems via the [bug report template](https://github.com/AtlasAnalyticsLab/AtlasPatch/issues/new?template=bug_report.md) so we can reproduce and fix them quickly.
- Suggest enhancements through the [feature request template](https://github.com/AtlasAnalyticsLab/AtlasPatch/issues/new?template=feature_request.md) with your use case and proposal.
- When opening a PR, fill out the [pull request template](.github/pull_request_template.md) and run the listed checks (lint, format, type-check, tests).

## Citation

If you use AtlasPatch in your research, please cite it:

```
@software{atlaspatch,
  author  = {Atlas Analytics Lab},
  title   = {AtlasPatch},
  year    = {2025},
  url     = {https://github.com/AtlasAnalyticsLab/AtlasPatch}
}
```

## License

AtlasPatch is licensed under the **PolyForm Noncommercial License 1.0.0**, which strictly prohibits commercial use of this software or any derivative works. This applies to all forms of commercialization, including selling the software, offering it as a commercial service, using it in commercial products, or creating forked versions for commercial purposes. However, the license explicitly permits use for research, experimentation, and non-commercial purposes. Personal use for research, hobby projects, and educational purposes is allowed, as is use by academic institutions, educational organizations, public research organizations, and non-profit entities regardless of their funding sources. If you wish to use AtlasPatch commercially, you must obtain a separate commercial license from the authors. For the complete license text and detailed terms, see the [LICENSE](./LICENSE) file in this repository.

# Future Updates

## Slide Encoders
- We plan to add slide-level encoders (open for extension): TITAN, PRISM, GigaPath, Madeleine.

## Deployment
- Streamlined install and packaging flows (e.g., `pip install atlas_patch` with `uv` support).
