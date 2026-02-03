# Changelog

All notable changes to AtlasPatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-03

### Added

#### Core Pipeline
- SAM2-based tissue segmentation finetuned on ~35,000 diverse WSI thumbnails
- Four-checkpoint pipeline architecture:
  - `detect-tissue`: tissue detection with mask visualization
  - `segment-and-get-coords`: segmentation + patch coordinate extraction
  - `process`: full pipeline with feature embedding
  - `--save-images`: optional patch image export
- HDF5 output format with coordinates (`coords`) and features (`features/<encoder>`)
- Slide passport metadata in H5 files (vendor, MPP, staining info)
- Per-slide lock files for safe parallel job execution
- MPP override via CSV for slides with missing/incorrect metadata

#### Feature Extractors (66)
- **Natural image backbones**: ResNet (18/34/50/101/152), ConvNeXt (tiny/small/base/large), ViT (B/L/H)
- **DINOv2**: small, base, large, giant
- **DINOv3**: vits16, vitb16, vitl16, vith16_plus, vit7b16 (+ SAT variants)
- **Pathology encoders**: UNI v1/v2, Phikon v1/v2, Virchow v1/v2, GigaPath, CHIEF-CTransPath, Midnight, OpenMidnight, MUSK, PathOrchestra, H-optimus-0/1, H0-mini, CONCH v1/v1.5, Hibou B/L
- **Lunit models**: ResNet50 (BT/SwAV/MoCov2), ViT-S (patch16/patch8 DINO)
- **CLIP variants**: RN50/101/50x4/50x16/50x64, ViT-B/L
- **Medical CLIP**: PLIP, MedSigLIP, Quilt (B-32/B-16/B-16-PMB), BiomedCLIP, OmiCLIP
- Custom encoder plugin system via `--feature-plugin`

#### Visualization
- Tissue mask overlays
- Contour visualization
- Patch grid overlays
- Configurable output directory structure

#### CLI & Configuration
- `atlaspatch` CLI with subcommands
- Configurable patch size, target magnification, step size
- Device selection for segmentation and feature extraction
- Batch size controls for segmentation and feature extraction
- Precision options (float32, float16, bfloat16)
- Fast mode (default) for high-throughput processing
- Content filtering options (white/black thresholds)
- Recursive directory processing
- Skip existing / force overwrite modes

#### HPC Support
- SLURM job script templates (`jobs/atlaspatch_patch.slurm.sh`, `jobs/atlaspatch_features.slurm.sh`)
- Multi-GPU support via separate device flags
- Configurable worker counts for parallel processing

#### Documentation
- Comprehensive README with usage examples
- Pipeline checkpoint diagrams
- Feature extractor reference table
- FAQ section for common issues
- Issue templates (bug report, feature request)
- Pull request template

### Fixed
- Contour scaling from mask resolution to thumbnail
- H-optimus transformation pipeline
- MUSK model registration

[1.0.0]: https://github.com/AtlasAnalyticsLab/AtlasPatch/releases/tag/v1.0.0
