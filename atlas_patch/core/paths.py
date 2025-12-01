from __future__ import annotations

from pathlib import Path

from atlas_patch.core.config import ExtractionConfig, OutputConfig
from atlas_patch.core.models import Slide


def build_run_root(output_cfg: OutputConfig, extraction_cfg: ExtractionConfig) -> Path:
    """Return the root output directory for a run.
    """
    return output_cfg.output_root


def patch_h5_path(slide: Slide, output_cfg: OutputConfig, extraction_cfg: ExtractionConfig) -> Path:
    run_root = build_run_root(output_cfg, extraction_cfg)
    return run_root / "patches" / f"{slide.stem}.h5"


def find_existing_patch(
    slide: Slide, output_cfg: OutputConfig, extraction_cfg: ExtractionConfig
) -> Path | None:
    """Check for existing patch outputs at the current layout."""
    path = patch_h5_path(slide, output_cfg, extraction_cfg)
    return path if path.exists() else None


def images_dir(slide: Slide, output_cfg: OutputConfig, extraction_cfg: ExtractionConfig) -> Path:
    run_root = build_run_root(output_cfg, extraction_cfg)
    return run_root / "images" / slide.stem


def visualization_dir(output_cfg: OutputConfig, extraction_cfg: ExtractionConfig) -> Path:
    run_root = build_run_root(output_cfg, extraction_cfg)
    return run_root / "visualization"


def patch_lock_path(
    slide: Slide, output_cfg: OutputConfig, extraction_cfg: ExtractionConfig
) -> Path:
    run_root = build_run_root(output_cfg, extraction_cfg)
    return run_root / "patches" / f"{slide.stem}.lock"
