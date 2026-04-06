from __future__ import annotations

import os
from pathlib import Path

from atlas_patch.core.config import ExtractionConfig, OutputConfig
from atlas_patch.core.models import Slide

PATCH_FEATURE_GROUP = "features"
SLIDE_FEATURE_GROUP = "slide_features"
PATIENT_FEATURES_DIRNAME = "patient_features"


def normalize_encoder_name(name: str) -> str:
    value = str(name).strip().lower()
    if not value:
        raise ValueError("encoder name must be a non-empty string")
    return value


def _normalize_dataset_component(name: str, *, prefix: str) -> str:
    value = str(name).strip()
    if value.lower().startswith(f"{prefix}/"):
        value = value.split("/", 1)[1]
    return normalize_encoder_name(value)


def _validate_output_stem(value: str, *, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    if cleaned in {".", ".."}:
        raise ValueError(f"{field_name} cannot be '.' or '..'")
    separators = [os.sep]
    if os.altsep is not None:
        separators.append(os.altsep)
    if any(sep in cleaned for sep in separators):
        raise ValueError(f"{field_name} cannot contain path separators")
    return cleaned


def build_run_root(output_cfg: OutputConfig, extraction_cfg: ExtractionConfig) -> Path:
    """Return the root output directory for a run.
    """
    return output_cfg.output_root


def patch_h5_path(slide: Slide, output_cfg: OutputConfig, extraction_cfg: ExtractionConfig) -> Path:
    run_root = build_run_root(output_cfg, extraction_cfg)
    return run_root / "patches" / f"{slide.stem}.h5"


def patch_feature_dataset_key(encoder_name: str) -> str:
    return f"{PATCH_FEATURE_GROUP}/{_normalize_dataset_component(encoder_name, prefix=PATCH_FEATURE_GROUP)}"


def slide_feature_dataset_key(encoder_name: str) -> str:
    return f"{SLIDE_FEATURE_GROUP}/{_normalize_dataset_component(encoder_name, prefix=SLIDE_FEATURE_GROUP)}"


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


def slide_append_lock_path(
    slide: Slide, output_cfg: OutputConfig, extraction_cfg: ExtractionConfig
) -> Path:
    """Serialize all writes against the canonical per-slide patch H5."""
    return patch_lock_path(slide, output_cfg, extraction_cfg)


def patient_features_dir(output_cfg: OutputConfig) -> Path:
    return output_cfg.output_root / PATIENT_FEATURES_DIRNAME


def patient_encoder_dir(output_cfg: OutputConfig, encoder_name: str) -> Path:
    return patient_features_dir(output_cfg) / normalize_encoder_name(encoder_name)


def patient_embedding_path(output_cfg: OutputConfig, encoder_name: str, case_id: str) -> Path:
    case_stem = _validate_output_stem(case_id, field_name="case_id")
    return patient_encoder_dir(output_cfg, encoder_name) / f"{case_stem}.h5"


def patient_lock_path(output_cfg: OutputConfig, encoder_name: str, case_id: str) -> Path:
    case_stem = _validate_output_stem(case_id, field_name="case_id")
    return patient_encoder_dir(output_cfg, encoder_name) / f"{case_stem}.lock"
