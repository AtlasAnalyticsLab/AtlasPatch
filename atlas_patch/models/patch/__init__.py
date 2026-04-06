from __future__ import annotations

from importlib import import_module

import torch

from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry
from atlas_patch.models.patch.custom import (
    CustomEncoderComponents,
    CustomEncoderLoader,
    register_custom_encoder,
    register_feature_extractors_from_module,
)

__all__ = [
    "PatchFeatureExtractorRegistry",
    "build_default_registry",
    "CustomEncoderComponents",
    "CustomEncoderLoader",
    "register_custom_encoder",
    "register_feature_extractors_from_module",
]

_REGISTRARS: tuple[tuple[str, str], ...] = (
    ("atlas_patch.models.patch.resnet", "register_resnets"),
    ("atlas_patch.models.patch.convnext", "register_convnexts"),
    ("atlas_patch.models.patch.vit", "register_vits"),
    ("atlas_patch.models.patch.dinov2", "register_dinov2_models"),
    ("atlas_patch.models.patch.dinov3", "register_dinov3_models"),
    ("atlas_patch.models.patch.clip", "register_openai_clip_models"),
    ("atlas_patch.models.patch.conch", "register_conch_models"),
    ("atlas_patch.models.patch.omiclip", "register_omiclip_model"),
    ("atlas_patch.models.patch.quilt", "register_quilt_models"),
    ("atlas_patch.models.patch.uni", "register_uni_models"),
    ("atlas_patch.models.patch.lunit", "register_lunit_models"),
    ("atlas_patch.models.patch.plip", "register_plip_model"),
    ("atlas_patch.models.patch.medsiglip", "register_medsiglip_model"),
    ("atlas_patch.models.patch.musk", "register_musk_model"),
    ("atlas_patch.models.patch.openmidnight", "register_openmidnight_model"),
    ("atlas_patch.models.patch.pathorchestra", "register_pathorchestra_model"),
    ("atlas_patch.models.patch.hoptimus", "register_hoptimus_models"),
    ("atlas_patch.models.patch.hibou", "register_hibou_models"),
    ("atlas_patch.models.patch.biomedclip", "register_biomedclip_model"),
    ("atlas_patch.models.patch.phikon", "register_phikon_models"),
    ("atlas_patch.models.patch.virchow", "register_virchow_models"),
    ("atlas_patch.models.patch.gigapath", "register_prov_gigapath_model"),
    ("atlas_patch.models.patch.midnight", "register_midnight_model"),
    ("atlas_patch.models.patch.chief_ctranspath", "register_chief_ctranspath_model"),
)


def build_default_registry(
    *,
    device: str | torch.device = "cuda",
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
) -> PatchFeatureExtractorRegistry:
    """Factory that registers the built-in extractors."""
    dev = torch.device(device)
    registry = PatchFeatureExtractorRegistry()
    for module_name, registrar_name in _REGISTRARS:
        module = import_module(module_name)
        registrar = getattr(module, registrar_name)
        registrar(registry, device=dev, num_workers=num_workers, dtype=dtype)
    return registry
