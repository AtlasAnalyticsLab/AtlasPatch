from __future__ import annotations

import torch

from atlas_patch.models.patch.biomedclip import register_biomedclip_model
from atlas_patch.models.patch.clip import register_openai_clip_models
from atlas_patch.models.patch.conch import register_conch_models
from atlas_patch.models.patch.convnext import register_convnexts
from atlas_patch.models.patch.dinov2 import register_dinov2_models
from atlas_patch.models.patch.dinov3 import register_dinov3_models
from atlas_patch.models.patch.gigapath import register_prov_gigapath_model
from atlas_patch.models.patch.hibou import register_hibou_models
from atlas_patch.models.patch.hoptimus import register_hoptimus_models
from atlas_patch.models.patch.lunit import register_lunit_models
from atlas_patch.models.patch.medsiglip import register_medsiglip_model
from atlas_patch.models.patch.midnight import register_midnight_model
from atlas_patch.models.patch.musk import register_musk_model
from atlas_patch.models.patch.omiclip import register_omiclip_model
from atlas_patch.models.patch.openmidnight import register_openmidnight_model
from atlas_patch.models.patch.pathorchestra import register_pathorchestra_model
from atlas_patch.models.patch.phikon import register_phikon_models
from atlas_patch.models.patch.plip import register_plip_model
from atlas_patch.models.patch.quilt import register_quilt_models
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry
from atlas_patch.models.patch.resnet import register_resnets
from atlas_patch.models.patch.uni import register_uni_models
from atlas_patch.models.patch.virchow import register_virchow_models
from atlas_patch.models.patch.vit import register_vits
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


def build_default_registry(
    *,
    device: str | torch.device = "cuda",
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
) -> PatchFeatureExtractorRegistry:
    """Factory that registers the built-in extractors."""
    dev = torch.device(device)
    registry = PatchFeatureExtractorRegistry()
    register_resnets(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_convnexts(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_vits(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_dinov2_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_dinov3_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_openai_clip_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_conch_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_omiclip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_quilt_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_uni_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_lunit_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_plip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_medsiglip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_musk_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_openmidnight_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_pathorchestra_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_hoptimus_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_hibou_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_biomedclip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_phikon_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_virchow_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_prov_gigapath_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_midnight_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    return registry
