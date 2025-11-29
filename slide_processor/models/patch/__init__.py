from __future__ import annotations

import torch

from slide_processor.models.patch.biomedclip import register_biomedclip_model
from slide_processor.models.patch.clip import register_openai_clip_models
from slide_processor.models.patch.convnext import register_convnexts
from slide_processor.models.patch.hibou import register_hibou_models
from slide_processor.models.patch.hoptimus import register_hoptimus_models
from slide_processor.models.patch.lunit import register_lunit_models
from slide_processor.models.patch.medsiglip import register_medsiglip_model
from slide_processor.models.patch.pathorchestra import register_pathorchestra_model
from slide_processor.models.patch.plip import register_plip_model
from slide_processor.models.patch.quilt import register_quilt_models
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry
from slide_processor.models.patch.resnet import register_resnets
from slide_processor.models.patch.uni import register_uni_models
from slide_processor.models.patch.vit import register_vits

__all__ = ["PatchFeatureExtractorRegistry", "build_default_registry"]


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
    register_openai_clip_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_quilt_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_uni_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_lunit_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_plip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_medsiglip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_pathorchestra_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_hoptimus_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_hibou_models(registry, device=dev, num_workers=num_workers, dtype=dtype)
    register_biomedclip_model(registry, device=dev, num_workers=num_workers, dtype=dtype)
    return registry
