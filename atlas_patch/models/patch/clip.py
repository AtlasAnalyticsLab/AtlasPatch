from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

# (public name, open_clip model id, pretrained tag, expected dim)
_OPENAI_CLIP_SPECS = [
    ("clip_rn50", "RN50", "openai", 1024),
    ("clip_rn101", "RN101", "openai", 512),
    ("clip_rn50x4", "RN50x4", "openai", 640),
    ("clip_rn50x16", "RN50x16", "openai", 768),
    ("clip_rn50x64", "RN50x64", "openai", 1024),
    ("clip_vit_b_32", "ViT-B-32", "openai", 512),
    ("clip_vit_b_16", "ViT-B-16", "openai", 512),
    ("clip_vit_l_14", "ViT-L-14", "openai", 768),
    ("clip_vit_l_14_336", "ViT-L-14-336", "openai", 768),
]


def _build_openai_clip_extractor(
    *,
    name: str,
    model_name: str,
    pretrained_tag: str,
    expected_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int,
) -> PatchFeatureExtractor:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_tag
    )
    model = model.to(device=device, dtype=dtype).eval()
    # open_clip exposes the projection size as visual.output_dim; fall back to text projection.
    emb_dim = int(getattr(model.visual, "output_dim", expected_dim))
    if emb_dim != expected_dim:
        logger.debug(
            "CLIP extractor '%s' using embed dim %s (expected %s)", name, emb_dim, expected_dim
        )

    return PatchFeatureExtractor(
        name=name,
        model=model,
        embedding_dim=emb_dim,
        preprocess=preprocess,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
        non_blocking=True,
        forward_fn=lambda x, m=model: m.encode_image(x),
    )


def register_openai_clip_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, model_name, tag, emb_dim in _OPENAI_CLIP_SPECS:
        registry.register(
            name,
            lambda n=name, mn=model_name, t=tag, d=emb_dim: _build_openai_clip_extractor(
                name=n,
                model_name=mn,
                pretrained_tag=t,
                expected_dim=d,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
