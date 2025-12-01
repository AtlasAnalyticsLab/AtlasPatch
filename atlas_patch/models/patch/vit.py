from __future__ import annotations

import torch
from torchvision import models

from atlas_patch.models.patch.base import build_torchvision_extractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

_VIT_SPECS = [
    ("vit_b_16", models.vit_b_16, models.ViT_B_16_Weights, 768),
    ("vit_b_32", models.vit_b_32, models.ViT_B_32_Weights, 768),
    ("vit_l_16", models.vit_l_16, models.ViT_L_16_Weights, 1024),
    ("vit_l_32", models.vit_l_32, models.ViT_L_32_Weights, 1024),
    ("vit_h_14", models.vit_h_14, models.ViT_H_14_Weights, 1280),
]


def register_vits(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, ctor, weights_enum, emb_dim in _VIT_SPECS:
        registry.register(
            name,
            lambda n=name, c=ctor, w=weights_enum, d=emb_dim: build_torchvision_extractor(
                name=n,
                model_ctor=c,
                weights_enum=w,
                head_attr="heads",
                embedding_dim=d,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
