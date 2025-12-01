from __future__ import annotations

import torch
from torchvision import models

from slide_processor.models.patch.base import build_torchvision_extractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

_CONVNEXT_SPECS = [
    ("convnext_tiny", models.convnext_tiny, models.ConvNeXt_Tiny_Weights, 768),
    ("convnext_small", models.convnext_small, models.ConvNeXt_Small_Weights, 768),
    ("convnext_base", models.convnext_base, models.ConvNeXt_Base_Weights, 1024),
    ("convnext_large", models.convnext_large, models.ConvNeXt_Large_Weights, 1536),
]


def register_convnexts(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, ctor, weights_enum, emb_dim in _CONVNEXT_SPECS:
        registry.register(
            name,
            lambda n=name, c=ctor, w=weights_enum, d=emb_dim: build_torchvision_extractor(
                name=n,
                model_ctor=c,
                weights_enum=w,
                head_attr="classifier",
                embedding_dim=d,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
