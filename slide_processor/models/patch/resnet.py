from __future__ import annotations

import torch
from torchvision import models

from slide_processor.models.patch.base import build_torchvision_extractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

_RESNET_SPECS = [
    ("resnet18", models.resnet18, models.ResNet18_Weights, 512),
    ("resnet34", models.resnet34, models.ResNet34_Weights, 512),
    ("resnet50", models.resnet50, models.ResNet50_Weights, 2048),
    ("resnet101", models.resnet101, models.ResNet101_Weights, 2048),
    ("resnet152", models.resnet152, models.ResNet152_Weights, 2048),
]


def register_resnets(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, ctor, weights_enum, emb_dim in _RESNET_SPECS:
        registry.register(
            name,
            lambda n=name, c=ctor, w=weights_enum, d=emb_dim: build_torchvision_extractor(
                name=n,
                model_ctor=c,
                weights_enum=w,
                head_attr="fc",
                embedding_dim=d,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
