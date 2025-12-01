from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_LUNIT_SPECS = [
    ("lunit_resnet50_bt", "hf-hub:1aurent/resnet50.lunit_bt", 2048),
    ("lunit_resnet50_swav", "hf-hub:1aurent/resnet50.lunit_swav", 2048),
    ("lunit_resnet50_mocov2", "hf-hub:1aurent/resnet50.lunit_mocov2", 2048),
    ("lunit_vit_small_patch16_dino", "hf-hub:1aurent/vit_small_patch16_224.lunit_dino", 384),
    ("lunit_vit_small_patch8_dino", "hf-hub:1aurent/vit_small_patch8_224.lunit_dino", 384),
]


class LunitEncoder(PatchFeatureExtractor):
    """Pathology-specific encoders from Lunit Inc (Benchmarking Self-Supervised Learning on Diverse Pathology Datasets).

    Includes ResNet-50 checkpoints trained with Barlow Twins, SwAV, and MoCo v2 plus ViT-Small DINOv2
    models (patch16/patch8), all hosted on Hugging Face under `1aurent/*` for pathology imagery.
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        fallback_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))
        self.model_id = model_id

        try:
            model = timm.create_model(
                model_id,
                pretrained=True,
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                f"Failed to load Lunit encoder '{model_id}' via timm. "
                "Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        emb_dim = int(getattr(model, "num_features", fallback_dim))

        super().__init__(
            name=name,
            model=model,
            embedding_dim=emb_dim,
            preprocess=transform,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


def register_lunit_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, model_id, emb_dim in _LUNIT_SPECS:
        registry.register(
            name,
            lambda n=name, mid=model_id, d=emb_dim: LunitEncoder(
                name=n,
                model_id=mid,
                fallback_dim=d,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
