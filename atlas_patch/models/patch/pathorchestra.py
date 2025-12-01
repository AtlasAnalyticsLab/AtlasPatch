from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "hf-hub:AI4Pathology/PathOrchestra"
_EMB_DIM = 512  # TODO: verify this


class PathOrchestraEncoder(PatchFeatureExtractor):
    """PathOrchestra.

    Paper: "PathOrchestra: A comprehensive foundation model for computational pathology with over 100 diverse clinical-grade tasks"
    https://arxiv.org/abs/2503.24345
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm
        from torchvision import transforms

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = timm.create_model(
                _MODEL_ID,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        except Exception as e:
            msg = (
                "Failed to load PathOrchestra (AI4Pathology/PathOrchestra) via timm. "
                "Ensure HuggingFace access (token for gated weights) and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        super().__init__(
            name="pathorchestra",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=transform,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


def register_pathorchestra_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "pathorchestra",
        lambda: PathOrchestraEncoder(device=device, dtype=dtype, num_workers=num_workers),
    )
