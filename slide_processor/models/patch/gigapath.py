from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "hf_hub:prov-gigapath/prov-gigapath"
_EMB_DIM = 1536


def _build_preprocess():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class ProvGigaPathExtractor(PatchFeatureExtractor):
    """Prov-GigaPath encoder from \"A whole-slide foundation model for digital pathology from real-world data\" (https://www.nature.com/articles/s41586-024-07441-w)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = timm.create_model(_MODEL_ID, pretrained=True)
        except Exception as e:
            msg = (
                f"Failed to load Prov-GigaPath ({_MODEL_ID}) via timm. "
                "Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess()

        super().__init__(
            name="prov_gigapath",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


def register_prov_gigapath_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "prov_gigapath",
        lambda: ProvGigaPathExtractor(device=device, dtype=dtype, num_workers=num_workers),
    )
