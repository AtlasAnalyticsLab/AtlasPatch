from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_PHIKON_DIM = 768
_PHIKON_V2_DIM = 1024


def _build_preprocess(processor):
    def _preprocess(pil_img):
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return _preprocess


class Phikon(PatchFeatureExtractor):
    """Phikon encoder from \"Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling\" (https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from transformers import AutoImageProcessor, ViTModel

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = ViTModel.from_pretrained(
                "owkin/phikon",
                add_pooling_layer=False,
            )
            processor = AutoImageProcessor.from_pretrained("owkin/phikon", use_fast=True)
        except Exception as e:
            msg = (
                "Failed to load Phikon (owkin/phikon). Ensure HuggingFace access/token and cache "
                "permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(processor)

        def _forward(x, m=model):
            outputs = m(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]

        super().__init__(
            name="phikon_v1",
            model=model,
            embedding_dim=_PHIKON_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


class PhikonV2(PatchFeatureExtractor):
    """Phikon-v2 encoder from \"Phikon-v2, A large and public feature extractor for biomarker prediction\" (https://arxiv.org/abs/2409.09173)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = AutoModel.from_pretrained(
                "owkin/phikon-v2",
            )
            processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2", use_fast=True)
        except Exception as e:  # noqa: BLE001
            msg = (
                "Failed to load Phikon-v2 (owkin/phikon-v2). Ensure HuggingFace access/token and "
                "cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(processor)

        def _forward(x, m=model):
            outputs = m(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]

        super().__init__(
            name="phikon_v2",
            model=model,
            embedding_dim=_PHIKON_V2_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_phikon_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "phikon_v1",
        lambda: Phikon(
            device=device,
            dtype=dtype,
            num_workers=num_workers,
        ),
    )
    registry.register(
        "phikon_v2",
        lambda: PhikonV2(
            device=device,
            dtype=dtype,
            num_workers=num_workers,
        ),
    )
