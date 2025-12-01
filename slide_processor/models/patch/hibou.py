from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_HIBOU_SPECS = [
    ("hibou_b", "histai/hibou-B", 768),
    ("hibou_l", "histai/hibou-L", 1024),
]


def _build_preprocess(processor):
    def _preprocess(pil_img):
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return _preprocess


class HibouEncoder(PatchFeatureExtractor):
    """Hibou DINOv2-based pathology vision encoders (Base/Large).

    Paper: "Hibou: A Family of Foundational Vision Transformers for Pathology"
    https://arxiv.org/abs/2406.05074
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        embedding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))
        self.model_id = model_id

        try:
            processor = AutoImageProcessor.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            msg = f"Failed to load Hibou model '{model_id}'. Ensure HuggingFace access/token and cache permissions."
            raise RuntimeError(msg) from e

        if not hasattr(model, "forward"):
            raise RuntimeError(
                f"Loaded Hibou model '{model_id}' does not expose a forward method for feature extraction."
            )

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(processor)

        def _forward(x, m=model):
            outputs = m(pixel_values=x)
            return outputs.pooler_output

        super().__init__(
            name=name,
            model=model,
            embedding_dim=int(embedding_dim),
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_hibou_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, model_id, emb_dim in _HIBOU_SPECS:
        registry.register(
            name,
            lambda n=name, mid=model_id, d=emb_dim: HibouEncoder(
                name=n,
                model_id=mid,
                embedding_dim=d,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
