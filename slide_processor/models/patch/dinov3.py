from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_DINOV3_SPECS = [
    ("dinov3_vits16", "facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    ("dinov3_vits16_plus", "facebook/dinov3-vits16plus-pretrain-lvd1689m", 384),
    ("dinov3_vitb16", "facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
    ("dinov3_vitl16", "facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
    ("dinov3_vitl16_sat", "facebook/dinov3-vitl16-pretrain-sat493m", 1024),
    ("dinov3_vith16_plus", "facebook/dinov3-vith16plus-pretrain-lvd1689m", 1280),
    ("dinov3_vit7b16", "facebook/dinov3-vit7b16-pretrain-lvd1689m", 4096),
    ("dinov3_vit7b16_sat", "facebook/dinov3-vit7b16-pretrain-sat493m", 4096),
]


def _build_preprocess(processor):
    def _preprocess(pil_img):
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return _preprocess


class DINOv3Encoder(PatchFeatureExtractor):
    """DINOv3 ViT patch encoder from \"DINOv3\" (https://arxiv.org/abs/2508.10104)."""

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

        try:
            processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            model = AutoModel.from_pretrained(model_id)
        except Exception as e:
            msg = (
                f"Failed to load DINOv3 backbone '{model_id}'. Ensure you have accepted the model "
                "license on Hugging Face and that cache permissions are available."
            )
            raise RuntimeError(msg) from e

        actual_dim = int(getattr(model.config, "hidden_size", embedding_dim))
        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(processor)

        def _forward(x, m=model):
            outputs = m(pixel_values=x)
            return outputs.pooler_output

        super().__init__(
            name=name,
            model=model,
            embedding_dim=actual_dim,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_dinov3_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, model_id, emb_dim in _DINOV3_SPECS:
        registry.register(
            name,
            lambda n=name, mid=model_id, ed=emb_dim: DINOv3Encoder(
                name=n,
                model_id=mid,
                embedding_dim=ed,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
