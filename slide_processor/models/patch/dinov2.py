from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_DINOV2_SPECS = [
    ("dinov2_small", "facebook/dinov2-small", 384),
    ("dinov2_base", "facebook/dinov2-base", 768),
    ("dinov2_large", "facebook/dinov2-large", 1024),
    ("dinov2_giant", "facebook/dinov2-giant", 1536),
]


def _build_preprocess(processor):
    def _preprocess(pil_img):
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return _preprocess


class DinoV2Encoder(PatchFeatureExtractor):
    """DINOv2 patch encoder from \"DINOv2: Learning Robust Visual Features without Supervision\" (https://arxiv.org/abs/2304.07193)."""

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
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
        except Exception as e:  # noqa: BLE001
            msg = (
                f"Failed to load DINOv2 backbone '{model_id}'. Ensure HuggingFace access/token and "
                "cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(processor)

        def _forward(x, m=model):
            outputs = m(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]

        super().__init__(
            name=name,
            model=model,
            embedding_dim=embedding_dim,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_dinov2_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, model_id, emb_dim in _DINOV2_SPECS:
        registry.register(
            name,
            lambda n=name, mid=model_id, ed=emb_dim: DinoV2Encoder(
                name=n,
                model_id=mid,
                embedding_dim=ed,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
            ),
        )
