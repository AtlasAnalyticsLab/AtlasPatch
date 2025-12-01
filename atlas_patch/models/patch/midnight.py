from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "kaiko-ai/midnight"


def _build_preprocess():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class Midnight(PatchFeatureExtractor):
    """Midnight encoder from \"Training state-of-the-art pathology foundation models with orders of magnitude less data\" (https://arxiv.org/abs/2504.05186)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from transformers import AutoModel

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = AutoModel.from_pretrained(_MODEL_ID)
        except Exception as e:
            msg = (
                f"Failed to load Midnight ({_MODEL_ID}). Ensure HuggingFace access/token and cache "
                "permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess()
        emb_dim = 3072

        def _forward(x, m=model):
            outputs = m(x)
            cls_token = outputs.last_hidden_state[:, 0, :]
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            return torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)

        super().__init__(
            name="midnight",
            model=model,
            embedding_dim=emb_dim,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_midnight_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "midnight",
        lambda: Midnight(device=device, dtype=dtype, num_workers=num_workers),
    )
