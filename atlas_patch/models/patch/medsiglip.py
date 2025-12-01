from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "google/medsiglip-448"
_EMB_DIM = 1152


class MedSigLip(PatchFeatureExtractor):
    """MedSigLip: SigLip variant for medical image analysis (image tower only).

    Paper: "MedGemma Technical Report"
    https://arxiv.org/abs/2507.05201
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = AutoModel.from_pretrained(_MODEL_ID)
            processor = AutoProcessor.from_pretrained(_MODEL_ID, use_fast=True)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to load MedSigLip ({_MODEL_ID}). Ensure HuggingFace access and cache permissions."
            raise RuntimeError(msg) from e

        if not hasattr(model, "get_image_features"):
            raise RuntimeError(
                f"Loaded model '{_MODEL_ID}' does not expose get_image_features; cannot be used for patch embeddings."
            )

        model = model.to(device=self.device, dtype=self.dtype).eval()

        def _preprocess(pil_img):
            inputs = processor(images=pil_img, padding="max_length", return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        super().__init__(
            name="medsiglip",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=_preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=lambda x, m=model: m.get_image_features(pixel_values=x),
        )


def register_medsiglip_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "medsiglip",
        lambda: MedSigLip(device=device, dtype=dtype, num_workers=num_workers),
    )
