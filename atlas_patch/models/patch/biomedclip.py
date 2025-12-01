from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_MODEL_NAME = f"hf-hub:{_MODEL_ID}"
_EMB_DIM = 512


class BioMedCLIPExtractor(PatchFeatureExtractor):
    """BioMedCLIP (PubMedBERT-ViT-B/16) patch encoder for medical images.

    Paper: "BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs"
    https://aka.ms/biomedclip-paper
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import open_clip

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model, preprocess = open_clip.create_model_from_pretrained(_MODEL_NAME)
        except Exception as e:
            msg = (
                "Failed to load BioMedCLIP "
                f"({_MODEL_ID}). Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        if not hasattr(model, "encode_image"):
            raise RuntimeError(
                f"Loaded model '{_MODEL_ID}' does not expose encode_image; cannot be used for patch embeddings."
            )

        model = model.to(device=self.device, dtype=self.dtype).eval()
        emb_dim = 512

        super().__init__(
            name="biomedclip",
            model=model,
            embedding_dim=emb_dim,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=lambda x, m=model: m.encode_image(x),
        )


def register_biomedclip_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "biomedclip",
        lambda: BioMedCLIPExtractor(device=device, dtype=dtype, num_workers=num_workers),
    )
