from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)


class PLIPExtractor(PatchFeatureExtractor):
    """Pathology Language and Image Pre-Training (PLIP) extractor via HuggingFace.

    Paper: "A visual-language foundation model for pathology image analysis using medical Twitter"
    https://github.com/PathologyFoundation/plip
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            self.model = CLIPModel.from_pretrained("vinid/plip")
            self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        except Exception as e:
            msg = (
                "Failed to load PLIP (vinid/plip). Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = self.model.to(device=self.device, dtype=self.dtype).eval()
        emb_dim = 512

        preprocess = self._build_preprocess()

        super().__init__(
            name="plip",
            model=model,
            embedding_dim=emb_dim,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=lambda x, m=model: m.get_image_features(pixel_values=x),
        )

    def _build_preprocess(self):
        processor = self.processor

        def _preprocess(pil_img):
            inputs = processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return _preprocess


def register_plip_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "plip",
        lambda: PLIPExtractor(device=device, dtype=dtype, num_workers=num_workers),
    )
