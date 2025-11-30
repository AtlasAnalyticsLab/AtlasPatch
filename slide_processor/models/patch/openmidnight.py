from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "SophontAI/OpenMidnight"
_EMB_DIM = 1536


def _build_preprocess():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class OpenMidnight(PatchFeatureExtractor):
    """OpenMidnight encoder from \"How to Train a State-of-the-Art Pathology Foundation Model with $1.6k\" (https://sophontai.com/blog/openmidnight)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from huggingface_hub import hf_hub_download

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            # Load the base dinov2 model
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitg14_reg", weights=None
            )

            # Download OpenMidnight weights
            download_location = hf_hub_download(
                repo_id=_MODEL_ID, filename="teacher_checkpoint_load.pt"
            )

            # Load OpenMidnight weights
            checkpoint = torch.load(download_location, map_location="cpu")

            # Handle positional embedding interpolation from 392 to 224 resolution
            pos_embed = checkpoint["pos_embed"]
            model.pos_embed = torch.nn.parameter.Parameter(pos_embed)
            model.load_state_dict(checkpoint)

        except Exception as e:
            msg = (
                f"Failed to load OpenMidnight ({_MODEL_ID}). "
                "Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess()

        super().__init__(
            name="openmidnight",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


def register_openmidnight_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "openmidnight",
        lambda: OpenMidnight(device=device, dtype=dtype, num_workers=num_workers),
    )
