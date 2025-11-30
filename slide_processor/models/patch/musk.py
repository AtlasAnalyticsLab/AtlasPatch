from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "xiangjx/musk"
_EMB_DIM = 1024


def _build_preprocess():
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(384, interpolation=3, antialias=True),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )


class MUSK(PatchFeatureExtractor):
    """MUSK encoder from \"MUSK: A Vision-Language Foundation Model for Precision Oncology\" (https://www.nature.com/articles/s41586-024-08378-w)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from musk import modeling, utils
        from timm.models import create_model

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        except Exception as e:
            msg = (
                f"Failed to load MUSK ({_MODEL_ID}) via timm and musk utils. "
                "Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess()

        def _forward(x, m=model):
            with torch.inference_mode():
                image_embeddings = m(
                    image=x,
                    with_head=False,
                    out_norm=False,
                    ms_aug=True,
                    return_global=True,
                )[0]
            return image_embeddings

        super().__init__(
            name="musk",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_musk_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "musk",
        lambda: MUSK(device=device, dtype=dtype, num_workers=num_workers),
    )
