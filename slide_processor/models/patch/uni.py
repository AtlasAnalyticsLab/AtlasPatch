from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)


class UNIV1(PatchFeatureExtractor):
    """UNI (MahmoodLab/UNI) feature extractor using timm hub weights."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            self.model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
                num_classes=0,
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                "Failed to load UNI (MahmoodLab/uni) via timm. "
                "Ensure you have access (HuggingFace token) and internet/cache permissions."
            )
            raise RuntimeError(msg) from e

        model = self.model.to(device=self.device, dtype=self.dtype).eval()
        cfg = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**cfg)
        emb_dim = 1024

        super().__init__(
            name="uni",
            model=model,
            embedding_dim=emb_dim,
            preprocess=transform,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


class UNIV2(PatchFeatureExtractor):
    """UNI2-h (MahmoodLab/UNI2-h) feature extractor using timm hub weights."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        try:
            self.model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h",
                pretrained=True,
                **timm_kwargs,
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                "Failed to load UNI2-h (MahmoodLab/UNI2-h) via timm. "
                "Ensure you have access (HuggingFace token) and internet/cache permissions."
            )
            raise RuntimeError(msg) from e

        model = self.model.to(device=self.device, dtype=self.dtype).eval()
        cfg = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**cfg)
        logger.debug("UNI2-h transform: %s", transform)
        emb_dim = 1536

        super().__init__(
            name="uni2_h",
            model=model,
            embedding_dim=emb_dim,
            preprocess=transform,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


def register_uni_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "uni",
        lambda: UNIV1(device=device, dtype=dtype, num_workers=num_workers),
    )
    registry.register(
        "uni2_h",
        lambda: UNIV2(device=device, dtype=dtype, num_workers=num_workers),
    )
