from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_EMB_DIM = 2560


def _build_preprocess(model):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    return create_transform(**cfg)


class Virchow(PatchFeatureExtractor):
    """Virchow encoder (paige-ai/Virchow) from \"Virchow: A Million-Slide Digital Pathology Foundation Model\" (https://arxiv.org/abs/2309.07778)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm
        from timm.layers import SwiGLUPacked

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                "Failed to load Virchow (paige-ai/Virchow) via timm. "
                "Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(model)

        def _forward(x, m=model):
            output = m(x)
            cls_token = output[:, 0]
            patch_tokens = output[:, 1:]
            return torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)

        super().__init__(
            name="virchow_v1",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


class Virchow2(PatchFeatureExtractor):
    """Virchow2 encoder (paige-ai/Virchow2) from \"Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology\" (https://arxiv.org/abs/2408.00738)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm
        from timm.layers import SwiGLUPacked

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        except Exception as e:
            msg = (
                "Failed to load Virchow2 (paige-ai/Virchow2) via timm. "
                "Ensure HuggingFace access and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_preprocess(model)

        def _forward(x, m=model):
            output = m(x)
            cls_token = output[:, 0]
            patch_tokens = output[:, 5:]  # tokens 1-4 are register tokens
            return torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)

        super().__init__(
            name="virchow_v2",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_virchow_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "virchow_v1",
        lambda: Virchow(device=device, dtype=dtype, num_workers=num_workers),
    )
    registry.register(
        "virchow_v2",
        lambda: Virchow2(device=device, dtype=dtype, num_workers=num_workers),
    )
