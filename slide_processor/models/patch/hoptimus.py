from __future__ import annotations

import logging

import torch

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_EMB_DIM = 1536


def _build_hoptimus_transform():
    """
    This is only for HOptimus-0 and HOptimus-1, H0-Mini uses a different transformation
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ]
    )


class _HOptimusBase(PatchFeatureExtractor):
    """Common loader for H-optimus ViT vision foundation models via timm."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import timm

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))
        self.model_id = model_id

        try:
            model = timm.create_model(
                model_id,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        except Exception as e:
            msg = (
                f"Failed to load H-optimus model '{model_id}' via timm. "
                "Ensure HuggingFace access (token for gated weights) and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = _build_hoptimus_transform()

        super().__init__(
            name=name,
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
        )


class HOptimus0(_HOptimusBase):
    """H-optimus-0 vision foundation model (bioptimus/H-optimus-0)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            name="h_optimus_0",
            model_id="hf-hub:bioptimus/H-optimus-0",
            device=device,
            dtype=dtype,
            num_workers=num_workers,
        )


class HOptimus1(_HOptimusBase):
    """H-optimus-1 vision foundation model (bioptimus/H-optimus-1)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            name="h_optimus_1",
            model_id="hf-hub:bioptimus/H-optimus-1",
            device=device,
            dtype=dtype,
            num_workers=num_workers,
        )


class H0Mini(PatchFeatureExtractor):
    """H0-mini distilled H-optimus variant with CLS + mean patch concatenation.

    Paper: "Distilling foundation models for robust and efficient models in digital pathology" (H0-mini; https://doi.org/10.48550/arXiv.2501.16239)
    """

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
            model = timm.create_model(
                "hf-hub:bioptimus/H0-mini",
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        except Exception as e:
            msg = (
                "Failed to load H0-mini (bioptimus/H0-mini) via timm. "
                "Ensure HuggingFace access (token for gated weights) and cache permissions."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        def _forward(x, m=model):
            output = m(x)
            cls_features = output[:, 0]
            patch_token_features = output[:, m.num_prefix_tokens :]  # drop any prefix tokens
            concatenated = torch.cat([cls_features, patch_token_features.mean(1)], dim=-1)
            return concatenated

        super().__init__(
            name="h0_mini",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_hoptimus_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "h_optimus_0",
        lambda: HOptimus0(device=device, dtype=dtype, num_workers=num_workers),
    )
    registry.register(
        "h_optimus_1",
        lambda: HOptimus1(device=device, dtype=dtype, num_workers=num_workers),
    )
    registry.register(
        "h0_mini",
        lambda: H0Mini(device=device, dtype=dtype, num_workers=num_workers),
    )
