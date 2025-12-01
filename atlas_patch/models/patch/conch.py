from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry
from atlas_patch.utils.hf import import_module_from_hf

logger = logging.getLogger(__name__)

_CONCH_V1_DIM = 512
_CONCH_V15_DIM = 768
_CONCH_V1_MODEL_ID = "conch_ViT-B-16"
_CONCH_V1_REPO = "hf_hub:MahmoodLab/conch"
_CONCH_V15_REPO = "MahmoodLab/TITAN"


class CONCHV1(PatchFeatureExtractor):
    """CONCH v1 encoder (MahmoodLab/CONCH) via open_clip custom weights."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        import conch.open_clip_custom as open_clip_custom

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            model, preprocess = open_clip_custom.create_model_from_pretrained(
                _CONCH_V1_MODEL_ID,
                _CONCH_V1_REPO,
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                "Failed to load CONCH v1 (MahmoodLab/CONCH). "
                "Ensure the conch package is installed and HuggingFace token access is configured."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()

        def _forward(x, m=model):
            with torch.inference_mode():
                return m.encode_image(x, proj_contrast=False, normalize=False)

        super().__init__(
            name="conch_v1",
            model=model,
            embedding_dim=_CONCH_V1_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


class CONCHV15(PatchFeatureExtractor):
    """CONCH v1.5 encoder from TITAN repo (MahmoodLab/TITAN)."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            titan_cfg = import_module_from_hf(_CONCH_V15_REPO, "configuration_titan.py")
            conch_cfg = titan_cfg.ConchConfig()
            conch_module = import_module_from_hf(_CONCH_V15_REPO, "conch_v1_5.py")
            model, preprocess = conch_module.build_conch(conch_cfg)
        except Exception as e:
            logger.exception("Failed to load CONCH v1.5 (MahmoodLab/TITAN)")
            msg = (
                "Failed to load CONCH v1.5 (MahmoodLab/TITAN). "
                f"Root error: {e}. Ensure gated repo access and dependencies (einops_exts, etc.) "
                "are available."
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()

        def _forward(x, m=model):
            with torch.inference_mode():
                out = m(x)
            if isinstance(out, tuple):
                logger.debug(
                    "CONCH v1.5 raw output shapes: %s",
                    [getattr(t, "shape", None) for t in out],
                )
                out = out[0]
            else:
                logger.debug("CONCH v1.5 output shape: %s", getattr(out, "shape", None))
            return out

        super().__init__(
            name="conch_v15",
            model=model,
            embedding_dim=_CONCH_V15_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_conch_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "conch_v1",
        lambda: CONCHV1(device=device, dtype=dtype, num_workers=num_workers),
    )
    registry.register(
        "conch_v15",
        lambda: CONCHV15(device=device, dtype=dtype, num_workers=num_workers),
    )
