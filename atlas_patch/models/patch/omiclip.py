from __future__ import annotations

import logging

import torch

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_MODEL_ID = "WangGuangyuLab/Loki"
_CHECKPOINT_NAME = "checkpoint.pt"
_MODEL_NAME = "coca_ViT-L-14"
_FALLBACK_DIM = 768


class OmiCLIP(PatchFeatureExtractor):
    """OmiCLIP encoder from \"A visual-omics foundation model to bridge histopathology with spatial transcriptomics\"."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        from importlib.metadata import version

        from huggingface_hub import hf_hub_download
        from open_clip import create_model_from_pretrained
        from packaging.version import Version

        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        try:
            ckpt_path = hf_hub_download(_MODEL_ID, _CHECKPOINT_NAME)
            opts: dict = {"pretrained": ckpt_path}
            if Version(version("open_clip_torch")) >= Version("3.0.0"):
                opts["weights_only"] = False
            else:
                opts["load_weights_only"] = False

            model, preprocess = create_model_from_pretrained(_MODEL_NAME, **opts)
        except Exception as e:
            logger.exception("Failed to load OmiCLIP (%s)", _MODEL_ID)
            msg = (
                f"Failed to load OmiCLIP ({_MODEL_ID}). "
                "Ensure HuggingFace access and open_clip_torch are available. "
                f"Root error: {e}"
            )
            raise RuntimeError(msg) from e

        model = model.to(device=self.device, dtype=self.dtype).eval()
        emb_dim = 768

        def _forward(x, m=model):
            with torch.inference_mode():
                return m.encode_image(x)

        super().__init__(
            name="omiclip",
            model=model,
            embedding_dim=emb_dim,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_omiclip_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "omiclip",
        lambda: OmiCLIP(device=device, dtype=dtype, num_workers=num_workers),
    )
