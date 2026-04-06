from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from atlas_patch.models.slide.base import SlideEncoder, SlideEncoderSpec
from atlas_patch.models.slide.common import coerce_slide_embedding, resolve_slide_device
from atlas_patch.models.slide.registry import SlideEncoderRegistry
from atlas_patch.utils.feature_h5 import load_patch_feature_data

_MODEL_ID = "paige-ai/Prism"
_PATCH_FEATURE_DIM = 2560


def _load_prism_model(*, device: torch.device, dtype: torch.dtype):
    if sys.version_info < (3, 10):
        raise RuntimeError("PRISM requires Python 3.10 or newer.")

    try:
        import environs  # noqa: F401
        import sacremoses  # noqa: F401
        from transformers import AutoModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PRISM requires optional slide-encoder dependencies. "
            "Install `atlas-patch[prism]` or `atlas-patch[slide-encoders]`."
        ) from exc

    model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
    if hasattr(model, "text_decoder"):
        model.text_decoder = None
    return model.to(device=device, dtype=dtype).eval()


class PrismSlideEncoder(SlideEncoder):
    spec = SlideEncoderSpec(
        name="prism",
        embedding_dim=1280,
        patch_encoder_name="virchow_v1",
        patch_size=224,
    )

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = resolve_slide_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = _load_prism_model(device=self.device, dtype=self.dtype)

    def encode_slide(self, patch_h5_path: Path) -> np.ndarray:
        patch_data = load_patch_feature_data(patch_h5_path, self.required_patch_encoder)
        if patch_data.patch_size != self.required_patch_size:
            raise ValueError(
                f"{patch_h5_path} has patch_size={patch_data.patch_size}, "
                f"but PRISM requires {self.required_patch_size}."
            )
        if patch_data.features.shape[1] != _PATCH_FEATURE_DIM:
            raise ValueError(
                f"{patch_h5_path} has feature dim {patch_data.features.shape[1]}, "
                f"but PRISM expects {_PATCH_FEATURE_DIM} from '{self.required_patch_encoder}'."
            )

        features = torch.from_numpy(patch_data.features).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        tile_mask = torch.ones(
            (1, patch_data.num_patches),
            device=self.device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            output = self.model.slide_representations(features, tile_mask=tile_mask)
        if not isinstance(output, dict) or "image_embedding" not in output:
            raise ValueError("PRISM did not return an 'image_embedding' entry.")
        return coerce_slide_embedding(
            output["image_embedding"],
            expected_dim=self.embedding_dim,
            encoder_name="PRISM",
        )

    def cleanup(self) -> None:
        try:
            self.model.cpu()
        except Exception:
            pass


def register_prism_slide_encoder(
    registry: SlideEncoderRegistry,
    *,
    device: str | torch.device = "cuda",
) -> None:
    registry.register(
        PrismSlideEncoder.spec,
        lambda: PrismSlideEncoder(device=device),
    )
