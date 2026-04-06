from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from atlas_patch.models.common import coerce_model_embedding, resolve_model_device
from atlas_patch.models.slide.base import SlideEncoder, SlideEncoderSpec
from atlas_patch.models.slide.registry import SlideEncoderRegistry
from atlas_patch.utils.feature_h5 import load_patch_feature_data

_MODEL_ID = "MahmoodLab/TITAN"
_PATCH_FEATURE_DIM = 768


def _load_titan_model(*, device: torch.device, dtype: torch.dtype):
    try:
        from transformers import AutoModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TITAN requires optional slide-encoder dependencies. "
            "Install `atlas-patch[titan]` or `atlas-patch[slide-encoders]`."
        ) from exc

    model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return model.to(device=device, dtype=dtype).eval()


class TitanSlideEncoder(SlideEncoder):
    spec = SlideEncoderSpec(
        name="titan",
        embedding_dim=768,
        patch_encoder_name="conch_v15",
        patch_size=512,
    )

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = resolve_model_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = _load_titan_model(device=self.device, dtype=self.dtype)

    def encode_slide(self, patch_h5_path: Path) -> np.ndarray:
        patch_data = load_patch_feature_data(patch_h5_path, self.required_patch_encoder)
        if patch_data.patch_size != self.required_patch_size:
            raise ValueError(
                f"{patch_h5_path} has patch_size={patch_data.patch_size}, "
                f"but TITAN requires {self.required_patch_size}."
            )
        if patch_data.features.shape[1] != _PATCH_FEATURE_DIM:
            raise ValueError(
                f"{patch_h5_path} has feature dim {patch_data.features.shape[1]}, "
                f"but TITAN expects {_PATCH_FEATURE_DIM} from '{self.required_patch_encoder}'."
            )

        features = torch.from_numpy(patch_data.features).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        coords = torch.from_numpy(patch_data.coords[:, :2]).unsqueeze(0).to(
            device=self.device,
            dtype=torch.int64,
        )
        with torch.inference_mode():
            embedding = self.model.encode_slide_from_patch_features(
                features,
                coords,
                int(patch_data.patch_size_level0),
            )
        return coerce_model_embedding(
            embedding,
            expected_dim=self.embedding_dim,
            label="TITAN",
        )

    def cleanup(self) -> None:
        try:
            self.model.cpu()
        except Exception:
            pass


def register_titan_slide_encoder(
    registry: SlideEncoderRegistry,
    *,
    device: str | torch.device = "cuda",
) -> None:
    registry.register(
        TitanSlideEncoder.spec,
        lambda: TitanSlideEncoder(device=device),
    )
