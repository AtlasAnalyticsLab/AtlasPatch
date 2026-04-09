from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from atlas_patch.models.common import coerce_model_embedding, model_autocast, resolve_model_device
from atlas_patch.models.slide.base import SlideEncoder, SlideEncoderSpec
from atlas_patch.models.slide.registry import SlideEncoderRegistry
from atlas_patch.utils.hf import download_hf_file, load_remote_class
from atlas_patch.utils.feature_h5 import load_patch_feature_data

_MODEL_ID = "paige-ai/Prism"
_PATCH_FEATURE_DIM = 2560


def _load_prism_model(*, device: torch.device, dtype: torch.dtype):
    if sys.version_info < (3, 10):
        raise RuntimeError("PRISM requires Python 3.10 or newer.")

    try:
        import environs  # noqa: F401
        import sacremoses  # noqa: F401
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PRISM requires optional slide-encoder dependencies. "
            "Install `atlas-patch[prism]` or `atlas-patch[slide-encoders]`."
        ) from exc

    config_class = load_remote_class(_MODEL_ID, "configuring_prism.PrismConfig")
    model_class = load_remote_class(_MODEL_ID, "modeling_prism.Prism")
    config = config_class.from_json_file(str(download_hf_file(_MODEL_ID, "config.json")))
    config.tie_word_embeddings = False
    config.biogpt_config.tie_word_embeddings = False

    model = model_class(config)
    state_dict = load_file(str(download_hf_file(_MODEL_ID, "model.safetensors")))
    # PRISM shares repeated layer weights, so the checkpoint only stores one copy per shared block.
    _, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        joined = ", ".join(sorted(unexpected))
        raise RuntimeError(f"PRISM checkpoint contains unexpected parameter keys: {joined}")
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
        self.device = resolve_model_device(device)
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
        )
        with torch.inference_mode(), model_autocast(self.device, self.dtype):
            output = self.model.slide_representations(features)
        if not isinstance(output, dict) or "image_embedding" not in output:
            raise ValueError("PRISM did not return an 'image_embedding' entry.")
        return coerce_model_embedding(
            output["image_embedding"],
            expected_dim=self.embedding_dim,
            label="PRISM",
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
