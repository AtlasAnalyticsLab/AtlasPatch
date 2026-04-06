from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from atlas_patch.models.common import coerce_model_embedding, resolve_model_device
from atlas_patch.models.moozy_api import load_moozy_embedding, run_moozy_public_api
from atlas_patch.models.slide.base import SlideEncoder, SlideEncoderSpec
from atlas_patch.models.slide.registry import SlideEncoderRegistry


class MOOZYSlideEncoder(SlideEncoder):
    spec = SlideEncoderSpec(
        name="moozy",
        embedding_dim=768,
        patch_encoder_name="lunit_vit_small_patch8_dino",
        patch_size=224,
    )

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = resolve_model_device(device)

    def encode_slide(self, patch_h5_path: Path) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="atlaspatch_moozy_") as tmp_dir:
            output_path = Path(tmp_dir) / "case_embedding.h5"
            run_moozy_public_api(
                [str(patch_h5_path)],
                str(output_path),
                device=self.device,
                mixed_precision=False,
            )
            embedding = load_moozy_embedding(output_path)
        return coerce_model_embedding(
            embedding,
            expected_dim=self.embedding_dim,
            label="MOOZY",
        )


def register_moozy_slide_encoder(
    registry: SlideEncoderRegistry,
    *,
    device: str | torch.device = "cuda",
) -> None:
    registry.register(
        MOOZYSlideEncoder.spec,
        lambda: MOOZYSlideEncoder(device=device),
    )
