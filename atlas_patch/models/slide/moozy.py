from __future__ import annotations

from contextlib import nullcontext
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from atlas_patch.models.slide.base import SlideEncoder, SlideEncoderSpec
from atlas_patch.models.slide.common import coerce_slide_embedding, resolve_slide_device
from atlas_patch.models.slide.registry import SlideEncoderRegistry

_EMBEDDING_DATASET = "features"


def _run_moozy_public_api(
    slide_paths: list[str],
    output_path: str,
    *,
    device: torch.device,
    mixed_precision: bool = False,
) -> None:
    try:
        from moozy.encoding import run_encoding
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MOOZY slide encoding requires the optional `moozy` package. "
            "Install `atlas-patch[moozy]`, `atlas-patch[slide-encoders]`, or `pip install moozy`."
        ) from exc

    if device.type == "cpu" and torch.cuda.is_available():
        raise RuntimeError(
            "MOOZY's public `run_encoding` API does not support forcing CPU when CUDA is "
            "available. Use `--device cuda` or run in a CPU-only environment."
        )

    cuda_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with cuda_ctx:
        run_encoding(
            slide_paths=slide_paths,
            output_path=output_path,
            mixed_precision=mixed_precision,
        )


def _load_moozy_embedding(output_path: Path) -> np.ndarray:
    with h5py.File(output_path, "r") as handle:
        dataset = handle.get(_EMBEDDING_DATASET)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{output_path} is missing required dataset '{_EMBEDDING_DATASET}'.")
        return np.asarray(dataset[()], dtype=np.float32)


class MOOZYSlideEncoder(SlideEncoder):
    spec = SlideEncoderSpec(
        name="moozy",
        embedding_dim=768,
        patch_encoder_name="lunit_vit_small_patch8_dino",
        patch_size=224,
    )

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = resolve_slide_device(device)

    def encode_slide(self, patch_h5_path: Path) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="atlaspatch_moozy_") as tmp_dir:
            output_path = Path(tmp_dir) / "case_embedding.h5"
            _run_moozy_public_api(
                [str(patch_h5_path)],
                str(output_path),
                device=self.device,
                mixed_precision=False,
            )
            embedding = _load_moozy_embedding(output_path)
        return coerce_slide_embedding(
            embedding,
            expected_dim=self.embedding_dim,
            encoder_name="MOOZY",
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
