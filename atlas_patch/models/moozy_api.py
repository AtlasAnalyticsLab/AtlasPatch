from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch

_EMBEDDING_DATASET = "features"


def run_moozy_public_api(
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
            "MOOZY encoding requires the optional `moozy` package. "
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


def load_moozy_embedding(output_path: str | Path) -> np.ndarray:
    path = Path(output_path)
    with h5py.File(path, "r") as handle:
        dataset = handle.get(_EMBEDDING_DATASET)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{path} is missing required dataset '{_EMBEDDING_DATASET}'.")
        return np.asarray(dataset[()], dtype=np.float32)
