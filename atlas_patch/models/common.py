from __future__ import annotations

import logging

import numpy as np
import torch

from atlas_patch.utils.feature_h5 import coerce_1d_embedding

logger = logging.getLogger("atlas_patch.models")


def resolve_model_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        logger.warning("Model inference requested on CUDA but unavailable; using CPU instead.")
        return torch.device("cpu")
    return resolved


def coerce_model_embedding(
    output: torch.Tensor | np.ndarray,
    *,
    expected_dim: int,
    label: str,
) -> np.ndarray:
    tensor = torch.as_tensor(output)
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected a single {label} embedding, got shape {tuple(tensor.shape)}")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"Expected a 1-D {label} embedding, got shape {tuple(tensor.shape)}")
    vector = coerce_1d_embedding(
        tensor.detach().to(device="cpu", dtype=torch.float32).numpy(),
        label=label,
    )
    if vector.shape[0] != expected_dim:
        raise ValueError(f"{label} embedding dim mismatch: expected {expected_dim}, got {vector.shape[0]}")
    return vector
