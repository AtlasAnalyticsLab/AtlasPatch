from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalize_encoder_name(name: str) -> str:
    value = str(name).strip().lower()
    if not value:
        raise ValueError("encoder name must be a non-empty string")
    return value


def _normalize_patch_size(patch_size: int) -> int:
    value = int(patch_size)
    if value <= 0:
        raise ValueError("patch_size must be > 0")
    return value


@dataclass(frozen=True)
class SlideEncoderSpec:
    name: str
    embedding_dim: int
    patch_encoder_name: str
    patch_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_encoder_name(self.name))
        embedding_dim = int(self.embedding_dim)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        object.__setattr__(self, "embedding_dim", embedding_dim)
        object.__setattr__(
            self,
            "patch_encoder_name",
            _normalize_encoder_name(self.patch_encoder_name),
        )
        object.__setattr__(self, "patch_size", _normalize_patch_size(self.patch_size))


class SlideEncoder(ABC):
    """Base interface for slide-level encoders operating on AtlasPatch H5 artifacts."""

    spec: SlideEncoderSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def embedding_dim(self) -> int:
        return self.spec.embedding_dim

    @property
    def required_patch_encoder(self) -> str:
        return self.spec.patch_encoder_name

    @property
    def required_patch_size(self) -> int:
        return self.spec.patch_size

    @abstractmethod
    def encode_slide(self, patch_h5_path: Path) -> np.ndarray:
        """Return a 1-D embedding for the provided per-slide patch H5."""
        raise NotImplementedError
