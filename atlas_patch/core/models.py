from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Slide:
    path: Path
    mpp: float | None = None
    backend: str | None = None

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class Mask:
    data: np.ndarray
    source_shape: tuple[int, int]


@dataclass
class ExtractionResult:
    slide: Slide
    h5_path: Path
    num_patches: int
    image_dir: Path | None = None
    visualizations: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    coords: np.ndarray | None = None  # Optional in-memory coords for visualization
    patch_size_level0: int | None = None


@dataclass
class SlideEmbeddingResult:
    slide: Slide
    h5_path: Path
    encoder_name: str
    dataset_key: str
    embedding_dim: int
    num_patches: int
    source_patch_encoder: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientEmbeddingResult:
    case_id: str
    h5_path: Path
    encoder_name: str
    embedding_dim: int
    num_slides: int
    source_patch_encoder: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientCase:
    case_id: str
    slides: tuple[Slide, ...]

    @property
    def num_slides(self) -> int:
        return len(self.slides)
