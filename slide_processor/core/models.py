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
