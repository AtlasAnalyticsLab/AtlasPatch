from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

import numpy as np

from slide_processor.core.models import ExtractionResult, Mask, Slide
from slide_processor.core.wsi.iwsi import IWSI


class SegmentationService(ABC):
    @abstractmethod
    def segment_thumbnail(self, wsi: IWSI) -> Mask: ...

    @abstractmethod
    def segment_batch(self, wsis: Sequence[IWSI]) -> list[Mask]: ...


class ExtractionService(ABC):
    @abstractmethod
    def extract(self, wsi: IWSI, mask: np.ndarray, *, slide: Slide) -> ExtractionResult: ...


class FeatureEmbeddingService(ABC):
    @abstractmethod
    def embed_features(self, result: ExtractionResult, *, wsi: IWSI) -> ExtractionResult: ...


class VisualizationService(ABC):
    @abstractmethod
    def visualize(self, result: ExtractionResult, *, wsi: IWSI, mask: np.ndarray) -> None: ...


class MPPResolver(Protocol):
    def resolve(self, slide: Slide) -> float | None: ...


class WSILoader(Protocol):
    def open(self, slide: Slide) -> IWSI: ...
