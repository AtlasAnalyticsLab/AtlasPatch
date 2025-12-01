from __future__ import annotations

from typing import Callable, Iterable, Mapping

from .base import FeatureExtractor


class PatchFeatureExtractorRegistry:
    """Registry of feature extractor builders."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[], FeatureExtractor]] = {}

    def register(self, name: str, builder: Callable[[], FeatureExtractor]) -> None:
        key = name.lower()
        if key in self._builders:
            raise ValueError(f"Feature extractor '{name}' already registered.")
        self._builders[key] = builder

    def available(self) -> list[str]:
        return sorted(self._builders.keys())

    def create(self, name: str) -> FeatureExtractor:
        key = name.lower()
        if key not in self._builders:
            raise KeyError(f"Unknown feature extractor '{name}'. Available: {self.available()}")
        return self._builders[key]()

    def create_many(self, names: Iterable[str]) -> list[FeatureExtractor]:
        extractors: list[FeatureExtractor] = []
        for name in names:
            extractors.append(self.create(name))
        return extractors

    def as_mapping(self) -> Mapping[str, Callable[[], FeatureExtractor]]:
        return dict(self._builders)
