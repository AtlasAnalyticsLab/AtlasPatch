from __future__ import annotations

import logging
from typing import Callable, Mapping

from atlas_patch.core.paths import normalize_encoder_name
from atlas_patch.models.slide.base import SlideEncoder, SlideEncoderSpec

logger = logging.getLogger(__name__)


class SlideEncoderRegistry:
    """Registry of slide encoder builders plus lightweight metadata specs."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[], SlideEncoder]] = {}
        self._specs: dict[str, SlideEncoderSpec] = {}

    def register(self, spec: SlideEncoderSpec, builder: Callable[[], SlideEncoder]) -> None:
        key = normalize_encoder_name(spec.name)
        if key in self._builders:
            raise ValueError(f"Slide encoder '{spec.name}' already registered.")
        self._builders[key] = builder
        self._specs[key] = spec

    def available(self) -> list[str]:
        return sorted(self._builders.keys())

    def get_spec(self, name: str) -> SlideEncoderSpec:
        key = normalize_encoder_name(name)
        if key not in self._specs:
            raise KeyError(f"Unknown slide encoder '{name}'. Available: {self.available()}")
        return self._specs[key]

    def specs(self) -> Mapping[str, SlideEncoderSpec]:
        return dict(self._specs)

    def create(self, name: str) -> SlideEncoder:
        key = normalize_encoder_name(name)
        if key not in self._builders:
            raise KeyError(f"Unknown slide encoder '{name}'. Available: {self.available()}")
        builder = self._builders[key]
        try:
            encoder = builder()
        except ModuleNotFoundError as exc:
            logger.exception("Missing optional dependency for slide encoder '%s'", name)
            dependency = exc.name or "unknown dependency"
            raise RuntimeError(
                f"Slide encoder '{name}' requires optional dependency '{dependency}'. "
                "Install the relevant optional extras documented in the README."
            ) from exc
        except Exception:
            logger.exception("Failed to create slide encoder '%s'", name)
            raise
        if normalize_encoder_name(encoder.name) != key:
            raise RuntimeError(
                f"Slide encoder builder for '{name}' returned '{encoder.name}', which does not match."
            )
        return encoder
