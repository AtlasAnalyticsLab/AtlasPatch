from __future__ import annotations

from importlib import import_module

import torch

from atlas_patch.models.slide.base import (
    SlideEncoder,
    SlideEncoderSpec,
)
from atlas_patch.models.slide.registry import SlideEncoderRegistry

__all__ = [
    "SlideEncoder",
    "SlideEncoderRegistry",
    "SlideEncoderSpec",
    "build_default_registry",
]

_REGISTRARS: tuple[tuple[str, str], ...] = (
    ("atlas_patch.models.slide.titan", "register_titan_slide_encoder"),
    ("atlas_patch.models.slide.prism", "register_prism_slide_encoder"),
    ("atlas_patch.models.slide.moozy", "register_moozy_slide_encoder"),
)


def build_default_registry(
    *,
    device: str | torch.device = "cuda",
) -> SlideEncoderRegistry:
    """Factory that registers the built-in slide encoders."""
    registry = SlideEncoderRegistry()
    for module_name, registrar_name in _REGISTRARS:
        module = import_module(module_name)
        registrar = getattr(module, registrar_name)
        registrar(registry, device=device)
    return registry
