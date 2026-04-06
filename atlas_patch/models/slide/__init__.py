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


def build_default_registry() -> SlideEncoderRegistry:
    """Return an empty slide-encoder registry for phased population."""
    return SlideEncoderRegistry()
