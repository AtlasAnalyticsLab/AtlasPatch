"""WSI module for slide processing (core)."""

from .image_wsi import ImageWSI
from .iwsi import IWSI
from .openslide_wsi import OpenSlideWSI
from .wsi_factory import WSIFactory

__all__ = [
    "IWSI",
    "ImageWSI",
    "OpenSlideWSI",
    "WSIFactory",
]
