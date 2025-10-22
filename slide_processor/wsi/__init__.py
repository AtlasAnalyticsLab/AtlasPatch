"""WSI module for slide processing."""

from .iwsi import IWSI
from .image_wsi import ImageWSI
from .openslide_wsi import OpenSlideWSI
from .wsi_factory import WSIFactory

__all__ = [
    "IWSI",
    "ImageWSI",
    "OpenSlideWSI",
    "WSIFactory",
]
