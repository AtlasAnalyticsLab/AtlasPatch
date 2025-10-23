"""General utilities used across slide_processor.

Exports helpers for HDF5 I/O, image/patch checks, and contour handling.
"""

from .contours import FourPointContainment, mask_to_contours, scale_contours
from .h5 import PatchesH5, load_patches_h5, save_h5
from .image import is_black_patch, is_white_patch

__all__ = [
    "save_h5",
    "PatchesH5",
    "load_patches_h5",
    "is_black_patch",
    "is_white_patch",
    "FourPointContainment",
    "mask_to_contours",
    "scale_contours",
]
