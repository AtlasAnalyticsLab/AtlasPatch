"""General utilities used across slide_processor.

Exports helpers for HDF5 I/O, image/patch checks, contour handling, file discovery, and progress reporting.
"""

from .contours import FourPointContainment, mask_to_contours, scale_contours
from .image import is_black_patch, is_white_patch
from .params import get_wsi_files
from .progress import ProgressReporter

__all__ = [
    "is_black_patch",
    "is_white_patch",
    "FourPointContainment",
    "mask_to_contours",
    "scale_contours",
    "get_wsi_files",
    "ProgressReporter",
]
