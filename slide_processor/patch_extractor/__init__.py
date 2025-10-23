"""Patch extraction utilities for WSIs.

This module provides contour extraction from masks and patchification logic
integrated with the existing `wsi` backends and HDF5 saving utilities.
"""

from slide_processor.utils.contours import FourPointContainment, mask_to_contours

from .patch_extractor import PatchExtractor

__all__ = [
    "FourPointContainment",
    "PatchExtractor",
    "mask_to_contours",
]
