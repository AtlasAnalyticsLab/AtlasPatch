"""General utilities used across slide_processor.

Exports helpers for HDF5 I/O, image/patch checks, contour handling, and file discovery.
"""

from .contours import FourPointContainment, mask_to_contours, scale_contours
from .h5 import H5AppendWriter
from .logging import SuppressEmbeddingLogs, install_embedding_log_filter
from .image import is_black_patch, is_white_patch
from .params import get_wsi_files

__all__ = [
    "is_black_patch",
    "is_white_patch",
    "FourPointContainment",
    "mask_to_contours",
    "scale_contours",
    "H5AppendWriter",
    "get_wsi_files",
    "SuppressEmbeddingLogs",
    "install_embedding_log_filter",
]
