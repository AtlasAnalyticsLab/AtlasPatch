"""General utilities used across atlas_patch.

Exports helpers for HDF5 I/O, image/patch checks, contour handling, and file discovery.
"""

from .contours import FourPointContainment, mask_to_contours, scale_contours
from .features import (
    get_existing_features,
    missing_features,
    parse_feature_list,
)
from .h5 import H5AppendWriter
from .image import is_black_patch, is_white_patch
from .logging_utils import SuppressEmbeddingLogs, configure_logging, install_embedding_log_filter
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
    "configure_logging",
    "install_embedding_log_filter",
    "parse_feature_list",
    "get_existing_features",
    "missing_features",
]
