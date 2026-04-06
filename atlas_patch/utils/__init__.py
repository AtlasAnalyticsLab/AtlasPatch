"""General utilities used across atlas_patch.

Exports helpers for HDF5 I/O, image/patch checks, contour handling, and file discovery.
"""

from .contours import FourPointContainment, mask_to_contours, scale_contours
from .features import (
    get_existing_features,
    missing_features,
    parse_feature_list,
)
from .feature_h5 import (
    PatchFeatureData,
    append_slide_embedding,
    load_patch_feature_data,
    write_patient_embedding_h5,
)
from .h5 import H5AppendWriter
from .hf import import_module_from_hf
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
    "PatchFeatureData",
    "load_patch_feature_data",
    "append_slide_embedding",
    "write_patient_embedding_h5",
    "import_module_from_hf",
]
