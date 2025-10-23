"""High level processing pipelines for slide_processor.

Current pipelines:
- segment_and_patchify: segment WSI (SAM2) at thumbnail, patchify tissue, and save HDF5.

Returns HDF5 path as string or None if nothing saved.
"""

from .patchify import (
    PatchifyParams,
    SegmentParams,
    segment_and_patchify,
)

__all__ = [
    "PatchifyParams",
    "SegmentParams",
    "segment_and_patchify",
]
