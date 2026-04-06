"""Service implementations for segmentation, extraction, visualization, and WSI access."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "PatchExtractionService": "atlas_patch.services.extraction",
    "CSVMPPResolver": "atlas_patch.services.mpp",
    "SAM2SegmentationService": "atlas_patch.services.segmentation",
    "DefaultVisualizationService": "atlas_patch.services.visualization",
    "DefaultWSILoader": "atlas_patch.services.wsi_loader",
}

__all__ = [
    "PatchExtractionService",
    "CSVMPPResolver",
    "SAM2SegmentationService",
    "DefaultVisualizationService",
    "DefaultWSILoader",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
