from __future__ import annotations

import logging

import h5py
import numpy as np

from slide_processor.core.config import ExtractionConfig, OutputConfig, VisualizationConfig
from slide_processor.core.models import ExtractionResult
from slide_processor.core.paths import build_run_root
from slide_processor.core.wsi.iwsi import IWSI
from slide_processor.services.interfaces import VisualizationService
from slide_processor.utils.contours import mask_to_contours
from slide_processor.utils.visualization import (
    visualize_contours_on_thumbnail,
    visualize_mask_on_thumbnail,
    visualize_patches_on_thumbnail,
)

logger = logging.getLogger("slide_processor.visualization_service")


class DefaultVisualizationService(VisualizationService):
    """Composite visualizer that delegates to per-type visualizers."""

    def __init__(
        self,
        output_cfg: OutputConfig,
        extraction_cfg: ExtractionConfig,
        vis_cfg: VisualizationConfig | None = None,
    ) -> None:
        self.output_cfg = output_cfg
        self.extraction_cfg = extraction_cfg
        self.vis_cfg = vis_cfg or VisualizationConfig()

    def visualize(self, result: ExtractionResult, *, wsi: IWSI, mask: np.ndarray) -> None:
        if not (
            self.output_cfg.visualize_grids
            or self.output_cfg.visualize_mask
            or self.output_cfg.visualize_contours
        ):
            return

        vis_dir = build_run_root(self.output_cfg, self.extraction_cfg) / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)

        if self.output_cfg.visualize_grids:
            try:
                coords = result.coords
                psize_l0 = result.patch_size_level0
                if coords is None or psize_l0 is None:
                    with h5py.File(result.h5_path, "r") as f:
                        coords = f["coords"][:]
                        psize_l0 = int(f.attrs["patch_size_level0"])
                if coords is None or psize_l0 is None:
                    raise ValueError("Coordinates or patch size missing for grid visualization")
                coords_xy = coords[:, :2] if coords.ndim == 2 and coords.shape[1] >= 2 else coords
                info = {
                    "patch_size": self.extraction_cfg.patch_size,
                    "step_size": self.extraction_cfg.step_size or self.extraction_cfg.patch_size,
                    "tissue_thresh": self.extraction_cfg.tissue_threshold,
                }
                path = visualize_patches_on_thumbnail(
                    coords=coords_xy,
                    patch_size_level0=psize_l0,
                    wsi=wsi,
                    output_dir=vis_dir,
                    thumbnail_size=self.vis_cfg.thumbnail_size,
                    info=info,
                )
                result.visualizations["grids"] = path
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to visualize grids for %s: %s", result.slide.path.name, e)

        if self.output_cfg.visualize_mask:
            try:
                path = visualize_mask_on_thumbnail(
                    mask=mask,
                    wsi=wsi,
                    output_dir=vis_dir,
                    thumbnail_size=self.vis_cfg.thumbnail_size,
                )
                result.visualizations["mask"] = path
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to visualize mask for %s: %s", result.slide.path.name, e)

        if self.output_cfg.visualize_contours:
            try:
                tcs_t, hcs_t = mask_to_contours(
                    mask, tissue_area_thresh=self.extraction_cfg.tissue_threshold
                )
                path = visualize_contours_on_thumbnail(
                    tissue_contours=tcs_t,
                    holes_contours=hcs_t,
                    wsi=wsi,
                    output_dir=vis_dir,
                    thumbnail_size=self.vis_cfg.thumbnail_size,
                )
                result.visualizations["contours"] = path
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to visualize contours for %s: %s", result.slide.path.name, e)
