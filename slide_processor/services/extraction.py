from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from slide_processor.core.config import ExtractionConfig, OutputConfig
from slide_processor.core.models import ExtractionResult, Slide
from slide_processor.core.paths import build_run_root, images_dir, patch_h5_path
from slide_processor.core.wsi.iwsi import IWSI
from slide_processor.services.interfaces import ExtractionService
from slide_processor.services.storage import H5PatchWriter
from slide_processor.utils.contours import FourPointContainment, mask_to_contours, scale_contours
from slide_processor.utils.image import is_black_patch, is_white_patch

logger = logging.getLogger("slide_processor.extraction_service")


class PatchExtractionService(ExtractionService):
    """Extracts patch coordinates (and optional images) from WSIs given a tissue mask."""

    def __init__(self, extraction_cfg: ExtractionConfig, output_cfg: OutputConfig) -> None:
        self.cfg = extraction_cfg.validated()
        self.output_cfg = output_cfg.validated()

    # --- contour and geometry helpers ---------------------------------------------
    def _prepare_contours(self, mask: np.ndarray, wsi: IWSI):
        tissue_contours_t, holes_contours_t = mask_to_contours(
            mask, tissue_area_thresh=self.cfg.tissue_threshold
        )

        W, H = wsi.get_size(lv=0)
        mh, mw = mask.shape[:2]
        sx = W / float(mw)
        sy = H / float(mh)

        tissue_contours = scale_contours(tissue_contours_t, sx, sy)
        holes_contours = [scale_contours(hs, sx, sy) for hs in holes_contours_t]
        return tissue_contours, holes_contours

    def _prepare_geometry(self, wsi: IWSI) -> tuple[int, tuple[int, int], int, int, int]:
        """Return level, read size at level, patch footprint, stride, and patch size at level 0."""
        src_mag = wsi.mag
        tgt_mag = self.cfg.target_magnification
        if src_mag is None:
            raise ValueError("WSI base magnification is required for patch extraction.")
        if int(tgt_mag) > int(src_mag):
            raise ValueError(f"Requested magnification {tgt_mag}x exceeds available {src_mag}x.")

        desired_downsample = float(src_mag) / float(tgt_mag)
        level, _ = wsi.optimal_level(desired_downsample)
        downsamples = wsi.ds or [1.0]
        level_ds = float(downsamples[level])

        patch_size_src = int(round(self.cfg.patch_size * desired_downsample))
        step_src = int(round((self.cfg.step_size or self.cfg.patch_size) * desired_downsample))
        patch_size_level0 = int(self.cfg.patch_size * int(src_mag) // int(tgt_mag))

        read_w = max(1, int(round(patch_size_src / level_ds)))
        read_h = read_w
        return level, (read_w, read_h), patch_size_src, step_src, patch_size_level0

    # --- iteration -----------------------------------------------------------------
    def _in_tissue(
        self,
        pt: tuple[int, int],
        contour: np.ndarray,
        holes: Sequence[np.ndarray],
        *,
        patch_size: int,
    ) -> bool:
        checker = FourPointContainment(contour=contour, patch_size=patch_size, center_shift=0.5)
        cx = pt[0] + patch_size // 2
        cy = pt[1] + patch_size // 2
        for hole in holes:
            if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
                return False
        return checker(pt)

    def _iter_patch_entries(
        self,
        wsi: IWSI,
        tissue_contours: Sequence[np.ndarray],
        holes_contours: Sequence[Sequence[np.ndarray]],
        *,
        include_patch: bool,
    ) -> Iterable[tuple[int, int, int, int, int, np.ndarray | None]]:
        """Yield (x, y, read_w, read_h, level, patch_or_none)."""
        level, (read_w, read_h), patch_size_src, step_src, _ = self._prepare_geometry(wsi)
        for contour, holes in zip(tissue_contours, holes_contours):
            x0, y0, ww, hh = cv2.boundingRect(contour)
            stop_x, stop_y = x0 + ww, y0 + hh
            for y in range(y0, stop_y, step_src):
                for x in range(x0, stop_x, step_src):
                    if not self._in_tissue((x, y), contour, holes, patch_size=patch_size_src):
                        continue

                    if self.cfg.fast_mode and not include_patch:
                        yield x, y, int(read_w), int(read_h), int(level), None
                        continue

                    patch_any = wsi.extract((x, y), lv=level, wh=(read_w, read_h), mode="array")
                    if not isinstance(patch_any, np.ndarray):
                        continue
                    patch = patch_any
                    if (
                        patch.shape[0] != self.cfg.patch_size
                        or patch.shape[1] != self.cfg.patch_size
                    ):
                        patch = cv2.resize(patch, (self.cfg.patch_size, self.cfg.patch_size))

                    if not self.cfg.fast_mode:
                        if is_black_patch(patch, rgb_thresh=self.cfg.black_threshold):
                            continue
                        if is_white_patch(patch, sat_thresh=self.cfg.white_threshold):
                            continue

                    yield (
                        x,
                        y,
                        int(read_w),
                        int(read_h),
                        int(level),
                        patch if include_patch else None,
                    )

    # --- public API ----------------------------------------------------------------
    def extract(self, wsi: IWSI, mask: np.ndarray, *, slide: Slide) -> ExtractionResult:
        tissue_contours, holes_contours = self._prepare_contours(mask, wsi)

        run_root = build_run_root(self.output_cfg, self.cfg)
        patches_root = run_root / "patches"
        patches_root.mkdir(parents=True, exist_ok=True)
        out_h5 = patch_h5_path(slide, self.output_cfg, self.cfg)

        img_dir: Path | None = None
        if self.output_cfg.save_images:
            img_dir = images_dir(slide, self.output_cfg, self.cfg)
            img_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Extracting patches for %s to %s", slide.path.name, out_h5)

        level, _, _, _, patch_size_level0 = self._prepare_geometry(wsi)
        level0_width, level0_height = wsi.get_size(lv=0)
        step = self.cfg.step_size or self.cfg.patch_size
        overlap = max(0, int(self.cfg.patch_size) - int(step))

        writer = H5PatchWriter(
            chunk_rows=self.cfg.write_batch,
            patch_size=self.cfg.patch_size,
            patch_size_level0=patch_size_level0,
            level0_mag=int(wsi.mag) if wsi.mag is not None else 0,
            target_mag=self.cfg.target_magnification,
            level0_wh=(int(level0_width), int(level0_height)),
            overlap=overlap,
            slide_stem=slide.stem,
            wsi_path=str(wsi.path),
        )

        entries = self._iter_patch_entries(
            wsi=wsi,
            tissue_contours=tissue_contours,
            holes_contours=holes_contours,
            include_patch=bool(img_dir),
        )

        if img_dir is None:
            total, coords_viz = writer.write_coords(
                output_path=out_h5,
                entries=entries,
                batch=self.cfg.write_batch,
                collect_coords=False,
            )
        else:
            total, coords_viz = writer.write_coords_and_images(
                output_path=out_h5,
                entries=entries,
                image_dir=img_dir,
                batch=self.cfg.write_batch,
                collect_coords=False,
            )

        return ExtractionResult(
            slide=slide,
            h5_path=out_h5,
            num_patches=int(total),
            image_dir=img_dir,
            coords=None,
            patch_size_level0=patch_size_level0,
        )
