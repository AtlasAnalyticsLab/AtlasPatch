from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, cast

import cv2
import numpy as np
from PIL import Image

from slide_processor.utils.contours import FourPointContainment
from slide_processor.utils.h5 import save_h5
from slide_processor.utils.image import is_black_patch, is_white_patch
from slide_processor.wsi.iwsi import IWSI


@dataclass
class PatchExtractor:
    """Patch extraction over tissue contours."""

    target_mag: int
    patch_size: int = 256
    step_size: int = 256
    white_thresh: int = 15
    black_thresh: int = 50
    center_shift: float = 0.5

    def _point_in_holes(
        self, pt: tuple[int, int], holes: Sequence[np.ndarray], patch_size_src: int
    ) -> bool:
        cx = pt[0] + patch_size_src // 2
        cy = pt[1] + patch_size_src // 2
        for hole in holes:
            if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
                return True
        return False

    def _in_tissue(
        self,
        pt: tuple[int, int],
        contour: np.ndarray,
        holes: Sequence[np.ndarray],
        *,
        patch_size_src: int,
    ) -> bool:
        check_fn = FourPointContainment(
            contour=contour,
            patch_size=patch_size_src,
            center_shift=self.center_shift,
        )
        return check_fn(pt) and not self._point_in_holes(pt, holes, patch_size_src)

    def _prepare_geometry(self, wsi: IWSI) -> tuple[int, tuple[int, int], int, int]:
        """Compute reading geometry for multi-resolution extraction.

        Returns: (level, read_wh, patch_size_src_level0, step_size_src_level0)
        - level: pyramid level to read from
        - read_wh: (w, h) size to read at that level
        - patch_size_src_level0: patch footprint at level 0 (in px)
        - step_size_src_level0: step/stride footprint at level 0 (in px)
        """
        # Determine source and target magnifications
        src_mag = wsi.mag
        tgt_mag = self.target_mag
        if src_mag is None:
            raise ValueError(
                "Target magnification requested but WSI base magnification is unknown."
            )
        if int(tgt_mag) > int(src_mag):
            raise ValueError(
                f"Requested magnification {tgt_mag}x is higher than available {src_mag}x."
            )
        desired_downsample = float(src_mag) / float(tgt_mag)

        # Determine best pyramid level
        level, _ = wsi.optimal_level(desired_downsample)

        # Level downsample factor
        downsamples = wsi.ds or [1.0]
        level_ds = float(downsamples[level])

        # Patch and step footprints at level 0
        ps_src = int(round(self.patch_size * desired_downsample))
        ss_src = int(round(self.step_size * desired_downsample))

        # Read size at computed level
        read_w = max(1, int(round(ps_src / level_ds)))
        read_h = read_w

        return level, (read_w, read_h), ps_src, ss_src

    def iter_patches(
        self,
        wsi: IWSI,
        contour: np.ndarray,
        holes: Sequence[np.ndarray],
        *,
        fast_mode: bool = False,
    ) -> Iterable[tuple[int, int, np.ndarray, tuple[int, int, int]]]:
        """Iterate (x, y, patch_rgb) for patches within the given contour.

        When fast_mode=True, skips white/black content filtering.
        """
        x0, y0, ww, hh = cv2.boundingRect(contour)
        W, H = wsi.get_size(lv=0)

        # Geometry according to requested magnification
        level, read_wh, ps_src, ss_src = self._prepare_geometry(wsi)
        read_w, read_h = read_wh

        # Always allow patches at boundaries with padding
        stop_x = x0 + ww
        stop_y = y0 + hh

        for y in range(y0, stop_y, ss_src):
            for x in range(x0, stop_x, ss_src):
                if not self._in_tissue((x, y), contour, holes, patch_size_src=ps_src):
                    continue

                patch_any = wsi.extract((x, y), lv=level, wh=(read_w, read_h), mode="array")
                if patch_any is None or not isinstance(patch_any, np.ndarray):
                    continue
                patch = cast(np.ndarray, patch_any)

                # Resize to target output patch size if needed
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))

                # Filter out uninformative patches unless in fast mode
                if not fast_mode:
                    if is_black_patch(patch, rgb_thresh=self.black_thresh):
                        continue
                    if is_white_patch(patch, sat_thresh=self.white_thresh):
                        continue

                yield x, y, patch, (read_w, read_h, level)

    def extract_to_h5(
        self,
        wsi: IWSI,
        tissue_contours: Sequence[np.ndarray],
        holes_contours: Sequence[Sequence[np.ndarray]],
        output_path: str,
        *,
        image_output_dir: str | None = None,
        fast_mode: bool = False,
        batch: int = 512,
    ) -> str | None:
        """Extract patches and save to HDF5 using utils.save_h5.

        Datasets:
          - "coords": (N, 2) int32  (x, y)
          - "coords_ext": (N, 5) int32  (x, y, w, h, level)
        Root attributes set: patch_size, wsi_path, num_patches
        """
        logger = logging.getLogger("slide_processor.patch_extractor")
        total = 0
        first_write = True
        patch_imgs_dir: Path | None = None
        stem = Path(wsi.path).stem
        logger.info(f"{stem}: start extraction (batch={batch}, fast_mode={fast_mode})")

        if image_output_dir is not None:
            patch_imgs_dir = Path(image_output_dir)
            patch_imgs_dir.mkdir(parents=True, exist_ok=True)

        # Iterate contours and write in batches
        buf_xy: list[tuple[int, int]] = []
        buf_ext: list[tuple[int, int, int, int, int]] = []

        def flush() -> None:
            nonlocal first_write, total
            if not buf_xy:
                return
            # Prepare buffers
            coords = np.asarray(buf_xy, dtype=np.int32)
            coords_ext = np.asarray(buf_ext, dtype=np.int32)
            attrs = (
                {
                    "coords": {"description": "(x, y) coordinates at level 0"},
                    "coords_ext": {
                        "description": "(x, y, w, h, level) at extraction",
                    },
                }
                if first_write
                else None
            )

            mode = "w" if first_write else "a"
            file_attrs = None
            if first_write:
                src_mag = wsi.mag
                tgt_mag = self.target_mag
                assert src_mag is not None, "WSI base magnification is required to write metadata"
                patch_size_level0 = int(self.patch_size * int(src_mag) // int(tgt_mag))

                file_attrs = {
                    "patch_size": int(self.patch_size),
                    "wsi_path": wsi.path,
                    "level0_magnification": int(src_mag),
                    "target_magnification": int(tgt_mag),
                    "patch_size_level0": int(patch_size_level0),
                }

            # Build payload according to configuration
            assets: dict[str, np.ndarray] = {"coords": coords, "coords_ext": coords_ext}

            save_h5(output_path, assets, attributes=attrs, mode=mode, file_attrs=file_attrs)

            total += int(coords.shape[0])
            buf_xy.clear()
            buf_ext.clear()

        for cont, holes in zip(tissue_contours, holes_contours):
            level, (read_w, read_h), ps_src, ss_src = self._prepare_geometry(wsi)

            if fast_mode:
                # Coordinate-only fast path: avoid reading patches entirely
                x0, y0, ww, hh = cv2.boundingRect(cont)
                W, H = wsi.get_size(lv=0)
                # Always allow patches at boundaries with padding
                stop_x = x0 + ww
                stop_y = y0 + hh

                for y in range(y0, stop_y, ss_src):
                    for x in range(x0, stop_x, ss_src):
                        if not self._in_tissue((x, y), cont, holes, patch_size_src=ps_src):
                            continue
                        buf_xy.append((x, y))
                        buf_ext.append((x, y, int(read_w), int(read_h), int(level)))
                        flush_count = len(buf_xy)
                        if flush_count >= batch:
                            flush()
                            first_write = False
            else:
                for x, y, _, (rw, rh, lv) in self.iter_patches(
                    wsi, cont, holes, fast_mode=fast_mode
                ):
                    buf_xy.append((x, y))
                    buf_ext.append((x, y, int(rw), int(rh), int(lv)))

                    # Flush based on coordinate buffer size
                    flush_count = len(buf_xy)
                    if flush_count >= batch:
                        flush()
                        first_write = False

        # final flush
        flush()
        if not first_write:
            # write final attrs
            save_h5(output_path, {}, mode="a", file_attrs={"num_patches": int(total)})

            # optional image export (read back from file for consistency)
            if patch_imgs_dir is not None and total > 0:
                # Read coords from H5 and re-extract
                import h5py

                with h5py.File(output_path, "r") as f:
                    coords_d = f["coords"]
                    for i in range(int(coords_d.shape[0])):
                        x, y = (int(v) for v in coords_d[i])
                        # Use coords_ext to recover proper read size/level
                        rw, rh, lv = (int(v) for v in f["coords_ext"][i][2:5])
                        arr_any = wsi.extract((x, y), lv=lv, wh=(rw, rh), mode="array")
                        if isinstance(arr_any, np.ndarray):
                            # Resize to target patch size for consistency
                            if (
                                arr_any.shape[0] != self.patch_size
                                or arr_any.shape[1] != self.patch_size
                            ):
                                arr_any = cv2.resize(arr_any, (self.patch_size, self.patch_size))
                            out_name = f"{stem}_x{x}_y{y}.png"
                            Image.fromarray(arr_any).save(str(patch_imgs_dir / out_name))

            logger.info(f"{stem}: extraction complete, total patches={total}")
            return output_path

        return None
