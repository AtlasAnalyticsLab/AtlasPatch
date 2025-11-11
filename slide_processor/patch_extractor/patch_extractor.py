from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import cv2
import numpy as np
from PIL import Image

from slide_processor.utils.contours import FourPointContainment
from slide_processor.utils.h5 import H5AppendWriter
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

    def iter_filtered_patches(
        self,
        wsi: IWSI,
        contour: np.ndarray,
        holes: Sequence[np.ndarray],
    ) -> Iterable[tuple[int, int, np.ndarray, tuple[int, int, int]]]:
        """Iterate (x, y, patch_rgb) for patches within the given contour with filtering."""
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

                # Filter out uninformative patches
                if is_black_patch(patch, rgb_thresh=self.black_thresh):
                    continue
                if is_white_patch(patch, sat_thresh=self.white_thresh):
                    continue

                yield x, y, patch, (read_w, read_h, level)

    def _initialize_writer(self, output_path: str, wsi: IWSI, batch: int) -> H5AppendWriter:
        """Create and seed the HDF5 writer with empty datasets and file-level attrs."""
        writer = H5AppendWriter(output_path, chunk_rows=max(1, int(batch)))
        src_mag = wsi.mag
        tgt_mag = self.target_mag
        assert src_mag is not None, "WSI base magnification is required to write metadata"

        # Patch footprint at level 0 for the target patch size/magnification
        patch_size_level0 = int(self.patch_size * int(src_mag) // int(tgt_mag))

        # Create empty datasets so file exists even if no patches are written
        empty_coords = np.empty((0, 2), dtype=np.int32)
        empty_coords_ext = np.empty((0, 5), dtype=np.int32)
        level0_width, level0_height = wsi.get_size(lv=0)
        step = self.step_size if self.step_size else self.patch_size
        overlap = max(0, int(self.patch_size) - int(step))
        coords_metadata: dict[str, Any] = {
            "description": "(x, y) coordinates at level 0",
            "patch_size": int(self.patch_size),
            "patch_size_level0": int(patch_size_level0),
            "level0_magnification": int(src_mag),
            "target_magnification": int(tgt_mag),
            "overlap": int(overlap),
            "name": Path(wsi.path).stem,
            "savetodir": str(Path(output_path).resolve().parent),
            "level0_width": int(level0_width),
            "level0_height": int(level0_height),
        }
        dset_attrs: dict[str, Mapping[str, Any]] = {
            "coords": coords_metadata,
            "coords_ext": {
                "description": "(x, y, w, h, level) at extraction",
            },
        }
        writer.append(
            {"coords": empty_coords, "coords_ext": empty_coords_ext}, attributes=dset_attrs
        )

        # File-level attributes
        writer.update_file_attrs(
            {
                "patch_size": coords_metadata["patch_size"],
                "patch_size_level0": coords_metadata["patch_size_level0"],
                "level0_magnification": coords_metadata["level0_magnification"],
                "target_magnification": coords_metadata["target_magnification"],
                "overlap": coords_metadata["overlap"],
                "level0_width": coords_metadata["level0_width"],
                "level0_height": coords_metadata["level0_height"],
                "wsi_path": wsi.path,
            }
        )
        return writer

    def iter_patch_coords(
        self,
        wsi: IWSI,
        tissue_contours: Sequence[np.ndarray],
        holes_contours: Sequence[Sequence[np.ndarray]],
        *,
        fast_mode: bool,
    ) -> Iterable[tuple[int, int, int, int, int]]:
        """Yield (x, y, read_w, read_h, level) for all patches to be saved.

        - In slow mode, reuses `iter_patches` to apply content filtering.
        - In fast mode, computes coordinates only (no patch reads) using geometry.
        """
        level, (read_w, read_h), ps_src, ss_src = self._prepare_geometry(wsi)

        if fast_mode:
            for cont, holes in zip(tissue_contours, holes_contours):
                yield from self._iter_coords_fast_for_contour(
                    wsi,
                    cont,
                    holes,
                    level=level,
                    read_w=read_w,
                    read_h=read_h,
                    ps_src=ps_src,
                    ss_src=ss_src,
                )
        else:
            for cont, holes in zip(tissue_contours, holes_contours):
                for x, y, _patch, (rw, rh, lv) in self.iter_filtered_patches(wsi, cont, holes):
                    yield x, y, int(rw), int(rh), int(lv)

    def _iter_coords_fast_for_contour(
        self,
        wsi: IWSI,
        cont: np.ndarray,
        holes: Sequence[np.ndarray],
        *,
        level: int,
        read_w: int,
        read_h: int,
        ps_src: int,
        ss_src: int,
    ) -> Iterable[tuple[int, int, int, int, int]]:
        """Fast path: compute coordinates only for a single contour."""
        x0, y0, ww, hh = cv2.boundingRect(cont)
        stop_x = x0 + ww
        stop_y = y0 + hh

        for y in range(y0, stop_y, ss_src):
            for x in range(x0, stop_x, ss_src):
                if not self._in_tissue((x, y), cont, holes, patch_size_src=ps_src):
                    continue
                yield x, y, int(read_w), int(read_h), int(level)

    def save_coords_to_h5(
        self,
        wsi: IWSI,
        coords_iter: Iterable[tuple[int, int, int, int, int]],
        output_path: str,
        *,
        batch: int = 512,
    ) -> tuple[str, int]:
        """Write coords iterator to HDF5 using buffered appends.

        Returns (h5_path, num_patches).
        """
        writer = self._initialize_writer(output_path, wsi, batch)

        total = 0
        buf_xy: list[tuple[int, int]] = []
        buf_ext: list[tuple[int, int, int, int, int]] = []

        try:
            for x, y, rw, rh, lv in coords_iter:
                buf_xy.append((x, y))
                buf_ext.append((x, y, int(rw), int(rh), int(lv)))
                if len(buf_xy) >= batch:
                    coords = np.asarray(buf_xy, dtype=np.int32)
                    coords_ext = np.asarray(buf_ext, dtype=np.int32)
                    writer.append({"coords": coords, "coords_ext": coords_ext})
                    total += int(coords.shape[0])
                    buf_xy.clear()
                    buf_ext.clear()

            if buf_xy:
                coords = np.asarray(buf_xy, dtype=np.int32)
                coords_ext = np.asarray(buf_ext, dtype=np.int32)
                writer.append({"coords": coords, "coords_ext": coords_ext})
                total += int(coords.shape[0])

            writer.update_file_attrs({"num_patches": int(total)})
            writer.close()
        except Exception:
            try:
                writer.abort()
            except Exception:
                pass
            raise

        return output_path, int(total)

    def _export_patch_images(
        self,
        wsi: IWSI,
        h5_path: str,
        image_dir: Path,
    ) -> None:
        """Re-read coordinates from H5 and export per-patch PNGs to `image_dir`."""
        import h5py

        stem = Path(wsi.path).stem
        with h5py.File(h5_path, "r") as f:
            coords_d = f["coords"]
            coords_ext = f["coords_ext"]
            n = int(coords_d.shape[0])
            for i in range(n):
                x, y = (int(v) for v in coords_d[i])
                rw, rh, lv = (int(v) for v in coords_ext[i][2:5])
                arr_any = wsi.extract((x, y), lv=lv, wh=(rw, rh), mode="array")
                if not isinstance(arr_any, np.ndarray):
                    continue
                if arr_any.shape[0] != self.patch_size or arr_any.shape[1] != self.patch_size:
                    arr_any = cv2.resize(arr_any, (self.patch_size, self.patch_size))
                out_name = f"{stem}_x{x}_y{y}.png"
                Image.fromarray(arr_any).save(str(image_dir / out_name))

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
        """Extract patches and save to HDF5.

        Datasets:
          - "coords": (N, 2) int32  (x, y)
          - "coords_ext": (N, 5) int32  (x, y, w, h, level)
        Root attributes set: patch_size, wsi_path, num_patches
        """
        logger = logging.getLogger("slide_processor.patch_extractor")
        stem = Path(wsi.path).stem
        logger.info(f"{stem}: start extraction (batch={batch}, fast_mode={fast_mode})")

        # Prepare optional image output directory
        patch_imgs_dir: Path | None = None
        if image_output_dir is not None:
            patch_imgs_dir = Path(image_output_dir)
            patch_imgs_dir.mkdir(parents=True, exist_ok=True)

        # Compose: generate coords, then save to H5
        coords_it = self.iter_patch_coords(
            wsi, tissue_contours, holes_contours, fast_mode=fast_mode
        )
        _, total = self.save_coords_to_h5(wsi, coords_it, output_path, batch=batch)

        # Optional image export
        if patch_imgs_dir is not None and total > 0:
            self._export_patch_images(wsi, output_path, patch_imgs_dir)

        logger.info(f"{stem}: extraction complete, total patches={total}")
        return output_path
