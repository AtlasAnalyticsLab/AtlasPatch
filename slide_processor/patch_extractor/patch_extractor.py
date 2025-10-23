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

    patch_size: int = 256
    step_size: int = 256
    white_thresh: int = 15
    black_thresh: int = 50
    use_padding: bool = True
    center_shift: float = 0.5
    require_all_points: bool = False  # True -> hard, False -> easy

    def _point_in_holes(self, pt: tuple[int, int], holes: Sequence[np.ndarray]) -> bool:
        cx = pt[0] + self.patch_size // 2
        cy = pt[1] + self.patch_size // 2
        for hole in holes:
            if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
                return True
        return False

    def _in_tissue(
        self, pt: tuple[int, int], contour: np.ndarray, holes: Sequence[np.ndarray]
    ) -> bool:
        check_fn = FourPointContainment(
            contour=contour,
            patch_size=self.patch_size,
            center_shift=self.center_shift,
            require_all=self.require_all_points,
        )
        return check_fn(pt) and not self._point_in_holes(pt, holes)

    def iter_patches(
        self,
        wsi: IWSI,
        contour: np.ndarray,
        holes: Sequence[np.ndarray],
        *,
        fast_mode: bool = False,
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        """Iterate (x, y, patch_rgb) for patches within the given contour.

        When fast_mode=True, skips white/black content filtering.
        """
        x0, y0, ww, hh = cv2.boundingRect(contour)
        W, H = wsi.get_size(lv=0)

        if self.use_padding:
            stop_x = x0 + ww
            stop_y = y0 + hh
        else:
            stop_x = min(x0 + ww, W - self.patch_size)
            stop_y = min(y0 + hh, H - self.patch_size)

        for y in range(y0, stop_y, self.step_size):
            for x in range(x0, stop_x, self.step_size):
                if not self._in_tissue((x, y), contour, holes):
                    continue

                patch_any = wsi.extract(
                    (x, y), lv=0, wh=(self.patch_size, self.patch_size), mode="array"
                )
                if patch_any is None or not isinstance(patch_any, np.ndarray):
                    continue
                patch = cast(np.ndarray, patch_any)

                # Filter out uninformative patches unless in fast mode
                if not fast_mode:
                    if is_black_patch(patch, rgb_thresh=self.black_thresh):
                        continue
                    if is_white_patch(patch, sat_thresh=self.white_thresh):
                        continue

                yield x, y, patch

    def extract_to_h5(
        self,
        wsi: IWSI,
        tissue_contours: Sequence[np.ndarray],
        holes_contours: Sequence[Sequence[np.ndarray]],
        output_path: str,
        *,
        image_output_dir: str | None = None,
        store_images: bool = True,
        fast_mode: bool = False,
        batch: int = 512,
    ) -> str | None:
        """Extract patches and save to HDF5 using utils.save_h5.

        Datasets:
          - "imgs": (N, H, W, 3) uint8 (optional, when store_images=True)
          - "coords": (N, 2) int32  (x, y)
          - "coords_ext": (N, 5) int32  (x, y, w, h, level)
        Root attributes set: patch_size, wsi_path, num_patches
        """
        logger = logging.getLogger("slide_processor.patch_extractor")
        total = 0
        first_write = True
        patch_imgs_dir: Path | None = None
        stem = Path(wsi.path).stem
        logger.info(
            f"{stem}: start extraction (batch={batch}, store_images={store_images}, fast_mode={fast_mode})"
        )

        if image_output_dir is not None:
            patch_imgs_dir = Path(image_output_dir)
            patch_imgs_dir.mkdir(parents=True, exist_ok=True)

        # Iterate contours and write in batches
        buf_imgs: list[np.ndarray] = []
        buf_xy: list[tuple[int, int]] = []
        buf_ext: list[tuple[int, int, int, int, int]] = []

        def flush() -> None:
            nonlocal first_write, total
            if not buf_imgs:
                # When not storing images, rely on coords buffer for flushing
                if not buf_xy:
                    return
            # Prepare buffers
            imgs = np.asarray(buf_imgs, dtype=np.uint8) if store_images else None
            coords = np.asarray(buf_xy, dtype=np.int32)
            coords_ext = np.asarray(buf_ext, dtype=np.int32)
            attrs = (
                {
                    "imgs": {"description": "RGB patches"},
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
                file_attrs = {"patch_size": int(self.patch_size), "wsi_path": wsi.path}

            # Build payload according to configuration
            assets: dict[str, np.ndarray] = {"coords": coords, "coords_ext": coords_ext}
            if store_images:
                assert imgs is not None
                assets["imgs"] = imgs

            save_h5(output_path, assets, attributes=attrs, mode=mode, file_attrs=file_attrs)

            total += int(coords.shape[0])
            buf_imgs.clear()
            buf_xy.clear()
            buf_ext.clear()

        for cont, holes in zip(tissue_contours, holes_contours):
            if fast_mode and not store_images:
                # Coordinate-only fast path: avoid reading patches entirely
                x0, y0, ww, hh = cv2.boundingRect(cont)
                W, H = wsi.get_size(lv=0)
                if self.use_padding:
                    stop_x = x0 + ww
                    stop_y = y0 + hh
                else:
                    stop_x = min(x0 + ww, W - self.patch_size)
                    stop_y = min(y0 + hh, H - self.patch_size)

                for y in range(y0, stop_y, self.step_size):
                    for x in range(x0, stop_x, self.step_size):
                        if not self._in_tissue((x, y), cont, holes):
                            continue
                        buf_xy.append((x, y))
                        buf_ext.append((x, y, int(self.patch_size), int(self.patch_size), 0))
                        flush_count = len(buf_xy)
                        if flush_count >= batch:
                            flush()
                            first_write = False
            else:
                for x, y, patch in self.iter_patches(wsi, cont, holes, fast_mode=fast_mode):
                    if store_images:
                        buf_imgs.append(patch)
                    buf_xy.append((x, y))
                    buf_ext.append((x, y, int(self.patch_size), int(self.patch_size), 0))

                    # Decide flush condition based on whether images are persisted
                    flush_count = len(buf_imgs) if store_images else len(buf_xy)
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
                if store_images:
                    import h5py  # local import to avoid global dependency when unused

                    with h5py.File(output_path, "r") as f:
                        imgs_d = f["imgs"]
                        coords_d = f["coords"]
                        for i in range(total):
                            arr = imgs_d[i]
                            x, y = coords_d[i]
                            out_name = f"{stem}_x{x}_y{y}.png"
                            Image.fromarray(arr).save(str(patch_imgs_dir / out_name))
                else:
                    # When not storing images, read coords from H5 and re-extract
                    import h5py

                    with h5py.File(output_path, "r") as f:
                        coords_d = f["coords"]
                        for i in range(int(coords_d.shape[0])):
                            x, y = (int(v) for v in coords_d[i])
                            arr_any = wsi.extract(
                                (x, y), lv=0, wh=(self.patch_size, self.patch_size), mode="array"
                            )
                            if isinstance(arr_any, np.ndarray):
                                out_name = f"{stem}_x{x}_y{y}.png"
                                Image.fromarray(arr_any).save(str(patch_imgs_dir / out_name))

            logger.info(f"{stem}: extraction complete, total patches={total}")
            return output_path

        return None
