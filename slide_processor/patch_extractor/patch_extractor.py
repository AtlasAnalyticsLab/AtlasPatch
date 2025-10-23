from __future__ import annotations

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
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        """Iterate (x, y, patch_rgb) for patches within the given contour."""
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

                # Filter out uninformative patches
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
        batch: int = 512,
    ) -> str | None:
        """Extract patches and save to HDF5 using utils.save_h5.

        Datasets:
          - "imgs": (N, H, W, 3) uint8
          - "coords": (N, 2) int32  (x, y)
        Root attributes set: patch_size, wsi_path, num_patches
        """
        total = 0
        first_write = True
        patch_imgs_dir: Path | None = None
        stem = Path(wsi.path).stem

        if image_output_dir is not None:
            patch_imgs_dir = Path(image_output_dir)
            patch_imgs_dir.mkdir(parents=True, exist_ok=True)

        # Iterate contours and write in batches
        buf_imgs: list[np.ndarray] = []
        buf_xy: list[tuple[int, int]] = []

        def flush() -> None:
            nonlocal first_write, total
            if not buf_imgs:
                return
            imgs = np.asarray(buf_imgs, dtype=np.uint8)
            coords = np.asarray(buf_xy, dtype=np.int32)
            attrs = (
                {
                    "imgs": {"description": "RGB patches"},
                    "coords": {"description": "(x, y) coordinates at level 0"},
                }
                if first_write
                else None
            )

            mode = "w" if first_write else "a"
            file_attrs = None
            if first_write:
                file_attrs = {"patch_size": int(self.patch_size), "wsi_path": wsi.path}

            save_h5(
                output_path,
                {"imgs": imgs, "coords": coords},
                attributes=attrs,
                mode=mode,
                file_attrs=file_attrs,
            )

            total += len(imgs)
            buf_imgs.clear()
            buf_xy.clear()

        for cont, holes in zip(tissue_contours, holes_contours):
            for x, y, patch in self.iter_patches(wsi, cont, holes):
                buf_imgs.append(patch)
                buf_xy.append((x, y))

                if len(buf_imgs) >= batch:
                    flush()
                    first_write = False

        # final flush
        flush()
        if not first_write:
            # write final attrs
            save_h5(output_path, {}, mode="a", file_attrs={"num_patches": int(total)})

            # optional image export (read back from file for consistency)
            if patch_imgs_dir is not None and total > 0:
                import h5py  # local import to avoid global dependency when unused

                with h5py.File(output_path, "r") as f:
                    imgs = f["imgs"]
                    coords = f["coords"]
                    for i in range(total):
                        arr = imgs[i]
                        x, y = coords[i]
                        out_name = f"{stem}_x{x}_y{y}.png"
                        Image.fromarray(arr).save(str(patch_imgs_dir / out_name))

            return output_path

        return None
