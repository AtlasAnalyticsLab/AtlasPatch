from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from atlas_patch.core.wsi.iwsi import IWSI
from atlas_patch.utils.contours import scale_contours


def visualize_contours_on_thumbnail(
    *,
    tissue_contours: Sequence[np.ndarray],
    holes_contours: Sequence[Sequence[np.ndarray]],
    wsi: IWSI,
    output_dir: Path,
    thumbnail_size: int,
    mask_shape: tuple[int, int] | None = None,
) -> Path:
    """Visualize tissue and hole contours on a thumbnail."""
    thumb = wsi.get_thumb((thumbnail_size, thumbnail_size)).convert("RGB")
    tw, th = thumb.width, thumb.height

    # Scale contours from mask resolution to thumbnail resolution
    if mask_shape is not None:
        mask_h, mask_w = mask_shape
        sx = float(tw) / float(mask_w)
        sy = float(th) / float(mask_h)
    else:
        # Fallback to level-0 scaling (legacy behavior)
        W0, H0 = wsi.get_size(lv=0)
        sx = float(tw) / float(W0)
        sy = float(th) / float(H0)

    tcs = scale_contours(list(tissue_contours), sx, sy)
    holes_flat = [h for hs in holes_contours for h in hs]
    hcs = scale_contours(holes_flat, sx, sy)

    canvas = np.array(thumb.convert("RGB"))
    if len(tcs) > 0:
        cv2.polylines(canvas, tcs, isClosed=True, color=(255, 0, 0), thickness=2)
    if len(hcs) > 0:
        cv2.polylines(canvas, hcs, isClosed=True, color=(0, 0, 255), thickness=1)

    out_img = Image.fromarray(canvas)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(wsi.path).stem}_contours.png"
    out_img.save(out_path, quality=95)
    return out_path
