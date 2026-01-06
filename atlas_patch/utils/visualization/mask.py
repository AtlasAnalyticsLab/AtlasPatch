from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from atlas_patch.core.wsi.iwsi import IWSI


def visualize_mask_on_thumbnail(
    *,
    mask: np.ndarray,
    wsi: IWSI,
    output_dir: Path,
    thumbnail_size: int,
) -> Path:
    """Visualize the predicted mask as a semi-transparent overlay on the WSI thumbnail.

    Also saves a black/white mask preview with the same thumbnail sizing.
    """
    thumb = wsi.get_thumb((thumbnail_size, thumbnail_size)).convert("RGB")

    mask_float = (mask.astype(np.float32) > 0.5).astype(np.float32)
    mh, mw = mask_float.shape[:2]
    if (mw, mh) != (thumb.width, thumb.height):
        m_img = Image.fromarray((mask_float * 255).astype(np.uint8), mode="L")
        m_img = m_img.resize((thumb.width, thumb.height), resample=Image.Resampling.NEAREST)
        mask_float = np.asarray(m_img, dtype=np.float32) / 255.0

    output_dir.mkdir(parents=True, exist_ok=True)
    bw_path = output_dir / f"{Path(wsi.path).stem}_mask_bw.png"
    Image.fromarray((mask_float * 255).astype(np.uint8), mode="L").save(bw_path)

    alpha = 80
    mask_rgba = Image.fromarray((mask_float * alpha).astype(np.uint8), mode="L")
    green_layer = Image.new("RGBA", thumb.size, (0, 255, 0, 0))
    green_layer.putalpha(mask_rgba)

    out_img = thumb.convert("RGBA")
    out_img = Image.alpha_composite(out_img, green_layer)

    out_path = output_dir / f"{Path(wsi.path).stem}_mask.png"
    out_img.convert("RGB").save(out_path, quality=95)
    return out_path
