from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from PIL.ImageFont import ImageFont as PILImageFont

from atlas_patch.core.wsi.iwsi import IWSI


def _draw_info_box(image: Image.Image, text: str, padding: int = 10) -> None:
    draw = ImageDraw.Draw(image, "RGBA")

    font: FreeTypeFont | PILImageFont
    font = ImageFont.load_default()

    lines = text.split("\n")
    line_height = 16
    max_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_width = max(max_width, bbox[2] - bbox[0])

    text_height = len(lines) * line_height
    box_width = max_width + 2 * padding
    box_height = text_height + 2 * padding

    x1 = image.width - box_width - 10
    y1 = 10
    x2 = image.width - 10
    y2 = y1 + box_height

    draw.rectangle(((x1, y1), (x2, y2)), fill=(255, 255, 255, 230), outline=(0, 0, 0, 255), width=2)

    text_x = x1 + padding
    text_y = y1 + padding
    for i, line in enumerate(lines):
        draw.text((text_x, text_y + i * line_height), line, fill=(0, 0, 0, 255), font=font)


def visualize_patches_on_thumbnail(
    *,
    coords: np.ndarray,
    patch_size_level0: int,
    wsi: IWSI,
    output_dir: Path,
    thumbnail_size: int,
    info: dict[str, Any] | None = None,
) -> Path:
    """Overlay patch boxes onto a thumbnail without reading from H5."""
    thumbnail = wsi.get_thumb((thumbnail_size, thumbnail_size)).convert("RGB")
    W0, H0 = wsi.get_size(lv=0)
    downsample_x = W0 / thumbnail.width
    downsample_y = H0 / thumbnail.height

    coords_thumb = coords.astype(np.float32)
    coords_thumb[:, 0] = coords_thumb[:, 0] / float(downsample_x)
    coords_thumb[:, 1] = coords_thumb[:, 1] / float(downsample_y)
    patch_size_thumb_x = float(patch_size_level0) / float(downsample_x)
    patch_size_thumb_y = float(patch_size_level0) / float(downsample_y)

    draw = ImageDraw.Draw(thumbnail, "RGBA")
    for coord_x, coord_y in coords_thumb.astype(float):
        x0 = int(coord_x)
        y0 = int(coord_y)
        x1 = int(coord_x + patch_size_thumb_x)
        y1 = int(coord_y + patch_size_thumb_y)
        draw.rectangle(((x0, y0), (x1, y1)), outline=(0, 0, 0), width=1)

    info_lines = [
        f"Patches Extracted: {len(coords)}",
        f"WSI Size: {W0} x {H0}",
    ]
    if info:
        if "patch_size" in info:
            info_lines.append(f"Patch Size: {info['patch_size']}")
        if "step_size" in info:
            info_lines.append(f"Step Size: {info['step_size']}")
        if "tissue_thresh" in info:
            info_lines.append(f"Tissue Threshold: {info['tissue_thresh']}")

    _draw_info_box(thumbnail, "\n".join(info_lines))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(wsi.path).stem}.png"
    thumbnail.save(out_path, quality=95)
    return out_path
