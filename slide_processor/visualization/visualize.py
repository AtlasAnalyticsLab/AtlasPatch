"""Visualization functions for patch extraction results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from PIL.ImageFont import ImageFont as PILImageFont

logger = logging.getLogger(__name__)


def visualize_patches_on_thumbnail(
    hdf5_path: str,
    wsi,
    output_dir: str,
    cli_args: dict[str, Any] | None = None,
) -> str:
    """
    Visualize patch locations overlaid on WSI thumbnail with processing information.

    Creates a single image with the WSI thumbnail showing extracted patch locations as
    green rectangles, with an information panel overlaid in the top-right corner.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file containing extracted patches and coordinates.
    wsi : IWSI
        Opened WSI object implementing get_thumb/get_size and exposing `.path`.
    output_dir : str
        Directory where the visualization image will be saved.
    cli_args : dict[str, Any] | None, default None
        Dictionary of CLI arguments used for extraction. If provided, key parameters
        are displayed in the info box.

    Returns
    -------
    str
        Path to the saved visualization image.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file or WSI file does not exist.
    KeyError
        If required datasets are missing from the HDF5 file.
    """
    logger.info("Creating patch overlay visualization on thumbnail...")

    # Read coordinates and metadata from HDF5
    with h5py.File(hdf5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' dataset not found in {hdf5_path}")
        coords = f["coords"][:]
        # Always use level-0 patch size from file attrs
        patch_size = int(f.attrs["patch_size_level0"])  # raises KeyError if missing

    num_patches = len(coords)
    logger.debug(f"Found {num_patches} patches in HDF5 file")

    # Load WSI thumbnail via project abstraction (aspect-preserving)
    thumbnail = wsi.get_thumb((1024, 1024))
    thumbnail_image = thumbnail.convert("RGB")
    wsi_dims = wsi.get_size(lv=0)

    # Calculate downsample factors
    downsample_x = wsi_dims[0] / thumbnail_image.width
    downsample_y = wsi_dims[1] / thumbnail_image.height

    # Scale coordinates to thumbnail space
    coords_thumb = coords.astype(np.float32)
    coords_thumb[:, 0] = coords_thumb[:, 0] / downsample_x
    coords_thumb[:, 1] = coords_thumb[:, 1] / downsample_y
    patch_size_thumb_x = patch_size / downsample_x
    patch_size_thumb_y = patch_size / downsample_y

    # Draw rectangles directly on thumbnail
    draw = ImageDraw.Draw(thumbnail_image, "RGBA")

    # Draw all patches as green rectangles
    for coord in coords_thumb:
        x0 = int(coord[0])
        y0 = int(coord[1])
        x1 = int(coord[0] + patch_size_thumb_x)
        y1 = int(coord[1] + patch_size_thumb_y)

        # Draw green rectangle with transparent fill
        draw.rectangle(((x0, y0), (x1, y1)), outline=(0, 255, 0), width=1)

    # Build information text
    info_lines = []
    info_lines.append(f"Patches Extracted: {num_patches}")
    info_lines.append(f"WSI Size: {wsi_dims[0]} x {wsi_dims[1]}")

    # Parameters used
    if cli_args is not None:
        info_lines.append(f"Patch Size: {cli_args.get('patch_size', 'N/A')}")
        info_lines.append(f"Step Size: {cli_args.get('step_size', 'N/A')}")
        info_lines.append(f"Tissue Threshold: {cli_args.get('tissue_thresh', 'N/A')}")

    info_text = "\n".join(info_lines)

    # Draw info box in top-right corner
    _draw_info_box(thumbnail_image, info_text)

    # Save the image
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wsi.path).stem
    output_path = out_dir / f"{stem}.png"
    thumbnail_image.save(output_path, quality=95)

    logger.info(f"Saved patch overlay visualization: {output_path}")
    return str(output_path)


def _draw_info_box(image: Image.Image, text: str, padding: int = 10) -> None:
    """
    Draw a semi-transparent white box with text in the top-right corner of the image.

    Parameters
    ----------
    image : Image.Image
        PIL Image object to draw on (modified in-place).
    text : str
        Multi-line text to display in the box.
    padding : int
        Padding around text in pixels.
    """
    draw = ImageDraw.Draw(image, "RGBA")

    # Try to use a default font, fall back to default if not available
    font: FreeTypeFont | PILImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    # Calculate text bounding box
    lines = text.split("\n")
    line_height = 16
    max_width = 0

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        max_width = max(max_width, line_width)

    text_height = len(lines) * line_height
    box_width = max_width + 2 * padding
    box_height = text_height + 2 * padding

    # Position in top-right corner
    x1 = image.width - box_width - 10
    y1 = 10
    x2 = image.width - 10
    y2 = y1 + box_height

    # Draw semi-transparent white rectangle
    draw.rectangle(((x1, y1), (x2, y2)), fill=(255, 255, 255, 230), outline=(0, 0, 0, 255), width=2)

    # Draw text
    text_x = x1 + padding
    text_y = y1 + padding
    for i, line in enumerate(lines):
        draw.text((text_x, text_y + i * line_height), line, fill=(0, 0, 0, 255), font=font)
