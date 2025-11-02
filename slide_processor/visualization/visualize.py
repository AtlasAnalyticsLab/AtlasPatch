"""Visualization functions for patch extraction results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
from matplotlib.collections import PatchCollection

logger = logging.getLogger(__name__)


def visualize_patches_on_thumbnail(
    hdf5_path: str,
    wsi_path: str,
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
    wsi_path : str
        Path to the whole slide image file.
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

    # Load WSI and get thumbnail
    wsi = openslide.OpenSlide(wsi_path)
    thumbnail = wsi.get_thumbnail((1024, 1024))
    thumbnail_image = np.array(thumbnail)
    wsi_dims = wsi.dimensions
    wsi.close()

    # Calculate downsample factors
    downsample_x = wsi_dims[0] / thumbnail_image.shape[1]
    downsample_y = wsi_dims[1] / thumbnail_image.shape[0]

    # Scale coordinates to thumbnail space
    coords_thumb = coords.astype(np.float32).copy()
    coords_thumb[:, 0] = coords_thumb[:, 0] / downsample_x
    coords_thumb[:, 1] = coords_thumb[:, 1] / downsample_y
    patch_size_thumb_x = patch_size / downsample_x
    patch_size_thumb_y = patch_size / downsample_y

    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(thumbnail_image)

    # Draw all patch rectangles efficiently using PatchCollection
    patches = [
        mpatches.Rectangle(
            (coord[0], coord[1]),
            patch_size_thumb_x,
            patch_size_thumb_y,
        )
        for coord in coords_thumb
    ]
    pc = PatchCollection(
        patches,
        edgecolors="lime",
        facecolors="none",
        alpha=0.6,
        linewidths=0.5,
    )
    ax.add_collection(pc)
    ax.axis("off")

    # Build information text
    info_lines = []
    info_lines.append(f"  Patches Extracted: {num_patches}")
    info_lines.append(f"  WSI Size: {wsi_dims[0]} x {wsi_dims[1]}")

    # Parameters used
    if cli_args is not None:
        info_lines.append(f"  Patch Size: {cli_args.get('patch_size', 'N/A')}")
        info_lines.append(f"  Step Size: {cli_args.get('step_size', 'N/A')}")
        info_lines.append(f"  Thumbnail Size: {cli_args.get('thumbnail_size', 'N/A')}")
        info_lines.append(f"  Tissue Threshold: {cli_args.get('tissue_thresh', 'N/A')}")

    info_text = "\n".join(info_lines)

    # Add text to top-right corner of the image
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "black",
            "linewidth": 1.5,
        },
    )
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wsi_path).stem
    output_path = out_dir / f"{stem}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved patch overlay visualization: {output_path}")
    return str(output_path)
