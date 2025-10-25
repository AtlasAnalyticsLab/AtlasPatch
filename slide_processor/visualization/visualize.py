"""Visualization functions for patch extraction results."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide

logger = logging.getLogger(__name__)


def visualize_patches_on_thumbnail(
    hdf5_path: str,
    wsi_path: str,
    output_dir: str,
    patch_size: int = 256,
    processing_time: float | None = None,
    cli_args: dict[str, Any] | None = None,
) -> str:
    """
    Visualize patch locations overlaid on WSI thumbnail with processing information.

    Creates a figure with the WSI thumbnail showing extracted patch locations as green
    rectangles, alongside an information panel displaying extraction statistics and
    processing parameters.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file containing extracted patches and coordinates.
    wsi_path : str
        Path to the whole slide image file.
    output_dir : str
        Directory where the visualization image will be saved.
    patch_size : int, default 256
        Size of extracted patches at level 0 (in pixels).
    processing_time : float | None, default None
        Time taken for patch extraction in seconds. If provided, displayed in info box.
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

    # Read coordinates from HDF5
    with h5py.File(hdf5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' dataset not found in {hdf5_path}")
        coords = f["coords"][:]

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

    # Create figure with two subplots: thumbnail + info box
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

    # Left subplot: Thumbnail with patches
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(thumbnail_image)

    # Draw all patch rectangles
    for coord in coords_thumb:
        rect = mpatches.Rectangle(
            (coord[0], coord[1]),
            patch_size_thumb_x,
            patch_size_thumb_y,
            linewidth=0.5,
            edgecolor="lime",
            facecolor="none",
            alpha=0.6,
        )
        ax_img.add_patch(rect)

    ax_img.set_title(
        f"Patch Locations on WSI Thumbnail\n{num_patches} patches extracted",
        fontsize=16,
        fontweight="bold",
    )
    ax_img.axis("off")

    # Right subplot: Information panel
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    # Build information text
    info_lines = []
    info_lines.append("Processing Information")
    info_lines.append("=" * 30)
    info_lines.append("")

    # Basic statistics
    info_lines.append("Extraction Results:")
    info_lines.append(f"  Patches Extracted: {num_patches}")
    info_lines.append(f"  WSI Size: {wsi_dims[0]} x {wsi_dims[1]}")

    if processing_time is not None:
        info_lines.append(f"  Time Taken: {processing_time:.2f}s")

    # Parameters used
    if cli_args is not None:
        info_lines.append("")
        info_lines.append("Parameters Used:")
        info_lines.append(f"  Patch Size: {cli_args.get('patch_size', 'N/A')}")
        info_lines.append(f"  Step Size: {cli_args.get('step_size', 'N/A')}")
        info_lines.append(f"  Thumbnail Size: {cli_args.get('thumbnail_size', 'N/A')}")
        info_lines.append(f"  Device: {cli_args.get('device', 'N/A')}")
        info_lines.append(f"  Tissue Threshold: {cli_args.get('tissue_thresh', 'N/A')}")
        info_lines.append(f"  White Threshold: {cli_args.get('white_thresh', 'N/A')}")
        info_lines.append(f"  Black Threshold: {cli_args.get('black_thresh', 'N/A')}")
        info_lines.append(f"  Require All Points: {cli_args.get('require_all_points', 'N/A')}")
        info_lines.append(f"  Use Padding: {cli_args.get('use_padding', 'N/A')}")
        info_lines.append(f"  Fast Mode: {cli_args.get('fast_mode', 'N/A')}")
        info_lines.append(f"  Save Images: {cli_args.get('save_images', 'N/A')}")
        info_lines.append(f"  Store H5 Images: {cli_args.get('h5_images', 'N/A')}")

    info_text = "\n".join(info_lines)

    # Add text to info panel with left alignment
    ax_info.text(
        0.05,
        0.95,
        info_text,
        transform=ax_info.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        bbox={"boxstyle": "round,pad=1", "facecolor": "lightgray", "alpha": 0.8},
    )

    # Suppress tight_layout warning for axes with text boxes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        plt.tight_layout()

    # Save visualization
    output_path = Path(output_dir) / "patches_on_thumbnail.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved patch overlay visualization: {output_path}")
    return str(output_path)


def visualize_random_patches(
    hdf5_path: str,
    output_dir: str,
    wsi_path: str | None = None,
    n_patches: int = 10,
) -> str:
    """
    Visualize random patches from HDF5 file in a grid layout.

    Creates a grid visualization showing randomly selected patches with their
    indices and coordinates. If the HDF5 file does not contain image data,
    patches are read directly from the WSI using stored coordinates.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file containing patches and/or coordinates.
    output_dir : str
        Directory where the visualization image will be saved.
    wsi_path : str | None, default None
        Path to WSI file. Required if HDF5 file does not contain image data
        (i.e., when --no-h5-images was used during extraction).
    n_patches : int, default 10
        Number of random patches to display in the grid.

    Returns
    -------
    str
        Path to the saved visualization image.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file or WSI file (when required) does not exist.
    ValueError
        If n_patches is less than 1, or if HDF5 lacks images and no WSI path provided.
    KeyError
        If required datasets are missing from the HDF5 file.
    """
    logger.info(f"Visualizing {n_patches} random patches from HDF5...")

    if n_patches < 1:
        raise ValueError(f"n_patches must be at least 1, got {n_patches}")

    with h5py.File(hdf5_path, "r") as f:
        total_patches = f["coords"].shape[0]

        # Adjust if requesting more patches than available
        if total_patches < n_patches:
            logger.warning(
                f"Requested {n_patches} patches but only {total_patches} available. "
                f"Showing all {total_patches} patches."
            )
            n_patches = total_patches

        # Select random indices
        random_indices = np.random.choice(total_patches, n_patches, replace=False)
        random_indices = np.sort(random_indices)

        # Check if images are stored in HDF5
        has_images = "imgs" in f

        if has_images:
            # Read patches directly from HDF5
            patches_list = []
            coords_list = []
            for idx in random_indices:
                patches_list.append(f["imgs"][idx])
                coords_list.append(f["coords"][idx])

            patches = np.array(patches_list)
            coords = np.array(coords_list)
        else:
            # Need to read patches from WSI using coords_ext
            if wsi_path is None:
                raise ValueError(
                    "HDF5 file does not contain image data. "
                    "Please provide wsi_path to read patches from the original WSI."
                )

            if "coords_ext" not in f:
                raise KeyError(
                    "'coords_ext' dataset not found in HDF5. "
                    "Cannot reconstruct patches without extended coordinates."
                )

            logger.info("Reading patches from WSI (no images stored in HDF5)...")
            coords_ext_list = []
            coords_list = []
            for idx in random_indices:
                coords_ext_list.append(f["coords_ext"][idx])
                coords_list.append(f["coords"][idx])

            coords_ext = np.array(coords_ext_list)
            coords = np.array(coords_list)

            # Load patches from WSI
            from slide_processor.wsi import WSIFactory

            wsi = WSIFactory.load(wsi_path)
            patches_list = []
            for coord_ext in coords_ext:
                x, y, w, h, level = coord_ext
                patch = wsi.extract(
                    xy=(int(x), int(y)), lv=int(level), wh=(int(w), int(h)), mode="array"
                )
                patches_list.append(patch)

            try:
                wsi.cleanup()
            except Exception:
                pass

            patches = np.array(patches_list)

    # Create visualization grid
    n_cols = 5
    n_rows = (n_patches + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_patches == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (patch, coord, ax) in enumerate(zip(patches, coords, axes)):
        ax.imshow(patch)
        ax.set_title(
            f"Patch {random_indices[idx]}\nCoord: ({coord[0]}, {coord[1]})",
            fontsize=9,
        )
        ax.axis("off")

    # Hide extra subplots
    for idx in range(n_patches, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    output_path = Path(output_dir) / "random_patches_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved random patches visualization: {output_path}")
    return str(output_path)
