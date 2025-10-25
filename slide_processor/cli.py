from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

from slide_processor.pipeline.patchify import (
    PatchifyParams,
    SegmentParams,
    _build_segmentation_predictor,
    segment_and_patchify,
)
from slide_processor.visualization import (
    visualize_patches_on_thumbnail,
    visualize_random_patches,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Suppress a noisy third-party info log emitted to the root logger.
# Example: "For numpy array image, we assume (HxWxC) format"
class _SuppressNumpyHxWxCFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        msg = record.getMessage()
        return "For numpy array image, we assume (HxWxC) format" not in msg


logging.getLogger().addFilter(_SuppressNumpyHxWxCFilter())
logger = logging.getLogger("slide_processor.cli")


def validate_path(ctx, param, value):
    """Validate that file/directory path exists."""
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Path does not exist: {value}")
    return str(path.absolute())


def validate_positive_int(ctx, param, value):
    """Validate that value is a positive integer."""
    if value is not None and value <= 0:
        raise click.BadParameter(f"{param.name} must be positive, got {value}")
    return value


def get_wsi_files(path: str) -> list[str]:
    """Get list of WSI files from path (file or directory).

    Supported formats:
    - OpenSlide: .svs, .tif, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .dcm
    - Image: .png, .jpg, .jpeg, .bmp, .webp, .gif
    """
    supported_exts = {
        ".svs",
        ".tif",
        ".tiff",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".bif",
        ".biff",
        ".dcm",
        ".dicom",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".gif",
    }
    path_obj = Path(path)

    if path_obj.is_file():
        if path_obj.suffix.lower() not in supported_exts:
            logger.warning(f"File may not be a supported WSI format: {path_obj.name}")
        return [str(path_obj)]

    # Directory: collect all supported files
    files: list[Path] = []
    for ext in supported_exts:
        files.extend(path_obj.glob(f"*{ext}"))
        files.extend(path_obj.glob(f"*{ext.upper()}"))

    if not files:
        raise click.ClickException(
            f"No WSI files found in directory: {path}\n"
            f"Supported formats: SVS, TIF, TIFF, NDPI, PNG, JPG, etc."
        )

    return sorted([str(f) for f in files])


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """SlideProcessor: Whole Slide Image processing with SAM2 segmentation.

    Process whole slide images by:
    1. Segmenting tissue using SAM2 model
    2. Extracting patches from segmented tissue regions
    3. Saving patches to HDF5 format for efficient data handling

    Examples:
        # Process single WSI file (YAML config path)
        slideproc process wsi.svs --checkpoint ckpt.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml

        # Process directory of WSI files
        slideproc process ./wsi_folder/ --checkpoint ckpt.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml

        # With custom patch settings
        slideproc process wsi.svs --checkpoint ckpt.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \\
            --patch-size 512 --step-size 256 --output ./output

        # Export individual patch images
        slideproc process wsi.svs --checkpoint ckpt.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \\
            --save-images --output ./output

    For detailed help on specific commands, run:
        slideproc <command> --help
    """


@cli.command()
@click.argument("wsi_path", type=click.Path(exists=True), callback=validate_path)
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True),
    required=True,
    callback=validate_path,
    help="Path to SAM2 model checkpoint (.pt file).",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    callback=validate_path,
    help=("Path to SAM2 YAML config file. Only filesystem paths are accepted."),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for HDF5 files and patches. [default: ./output]",
)
@click.option(
    "--patch-size",
    type=int,
    default=256,
    callback=validate_positive_int,
    help="Size of extracted patches in pixels. [default: 256]",
)
@click.option(
    "--step-size",
    type=int,
    default=None,
    callback=validate_positive_int,
    help="Step size for patch extraction (stride). Defaults to patch-size if not set.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default="cuda",
    help="Device for model inference. [default: cuda]",
)
@click.option(
    "--thumbnail-size",
    type=int,
    default=1024,
    callback=validate_positive_int,
    help="Size of thumbnail for segmentation (max dimension). [default: 1024]",
)
@click.option(
    "--tissue-thresh",
    type=float,
    default=0.01,
    help="Minimum tissue area threshold as percentage of image. [default: 0.01]",
)
@click.option(
    "--white-thresh",
    type=int,
    default=15,
    callback=validate_positive_int,
    help="Saturation threshold for filtering white patches. [default: 15]",
)
@click.option(
    "--black-thresh",
    type=int,
    default=50,
    callback=validate_positive_int,
    help="RGB threshold for filtering black patches. [default: 50]",
)
@click.option(
    "--require-all-points",
    is_flag=True,
    default=False,
    help="Require all 4 corner points inside tissue (strict mode). "
    "Default is to require any point inside (lenient mode).",
)
@click.option(
    "--use-padding",
    is_flag=True,
    default=True,
    help="Allow patches at image boundaries with padding. [default: True]",
)
@click.option(
    "--save-images",
    is_flag=True,
    default=False,
    help="Export individual patch images as PNG files.",
)
@click.option(
    "--h5-images/--no-h5-images",
    default=True,
    help=(
        "Store image arrays in the HDF5 file ('imgs' dataset). "
        "Disable to save only coordinates + metadata. [default: --h5-images]"
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging output.",
)
@click.option(
    "--fast-mode",
    is_flag=True,
    default=False,
    help=(
        "Skip per-patch content filtering to speed up extraction. "
        "May include more background patches."
    ),
)
@click.option(
    "--visualize",
    is_flag=True,
    default=False,
    help="Generate visualization of patches overlaid on WSI thumbnail with processing info.",
)
@click.option(
    "--show-random-patches",
    type=int,
    default=None,
    help="Visualize N random patches in a grid. Example: --show-random-patches 20",
)
def process(
    wsi_path: str,
    checkpoint: str,
    config: str,
    output: str,
    patch_size: int,
    step_size: int | None,
    device: str,
    thumbnail_size: int,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    require_all_points: bool,
    use_padding: bool,
    save_images: bool,
    h5_images: bool,
    verbose: bool,
    fast_mode: bool,
    visualize: bool,
    show_random_patches: int | None,
):
    """Process whole slide image(s) with tissue segmentation and patch extraction.

    This command processes WSI files by:
    1. Loading the WSI (supports .svs, .tiff, .png, .jpg, etc.)
    2. Generating a thumbnail and predicting tissue segmentation with SAM2
    3. Extracting tissue regions as contours
    4. Iterating over tissue regions to extract patches
    5. Saving patches to HDF5 format with coordinates

    WSI_PATH can be either:
    - A single WSI file (e.g., sample.svs)
    - A directory containing multiple WSI files (for batch processing)

    Examples:

        # Single file processing
        slideproc process sample.svs \\
            --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml

        # Batch processing (all .svs files in directory)
        slideproc process ./slides/ \\
            --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \\
            --output ./processed_slides

        # Custom patch settings with image export
        slideproc process sample.svs \\
            --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \\
            --patch-size 512 --step-size 256 \\
            --save-images --output ./results

        # Strict tissue requirement (all 4 corners must be in tissue)
        slideproc process sample.svs \\
            --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \\
            --require-all-points

        # Using CPU instead of GPU
        slideproc process sample.svs \\
            --checkpoint model.pt --config slide_processor/configs/sam2.1_hiera_b+.yaml \\
            --device cpu
    """
    if verbose:
        logging.getLogger("slide_processor").setLevel(logging.DEBUG)

    # Validate parameters
    if patch_size <= 0:
        raise click.ClickException("--patch-size must be positive")
    if step_size is not None and step_size <= 0:
        raise click.ClickException("--step-size must be positive")
    if tissue_thresh < 0 or tissue_thresh > 100:
        raise click.ClickException("--tissue-thresh must be between 0 and 100")

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Get list of WSI files to process
    try:
        wsi_files = get_wsi_files(wsi_path)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error reading WSI path: {e}") from e

    num_files = len(wsi_files)
    logger.info(f"Found {num_files} WSI file(s) to process")

    # Set step size to patch size if not specified
    effective_step_size = step_size if step_size is not None else patch_size

    # Create segmentation and patchification parameters
    seg_params = SegmentParams(
        checkpoint_path=Path(checkpoint),
        config_file=Path(config),
        device=device.lower(),
        thumbnail_max=thumbnail_size,
    )

    patch_params = PatchifyParams(
        patch_size=patch_size,
        step_size=effective_step_size,
        tissue_area_thresh=tissue_thresh,
        require_all_points=require_all_points,
        white_thresh=white_thresh,
        black_thresh=black_thresh,
        use_padding=use_padding,
    )

    # Process each WSI file
    successful = 0
    failed = 0

    # Build SAM2 predictor once and reuse across files
    predict_fn, thumb_max = _build_segmentation_predictor(seg_params)

    with click.progressbar(wsi_files, label="Processing WSI files") as pbar:
        for wsi_file in pbar:
            try:
                logger.info(f"Processing: {Path(wsi_file).name}")

                # Track processing time
                start_time = time.time()

                result_h5 = segment_and_patchify(
                    wsi_path=wsi_file,
                    output_dir=str(output_path),
                    seg=seg_params,
                    patch=patch_params,
                    save_images=save_images,
                    store_images=h5_images,
                    fast_mode=fast_mode,
                    predict_fn=predict_fn,
                    thumb_max=thumb_max,
                )

                processing_time = time.time() - start_time

                if result_h5:
                    successful += 1
                    logger.info(f"Saved patches to: {result_h5}")

                    # Generate visualizations if requested
                    stem = Path(wsi_file).stem
                    vis_output_dir = output_path / stem

                    # Prepare CLI arguments for info box
                    cli_args_dict = {
                        "patch_size": patch_size,
                        "step_size": effective_step_size,
                        "thumbnail_size": thumbnail_size,
                        "device": device,
                        "tissue_thresh": tissue_thresh,
                        "white_thresh": white_thresh,
                        "black_thresh": black_thresh,
                        "require_all_points": require_all_points,
                        "use_padding": use_padding,
                        "fast_mode": fast_mode,
                        "save_images": save_images,
                        "h5_images": h5_images,
                    }

                    if visualize:
                        try:
                            vis_path = visualize_patches_on_thumbnail(
                                hdf5_path=result_h5,
                                wsi_path=wsi_file,
                                output_dir=str(vis_output_dir),
                                patch_size=patch_size,
                                processing_time=processing_time,
                                cli_args=cli_args_dict,
                            )
                            logger.info(f"Visualization saved to: {vis_path}")
                        except Exception as e:
                            logger.error(f"Failed to generate visualization: {e}")
                            if verbose:
                                raise

                    if show_random_patches is not None:
                        try:
                            # Determine if we need to pass wsi_path
                            wsi_path_arg = None if h5_images else wsi_file

                            random_vis_path = visualize_random_patches(
                                hdf5_path=result_h5,
                                output_dir=str(vis_output_dir),
                                wsi_path=wsi_path_arg,
                                n_patches=show_random_patches,
                            )
                            logger.info(f"Random patches visualization saved to: {random_vis_path}")
                        except Exception as e:
                            logger.error(f"Failed to generate random patches visualization: {e}")
                            if verbose:
                                raise
                else:
                    logger.warning(f"No patches extracted from {Path(wsi_file).name}")

            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {Path(wsi_file).name}: {e}")
                if verbose:
                    raise

    if failed > 0 and failed == num_files:
        raise click.ClickException("All files failed to process. Check logs for details.")

    logger.info("Processing complete!")


@cli.command()
def info():
    """Display information about supported WSI formats and features."""
    click.echo("\n" + "=" * 70)
    click.echo("SlideProcessor - Supported Formats and Features")
    click.echo("=" * 70)

    click.echo("\nSupported WSI Formats (via OpenSlide):")
    click.echo("  • .svs   - Aperio SVS")
    click.echo("  • .tif   - TIFF/BigTIFF")
    click.echo("  • .tiff  - TIFF/BigTIFF")
    click.echo("  • .ndpi  - Hamamatsu NDPI")
    click.echo("  • .vms   - Ventana VMS")
    click.echo("  • .vmu   - Ventana VMU")
    click.echo("  • .scn   - Leica SCN")
    click.echo("  • .mrxs  - MIRAX MRXS")
    click.echo("  • .bif   - Olympus BIF")
    click.echo("  • .dcm   - DICOM DCM")

    click.echo("\nSupported Image Formats (fallback):")
    click.echo("  • .png")
    click.echo("  • .jpg  / .jpeg")
    click.echo("  • .bmp")
    click.echo("  • .webp")
    click.echo("  • .gif")

    click.echo("\nOutput Format:")
    click.echo("  • HDF5 file per WSI containing:")
    click.echo("    - 'imgs': (N, H, W, 3) uint8 RGB patches (optional)")
    click.echo("    - 'coords': (N, 2) int32 (x, y) coordinates")
    click.echo("    - 'coords_ext': (N, 5) int32 (x, y, w, h, level)")
    click.echo("    - Metadata: patch_size, wsi_path, num_patches")
    click.echo("  • Optional PNG patches in 'images/' subdirectory")

    click.echo("\n" + "=" * 70 + "\n")


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except click.ClickException as e:
        click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
