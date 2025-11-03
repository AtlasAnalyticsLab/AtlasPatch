from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from utils.params import get_wsi_files, validate_path, validate_positive_int
from utils.progress import ProgressReporter

from slide_processor.pipeline.patchify import (
    PatchifyParams,
    SegmentParams,
    segment_and_patchify,
)
from slide_processor.segmentation.sam2_segmentation import SAM2SegmentationModel
from slide_processor.visualization import (
    visualize_patches_on_thumbnail,
)
from slide_processor.wsi import WSIFactory

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # default to quiet output unless --verbose is used
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


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """SlideProcessor: Whole Slide Image processing with SAM2 segmentation.

    Process whole slide images by:
    1. Segmenting tissue using SAM2 model
    2. Extracting patches from segmented tissue regions
    3. Saving patches to HDF5 format for efficient data handling

    Examples:
        # Process single WSI file
        slideproc process wsi.svs --checkpoint ckpt.pt \\
            --patch-size 256 --target-mag 20

        # Process directory of WSI files
        slideproc process ./wsi_folder/ --checkpoint ckpt.pt \\
            --patch-size 256 --target-mag 20

        # With custom patch settings
        slideproc process wsi.svs --checkpoint ckpt.pt \\
            --patch-size 512 --step-size 256 --target-mag 20 --output ./output

        # Export individual patch images
        slideproc process wsi.svs --checkpoint ckpt.pt \\
            --save-images --patch-size 256 --target-mag 20 --output ./output

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
    "--output",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for results (patches/, images/, visualization/). [default: ./output]",
)
@click.option(
    "--patch-size",
    type=int,
    required=True,
    callback=validate_positive_int,
    help="Target size of extracted patches in pixels (required)",
)
@click.option(
    "--step-size",
    type=int,
    default=None,
    callback=validate_positive_int,
    help="Step size for patch extraction (stride). Defaults to patch-size if not set.",
)
@click.option(
    "--target-mag",
    type=click.Choice(["1", "2", "4", "5", "10", "20", "40", "60", "80"], case_sensitive=False),
    required=True,
    help=(
        "Target magnification for patch extraction (e.g., 40, 20, 10). "
        "Required. Coordinates are always saved at level 0."
    ),
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default="cuda",
    help="Device for model inference. [default: cuda]",
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
    "--save-images",
    is_flag=True,
    default=False,
    help="Export individual patch images as PNG files.",
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
    "--seg-batch-size",
    type=int,
    default=1,
    callback=validate_positive_int,
    help=(
        "Batch size for SAM2 thumbnail segmentation when processing a folder. "
        "Set >1 to enable batched inference (experimental). [default: 1]"
    ),
)
@click.option(
    "--write-batch",
    type=int,
    default=8192,
    callback=validate_positive_int,
    help=(
        "Rows per HDF5 append/flush when saving coordinates. Larger is faster but uses more memory. "
        "[default: 8192]"
    ),
)
def process(
    wsi_path: str,
    checkpoint: str,
    output: str,
    patch_size: int,
    step_size: int | None,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    save_images: bool,
    verbose: bool,
    fast_mode: bool,
    visualize: bool,
    target_mag: str,
    seg_batch_size: int,
    write_batch: int,
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
            --checkpoint model.pt --patch-size 256 --target-mag 20

        # Batch processing (all .svs files in directory)
        slideproc process ./slides/ \\
            --checkpoint model.pt --patch-size 256 --target-mag 20 \\
            --output ./processed_slides

        # Custom patch settings with image export
        slideproc process sample.svs \\
            --checkpoint model.pt --patch-size 512 --step-size 256 --target-mag 20 \\
            --save-images --output ./results

        # Using CPU instead of GPU
        slideproc process sample.svs \\
            --checkpoint model.pt --patch-size 256 --target-mag 20 \\
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
        device=device.lower(),
        thumbnail_max=1024,
    )

    patch_params = PatchifyParams(
        patch_size=patch_size,
        step_size=effective_step_size,
        target_magnification=int(target_mag),
        tissue_area_thresh=tissue_thresh,
        white_thresh=white_thresh,
        black_thresh=black_thresh,
    )

    # Process each WSI file
    successful = 0
    failed = 0

    # Build SAM2 model once and reuse across files
    sam2_model = SAM2SegmentationModel(
        checkpoint_path=seg_params.checkpoint_path,
        config_file=seg_params.config_file,
        device=seg_params.device,
    )

    def _predict_single(img):
        return sam2_model.predict_image(img, resize_to_input=True)

    predict_fn = _predict_single
    thumb_max = seg_params.thumbnail_max

    if verbose:
        # Verbose mode: log per-file, optionally batching segmentation
        pending: list[str] = []
        for wsi_file in wsi_files:
            try:
                logger.info(f"Processing: {Path(wsi_file).name}")

                stem = Path(wsi_file).stem
                existing_h5 = output_path / "patches" / f"{stem}.h5"
                if existing_h5.exists():
                    logger.info(f"Skipping {Path(wsi_file).name}: already exists -> {existing_h5}")
                    continue

                pending.append(wsi_file)

                def _process_batch(batch_files: list[str]) -> None:
                    nonlocal successful, failed
                    if not batch_files:
                        return
                    # Prepare thumbnails
                    thumbs = []
                    for f in batch_files:
                        wsi_tmp = WSIFactory.load(f)
                        try:
                            thumb = wsi_tmp.get_thumb((thumb_max, thumb_max))
                            thumbs.append(thumb)
                        finally:
                            try:
                                wsi_tmp.cleanup()
                            except Exception:
                                pass
                    # Predict masks (batched when seg_batch_size>1)
                    try:
                        if seg_batch_size > 1:
                            masks = sam2_model.predict_batch(thumbs, resize_to_input=True)
                        else:
                            masks = [predict_fn(t) for t in thumbs]
                    except Exception as e:
                        logger.error(f"Batch segmentation failed ({len(batch_files)} files): {e}")
                        # Fallback to per-image
                        masks = []
                        for t in thumbs:
                            try:
                                masks.append(predict_fn(t))
                            except Exception as ie:
                                logger.error(f"Segmentation failed for a thumbnail: {ie}")
                                masks.append(None)  # type: ignore

                    # Patchify each file with its mask
                    for f, m in zip(batch_files, masks):
                        try:
                            result_h5 = segment_and_patchify(
                                wsi_path=f,
                                output_dir=str(output_path),
                                seg=seg_params,
                                patch=patch_params,
                                save_images=save_images,
                                fast_mode=fast_mode,
                                predict_fn=predict_fn,
                                thumb_max=thumb_max,
                                mask_override=m if m is not None else None,
                                write_batch=write_batch,
                            )

                            if result_h5:
                                successful += 1
                                logger.info(f"Saved patches to: {result_h5}")

                                if visualize:
                                    vis_output_dir = output_path / "visualization"
                                    vis_output_dir.mkdir(parents=True, exist_ok=True)
                                    cli_args_dict = {
                                        "patch_size": patch_size,
                                        "step_size": effective_step_size,
                                        "thumbnail_size": 1024,
                                        "device": device,
                                        "tissue_thresh": tissue_thresh,
                                        "white_thresh": white_thresh,
                                        "black_thresh": black_thresh,
                                        "fast_mode": fast_mode,
                                        "save_images": save_images,
                                        "target_mag": int(target_mag),
                                    }
                                    try:
                                        vis_path = visualize_patches_on_thumbnail(
                                            hdf5_path=result_h5,
                                            wsi_path=f,
                                            output_dir=str(vis_output_dir),
                                            cli_args=cli_args_dict,
                                        )
                                        logger.info(f"Visualization saved to: {vis_path}")
                                    except Exception as ve:
                                        logger.warning(
                                            f"Visualization failed for {Path(f).name}: {ve}"
                                        )
                            else:
                                logger.warning(f"No patches extracted from {Path(f).name}")
                        except Exception as pe:
                            failed += 1
                            logger.error(f"Failed to process {Path(f).name}: {pe}")
                            raise

                # Flush batch
                if seg_batch_size > 1 and len(pending) >= seg_batch_size:
                    _process_batch(pending)
                    pending = []

            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {Path(wsi_file).name}: {e}")
                raise
        # Process any remaining files
        if pending:
            try:
                _process_batch(pending)
            except Exception:
                failed += 1
    else:
        reporter = ProgressReporter(num_files)

        with reporter.progress_bar("Processing WSI files") as pbar:
            pending = []
            for wsi_file in wsi_files:
                stem = Path(wsi_file).stem
                existing_h5 = output_path / "patches" / f"{stem}.h5"
                if existing_h5.exists():
                    reporter.update(success=True)
                    reporter.update_progress_bar(pbar)
                    continue

                pending.append(wsi_file)
                if seg_batch_size > 1 and len(pending) < seg_batch_size:
                    continue

                # Process current pending batch
                batch_files = pending
                pending = []

                # Build thumbnails
                thumbs = []
                for f in batch_files:
                    wsi_tmp = WSIFactory.load(f)
                    try:
                        thumbs.append(wsi_tmp.get_thumb((thumb_max, thumb_max)))
                    finally:
                        try:
                            wsi_tmp.cleanup()
                        except Exception:
                            pass

                try:
                    if seg_batch_size > 1:
                        masks = sam2_model.predict_batch(thumbs, resize_to_input=True)
                    else:
                        masks = [predict_fn(t) for t in thumbs]
                except Exception:
                    # Fallback to per-image segmentation if batch failed
                    masks = []
                    for t in thumbs:
                        try:
                            masks.append(predict_fn(t))
                        except Exception:
                            masks.append(None)  # type: ignore

                # Patchify each file
                for f, m in zip(batch_files, masks):
                    try:
                        result_h5 = segment_and_patchify(
                            wsi_path=f,
                            output_dir=str(output_path),
                            seg=seg_params,
                            patch=patch_params,
                            save_images=save_images,
                            fast_mode=fast_mode,
                            predict_fn=predict_fn,
                            thumb_max=thumb_max,
                            mask_override=m if m is not None else None,
                            write_batch=write_batch,
                        )

                        if result_h5:
                            reporter.update(success=True)
                        else:
                            reporter.update(success=False)

                        if visualize and result_h5:
                            vis_output_dir = output_path / "visualization"
                            vis_output_dir.mkdir(parents=True, exist_ok=True)

                            cli_args_dict = {
                                "patch_size": patch_size,
                                "step_size": effective_step_size,
                                "thumbnail_size": 1024,
                                "device": device,
                                "tissue_thresh": tissue_thresh,
                                "white_thresh": white_thresh,
                                "black_thresh": black_thresh,
                                "fast_mode": fast_mode,
                                "save_images": save_images,
                                "target_mag": int(target_mag),
                            }
                            try:
                                _ = visualize_patches_on_thumbnail(
                                    hdf5_path=result_h5,
                                    wsi_path=f,
                                    output_dir=str(vis_output_dir),
                                    cli_args=cli_args_dict,
                                )
                            except Exception as ve:
                                # Visualization failure should not mark the slide as failed processing
                                logger.warning(f"Visualization failed for {Path(f).name}: {ve}")
                    except Exception:
                        reporter.update(success=False)
                    finally:
                        reporter.update_progress_bar(pbar)

            # Process any remaining pending files
            if pending:
                thumbs = []
                for f in pending:
                    wsi_tmp = WSIFactory.load(f)
                    try:
                        thumbs.append(wsi_tmp.get_thumb((thumb_max, thumb_max)))
                    finally:
                        try:
                            wsi_tmp.cleanup()
                        except Exception:
                            pass

                try:
                    if seg_batch_size > 1:
                        masks = sam2_model.predict_batch(thumbs, resize_to_input=True)
                    else:
                        masks = [predict_fn(t) for t in thumbs]
                except Exception:
                    masks = []
                    for t in thumbs:
                        try:
                            masks.append(predict_fn(t))
                        except Exception:
                            masks.append(None)  # type: ignore

                for f, m in zip(pending, masks):
                    try:
                        result_h5 = segment_and_patchify(
                            wsi_path=f,
                            output_dir=str(output_path),
                            seg=seg_params,
                            patch=patch_params,
                            save_images=save_images,
                            fast_mode=fast_mode,
                            predict_fn=predict_fn,
                            thumb_max=thumb_max,
                            mask_override=m if m is not None else None,
                            write_batch=write_batch,
                        )
                        if result_h5:
                            reporter.update(success=True)
                        else:
                            reporter.update(success=False)
                        if visualize and result_h5:
                            vis_output_dir = output_path / "visualization"
                            vis_output_dir.mkdir(parents=True, exist_ok=True)
                            cli_args_dict = {
                                "patch_size": patch_size,
                                "step_size": effective_step_size,
                                "thumbnail_size": 1024,
                                "device": device,
                                "tissue_thresh": tissue_thresh,
                                "white_thresh": white_thresh,
                                "black_thresh": black_thresh,
                                "fast_mode": fast_mode,
                                "save_images": save_images,
                                "target_mag": int(target_mag),
                            }
                            try:
                                _ = visualize_patches_on_thumbnail(
                                    hdf5_path=result_h5,
                                    wsi_path=f,
                                    output_dir=str(vis_output_dir),
                                    cli_args=cli_args_dict,
                                )
                            except Exception as ve:
                                logger.warning(f"Visualization failed for {Path(f).name}: {ve}")
                    except Exception:
                        reporter.update(success=False)
                    finally:
                        reporter.update_progress_bar(pbar)

        # Check if all files failed in non-verbose mode
        if reporter.failed > 0 and reporter.failed == num_files:
            raise click.ClickException("All files failed to process. Check logs for details.")

    # Check if all files failed in verbose mode
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
    click.echo("    - 'coords': (N, 2) int32 (x, y) coordinates")
    click.echo("    - 'coords_ext': (N, 5) int32 (x, y, w, h, level)")
    click.echo(
        "    - Metadata: patch_size, wsi_path, num_patches, level0_magnification, target_magnification, patch_size_level0"
    )
    click.echo("  • HDF5 per WSI under 'patches/<stem>.h5'")
    click.echo("  • Optional per-patch PNGs under 'images/<stem>/' when '--save-images' is used")
    click.echo("  • Visualization PNG under 'visualization/<stem>.png' when '--visualize' is used")

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
