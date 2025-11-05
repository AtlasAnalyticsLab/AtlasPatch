from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from slide_processor.pipeline.orchestrator import (
    process_files_batch as _process_files_batch,
)
from slide_processor.pipeline.orchestrator import (
    process_files_pipeline as _process_files_pipeline,
)
from slide_processor.pipeline.patchify import (
    PatchifyParams,
    SegmentParams,
)
from slide_processor.segmentation.sam2_segmentation import SAM2SegmentationModel
from slide_processor.utils.params import (
    get_wsi_files,
    load_mpp_csv,
    validate_path,
    validate_positive_int,
)
from slide_processor.utils.progress import ProgressReporter

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # default to quiet output unless --verbose is used
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Suppress a noisy third-party info log emitted to the root logger.
class _SuppressNumpyHxWxCFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return "For numpy array image, we assume (HxWxC) format" not in record.getMessage()


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
    "--min-tissue-proportion",
    type=float,
    default=0.0,
    help="Minimum tissue area threshold as fraction of image (0-1). [default: 0.01]",
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
    "--visualize-grids",
    is_flag=True,
    default=False,
    help="Generate patch grid overlay visualization on WSI thumbnail.",
)
@click.option(
    "--visualize-mask",
    is_flag=True,
    default=False,
    help="Generate predicted tissue mask overlay visualization on thumbnail.",
)
@click.option(
    "--visualize-contours",
    is_flag=True,
    default=False,
    help="Generate tissue contours overlay visualization on thumbnail.",
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
@click.option(
    "--workers",
    type=int,
    default=1,
    callback=validate_positive_int,
    help=("CPU workers for processing multiple WSIs in parallel (per-WSI). [default: 1]"),
)
@click.option(
    "--recursive",
    is_flag=True,
    default=False,
    help="Recursively search for WSI files in directories.",
)
@click.option(
    "--mpp-csv",
    type=click.Path(exists=True),
    default=None,
    callback=validate_path,
    help="Path to CSV file with custom MPP values. CSV must have 'wsi' and 'mpp' columns.",
)
@click.option(
    "--pipeline",
    is_flag=True,
    default=False,
    help=(
        "Pipeline GPU segmentation with CPU patchification (overlap compute). "
        "Recommended when processing multiple slides."
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
    visualize_grids: bool,
    visualize_mask: bool,
    visualize_contours: bool,
    target_mag: str,
    seg_batch_size: int,
    write_batch: int,
    workers: int,
    recursive: bool,
    mpp_csv: str | None,
    pipeline: bool,
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

    # Load MPP CSV if provided
    mpp_dict = None
    if mpp_csv is not None:
        try:
            mpp_dict = load_mpp_csv(mpp_csv)
        except click.ClickException:
            raise

    # Validate parameters
    if patch_size <= 0:
        raise click.ClickException("--patch-size must be positive")
    if step_size is not None and step_size <= 0:
        raise click.ClickException("--step-size must be positive")
    if tissue_thresh < 0 or tissue_thresh > 1:
        raise click.ClickException("--tissue-thresh must be between 0 and 1")

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Get list of WSI files to process
    try:
        wsi_files = get_wsi_files(wsi_path, recursive=recursive)
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

    thumb_max = seg_params.thumbnail_max

    # Setup progress reporter or simple counters
    reporter = None if verbose else ProgressReporter(num_files)
    pbar = None
    pbar_cm = None

    try:
        if not verbose and reporter is not None:
            pbar_cm = reporter.progress_bar("Processing WSI files")
            pbar = pbar_cm.__enter__()

        # Pre-filter already processed files (update progress for skipped)
        to_process: list[str] = []
        for wsi_file in wsi_files:
            stem = Path(wsi_file).stem
            existing_h5 = output_path / "patches" / f"{stem}.h5"
            if existing_h5.exists():
                if verbose:
                    logger.info(f"Skipping {Path(wsi_file).name}: already exists -> {existing_h5}")
                elif reporter:
                    reporter.update(success=True)
                    reporter.update_progress_bar(pbar)
                continue
            to_process.append(wsi_file)

        if pipeline and len(to_process) > 0:
            # Pipeline GPU segmentation with CPU patchification
            s, f = _process_files_pipeline(
                wsi_files=to_process,
                sam2_model=sam2_model,
                output_path=output_path,
                seg_params=seg_params,
                patch_params=patch_params,
                save_images=save_images,
                fast_mode=fast_mode,
                thumb_max=thumb_max,
                write_batch=write_batch,
                visualize_grids=visualize_grids,
                visualize_mask=visualize_mask,
                visualize_contours=visualize_contours,
                patch_size=patch_size,
                effective_step_size=effective_step_size,
                device=device,
                tissue_thresh=tissue_thresh,
                white_thresh=white_thresh,
                black_thresh=black_thresh,
                target_mag=int(target_mag),
                verbose=verbose,
                reporter=reporter,
                pbar=pbar,
                wsi_workers=workers,
                seg_batch_size=seg_batch_size,
                mpp_dict=mpp_dict,
            )
            successful += s
            failed += f
        elif len(to_process) > 0:
            # Original non-pipelined path (batched segmentation followed by patchification)
            pending: list[str] = []
            for wsi_file in to_process:
                if verbose:
                    logger.info(f"Processing: {Path(wsi_file).name}")

                pending.append(wsi_file)
                # Flush batch if batch size reached (or force flush in verbose mode)
                should_flush = (seg_batch_size > 1 and len(pending) >= seg_batch_size) or (
                    not verbose and len(pending) >= seg_batch_size
                )
                if not should_flush and wsi_file != to_process[-1]:
                    continue

                # Process pending batch
                batch_files = pending
                pending = []

                # Open WSIs once with MPP override for thumbnails
                wsi_objs = []
                thumbs = []
                for bf in batch_files:
                    from slide_processor.utils.params import get_mpp_for_wsi as _get_mpp
                    from slide_processor.wsi import WSIFactory as _WSIFactory

                    mpp_value = _get_mpp(bf, mpp_dict)
                    wsi_obj = _WSIFactory.load(bf, mpp=mpp_value)
                    wsi_objs.append(wsi_obj)
                    thumbs.append(
                        wsi_obj.get_thumbnail_at_power(power=1.25, interpolation="optimise")
                    )
                masks = sam2_model.predict_batch(thumbs, resize_to_input=True)

                s, f = _process_files_batch(
                    batch_files,
                    masks,
                    seg_params,
                    patch_params,
                    output_path,
                    save_images,
                    fast_mode,
                    thumb_max,
                    write_batch,
                    visualize_grids,
                    visualize_mask,
                    visualize_contours,
                    patch_size,
                    effective_step_size,
                    device,
                    tissue_thresh,
                    white_thresh,
                    black_thresh,
                    int(target_mag),
                    verbose,
                    reporter,
                    pbar,
                    workers,
                    mpp_dict=mpp_dict,
                    wsis=wsi_objs,
                )
                successful += s
                failed += f

                # Cleanup GPU memory after batch
                del thumbs, masks
                if device.lower() == "cuda":
                    import torch as _torch

                    _torch.cuda.empty_cache()

            # Process any remaining pending files (shouldn't happen here)
            if pending:
                # Open WSIs once with MPP override for thumbnails
                wsi_objs = []
                thumbs = []
                for pf in pending:
                    from slide_processor.utils.params import get_mpp_for_wsi as _get_mpp
                    from slide_processor.wsi import WSIFactory as _WSIFactory

                    mpp_value = _get_mpp(pf, mpp_dict)
                    wsi_obj = _WSIFactory.load(pf, mpp=mpp_value)
                    wsi_objs.append(wsi_obj)
                    thumbs.append(
                        wsi_obj.get_thumbnail_at_power(power=1.25, interpolation="optimise")
                    )
                masks = sam2_model.predict_batch(thumbs, resize_to_input=True)

                s, f = _process_files_batch(
                    pending,
                    masks,
                    seg_params,
                    patch_params,
                    output_path,
                    save_images,
                    fast_mode,
                    thumb_max,
                    write_batch,
                    visualize_grids,
                    visualize_mask,
                    visualize_contours,
                    patch_size,
                    effective_step_size,
                    device,
                    tissue_thresh,
                    white_thresh,
                    black_thresh,
                    int(target_mag),
                    verbose,
                    reporter,
                    pbar,
                    workers,
                    mpp_dict=mpp_dict,
                    wsis=wsi_objs,
                )
                successful += s
                failed += f

                # Cleanup GPU memory after final batch
                del thumbs, masks
                if device.lower() == "cuda":
                    import torch as _torch

                    _torch.cuda.empty_cache()

    finally:
        if pbar_cm is not None and reporter is not None:
            pbar_cm.__exit__(None, None, None)

    # Check if all files failed
    if (reporter and reporter.failed > 0 and reporter.failed == num_files) or (
        verbose and failed > 0 and failed == num_files
    ):
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
    click.echo("  • Visualizations under 'visualization/':")
    click.echo("    - '<stem>.png' for --visualize-grids (patch grids)")
    click.echo("    - '<stem>_mask.png' for --visualize-mask (mask overlay)")
    click.echo("    - '<stem>_contours.png' for --visualize-contours (contour overlay)")

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
