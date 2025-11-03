from __future__ import annotations

import concurrent.futures as _fut
import logging
import sys
from pathlib import Path

import click
import numpy as _np

from slide_processor.pipeline.patchify import (
    PatchifyParams,
    SegmentParams,
    segment_and_patchify,
)
from slide_processor.segmentation.sam2_segmentation import SAM2SegmentationModel
from slide_processor.utils.params import (
    get_wsi_files,
    validate_path,
    validate_positive_int,
)
from slide_processor.utils.progress import ProgressReporter
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


def _load_thumbnails(batch_files: list[str], thumb_max: int) -> list:
    """Load thumbnails for a batch of WSI files (aspect-preserving)."""
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
    return thumbs


def _pack_mask(mask_arr):
    """Pack a predicted mask to compact uint8 for IPC."""
    if mask_arr is None:
        return None
    if isinstance(mask_arr, _np.ndarray):
        return (mask_arr > 0.5).astype(_np.uint8) if mask_arr.dtype != _np.uint8 else mask_arr
    return None


def _build_wsi_task(
    *,
    file_path: str,
    mask_arr,
    output_dir: Path,
    seg_params,
    patch_params,
    thumb_max: int,
    effective_step_size: int,
    save_images: bool,
    fast_mode: bool,
    write_batch: int,
    visualize: bool,
    device: str,
    patch_size: int,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    target_mag: int,
):
    """Build a serializable task payload for per-WSI worker processing."""
    return {
        "wsi_path": file_path,
        "mask": _pack_mask(mask_arr),
        "output_dir": str(output_dir),
        "seg": {
            "checkpoint": str(seg_params.checkpoint_path),
            "device": seg_params.device,
            "thumb_max": int(thumb_max),
        },
        "patch": {
            "patch_size": int(patch_params.patch_size),
            "step_size": int(effective_step_size),
            "target_mag": int(patch_params.target_magnification),
            "tissue_thresh": float(patch_params.tissue_area_thresh),
            "white_thresh": int(patch_params.white_thresh),
            "black_thresh": int(patch_params.black_thresh),
        },
        "opts": {
            "save_images": bool(save_images),
            "fast_mode": bool(fast_mode),
            "write_batch": int(write_batch),
            "visualize": bool(visualize),
            "device": device,
            "patch_size": int(patch_size),
            "step_size": int(effective_step_size),
            "tissue_thresh": float(tissue_thresh),
            "white_thresh": int(white_thresh),
            "black_thresh": int(black_thresh),
            "target_mag": int(target_mag),
        },
    }


def _run_wsi_tasks(tasks, *, max_workers: int, reporter, pbar, verbose: bool) -> tuple[int, int]:
    """Execute per-WSI tasks in a process pool and aggregate results."""
    successful, failed = 0, 0
    with _fut.ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        fut_map = {ex.submit(_process_wsi_worker, t): t["wsi_path"] for t in tasks}
        for fut in _fut.as_completed(fut_map):
            fpath = fut_map[fut]
            try:
                ok, msg = fut.result()
            except Exception as e:
                ok, msg = False, str(e)
            if ok:
                successful += 1
                if verbose:
                    logger.info(f"Saved patches to: {msg}")
                elif reporter:
                    reporter.update(success=True)
            else:
                failed += 1
                if verbose:
                    logger.error(f"Failed to process {Path(fpath).name}: {msg}")
                elif reporter:
                    reporter.update(success=False)
            if reporter and pbar:
                reporter.update_progress_bar(pbar)
    return successful, failed


def _process_files_batch(
    batch_files: list[str],
    masks,
    seg_params,
    patch_params,
    output_path: Path,
    save_images: bool,
    fast_mode: bool,
    thumb_max: int,
    write_batch: int,
    visualize: bool,
    patch_size: int,
    effective_step_size: int,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    target_mag: int,
    verbose: bool,
    reporter=None,
    pbar=None,
    wsi_workers: int = 1,
) -> tuple[int, int]:
    """Process a batch of files. Returns (successful, failed) counts."""
    successful, failed = 0, 0

    # Parallel per-WSI processing if requested
    if max(1, int(wsi_workers)) > 1 and len(batch_files) > 1:
        tasks = [
            _build_wsi_task(
                file_path=f,
                mask_arr=m,
                output_dir=output_path,
                seg_params=seg_params,
                patch_params=patch_params,
                thumb_max=thumb_max,
                effective_step_size=effective_step_size,
                save_images=save_images,
                fast_mode=fast_mode,
                write_batch=write_batch,
                visualize=visualize,
                device=device,
                patch_size=patch_size,
                tissue_thresh=tissue_thresh,
                white_thresh=white_thresh,
                black_thresh=black_thresh,
                target_mag=target_mag,
            )
            for f, m in zip(batch_files, masks)
        ]
        s_delta, f_delta = _run_wsi_tasks(
            tasks,
            max_workers=max(1, int(wsi_workers)),
            reporter=reporter,
            pbar=pbar,
            verbose=verbose,
        )
        successful += s_delta
        failed += f_delta
        return successful, failed

    for f, m in zip(batch_files, masks):
        try:
            wsi = WSIFactory.load(f)
            try:
                result_h5 = segment_and_patchify(
                    wsi=wsi,
                    output_dir=str(output_path),
                    seg=seg_params,
                    patch=patch_params,
                    save_images=save_images,
                    fast_mode=fast_mode,
                    thumb_max=thumb_max,
                    mask_override=m if m is not None else None,
                    write_batch=write_batch,
                )

                if result_h5:
                    successful += 1
                    if verbose:
                        logger.info(f"Saved patches to: {result_h5}")
                    elif reporter:
                        reporter.update(success=True)
                    if visualize:
                        _visualize_result(
                            result_h5,
                            wsi,
                            output_path,
                            patch_size,
                            effective_step_size,
                            device,
                            tissue_thresh,
                            white_thresh,
                            black_thresh,
                            fast_mode,
                            save_images,
                            target_mag,
                        )
                        if verbose:
                            logger.info(f"Visualization saved to: {result_h5}")
                else:
                    failed += 1
                    if verbose:
                        logger.warning(f"No patches extracted from {Path(f).name}")
                    elif reporter:
                        reporter.update(success=False)
            finally:
                try:
                    wsi.cleanup()
                except Exception:
                    pass
        except Exception as e:
            failed += 1
            if verbose:
                logger.error(f"Failed to process {Path(f).name}: {e}")
                raise
            elif reporter:
                reporter.update(success=False)
        finally:
            if reporter and pbar:
                reporter.update_progress_bar(pbar)

    return successful, failed


def _visualize_result(
    result_h5: str,
    wsi,
    output_path: Path,
    patch_size: int,
    step_size: int,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    fast_mode: bool,
    save_images: bool,
    target_mag: int,
) -> None:
    """Visualize patches on thumbnail."""
    vis_output_dir = output_path / "visualization"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    cli_args_dict = {
        "patch_size": patch_size,
        "step_size": step_size,
        "thumbnail_size": 1024,
        "device": device,
        "tissue_thresh": tissue_thresh,
        "white_thresh": white_thresh,
        "black_thresh": black_thresh,
        "fast_mode": fast_mode,
        "save_images": save_images,
        "target_mag": target_mag,
    }
    try:
        visualize_patches_on_thumbnail(
            hdf5_path=result_h5,
            wsi=wsi,
            output_dir=str(vis_output_dir),
            cli_args=cli_args_dict,
        )
    except Exception as e:
        logger.warning(f"Visualization failed for {Path(wsi.path).name}: {e}")


def _process_wsi_worker(task: dict) -> tuple[bool, str]:
    """Worker: process one WSI end-to-end after mask is computed.

    Returns (ok, message). On success, message is HDF5 path; on error, message is error str.
    """
    try:
        from pathlib import Path as _Path

        import numpy as _np

        from slide_processor.pipeline.patchify import (
            PatchifyParams as _PatchifyParams,
        )
        from slide_processor.pipeline.patchify import (
            SegmentParams as _SegmentParams,
        )
        from slide_processor.pipeline.patchify import (
            segment_and_patchify as _segment_and_patchify,
        )
        from slide_processor.visualization import visualize_patches_on_thumbnail as _viz
        from slide_processor.wsi import WSIFactory as _WSIFactory

        wsi = _WSIFactory.load(task["wsi_path"])
        try:
            segp = _SegmentParams(
                checkpoint_path=_Path(task["seg"]["checkpoint"]),
                device=task["seg"]["device"],
                thumbnail_max=int(task["seg"]["thumb_max"]),
            )
            patchp = _PatchifyParams(
                patch_size=int(task["patch"]["patch_size"]),
                step_size=int(task["patch"]["step_size"]),
                target_magnification=int(task["patch"]["target_mag"]),
                tissue_area_thresh=float(task["patch"]["tissue_thresh"]),
                white_thresh=int(task["patch"]["white_thresh"]),
                black_thresh=int(task["patch"]["black_thresh"]),
            )
            mask = task["mask"]
            if mask is not None and isinstance(mask, _np.ndarray) and mask.dtype != _np.uint8:
                mask = (mask > 0.5).astype(_np.uint8)

            out_h5 = _segment_and_patchify(
                wsi=wsi,
                output_dir=task["output_dir"],
                seg=segp,
                patch=patchp,
                save_images=bool(task["opts"]["save_images"]),
                fast_mode=bool(task["opts"]["fast_mode"]),
                thumb_max=int(task["seg"]["thumb_max"]),
                mask_override=mask,
                write_batch=int(task["opts"]["write_batch"]),
            )

            if not out_h5:
                return False, "No patches extracted"

            if bool(task["opts"]["visualize"]):
                cli_args_dict = {
                    "patch_size": int(task["opts"]["patch_size"]),
                    "step_size": int(task["opts"]["step_size"]),
                    "thumbnail_size": 1024,
                    "device": task["opts"]["device"],
                    "tissue_thresh": float(task["opts"]["tissue_thresh"]),
                    "white_thresh": int(task["opts"]["white_thresh"]),
                    "black_thresh": int(task["opts"]["black_thresh"]),
                    "fast_mode": bool(task["opts"]["fast_mode"]),
                    "save_images": bool(task["opts"]["save_images"]),
                    "target_mag": int(task["opts"]["target_mag"]),
                }
                _viz(
                    hdf5_path=out_h5,
                    wsi=wsi,
                    output_dir=str(_Path(task["output_dir"]) / "visualization"),
                    cli_args=cli_args_dict,
                )
            return True, out_h5
        finally:
            try:
                wsi.cleanup()
            except Exception:
                pass
    except Exception as e:
        return False, str(e)


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
@click.option(
    "--workers",
    type=int,
    default=1,
    callback=validate_positive_int,
    help=("CPU workers for processing multiple WSIs in parallel (per-WSI). [default: 1]"),
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
    workers: int,
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
    if tissue_thresh < 0 or tissue_thresh > 1:
        raise click.ClickException("--tissue-thresh must be between 0 and 1")

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

    thumb_max = seg_params.thumbnail_max

    # Setup progress reporter or simple counters
    reporter = None if verbose else ProgressReporter(num_files)
    pbar = None

    try:
        if not verbose and reporter is not None:
            pbar = reporter.progress_bar("Processing WSI files").__enter__()

        pending: list[str] = []
        for wsi_file in wsi_files:
            if verbose:
                logger.info(f"Processing: {Path(wsi_file).name}")

            stem = Path(wsi_file).stem
            existing_h5 = output_path / "patches" / f"{stem}.h5"
            if existing_h5.exists():
                if verbose:
                    logger.info(f"Skipping {Path(wsi_file).name}: already exists -> {existing_h5}")
                elif reporter:
                    reporter.update(success=True)
                    reporter.update_progress_bar(pbar)
                continue

            pending.append(wsi_file)
            # Flush batch if batch size reached (or force flush in verbose mode)
            should_flush = (seg_batch_size > 1 and len(pending) >= seg_batch_size) or (
                not verbose and len(pending) >= seg_batch_size
            )
            if not should_flush and wsi_file != wsi_files[-1]:
                continue

            # Process pending batch
            batch_files = pending
            pending = []

            thumbs = _load_thumbnails(batch_files, thumb_max)
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
                visualize,
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
            )
            successful += s
            failed += f

        # Process any remaining pending files
        if pending:
            thumbs = _load_thumbnails(pending, thumb_max)
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
                visualize,
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
            )
            successful += s
            failed += f

    finally:
        if pbar and reporter is not None:
            reporter.progress_bar("Processing WSI files").__exit__(None, None, None)

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
