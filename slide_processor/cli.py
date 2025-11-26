from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from slide_processor.core.config import (
    AppConfig,
    ExtractionConfig,
    OutputConfig,
    ProcessingConfig,
    SegmentationConfig,
    VisualizationConfig,
)
from slide_processor.orchestration.runner import ProcessingRunner
from slide_processor.services.extraction import PatchExtractionService
from slide_processor.services.mpp import CSVMPPResolver
from slide_processor.services.segmentation import SAM2SegmentationService
from slide_processor.services.visualization import DefaultVisualizationService
from slide_processor.services.wsi_loader import DefaultWSILoader
from slide_processor.utils import install_embedding_log_filter

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("slide_processor.cli")
install_embedding_log_filter()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "sam2.1_hiera_t.yaml"


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """SlideProcessor CLI.

    Processes WSI files by segmenting tissue with SAM2, extracting patches, and
    optionally exporting images/visualizations.
    """


@cli.command()
@click.argument("wsi_path", type=click.Path(exists=True))
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to SAM2 checkpoint (.pt).",
)
@click.option("--output", "-o", type=click.Path(), default="./output", show_default=True)
@click.option("--patch-size", type=int, required=True, help="Patch size at target magnification.")
@click.option(
    "--step-size",
    type=int,
    default=None,
    help="Stride between patches; defaults to patch size when omitted.",
)
@click.option(
    "--target-mag",
    type=click.IntRange(1, 120),
    required=True,
    help="Target magnification (e.g., 20, 40).",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default="cuda",
    show_default=True,
)
@click.option(
    "--tissue-thresh",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum tissue area fraction.",
)
@click.option(
    "--white-thresh",
    type=int,
    default=15,
    show_default=True,
    help="Saturation threshold for white filtering.",
)
@click.option(
    "--black-thresh",
    type=int,
    default=50,
    show_default=True,
    help="RGB threshold for black filtering.",
)
@click.option(
    "--seg-batch-size", type=int, default=1, show_default=True, help="Segmentation batch."
)
@click.option("--write-batch", type=int, default=8192, show_default=True, help="HDF5 write batch.")
@click.option(
    "--patch-workers",
    type=int,
    default=None,
    show_default=True,
    help="Parallel worker threads for per-slide patch extraction; defaults to CPU count.",
)
@click.option(
    "--max-open-slides",
    type=int,
    default=200,
    show_default=True,
    help="Upper bound on simultaneously open slides (segmentation + extraction).",
)
@click.option(
    "--fast-mode/--no-fast-mode",
    default=True,
    show_default=True,
    help="fast-mode skips per-patch content filtering; use --no-fast-mode to enable filtering.",
)
@click.option("--save-images", is_flag=True, help="Export individual patch PNGs.")
@click.option("--visualize-grids", is_flag=True, help="Render patch grid overlay.")
@click.option("--visualize-mask", is_flag=True, help="Render predicted mask overlay.")
@click.option("--visualize-contours", is_flag=True, help="Render contour overlay.")
@click.option("--recursive", is_flag=True, help="Recursively search directories for WSIs.")
@click.option("--mpp-csv", type=click.Path(exists=True), default=None, help="CSV with custom MPP.")
@click.option("--skip-existing/--force", default=True, show_default=True, help="Skip existing H5.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def segment_and_get_coords(
    wsi_path: str,
    checkpoint: str,
    output: str,
    patch_size: int,
    step_size: int | None,
    target_mag: int,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    seg_batch_size: int,
    write_batch: int,
    patch_workers: int | None,
    max_open_slides: int | None,
    fast_mode: bool,
    save_images: bool,
    visualize_grids: bool,
    visualize_mask: bool,
    visualize_contours: bool,
    recursive: bool,
    mpp_csv: str | None,
    skip_existing: bool,
    verbose: bool,
):
    """Segment, patchify, and optionally visualize WSI files."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("slide_processor").setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("slide_processor").setLevel(logging.WARNING)

    processing_cfg = ProcessingConfig(
        input_path=Path(wsi_path),
        recursive=recursive,
        mpp_csv=Path(mpp_csv) if mpp_csv else None,
    )
    segmentation_cfg = SegmentationConfig(
        checkpoint_path=Path(checkpoint),
        config_path=_default_config_path(),
        device=device.lower(),
        batch_size=seg_batch_size,
    )
    extraction_cfg = ExtractionConfig(
        patch_size=patch_size,
        step_size=step_size,
        target_magnification=target_mag,
        tissue_threshold=tissue_thresh,
        white_threshold=white_thresh,
        black_threshold=black_thresh,
        fast_mode=fast_mode,
        write_batch=write_batch,
        workers=patch_workers,
        max_open_slides=max_open_slides,
    )
    output_cfg = OutputConfig(
        output_root=Path(output),
        save_images=save_images,
        visualize_grids=visualize_grids,
        visualize_mask=visualize_mask,
        visualize_contours=visualize_contours,
        skip_existing=skip_existing,
    )
    app_cfg = AppConfig(
        processing=processing_cfg,
        segmentation=segmentation_cfg,
        extraction=extraction_cfg,
        output=output_cfg,
        visualization=VisualizationConfig(),
        device=device.lower(),
    ).validated()

    segmentation_service = SAM2SegmentationService(app_cfg.segmentation)
    extractor_service = PatchExtractionService(app_cfg.extraction, app_cfg.output)
    visualizer_service = None
    if visualize_grids or visualize_mask or visualize_contours:
        visualizer_service = DefaultVisualizationService(
            app_cfg.output, app_cfg.extraction, app_cfg.visualization
        )

    mpp_resolver = CSVMPPResolver(app_cfg.processing.mpp_csv)
    wsi_loader = DefaultWSILoader()

    runner = ProcessingRunner(
        config=app_cfg,
        segmentation=segmentation_service,
        extractor=extractor_service,
        visualizer=visualizer_service,
        mpp_resolver=mpp_resolver,
        wsi_loader=wsi_loader,
        show_progress=not verbose,
    )

    results, failures = runner.run()

    click.echo(f"Completed {len(results)} slide(s), failures: {len(failures)}")
    if verbose:
        for res in results:
            click.echo(f"[OK] {res.slide.path.name} -> {res.h5_path} (patches={res.num_patches})")
        for slide, err in failures:
            click.echo(f"[FAIL] {slide.path.name}: {err}", err=True)


@cli.command()
def info():
    """Display supported formats and output structure."""
    click.echo(
        "Supported WSI formats (OpenSlide): .svs, .tif, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .dcm"
    )
    click.echo("Image formats: .png, .jpg, .jpeg, .bmp, .webp, .gif")
    click.echo(
        "Outputs: HDF5 per slide under patches/<stem>.h5; optional PNGs under images/<stem>; visualizations under visualization/."
    )


def main():
    try:
        cli()
    except click.ClickException as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
