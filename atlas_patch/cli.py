from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import torch
from tqdm import tqdm

from atlas_patch.core.config import (
    AppConfig,
    ExtractionConfig,
    FeatureExtractionConfig,
    OutputConfig,
    ProcessingConfig,
    SegmentationConfig,
    VisualizationConfig,
)
from atlas_patch.core.models import Slide
from atlas_patch.models.patch import PatchFeatureExtractorRegistry, build_default_registry
from atlas_patch.models.patch.custom import register_feature_extractors_from_module
from atlas_patch.orchestration.runner import ProcessingRunner
from atlas_patch.services.extraction import PatchExtractionService
from atlas_patch.services.feature_embedding import PatchFeatureEmbeddingService, resolve_feature_dtype
from atlas_patch.services.mpp import CSVMPPResolver
from atlas_patch.services.segmentation import SAM2SegmentationService
from atlas_patch.services.visualization import DefaultVisualizationService
from atlas_patch.services.wsi_loader import DefaultWSILoader
from atlas_patch.utils import (
    configure_logging,
    install_embedding_log_filter,
    parse_feature_list,
)
from atlas_patch.utils.params import get_wsi_files
from atlas_patch.utils.visualization import visualize_mask_on_thumbnail

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("atlas_patch.cli")
install_embedding_log_filter()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "sam2.1_hiera_t.yaml"


FEATURE_EXTRACTOR_CHOICES = build_default_registry(device="cpu").available()


# Shared option sets -----------------------------------------------------------
_COMMON_OPTIONS: list = [
    click.argument("wsi_path", type=click.Path(exists=True)),
    click.option(
        "--output",
        "-o",
        type=click.Path(),
        required=True,
        help="Output directory root for generated artifacts.",
    ),
    click.option(
        "--patch-size", type=int, required=True, help="Patch size at target magnification."
    ),
    click.option(
        "--step-size",
        type=int,
        default=None,
        help="Stride between patches; defaults to patch size when omitted.",
    ),
    click.option(
        "--target-mag",
        type=click.IntRange(1, 120),
        required=True,
        help="Target magnification (e.g., 20, 40).",
    ),
    click.option(
        "--device",
        type=str,
        default="cuda",
        show_default=True,
        help="Segmentation device (e.g., cuda, cuda:0, cpu).",
    ),
    click.option(
        "--tissue-thresh",
        type=float,
        default=0.0,
        show_default=True,
        help="Minimum tissue area fraction.",
    ),
    click.option(
        "--white-thresh",
        type=int,
        default=15,
        show_default=True,
        help="Saturation threshold for white filtering.",
    ),
    click.option(
        "--black-thresh",
        type=int,
        default=50,
        show_default=True,
        help="RGB threshold for black filtering.",
    ),
    click.option(
        "--seg-batch-size", type=int, default=1, show_default=True, help="Segmentation batch."
    ),
    click.option(
        "--write-batch", type=int, default=8192, show_default=True, help="HDF5 write batch."
    ),
    click.option(
        "--patch-workers",
        type=int,
        default=None,
        show_default=True,
        help="Parallel worker threads for per-slide patch extraction; defaults to CPU count.",
    ),
    click.option(
        "--max-open-slides",
        type=int,
        default=200,
        show_default=True,
        help="Upper bound on simultaneously open slides (segmentation + extraction).",
    ),
    click.option(
        "--fast-mode/--no-fast-mode",
        default=True,
        show_default=True,
        help="fast-mode skips per-patch content filtering; use --no-fast-mode to enable filtering.",
    ),
    click.option("--save-images", is_flag=True, help="Export individual patch PNGs."),
    click.option("--visualize-grids", is_flag=True, help="Render patch grid overlay."),
    click.option("--visualize-mask", is_flag=True, help="Render predicted mask overlay."),
    click.option("--visualize-contours", is_flag=True, help="Render contour overlay."),
    click.option("--recursive", is_flag=True, help="Recursively search directories for WSIs."),
    click.option(
        "--mpp-csv", type=click.Path(exists=True), default=None, help="CSV with custom MPP."
    ),
    click.option(
        "--skip-existing/--force", default=True, show_default=True, help="Skip existing H5."
    ),
    click.option("--verbose", "-v", is_flag=True, help="Enable debug logging."),
]

_FEATURE_OPTIONS: list = [
    click.option(
        "--feature-device",
        type=str,
        default=None,
        help="Device for feature extraction; e.g. cuda, cuda:0, cpu. Defaults to --device.",
    ),
    click.option(
        "--feature-extractors",
        required=True,
        type=str,
        help="Space/comma separated feature extractors to run (available: "
        + ", ".join(FEATURE_EXTRACTOR_CHOICES)
        + "; add more via --feature-plugin).",
    ),
    click.option(
        "--feature-batch-size",
        type=int,
        default=32,
        show_default=True,
        help="Batch size used when embedding patches.",
    ),
    click.option(
        "--feature-num-workers",
        type=int,
        default=4,
        show_default=True,
        help="DataLoader worker count for feature extraction.",
    ),
    click.option(
        "--feature-precision",
        type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
        default="float16",
        show_default=True,
        help="Computation precision for feature extraction.",
    ),
    click.option(
        "--feature-plugin",
        "feature_plugins",
        type=click.Path(exists=True),
        multiple=True,
        help=(
            "Path(s) to Python modules that register custom feature extractors via "
            "register_feature_extractors(registry, device, dtype, num_workers)."
        ),
    ),
]


def _apply_options(func, options: list):
    for opt in reversed(options):
        func = opt(func)
    return func


def common_options(func):
    return _apply_options(func, _COMMON_OPTIONS)


def feature_options(func):
    return _apply_options(func, _FEATURE_OPTIONS)


def _run_pipeline(
    *,
    wsi_path: str,
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
    feature_cfg: FeatureExtractionConfig | None = None,
    registry: PatchFeatureExtractorRegistry | None = None,
) -> tuple[list, list]:
    configure_logging(verbose)

    processing_cfg = ProcessingConfig(
        input_path=Path(wsi_path),
        recursive=recursive,
        mpp_csv=Path(mpp_csv) if mpp_csv else None,
    )
    segmentation_cfg = SegmentationConfig(
        checkpoint_path=None,
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
        features=feature_cfg,
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

    results: list
    failures: list
    try:
        results, failures = runner.run()
    finally:
        segmentation_service.close()

    click.echo("Segmentation and patch coordinate extraction complete.")

    if app_cfg.features is not None:
        feature_service = PatchFeatureEmbeddingService(
            app_cfg.extraction, app_cfg.output, app_cfg.features, registry=registry
        )
        total_units = len(results) * len(app_cfg.features.extractors)
        feature_progress = tqdm(
            total=total_units,
            desc="Feature embedding",
            disable=verbose or total_units == 0,
        )
        try:
            failures.extend(
                feature_service.embed_all(results, wsi_loader=wsi_loader, progress=feature_progress)
            )
        finally:
            feature_progress.close()

    return results, failures


def _run_tissue_visualization(
    *,
    wsi_path: str,
    output: str,
    device: str,
    seg_batch_size: int,
    recursive: bool,
    mpp_csv: str | None,
    verbose: bool,
) -> tuple[list[tuple[Slide, Path]], list[tuple[Slide, Exception | str]]]:
    configure_logging(verbose)

    processing_cfg = ProcessingConfig(
        input_path=Path(wsi_path),
        recursive=recursive,
        mpp_csv=Path(mpp_csv) if mpp_csv else None,
    ).validated()
    segmentation_cfg = (
        SegmentationConfig(
            checkpoint_path=None,
            config_path=_default_config_path(),
            device=device.lower(),
            batch_size=seg_batch_size,
        )
        .validated()
    )
    vis_cfg = VisualizationConfig().validated()

    slide_paths = get_wsi_files(str(processing_cfg.input_path), recursive=processing_cfg.recursive)

    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    vis_dir = output_root / "visualization"

    mpp_resolver = CSVMPPResolver(processing_cfg.mpp_csv)
    wsi_loader = DefaultWSILoader()
    segmentation_service = SAM2SegmentationService(segmentation_cfg)

    results: list[tuple[Slide, Path]] = []
    failures: list[tuple[Slide, Exception | str]] = []
    progress = tqdm(total=len(slide_paths), disable=verbose, desc="Tissue detection")

    def _process_batch(batch: list[tuple[Slide, object]]) -> None:
        if not batch:
            return

        wsis = [w for _, w in batch]
        slides_batch = [s for s, _ in batch]

        try:
            masks = (
                segmentation_service.segment_batch(wsis)
                if len(wsis) > 1
                else [segmentation_service.segment_thumbnail(wsis[0])]
            )
        except Exception as e:  # noqa: BLE001
            for slide, wsi in batch:
                failures.append((slide, e))
                try:
                    wsi.cleanup()
                except Exception:
                    pass
                progress.update(1)
            return

        for slide, wsi, mask in zip(slides_batch, wsis, masks):
            try:
                out_path = visualize_mask_on_thumbnail(
                    mask=mask.data,
                    wsi=wsi,
                    output_dir=vis_dir,
                    thumbnail_size=vis_cfg.thumbnail_size,
                )
                results.append((slide, out_path))
            except Exception as e:  # noqa: BLE001
                failures.append((slide, e))
            finally:
                try:
                    wsi.cleanup()
                except Exception:
                    pass
            progress.update(1)

    try:
        batch: list[tuple[Slide, object]] = []
        for path_str in slide_paths:
            base_slide = Slide(path=Path(path_str))
            mpp_val = mpp_resolver.resolve(base_slide)
            slide = Slide(path=base_slide.path, mpp=mpp_val, backend=base_slide.backend)
            try:
                wsi = wsi_loader.open(slide)
            except Exception as e:  # noqa: BLE001
                failures.append((slide, e))
                progress.update(1)
                continue

            batch.append((slide, wsi))
            if len(batch) >= segmentation_cfg.batch_size:
                _process_batch(batch)
                batch = []

        if batch:
            _process_batch(batch)
    finally:
        try:
            segmentation_service.close()
        finally:
            progress.close()

    return results, failures


def _echo_results(
    results: list, failures: list, verbose: bool, feature_cfg: FeatureExtractionConfig | None
) -> None:
    click.echo(f"Completed {len(results)} slide(s), failures: {len(failures)}")
    if verbose:
        for res in results:
            feature_note = f" features={','.join(feature_cfg.extractors)}" if feature_cfg else ""
            click.echo(
                f"[OK] {res.slide.path.name} -> {res.h5_path} (patches={res.num_patches}){feature_note}"
            )
        for slide, err in failures:
            click.echo(f"[FAIL] {slide.path.name}: {err}", err=True)


def _echo_mask_results(
    results: list[tuple[Slide, Path]], failures: list[tuple[Slide, Exception | str]], verbose: bool
) -> None:
    click.echo(f"Created {len(results)} mask overlay(s), failures: {len(failures)}")
    if verbose:
        for slide, path in results:
            click.echo(f"[OK] {slide.path.name} -> {path}")
        for slide, err in failures:
            click.echo(f"[FAIL] {slide.path.name}: {err}", err=True)


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """AtlasPatch CLI.

    Processes WSI files by segmenting tissue with SAM2, extracting patches, and
    optionally exporting images/visualizations.
    """


@cli.command()
@common_options
def segment_and_get_coords(
    wsi_path: str,
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
    results, failures = _run_pipeline(
        wsi_path=wsi_path,
        output=output,
        patch_size=patch_size,
        step_size=step_size,
        target_mag=target_mag,
        device=device,
        tissue_thresh=tissue_thresh,
        white_thresh=white_thresh,
        black_thresh=black_thresh,
        seg_batch_size=seg_batch_size,
        write_batch=write_batch,
        patch_workers=patch_workers,
        max_open_slides=max_open_slides,
        fast_mode=fast_mode,
        save_images=save_images,
        visualize_grids=visualize_grids,
        visualize_mask=visualize_mask,
        visualize_contours=visualize_contours,
        recursive=recursive,
        mpp_csv=mpp_csv,
        skip_existing=skip_existing,
        verbose=verbose,
        feature_cfg=None,
    )
    _echo_results(results, failures, verbose, None)


@cli.command()
@click.argument("wsi_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory root for generated artifacts.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Segmentation device (e.g., cuda, cuda:0, cpu).",
)
@click.option(
    "--seg-batch-size",
    type=click.IntRange(1, None),
    default=1,
    show_default=True,
    help="Segmentation batch size for thumbnail inference.",
)
@click.option("--recursive", is_flag=True, help="Recursively search directories for WSIs.")
@click.option(
    "--mpp-csv", type=click.Path(exists=True), default=None, help="CSV with custom MPP."
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def detect_tissue(
    wsi_path: str,
    output: str,
    device: str,
    seg_batch_size: int,
    recursive: bool,
    mpp_csv: str | None,
    verbose: bool,
):
    """Run tissue segmentation only and export mask overlays."""
    results, failures = _run_tissue_visualization(
        wsi_path=wsi_path,
        output=output,
        device=device,
        seg_batch_size=seg_batch_size,
        recursive=recursive,
        mpp_csv=mpp_csv,
        verbose=verbose,
    )
    _echo_mask_results(results, failures, verbose)


@cli.command()
@feature_options
@common_options
def process(
    wsi_path: str,
    output: str,
    patch_size: int,
    step_size: int | None,
    target_mag: int,
    device: str,
    feature_device: str | None,
    feature_extractors: str,
    feature_batch_size: int,
    feature_num_workers: int,
    feature_precision: str,
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
    feature_plugins: tuple[str, ...],
):
    """Run segmentation, patch extraction, and feature embedding into a single H5."""
    feat_device = feature_device.lower() if feature_device else device.lower()
    torch_device = torch.device(feat_device)
    dtype = resolve_feature_dtype(torch_device, feature_precision.lower())
    registry = build_default_registry(
        device=torch_device, num_workers=feature_num_workers, dtype=dtype
    )
    for plugin in feature_plugins:
        register_feature_extractors_from_module(
            plugin,
            registry=registry,
            device=torch_device,
            dtype=dtype,
            num_workers=feature_num_workers,
        )

    available_extractors = registry.available()
    feats = parse_feature_list(feature_extractors, choices=available_extractors)
    feature_cfg = FeatureExtractionConfig(
        extractors=feats,
        batch_size=feature_batch_size,
        device=feat_device,
        num_workers=feature_num_workers,
        precision=feature_precision.lower(),
        plugins=[Path(p) for p in feature_plugins],
    )
    results, failures = _run_pipeline(
        wsi_path=wsi_path,
        output=output,
        patch_size=patch_size,
        step_size=step_size,
        target_mag=target_mag,
        device=device,
        tissue_thresh=tissue_thresh,
        white_thresh=white_thresh,
        black_thresh=black_thresh,
        seg_batch_size=seg_batch_size,
        write_batch=write_batch,
        patch_workers=patch_workers,
        max_open_slides=max_open_slides,
        fast_mode=fast_mode,
        save_images=save_images,
        visualize_grids=visualize_grids,
        visualize_mask=visualize_mask,
        visualize_contours=visualize_contours,
        recursive=recursive,
        mpp_csv=mpp_csv,
        skip_existing=skip_existing,
        verbose=verbose,
        feature_cfg=feature_cfg,
        registry=registry,
    )
    _echo_results(results, failures, verbose, feature_cfg)


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
