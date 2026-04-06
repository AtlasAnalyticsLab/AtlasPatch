from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from atlas_patch import __version__
from atlas_patch.core.config import (
    DEFAULT_FEATURE_BATCH_SIZE,
    DEFAULT_FEATURE_NUM_WORKERS,
    DEFAULT_FEATURE_PRECISION,
    AppConfig,
    FeatureExtractionConfig,
    SlideEncodingConfig,
    default_sam2_config_path,
)
from atlas_patch.core.models import Slide
from atlas_patch.core.paths import patch_h5_path
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry
from atlas_patch.models.slide.registry import SlideEncoderRegistry
from atlas_patch.utils import (
    configure_logging,
    install_embedding_log_filter,
    missing_features,
    parse_named_list,
)
from atlas_patch.utils.feature_h5 import read_patch_artifact_summary
from atlas_patch.utils.params import get_wsi_files
from atlas_patch.utils.visualization import visualize_mask_on_thumbnail

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
install_embedding_log_filter()

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

_FEATURE_RUNTIME_OPTIONS: list = [
    click.option(
        "--feature-device",
        type=str,
        default=None,
        help="Device for feature extraction; e.g. cuda, cuda:0, cpu. Defaults to --device.",
    ),
    click.option(
        "--feature-batch-size",
        type=int,
        default=DEFAULT_FEATURE_BATCH_SIZE,
        show_default=True,
        help="Batch size used when embedding patches.",
    ),
    click.option(
        "--feature-num-workers",
        type=int,
        default=DEFAULT_FEATURE_NUM_WORKERS,
        show_default=True,
        help="DataLoader worker count for feature extraction.",
    ),
    click.option(
        "--feature-precision",
        type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
        default=DEFAULT_FEATURE_PRECISION,
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
_FEATURE_EXTRACTOR_OPTION = click.option(
    "--feature-extractors",
    required=True,
    type=str,
    help=(
        "Space/comma separated feature extractors to run. "
        "The registry is resolved lazily at runtime; install `atlas-patch[patch-encoders]` "
        "for optional timm/transformers/open-clip based models and add more via --feature-plugin."
    ),
)
_FEATURE_OPTIONS: list = [
    _FEATURE_EXTRACTOR_OPTION,
    *_FEATURE_RUNTIME_OPTIONS,
]


def _apply_options(func, options: list):
    for opt in reversed(options):
        func = opt(func)
    return func


def common_options(func):
    return _apply_options(func, _COMMON_OPTIONS)


def feature_options(func):
    return _apply_options(func, _FEATURE_OPTIONS)


def feature_runtime_options(func):
    return _apply_options(func, _FEATURE_RUNTIME_OPTIONS)


def _build_patch_feature_registry(
    *,
    device: str,
    feature_device: str | None,
    feature_num_workers: int,
    feature_precision: str,
    feature_plugins: tuple[str, ...],
):
    import torch

    from atlas_patch.models.patch import build_default_registry
    from atlas_patch.models.patch.custom import register_feature_extractors_from_module
    from atlas_patch.services.feature_embedding import resolve_feature_dtype

    feat_device = feature_device.lower() if feature_device else device.lower()
    precision = feature_precision.lower()
    torch_device = torch.device(feat_device)
    dtype = resolve_feature_dtype(torch_device, precision)
    registry = build_default_registry(
        device=torch_device,
        num_workers=feature_num_workers,
        dtype=dtype,
    )
    for plugin in feature_plugins:
        register_feature_extractors_from_module(
            plugin,
            registry=registry,
            device=torch_device,
            dtype=dtype,
            num_workers=feature_num_workers,
        )
    return registry, feat_device, precision


def _resolve_slide_encoder_requirements(
    slide_registry: SlideEncoderRegistry,
    encoders: list[str],
) -> tuple[int, list[str]]:
    specs = [slide_registry.get_spec(name) for name in encoders]
    patch_sizes = {spec.patch_size for spec in specs}
    if len(patch_sizes) != 1:
        parts = ", ".join(f"{spec.name}:{spec.patch_size}" for spec in specs)
        raise click.ClickException(
            "Requested slide encoders require incompatible patch geometries for one canonical H5: "
            f"{parts}"
        )
    required_patch_encoders = sorted({spec.patch_encoder_name for spec in specs})
    return next(iter(patch_sizes)), required_patch_encoders


def _validate_existing_slide_outputs(app_cfg: AppConfig, slides: list[Slide]) -> None:
    if not app_cfg.output.skip_existing:
        return

    for slide in slides:
        h5_path = patch_h5_path(slide, app_cfg.output, app_cfg.extraction)
        if not h5_path.exists():
            continue

        try:
            summary = read_patch_artifact_summary(h5_path)
        except Exception:
            continue
        if summary.patch_size != app_cfg.extraction.patch_size:
            raise click.ClickException(
                f"Existing patch artifact {h5_path} has patch_size={summary.patch_size}, "
                f"but encode-slide requested --patch-size {app_cfg.extraction.patch_size}. "
                "Re-run with --force to rebuild the canonical H5."
            )
        if summary.target_magnification != app_cfg.extraction.target_magnification:
            raise click.ClickException(
                f"Existing patch artifact {h5_path} has target_mag={summary.target_magnification}, "
                f"but encode-slide requested --target-mag {app_cfg.extraction.target_magnification}. "
                "Re-run with --force to rebuild the canonical H5."
            )


def _collect_ready_slide_artifacts(
    app_cfg: AppConfig,
    slides: list[Slide],
    *,
    required_patch_encoders: list[str],
    failed_slide_paths: set[Path],
) -> tuple[list[tuple[Slide, Path]], list[tuple[Slide, Exception | str]]]:
    artifacts: list[tuple[Slide, Path]] = []
    failures: list[tuple[Slide, Exception | str]] = []

    for slide in slides:
        if slide.path.resolve() in failed_slide_paths:
            continue

        h5_path = patch_h5_path(slide, app_cfg.output, app_cfg.extraction)
        if not h5_path.exists():
            failures.append(
                (
                    slide,
                    RuntimeError(
                        f"No canonical patch artifact was produced for {slide.path.name} at {h5_path}."
                    ),
                )
            )
            continue

        try:
            summary = read_patch_artifact_summary(h5_path)
            if summary.patch_size != app_cfg.extraction.patch_size:
                raise ValueError(
                    f"{h5_path} has patch_size={summary.patch_size}, "
                    f"but encode-slide expects {app_cfg.extraction.patch_size}."
                )
            missing = missing_features(
                h5_path,
                required_patch_encoders,
                expected_total=summary.num_patches,
            )
            if missing:
                raise ValueError(
                    f"{h5_path} is missing required patch feature sets: {', '.join(missing)}"
                )
            artifacts.append((slide, h5_path))
        except Exception as exc:  # noqa: BLE001
            failures.append((slide, exc))

    return artifacts, failures


def _build_wsi_app_config(
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
    feature_cfg=None,
):
    from atlas_patch.core.config import (
        ExtractionConfig,
        OutputConfig,
        ProcessingConfig,
        SegmentationConfig,
        VisualizationConfig,
    )

    return AppConfig(
        processing=ProcessingConfig(
            input_path=Path(wsi_path),
            recursive=recursive,
            mpp_csv=Path(mpp_csv) if mpp_csv else None,
        ),
        segmentation=SegmentationConfig(
            checkpoint_path=None,
            config_path=default_sam2_config_path(),
            device=device.lower(),
            batch_size=seg_batch_size,
        ),
        extraction=ExtractionConfig(
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
        ),
        output=OutputConfig(
            output_root=Path(output),
            save_images=save_images,
            visualize_grids=visualize_grids,
            visualize_mask=visualize_mask,
            visualize_contours=visualize_contours,
            skip_existing=skip_existing,
        ),
        visualization=VisualizationConfig(),
        features=feature_cfg,
        device=device.lower(),
    ).validated()


def _run_pipeline_from_config(
    app_cfg: AppConfig,
    *,
    slides: list[Slide] | None = None,
    verbose: bool,
    registry: PatchFeatureExtractorRegistry | None = None,
    announce_completion: bool = True,
) -> tuple[list, list]:
    from tqdm import tqdm

    configure_logging(verbose)

    from atlas_patch.orchestration.runner import ProcessingRunner
    from atlas_patch.services.extraction import PatchExtractionService
    from atlas_patch.services.feature_embedding import PatchFeatureEmbeddingService
    from atlas_patch.services.mpp import CSVMPPResolver
    from atlas_patch.services.segmentation import SAM2SegmentationService
    from atlas_patch.services.visualization import DefaultVisualizationService
    from atlas_patch.services.wsi_loader import DefaultWSILoader

    segmentation_service = SAM2SegmentationService(app_cfg.segmentation)
    extractor_service = PatchExtractionService(app_cfg.extraction, app_cfg.output)
    visualizer_service = None
    if (
        app_cfg.output.visualize_grids
        or app_cfg.output.visualize_mask
        or app_cfg.output.visualize_contours
    ):
        visualizer_service = DefaultVisualizationService(
            app_cfg.output,
            app_cfg.extraction,
            app_cfg.visualization,
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

    try:
        results, failures = runner.run(slides=slides)
    finally:
        segmentation_service.close()

    if app_cfg.features is not None:
        feature_service = PatchFeatureEmbeddingService(
            app_cfg.extraction,
            app_cfg.output,
            app_cfg.features,
            registry=registry,
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

    if announce_completion:
        click.echo("Segmentation and patch coordinate extraction complete.")
    return results, failures


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
    announce_completion: bool = True,
) -> tuple[list, list]:
    app_cfg = _build_wsi_app_config(
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
        feature_cfg=feature_cfg,
    )
    return _run_pipeline_from_config(
        app_cfg,
        verbose=verbose,
        registry=registry,
        announce_completion=announce_completion,
    )


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
    from tqdm import tqdm

    from atlas_patch.core.config import ProcessingConfig, SegmentationConfig, VisualizationConfig
    from atlas_patch.core.models import Slide
    from atlas_patch.services.mpp import CSVMPPResolver
    from atlas_patch.services.segmentation import SAM2SegmentationService
    from atlas_patch.services.wsi_loader import DefaultWSILoader

    configure_logging(verbose)

    processing_cfg = ProcessingConfig(
        input_path=Path(wsi_path),
        recursive=recursive,
        mpp_csv=Path(mpp_csv) if mpp_csv else None,
    ).validated()
    segmentation_cfg = (
        SegmentationConfig(
            checkpoint_path=None,
            config_path=default_sam2_config_path(),
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


def _echo_slide_results(
    results: list, failures: list, verbose: bool
) -> None:
    click.echo(f"Completed {len(results)} slide embedding(s), failures: {len(failures)}")
    if verbose:
        for res in results:
            status = "reused" if res.metadata.get("reused") else "written"
            click.echo(
                f"[OK] {res.slide.path.name} -> {res.h5_path} ({res.dataset_key}, {status})"
            )
        for slide, err in failures:
            click.echo(f"[FAIL] {slide.path.name}: {err}", err=True)


@click.group()
@click.version_option(version=__version__)
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
    registry, feat_device, precision = _build_patch_feature_registry(
        device=device,
        feature_device=feature_device,
        feature_num_workers=feature_num_workers,
        feature_precision=feature_precision,
        feature_plugins=feature_plugins,
    )

    available_extractors = registry.available()
    feats = parse_named_list(
        feature_extractors,
        choices=available_extractors,
        item_label="feature extractor",
    )
    feature_cfg = FeatureExtractionConfig(
        extractors=feats,
        batch_size=feature_batch_size,
        device=feat_device,
        num_workers=feature_num_workers,
        precision=precision,
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
@feature_runtime_options
@common_options
@click.option(
    "--slide-encoders",
    required=True,
    type=str,
    help="Space/comma separated slide encoders to run.",
)
def encode_slide(
    wsi_path: str,
    output: str,
    patch_size: int,
    step_size: int | None,
    target_mag: int,
    device: str,
    feature_device: str | None,
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
    slide_encoders: str,
):
    """Run the WSI pipeline and append slide embeddings into canonical AtlasPatch H5 files."""
    from atlas_patch.models.slide import build_default_registry
    from atlas_patch.services.slide_embedding import SlideEmbeddingService

    slide_registry = build_default_registry(device=device.lower())
    available_encoders = slide_registry.available()
    encoders = parse_named_list(
        slide_encoders,
        choices=available_encoders,
        item_label="slide encoder",
    )
    required_patch_size, required_patch_encoders = _resolve_slide_encoder_requirements(
        slide_registry,
        encoders,
    )
    if patch_size != required_patch_size:
        raise click.ClickException(
            f"Requested slide encoders require patch_size={required_patch_size}, "
            f"but encode-slide was given --patch-size {patch_size}."
        )
    patch_registry, feat_device, precision = _build_patch_feature_registry(
        device=device,
        feature_device=feature_device,
        feature_num_workers=feature_num_workers,
        feature_precision=feature_precision,
        feature_plugins=feature_plugins,
    )
    available_extractors = patch_registry.available()
    missing_extractors = [name for name in required_patch_encoders if name not in available_extractors]
    if missing_extractors:
        joined = ", ".join(missing_extractors)
        raise click.ClickException(
            f"Required patch extractor(s) for the selected slide encoders are unavailable: {joined}"
        )

    feature_cfg = FeatureExtractionConfig(
        extractors=required_patch_encoders,
        batch_size=feature_batch_size,
        device=feat_device,
        num_workers=feature_num_workers,
        precision=precision,
        plugins=[Path(p) for p in feature_plugins],
    ).validated()
    slide_cfg = SlideEncodingConfig(
        encoders=encoders,
        device=device.lower(),
        skip_existing=skip_existing,
    ).validated()
    app_cfg = _build_wsi_app_config(
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
        feature_cfg=feature_cfg,
    )
    slides = [
        Slide(path=Path(path))
        for path in get_wsi_files(
            str(app_cfg.processing.input_path),
            recursive=app_cfg.processing.recursive,
        )
    ]
    _validate_existing_slide_outputs(app_cfg, slides)

    try:
        _, failures = _run_pipeline_from_config(
            app_cfg,
            slides=slides,
            verbose=verbose,
            registry=patch_registry,
            announce_completion=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    failed_slide_paths = {slide.path.resolve() for slide, _ in failures}
    artifacts, artifact_failures = _collect_ready_slide_artifacts(
        app_cfg,
        slides,
        required_patch_encoders=required_patch_encoders,
        failed_slide_paths=failed_slide_paths,
    )
    failures.extend(artifact_failures)
    results, embedding_failures = SlideEmbeddingService(
        app_cfg.extraction,
        app_cfg.output,
        slide_cfg,
        registry=slide_registry,
    ).embed_all(artifacts)
    failures.extend(embedding_failures)
    _echo_slide_results(results, failures, verbose)


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
    click.echo("Slide embeddings append into slide_features/<encoder> inside patches/<stem>.h5.")


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
