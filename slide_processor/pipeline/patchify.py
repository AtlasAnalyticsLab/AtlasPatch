from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from slide_processor.patch_extractor.patch_extractor import PatchExtractor
from slide_processor.utils.contours import mask_to_contours, scale_contours
from slide_processor.wsi import WSIFactory


@dataclass
class PatchifyParams:
    """Patch extraction configuration."""

    patch_size: int
    target_magnification: int
    step_size: int | None = None  # Defaults to patch_size when None
    tissue_area_thresh: float = 0.01
    require_all_points: bool = False
    white_thresh: int = 15
    black_thresh: int = 50
    use_padding: bool = True


@dataclass
class SegmentParams:
    """Segmentation configuration for SAM2."""

    checkpoint_path: Path
    config_file: Path
    device: str = "cuda"
    thumbnail_max: int = 1024


def _build_segmentation_predictor(seg: SegmentParams):
    """Create a SAM2 predictor callable and return it with thumbnail size."""
    from slide_processor.segmentation.sam2_segmentation import SAM2SegmentationModel

    model = SAM2SegmentationModel(
        checkpoint_path=seg.checkpoint_path,
        config_file=seg.config_file,
        device=seg.device,
    )

    def _predict(img) -> np.ndarray:
        return model.predict_image(img, resize_to_input=True)

    return _predict, seg.thumbnail_max


def segment_and_patchify(
    wsi_path: str,
    output_dir: str,
    *,
    seg: SegmentParams,
    patch: PatchifyParams,
    save_images: bool = False,
    store_images: bool = True,
    fast_mode: bool = False,
    predict_fn: Callable[[Any], np.ndarray] | None = None,
    thumb_max: int | None = None,
) -> str | None:
    """High-level pipeline: segment tissue and patchify WSI into an HDF5 file.

    Required parameters:
    - seg: SegmentParams (checkpoint, config, device, thumbnail_max)
    - patch: PatchifyParams (patch_size and target_magnification required; step_size defaults to patch_size if None)

    Behavior:
    - If `save_images` is True, PNGs are saved under `<output_dir>/<stem>/images/`.
    - Returns HDF5 path as string, or None if no patches were saved.
    """
    # Ensure required patch values
    if patch.patch_size is None or int(patch.patch_size) <= 0:
        raise ValueError("patch.patch_size must be a positive integer")
    if patch.step_size is None:
        patch.step_size = int(patch.patch_size)

    # Load WSI and setup thumb segmentation
    wsi = WSIFactory.load(wsi_path)
    W, H = wsi.get_size(lv=0)

    if predict_fn is None or thumb_max is None:
        predict_fn, thumb_max = _build_segmentation_predictor(seg)
    # mypy: ensure non-None after lazy init
    assert predict_fn is not None and thumb_max is not None
    thumb = wsi.get_thumb((thumb_max, thumb_max))

    # Predict binary mask (H, W) in [0, 1]
    mask = predict_fn(thumb)

    # Extract contours on thumbnail
    tissue_contours_t, holes_contours_t = mask_to_contours(
        mask, tissue_area_thresh=patch.tissue_area_thresh
    )

    # Scale contours to level 0
    ht, wt = mask.shape[:2]
    sx = W / float(wt)
    sy = H / float(ht)
    tissue_contours = scale_contours(tissue_contours_t, sx, sy)
    holes_contours = [scale_contours(hs, sx, sy) for hs in holes_contours_t]

    # Extractor and output path
    extractor = PatchExtractor(
        patch_size=patch.patch_size,
        step_size=patch.step_size,
        target_mag=patch.target_magnification,
        white_thresh=patch.white_thresh,
        black_thresh=patch.black_thresh,
        use_padding=patch.use_padding,
        require_all_points=patch.require_all_points,
    )

    stem = Path(wsi_path).stem
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = str(out_dir / f"{stem}.h5")

    img_dir = str(out_dir / "images") if save_images else None

    result_path = extractor.extract_to_h5(
        wsi,
        tissue_contours,
        holes_contours,
        out_h5,
        image_output_dir=img_dir,
        store_images=store_images,
        fast_mode=fast_mode,
        batch=512,
    )

    try:
        wsi.cleanup()
    except Exception:
        pass

    return result_path
