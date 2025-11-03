from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from slide_processor.patch_extractor.patch_extractor import PatchExtractor
from slide_processor.utils.contours import mask_to_contours, scale_contours
from slide_processor.wsi.iwsi import IWSI


@dataclass
class PatchifyParams:
    """Patch extraction configuration."""

    patch_size: int
    target_magnification: int
    step_size: int | None = None  # Defaults to patch_size when None
    tissue_area_thresh: float = 0.01
    white_thresh: int = 15
    black_thresh: int = 50


@dataclass
class SegmentParams:
    """Segmentation configuration for SAM2."""

    checkpoint_path: Path
    config_file: Path = None  # type: ignore
    device: str = "cuda"
    thumbnail_max: int = 1024

    def __post_init__(self):
        """Set default config file if not provided."""
        default_cfg = Path(__file__).resolve().parent.parent / "configs" / "sam2.1_hiera_t.yaml"
        if not default_cfg.exists():
            raise FileNotFoundError(f"Built-in SAM2 config not found: {default_cfg}")
        self.config_file = default_cfg


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
    wsi: IWSI,
    output_dir: str,
    *,
    seg: SegmentParams,
    patch: PatchifyParams,
    save_images: bool = False,
    fast_mode: bool = False,
    predict_fn: Callable[[Any], np.ndarray] | None = None,
    thumb_max: int | None = None,
    mask_override: np.ndarray | None = None,
    write_batch: int = 8192,
) -> str | None:
    """High-level pipeline: segment tissue and patchify WSI into an HDF5 file.

    Required parameters:
    - wsi: Opened WSI object (IWSI implementation) to operate on
    - seg: SegmentParams (checkpoint, config, device, thumbnail_max)
    - patch: PatchifyParams (patch_size and target_magnification required; step_size defaults to patch_size if None)

    Behavior:
    - HDF5 files are saved under `<output_dir>/patches/<stem>.h5`.
    - If `save_images` is True, per-patch PNGs are saved under
      `<output_dir>/images/<stem>/`.
    - Returns HDF5 path as string, or None if no patches were saved.
    """
    # Ensure required patch values
    if patch.patch_size is None or int(patch.patch_size) <= 0:
        raise ValueError("patch.patch_size must be a positive integer")
    if patch.step_size is None:
        patch.step_size = int(patch.patch_size)

    # Setup dimensions for segmentation/patchification
    W, H = wsi.get_size(lv=0)

    if mask_override is not None:
        mask = mask_override
        ht, wt = mask.shape[:2]
    else:
        if predict_fn is None or thumb_max is None:
            predict_fn, thumb_max = _build_segmentation_predictor(seg)
        # ensure non-None after lazy init
        assert predict_fn is not None and thumb_max is not None
        thumb = wsi.get_thumb((thumb_max, thumb_max))
        mask = predict_fn(thumb)
        ht, wt = mask.shape[:2]

    # Extract contours on thumbnail
    tissue_contours_t, holes_contours_t = mask_to_contours(
        mask, tissue_area_thresh=patch.tissue_area_thresh
    )

    # Scale contours to level 0
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
    )

    stem = Path(wsi.path).stem
    patches_root = Path(output_dir) / "patches"
    patches_root.mkdir(parents=True, exist_ok=True)
    out_h5 = str(patches_root / f"{stem}.h5")

    img_dir = str(Path(output_dir) / "images" / stem) if save_images else None

    result_path = extractor.extract_to_h5(
        wsi,
        tissue_contours,
        holes_contours,
        out_h5,
        image_output_dir=img_dir,
        fast_mode=fast_mode,
        batch=write_batch,
    )

    return result_path
