from __future__ import annotations

import cv2
import numpy as np


def is_black_patch(patch: np.ndarray, rgb_thresh: int = 40, min_fraction: float = 0.7) -> bool:
    """Return True if the patch is mostly black.

    Heuristic: convert to grayscale and check what fraction of pixels are below
    `rgb_thresh`. If that fraction >= `min_fraction`, treat as black.
    """
    if patch.ndim == 3 and patch.shape[2] == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch.astype(np.uint8)
    frac = float((gray < rgb_thresh).mean())
    return bool(frac >= float(min_fraction))


def is_white_patch(
    patch: np.ndarray,
    sat_thresh: int = 5,
    min_fraction: float = 0.7,
    value_thresh: int = 200,
) -> bool:
    """Return True if the patch is mostly white using HSV.

    Heuristic: convert to HSV and count pixels with low saturation (< sat_thresh)
    AND high value/brightness (>= value_thresh). If that fraction >= min_fraction,
    treat as white.
    """
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    s = patch_hsv[:, :, 1]
    v = patch_hsv[:, :, 2]
    mask = (s < sat_thresh) & (v >= value_thresh)
    frac = float(mask.mean())
    return bool(frac >= float(min_fraction))
