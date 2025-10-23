from __future__ import annotations

import cv2
import numpy as np


def is_black_patch(patch: np.ndarray, rgb_thresh: int = 40) -> bool:
    """Return True if the RGB patch is mostly black.

    A simple but fast heuristic: all channels below `rgb_thresh`.
    """
    return bool(np.all(patch < rgb_thresh))


def is_white_patch(patch: np.ndarray, sat_thresh: int = 5) -> bool:
    """Return True if the RGB patch is mostly white using low saturation.

    Converts RGB patch to HSV and checks for low saturation values.
    """
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return bool(np.all(patch_hsv[:, :, 1] < sat_thresh))
