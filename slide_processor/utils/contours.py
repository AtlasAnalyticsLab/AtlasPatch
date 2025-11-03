from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class FourPointContainment:
    """Four-point containment test around patch center (lenient any-point mode).

    Computes probe points around the patch center and returns True if any probe
    falls inside the provided contour.
    """

    contour: np.ndarray
    patch_size: int
    center_shift: float = 0.5

    def __call__(self, pt: tuple[int, int]) -> bool:
        cx = pt[0] + self.patch_size // 2
        cy = pt[1] + self.patch_size // 2
        shift = int(self.patch_size // 2 * self.center_shift)

        if shift > 0:
            probes = [
                (cx - shift, cy - shift),
                (cx + shift, cy + shift),
                (cx + shift, cy - shift),
                (cx - shift, cy + shift),
            ]
        else:
            probes = [(cx, cy)]

        inside = [cv2.pointPolygonTest(self.contour, p, False) >= 0 for p in probes]
        return any(inside)


def mask_to_contours(
    mask: np.ndarray,
    *,
    tissue_area_thresh: float = 0.01,
    filter_params: dict[str, int] | None = None,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Convert a binary mask (H, W) in [0, 1] to OpenCV contours.

    Returns a list of tissue contours and a parallel list of hole lists for each tissue contour.
    Holes are grouped per their parent contour using OpenCV hierarchy.
    """
    if filter_params is None:
        filter_params = {
            "a_t": 100,  # Minimum tissue contour area (in pixels)
            "a_h": 16,  # Minimum hole area
            "max_n_holes": 10,
        }

    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None or len(contours) == 0:
        return [], []

    # Normalize hierarchy to shape (N, 4): [next, prev, first_child, parent]
    # OpenCV returns either (1, N, 4) or (N, 1, 4) depending on version/platform.
    if hierarchy.ndim == 3:
        if hierarchy.shape[0] == 1:  # (1, N, 4)
            hierarchy = hierarchy[0]
        elif hierarchy.shape[1] == 1:  # (N, 1, 4)
            hierarchy = hierarchy[:, 0, :]
        else:
            # Unexpected shape; flatten conservatively
            hierarchy = hierarchy.reshape(-1, 4)
    elif hierarchy.ndim == 2 and hierarchy.shape[1] == 4:
        # already (N, 4)
        pass
    else:
        hierarchy = hierarchy.reshape(-1, 4)

    # Compute minimum tissue area threshold from fraction of image area (0..1)
    H, W = mask.shape[:2]
    image_area = float(H * W)
    min_area_threshold = tissue_area_thresh * image_area
    effective_min_area = max(min_area_threshold, float(filter_params["a_t"]))

    # Collect tissue contours (parent == -1) and holes (parent != -1)
    tissue_indices: list[int] = []
    holes_by_parent_index: dict[int, list[np.ndarray]] = {}

    for i, cont in enumerate(contours):
        area = cv2.contourArea(cont)
        parent = hierarchy[i][3]
        if parent == -1:
            if area >= effective_min_area:
                tissue_indices.append(i)
        else:
            if area >= filter_params["a_h"]:
                holes_by_parent_index.setdefault(parent, []).append(cont)

    # Limit total number of holes globally to avoid explosion
    all_holes = [h for hs in holes_by_parent_index.values() for h in hs]
    if len(all_holes) > filter_params["max_n_holes"]:
        all_holes_sorted = sorted(all_holes, key=cv2.contourArea, reverse=True)
        allowed = set(map(id, all_holes_sorted[: filter_params["max_n_holes"]]))
        for parent, hs in list(holes_by_parent_index.items()):
            holes_by_parent_index[parent] = [h for h in hs if id(h) in allowed]

    # Build outputs ordered by tissue_indices
    tissue_contours: list[np.ndarray] = []
    holes_per_tissue: list[list[np.ndarray]] = []
    for ti in tissue_indices:
        tissue_contours.append(contours[ti])
        holes_per_tissue.append(list(holes_by_parent_index.get(ti, [])))

    return tissue_contours, holes_per_tissue


def scale_contours(
    contours: Sequence[np.ndarray],
    sx: float,
    sy: float,
) -> list[np.ndarray]:
    """Scale a list of OpenCV contours by (sx, sy)."""
    out: list[np.ndarray] = []
    for c in contours:
        c = c.astype(np.float32)
        c[:, :, 0] *= sx
        c[:, :, 1] *= sy
        out.append(c.astype(np.int32))
    return out
