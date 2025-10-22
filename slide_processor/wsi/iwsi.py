from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image


class IWSI(ABC):
    """Base interface for whole slide image access."""

    def __init__(self, path: str, mpp: Optional[float] = None):
        """Initialize WSI interface.

        Parameters
        ----------
        path : str
            Path to the WSI file.
        mpp : float, optional
            Manual microns per pixel value.
        """
        self.path = path
        self._mpp_manual = mpp
        self._loaded = False

        self.w: Optional[int] = None
        self.h: Optional[int] = None
        self.nlvl: Optional[int] = None
        self.ds: Optional[list[float]] = None
        self.dims: Optional[list[Tuple[int, int]]] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.mpp: Optional[float] = None
        self.mag: Optional[int] = None

    def _ensure_loaded(self) -> None:
        """Ensure internal state is initialized."""
        if not self._loaded:
            self._setup()
            self._loaded = True

    @abstractmethod
    def _setup(self) -> None:
        """Perform setup and metadata extraction."""
        pass

    @abstractmethod
    def _extract_mpp(self) -> Optional[float]:
        """Extract microns per pixel value."""
        pass

    @abstractmethod
    def _extract_mag(self) -> Optional[int]:
        """Extract magnification value."""
        pass

    @abstractmethod
    def extract(
        self,
        xy: Tuple[int, int],
        lv: int,
        wh: Tuple[int, int],
        *,
        mode: Literal["array", "image"] = "array",
    ) -> Union[np.ndarray, Image.Image]:
        """Extract region from WSI.

        Parameters
        ----------
        xy : tuple of int
            (x, y) coordinates of top-left corner.
        lv : int
            Pyramid level to read from.
        wh : tuple of int
            (width, height) of region.
        mode : {"array", "image"}, default "array"
            Output format.

        Returns
        -------
        np.ndarray or PIL.Image.Image
            Extracted region.
        """
        pass

    @abstractmethod
    def get_size(self, lv: int = 0) -> Tuple[int, int]:
        """Get dimensions at specific level.

        Parameters
        ----------
        lv : int, default 0
            Pyramid level.

        Returns
        -------
        tuple of int
            (width, height) in pixels.
        """
        pass

    @abstractmethod
    def get_thumb(self, max_hw: Tuple[int, int]) -> Image.Image:
        """Generate thumbnail.

        Parameters
        ----------
        max_hw : tuple of int
            Maximum (width, height) for thumbnail.

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        pass

    def optimal_level(self, target_ds: float) -> Tuple[int, float]:
        """Find optimal pyramid level for target downsample.

        Parameters
        ----------
        target_ds : float
            Target downsample factor.

        Returns
        -------
        tuple of (int, float)
            (level, additional_downsample).
        """
        self._ensure_loaded()
        downsamples = self.ds or [1.0]

        for i, d in enumerate(downsamples):
            if abs(d - target_ds) < 0.01:
                return i, 1.0

        if target_ds >= downsamples[0]:
            best_i, best_d = 0, downsamples[0]
            for i, d in enumerate(downsamples):
                if d <= target_ds:
                    best_i, best_d = i, d
                else:
                    break
            return best_i, target_ds / best_d
        else:
            for i, d in enumerate(downsamples):
                if d >= target_ds:
                    return i, d / target_ds

        raise ValueError(f"No level for target downsample {target_ds}")

    def _infer_mag(self, m: float) -> int:
        """Infer magnification from microns per pixel.

        Parameters
        ----------
        m : float
            Microns per pixel.

        Returns
        -------
        int
            Inferred magnification.
        """
        thresholds = [
            (0.16, 80),
            (0.2, 60),
            (0.3, 40),
            (0.6, 20),
            (1.2, 10),
            (2.4, 5),
        ]
        for threshold, mag_val in thresholds:
            if m < threshold:
                return mag_val
        raise ValueError(f"Cannot infer magnification from mpp {m}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    def __repr__(self) -> str:
        if self._loaded:
            return f"<{self.__class__.__name__}: {self.w}x{self.h}>"
        return f"<{self.__class__.__name__}: loading pending>"
