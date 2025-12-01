from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import cv2
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

    @staticmethod
    def _clean_meta_value(val: Any) -> str | None:
        """Normalize metadata values to a trimmed string or None."""
        if val is None:
            return None
        try:
            text = str(val).strip()
        except Exception:
            return None
        return text or None

    @classmethod
    def _find_meta_value(
        cls, meta: Mapping[str, Any], keys: Sequence[str], *, contains: Sequence[str] | None = None
    ) -> str | None:
        if not meta:
            return None

        normalized: dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                continue
            try:
                lower = str(key).lower()
            except Exception:
                continue
            if lower not in normalized:
                normalized[lower] = value

        for key in keys:
            val = normalized.get(key.lower())
            text = cls._clean_meta_value(val)
            if text:
                return text

        if contains:
            for key in sorted(normalized):
                if any(token in key for token in contains):
                    text = cls._clean_meta_value(normalized[key])
                    if text:
                        return text

        return None

    def metadata_attrs(self) -> Dict[str, Any]:
        """Collect optional WSI metadata for downstream storage."""
        self._ensure_loaded()
        meta = self.meta or {}
        vendor = self._find_meta_value(
            meta,
            ["openslide.vendor", "tiff.make", "tiff.model", "hamamatsu.model", "leica.scanner"],
            contains=["vendor"],
        )
        institution = self._find_meta_value(
            meta,
            [
                "tiff.institution",
                "tiff.institutionname",
                "aperio.institution",
                "openslide.institution",
                "dicom.institutionname",
            ],
            contains=["institution"],
        )
        stain = self._find_meta_value(
            meta,
            [
                "aperio.stain",
                "aperio.staindescription",
                "openslide.stain",
                "hamamatsu.stain",
                "philips.stain",
            ],
            contains=["stain"],
        )

        attrs: Dict[str, Any] = {}
        if self.mpp is not None:
            attrs["mpp"] = self.mpp
        if self.mag is not None:
            attrs["magnification"] = int(self.mag)
        if vendor:
            attrs["vendor"] = vendor
        if institution:
            attrs["institution"] = institution
        if stain:
            attrs["stain"] = stain

        return attrs

    def get_thumbnail_at_power(
        self,
        *,
        power: float = 1.25,
        interpolation: str = "optimise",
    ) -> Image.Image:
        """Create a full-slide thumbnail at a fixed objective power.

        - Uses base magnification (self.mag) to compute target downsample ds = base_mag / power.
        - Selects an optimal pyramid level and resizes to exact output size.

        Parameters
        ----------
        power : float, default 1.25
            Objective power for thumbnail generation.
        interpolation : str, default "optimise"
            Interpolation policy: "optimise" uses AREA (downscale) or CUBIC (upscale).

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail at approximately (W0/ds, H0/ds).
        """
        self._ensure_loaded()

        if self.mag is None:
            raise ValueError(
                "WSI base magnification is unknown; cannot generate power-based thumbnail."
            )

        W0, H0 = self.get_size(lv=0)
        if W0 <= 0 or H0 <= 0:
            raise ValueError("Invalid WSI dimensions.")

        base_mag = float(self.mag)
        tgt_power = float(power)
        if tgt_power <= 0:
            raise ValueError("thumbnail power must be positive")

        # Target downsample factor at level 0
        ds_target = max(1e-6, base_mag / tgt_power)

        # Choose pyramid level closest to ds_target
        level, _ = self.optimal_level(ds_target)
        downsamples = self.ds or [1.0]
        ds_lvl = float(downsamples[level])

        # Read entire slide at computed level
        read_w = max(1, int(round(W0 / ds_lvl)))
        read_h = max(1, int(round(H0 / ds_lvl)))
        arr_any = self.extract((0, 0), lv=level, wh=(read_w, read_h), mode="array")
        if not isinstance(arr_any, np.ndarray):
            raise RuntimeError("Failed to read thumbnail region as array")
        arr = arr_any

        # Compute exact output size for ds_target
        out_w = max(1, int(round(W0 / ds_target)))
        out_h = max(1, int(round(H0 / ds_target)))

        if arr.shape[1] != out_w or arr.shape[0] != out_h:
            if interpolation == "optimise":
                # Downscale → area; upscale → cubic
                if out_w < arr.shape[1] or out_h < arr.shape[0]:
                    interp = cv2.INTER_AREA
                else:
                    interp = cv2.INTER_CUBIC
            elif interpolation == "area":
                interp = cv2.INTER_AREA
            elif interpolation == "cubic":
                interp = cv2.INTER_CUBIC
            elif interpolation == "linear":
                interp = cv2.INTER_LINEAR
            else:
                interp = cv2.INTER_LINEAR

            arr = cv2.resize(arr, (out_w, out_h), interpolation=interp)

        return Image.fromarray(arr)

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
