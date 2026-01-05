from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .iwsi import IWSI


class ImageWSI(IWSI):
    """Standard image backend."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ImageWSI.

        Parameters
        ----------
        mpp : float, required
            Microns per pixel value (mandatory for standard images).
            Must be within valid range [IWSI.MPP_MIN, IWSI.MPP_MAX].
        """
        mpp = kwargs.get("mpp")
        if mpp is None:
            raise ValueError("mpp parameter is required for standard images")
        if mpp <= 0:
            raise ValueError(f"mpp must be positive, got {mpp}")

        super().__init__(**kwargs)
        self._pil_img: Optional[Image.Image] = None

        self._mpp_value = self.validate_mpp(mpp, source="user-provided mpp")

    def _setup(self) -> None:
        """Initialize image and extract metadata."""
        try:
            self._load_image()

            if self._pil_img is None:
                raise RuntimeError("Image not loaded")

            self.w, self.h = self._pil_img.size
            self.nlvl = 1
            self.ds = [1.0]
            self.dims = [(self.w, self.h)]

            self.meta = {
                "format": self._pil_img.format or "unknown",
                "mode": self._pil_img.mode,
            }

            self.mpp = self._mpp_value

            self.mag = self._extract_mag()

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image not found: {self.path}") from e
        except Exception as e:
            raise RuntimeError(f"Setup failed: {e}") from e

    def _load_image(self) -> None:
        """Load and convert image to RGB."""
        if self._pil_img is None:
            try:
                self._pil_img = Image.open(self.path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot open: {self.path}: {e}") from e

    def _extract_mpp(self) -> Optional[float]:
        """Return configured MPP value."""
        return self._mpp_value

    def _extract_mag(self) -> Optional[int]:
        """Extract magnification from MPP."""
        if self.mpp is not None:
            try:
                return self._infer_mag(self.mpp)
            except ValueError:
                return None
        return None

    def extract(
        self,
        xy: Tuple[int, int],
        lv: int,
        wh: Tuple[int, int],
        *,
        mode: Literal["array", "image"] = "array",
    ) -> Union[np.ndarray, Image.Image]:
        """Extract region from image."""
        self._ensure_loaded()

        if lv != 0:
            raise ValueError("Standard images only support level 0")

        if self._pil_img is None:
            raise RuntimeError("Image not loaded")

        x, y = xy
        w, h = wh
        region = self._pil_img.crop((x, y, x + w, y + h)).convert("RGB")

        if mode == "image":
            return region
        elif mode == "array":
            return np.array(region)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_size(self, lv: int = 0) -> Tuple[int, int]:
        """Get image size."""
        self._ensure_loaded()

        if lv != 0:
            raise ValueError("Standard images only support level 0")

        if self._pil_img is None:
            raise RuntimeError("Image not loaded")

        if self.w is None or self.h is None:
            raise RuntimeError("Image dimensions not set")

        return (self.w, self.h)

    def get_thumb(self, max_hw: Tuple[int, int]) -> Image.Image:
        """Get thumbnail."""
        self._ensure_loaded()

        if self._pil_img is None:
            raise RuntimeError("Image not loaded")

        thumb = self._pil_img.copy()
        thumb.thumbnail(max_hw, Image.Resampling.LANCZOS)
        return thumb

    def cleanup(self) -> None:
        """Release resources."""
        if self._pil_img is not None:
            try:
                self._pil_img.close()
            except Exception:
                pass
            finally:
                self._pil_img = None

        self._loaded = False

    def __del__(self) -> None:
        self.cleanup()
