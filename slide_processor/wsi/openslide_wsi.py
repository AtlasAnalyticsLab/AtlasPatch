from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:
    raise ImportError(
        "OpenSlide is required. Install with: pip install openslide-python"
    )

from .iwsi import IWSI


class OpenSlideWSI(IWSI):
    """OpenSlide backend implementation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OpenSlideWSI."""
        super().__init__(**kwargs)
        self._oslide: Optional[openslide.OpenSlide] = None

    def _setup(self) -> None:
        """Initialize OpenSlide and extract metadata."""
        try:
            self._oslide = openslide.OpenSlide(self.path)

            self.w, self.h = self._oslide.dimensions
            self.nlvl = self._oslide.level_count
            self.ds = list(self._oslide.level_downsamples)
            self.dims = list(self._oslide.level_dimensions)
            self.meta = dict(self._oslide.properties)

            if self._mpp_manual is not None:
                self.mpp = self._mpp_manual
            else:
                self.mpp = self._extract_mpp()

            self.mag = self._extract_mag()

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.path}")
        except openslide.OpenSlideError as e:
            raise RuntimeError(f"OpenSlide error: {e}")
        except Exception as e:
            raise RuntimeError(f"Setup failed: {e}")

    def _extract_mpp(self) -> Optional[float]:
        """Extract MPP from metadata."""
        if self._oslide is None or self.meta is None:
            return None

        keys = [
            openslide.PROPERTY_NAME_MPP_X,
            "openslide.mirax.MPP",
            "aperio.MPP",
        ]

        for key in keys:
            if key in self.meta:
                try:
                    return round(float(self.meta[key]), 4)
                except (ValueError, TypeError):
                    continue

        try:
            x_res = self.meta.get("tiff.XResolution")
            unit = self.meta.get("tiff.ResolutionUnit")
            if x_res and unit:
                x_res_f = float(x_res)
                if unit.lower() == "centimeter":
                    return round(10000 / x_res_f, 4)
                elif unit.lower() == "inch":
                    return round(25400 / x_res_f, 4)
        except (ValueError, TypeError):
            pass

        return None

    def _extract_mag(self) -> Optional[int]:
        """Extract magnification."""
        if self._oslide is None or self.meta is None:
            return None

        obj_pow = self.meta.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if obj_pow:
            try:
                return int(float(obj_pow))
            except (ValueError, TypeError):
                pass

        if self.mpp is not None:
            try:
                return self._infer_mag(self.mpp)
            except ValueError:
                pass

        return None

    def extract(
        self,
        xy: Tuple[int, int],
        lv: int,
        wh: Tuple[int, int],
        *,
        mode: Literal["array", "image"] = "array",
    ) -> Union[np.ndarray, Image.Image]:
        """Extract region from slide."""
        self._ensure_loaded()

        if self._oslide is None:
            raise RuntimeError("OpenSlide not initialized")

        region = self._oslide.read_region(xy, lv, wh).convert("RGB")

        if mode == "image":
            return region
        elif mode == "array":
            return np.array(region)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_size(self, lv: int = 0) -> Tuple[int, int]:
        """Get size at level."""
        self._ensure_loaded()

        if self._oslide is None:
            raise RuntimeError("OpenSlide not initialized")

        if lv < 0 or lv >= self.nlvl:
            raise IndexError(f"Level {lv} out of range")

        return self.dims[lv]

    def get_thumb(self, max_hw: Tuple[int, int]) -> Image.Image:
        """Get thumbnail."""
        self._ensure_loaded()

        if self._oslide is None:
            raise RuntimeError("OpenSlide not initialized")

        return self._oslide.get_thumbnail(max_hw).convert("RGB")

    def cleanup(self) -> None:
        """Release resources."""
        if self._oslide is not None:
            try:
                self._oslide.close()
            except Exception:
                pass
            finally:
                self._oslide = None

        self._loaded = False

    def __del__(self) -> None:
        self.cleanup()
