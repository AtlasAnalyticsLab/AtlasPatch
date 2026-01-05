import re
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError as e:
    raise ImportError("OpenSlide is required. Install with: pip install openslide-python") from e

from .iwsi import IWSI


class OpenSlideWSI(IWSI):
    """OpenSlide backend implementation."""

    # Metadata keys for direct MPP lookup (in priority order)
    _MPP_KEYS = (
        openslide.PROPERTY_NAME_MPP_X,  # "openslide.mpp-x"
        "openslide.mpp-y",
        "openslide.mirax.MPP",
        "aperio.MPP",
        "hamamatsu.XResolution",
    )

    # Keys containing free-text that may embed MPP
    _MPP_TEXT_KEYS = ("openslide.comment", "tiff.ImageDescription")

    # Magnification keys for fallback MPP estimation
    _MAG_KEYS = ("aperio.AppMag", "openslide.objective-power", "hamamatsu.SourceLens")

    

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
                self.mpp = self.validate_mpp(self._mpp_manual, source="user-provided mpp")
            else:
                extracted = self._extract_mpp()
                if extracted is not None:
                    self.mpp = self.validate_mpp(extracted, source="slide metadata")
                else:
                    self.mpp = None

            self.mag = self._extract_mag()

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.path}") from e
        except openslide.OpenSlideError as e:
            raise RuntimeError(f"OpenSlide error: {e}") from e
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Setup failed: {e}") from e

    def _extract_mpp(self) -> Optional[float]:
        """Extract MPP from metadata using multiple strategies.

        Lookup order:
        1. Direct metadata keys (openslide.mpp-x, aperio.MPP, etc.)
        2. Regex parsing from comment/description fields
        3. TIFF resolution fields (XResolution + ResolutionUnit)
        4. Estimation from magnification (10.0 / mag)

        Returns
        -------
        float or None
            MPP in microns per pixel, rounded to 4 decimal places.
        """
        if self._oslide is None or self.meta is None:
            return None

        # 1. Direct key lookup
        for key in self._MPP_KEYS:
            if key in self.meta:
                try:
                    return round(float(self.meta[key]), 4)
                except (ValueError, TypeError):
                    continue

        # 2. Parse from text fields (comment, description)
        for key in self._MPP_TEXT_KEYS:
            parsed = self._parse_mpp_from_string(self.meta.get(key))
            if parsed is not None:
                return round(parsed, 4)

        # 3. TIFF resolution fallback
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

        # 4. Magnification-based estimation (10.0 / mag)
        for mag_key in self._MAG_KEYS:
            mag_val = self.meta.get(mag_key)
            if mag_val is not None:
                try:
                    mag = float(mag_val)
                    if mag > 0:
                        return round(10.0 / mag, 4)
                except (ValueError, TypeError):
                    continue

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

    @staticmethod
    def _parse_mpp_from_string(val: Optional[str]) -> Optional[float]:
        """Parse MPP value from a free-text metadata string.

        Searches for patterns like:
        - "mpp = 0.25" or "mpp: 0.25"
        - "microns per pixel 0.25"

        Parameters
        ----------
        val : str or None
            Text to parse.

        Returns
        -------
        float or None
            Parsed MPP value, or None if not found.
        """
        if not val:
            return None

        patterns = (
            r"mpp\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"microns?\s+per\s+pixel[^0-9]*([0-9]*\.?[0-9]+)",
        )

        for pattern in patterns:
            match = re.search(pattern, val, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

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

        if self.nlvl is None or self.dims is None:
            raise RuntimeError("Metadata not initialized")

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
