"""Factory for backend selection."""

import os
from pathlib import Path
from typing import Optional

from .image_wsi import ImageWSI
from .iwsi import IWSI
from .openslide_wsi import OpenSlideWSI


class WSIFactory:
    """WSI loader with backend selection."""

    _registry = {
        "openslide": OpenSlideWSI,
        "image": ImageWSI,
    }

    _formats = {
        ".svs": "openslide",
        ".tif": "openslide",
        ".tiff": "openslide",
        ".ndpi": "openslide",
        ".vms": "openslide",
        ".vmu": "openslide",
        ".scn": "openslide",
        ".mrxs": "openslide",
        ".bif": "openslide",
        ".biff": "openslide",
        ".dcm": "openslide",
        ".dicom": "openslide",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".bmp": "image",
        ".webp": "image",
        ".gif": "image",
    }

    @classmethod
    def register(cls, name: str, impl_class) -> None:
        """Register a backend implementation."""
        cls._registry[name] = impl_class

    @classmethod
    def map_extension(cls, ext: str, backend: str) -> None:
        """Map file extension to backend."""
        if backend not in cls._registry:
            raise ValueError(f"Unknown backend: {backend}")
        if not ext.startswith("."):
            ext = "." + ext
        cls._formats[ext.lower()] = backend

    @classmethod
    def detect(cls, path: str) -> Optional[str]:
        """Detect backend from file extension."""
        ext = Path(path).suffix.lower()
        return cls._formats.get(ext)

    @classmethod
    def load(
        cls, path: str, backend: Optional[str] = None, mpp: Optional[float] = None, **kwargs
    ) -> IWSI:
        """Load WSI with specified or detected backend.

        Parameters
        ----------
        path : str
            Path to WSI file.
        backend : str, optional
            Backend name. If None, auto-detect from extension.
        mpp : float, optional
            Custom microns per pixel value to override metadata extraction.
        **kwargs
            Arguments passed to backend constructor.

        Returns
        -------
        IWSI
            Loaded WSI instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if backend is None:
            backend = cls.detect(path)
            if backend is None:
                raise ValueError(f"No backend found for: {path}")
        elif backend not in cls._registry:
            raise ValueError(f"Unknown backend: {backend}")

        impl = cls._registry[backend]
        return impl(path=path, mpp=mpp, **kwargs)

    @classmethod
    def try_load(
        cls, path: str, backends: Optional[list] = None, mpp: Optional[float] = None, **kwargs
    ) -> IWSI:
        """Try multiple backends until one succeeds.

        Parameters
        ----------
        path : str
            Path to WSI file.
        backends : list, optional
            List of backends to try in order.
        mpp : float, optional
            Custom microns per pixel value to override metadata extraction.
        **kwargs
            Arguments passed to backend constructor.

        Returns
        -------
        IWSI
            Loaded WSI instance.

        Raises
        ------
        RuntimeError
            If all backends fail.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if backends is None:
            backends = list(cls._registry.keys())

        errors = []
        for b in backends:
            if b not in cls._registry:
                errors.append(f"{b}: not registered")
                continue

            try:
                return cls.load(path, backend=b, mpp=mpp, **kwargs)
            except Exception as e:
                errors.append(f"{b}: {str(e)}")

        msg = f"All backends failed for {path}:\n" + "\n".join(errors)
        raise RuntimeError(msg)
