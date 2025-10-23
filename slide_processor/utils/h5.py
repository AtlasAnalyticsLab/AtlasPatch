import json
import os
import uuid
from typing import Any, Mapping, Optional

import h5py
import numpy as np


def save_h5(
    save_path: str,
    assets: Mapping[str, np.ndarray],
    attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    mode: str = "w",
    file_attrs: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save a dictionary of arrays to an HDF5 file with optional dataset and file attributes.

    Parameters
    ----------
    save_path : str
        Destination path for the HDF5 file.
    assets : dict[str, np.ndarray]
        Mapping from dataset name to numpy array. If empty, only file-level attributes are written.
    attributes : dict[str, dict], optional
        Mapping from dataset name to a dict of attributes.
    mode : str, default "w"
        H5 file mode. "w"/"x" will write to a temporary file and atomically replace target.
    file_attrs : dict[str, Any], optional
        Attributes to save on the file root (e.g., {"patch_size": 256}).
    """
    target_path = save_path
    use_atomic = mode in ("w", "x")
    tmp_path = None

    def _encode_attr(value: Any) -> Any:
        if isinstance(value, dict):
            return json.dumps(value)
        if value is None:
            return "None"
        return value

    try:
        # Pick path for writing
        if use_atomic:
            dir_name = os.path.dirname(os.path.abspath(target_path)) or "."
            base_name = os.path.basename(target_path)
            tmp_name = f".{base_name}.tmp.{uuid.uuid4().hex}"
            tmp_path = os.path.join(dir_name, tmp_name)
            h5_path = tmp_path
        else:
            h5_path = target_path

        # Open file and write datasets/attributes
        with h5py.File(h5_path, mode) as file:
            # Datasets
            for key, val in assets.items():
                data_shape = val.shape
                if key not in file:
                    data_type = val.dtype
                    chunk_shape = (1,) + data_shape[1:]
                    maxshape = (None,) + data_shape[1:]
                    dset = file.create_dataset(
                        key,
                        shape=data_shape,
                        maxshape=maxshape,
                        chunks=chunk_shape,
                        dtype=data_type,
                    )
                    dset[:] = val
                    if attributes is not None and key in attributes:
                        for a_k, a_v in attributes[key].items():
                            dset.attrs[a_k] = _encode_attr(a_v)
                else:
                    dset = file[key]
                    dset.resize(len(dset) + data_shape[0], axis=0)
                    dset[-data_shape[0] :] = val

            # File-level attributes
            if file_attrs:
                for a_k, a_v in file_attrs.items():
                    file.attrs[a_k] = _encode_attr(a_v)

        if use_atomic and tmp_path is not None:
            os.replace(tmp_path, target_path)
    finally:
        # Cleanup temp file on failure
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class PatchesH5:
    """Lightweight reader for patch HDF5 files produced by PatchExtractor.

    Provides lazy access with indexing and length, and exposes metadata.

    Datasets expected:
      - "coords": (N, 2) int32
      - "coords_ext": (N, 5) int32 (x, y, w, h, level) [optional]
      - "imgs": (N, H, W, 3) uint8 [optional]

    Root attributes (optional):
      - "patch_size": int
      - "wsi_path": str
      - "num_patches": int
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._f: h5py.File | None = None

    def open(self) -> None:
        if self._f is None:
            self._f = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            finally:
                self._f = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # Internal
    def _require_open(self) -> h5py.File:
        if self._f is None:
            self.open()
        assert self._f is not None
        return self._f

    @property
    def images(self) -> "h5py.Dataset":
        f = self._require_open()
        return f["imgs"]

    @property
    def coords(self) -> "h5py.Dataset":
        f = self._require_open()
        return f["coords"]

    @property
    def coords_ext(self):
        f = self._require_open()
        if "coords_ext" in f:
            return f["coords_ext"]
        raise KeyError("'coords_ext' not found; older files may only contain 'coords'.")

    @property
    def has_images(self) -> bool:
        f = self._require_open()
        return "imgs" in f

    def __len__(self) -> int:
        f = self._require_open()
        if "imgs" in f:
            return int(f["imgs"].shape[0])
        if "coords" in f:
            return int(f["coords"].shape[0])
        return 0

    def __getitem__(self, idx):
        f = self._require_open()
        cds = f["coords"][idx]
        if "imgs" in f:
            imgs = f["imgs"][idx]
        else:
            imgs = None
        return imgs, cds

    @property
    def metadata(self) -> dict[str, Any]:
        f = self._require_open()
        return dict(f.attrs)

    @property
    def patch_size(self) -> int | None:
        md = self.metadata
        v = md.get("patch_size")
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    @property
    def wsi_path(self) -> str | None:
        md = self.metadata
        v = md.get("wsi_path")
        return str(v) if v is not None else None

    @property
    def num_patches(self) -> int | None:
        md = self.metadata
        v = md.get("num_patches")
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None


def load_patches_h5(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Eagerly load all patches and coordinates into memory with metadata.

    Returns: (imgs, coords, metadata)
    """
    with h5py.File(path, "r") as f:
        if "imgs" not in f:
            raise KeyError(
                "'imgs' dataset not found in H5. This file was saved without image arrays. "
                "Use PatchesH5 for coords/metadata or re-extract images from the source WSI using 'coords_ext'."
            )
        imgs = np.array(f["imgs"], dtype=np.uint8)
        coords = np.array(f["coords"], dtype=np.int32)
        meta = dict(f.attrs)
    return imgs, coords, meta
