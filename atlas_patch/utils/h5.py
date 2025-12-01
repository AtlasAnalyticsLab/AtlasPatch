import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import h5py
import numpy as np


@dataclass
class H5AppendWriter:
    """Incremental HDF5 writer with a single open/close and efficient chunking."""

    path: str
    chunk_rows: int = 8192

    def __post_init__(self) -> None:
        target_path = self.path
        self._target_path = os.path.abspath(target_path)
        dir_name = os.path.dirname(self._target_path) or "."
        base_name = os.path.basename(self._target_path)
        tmp_name = f".{base_name}.tmp.{uuid.uuid4().hex}"
        self._tmp_path: str | None = os.path.join(dir_name, tmp_name)
        h5_path = self._tmp_path

        self._f = h5py.File(h5_path, "w")
        self._created: set[str] = set()
        self._closed: bool = False

    def _ensure_dataset(
        self, key: str, sample: np.ndarray, attrs: Optional[Mapping[str, Any]]
    ) -> None:
        if key in self._created or key in self._f:
            return
        data_shape = sample.shape
        data_type = sample.dtype
        maxshape = (None,) + data_shape[1:]
        chunk_shape = (max(1, int(self.chunk_rows)),) + data_shape[1:]
        dset = self._f.create_dataset(
            key,
            shape=(0,) + data_shape[1:],
            maxshape=maxshape,
            chunks=chunk_shape,
            dtype=data_type,
        )
        if attrs is not None:
            for a_k, a_v in attrs.items():
                dset.attrs[a_k] = (
                    json.dumps(a_v) if isinstance(a_v, dict) else ("None" if a_v is None else a_v)
                )
        self._created.add(key)

    def append(
        self,
        assets: Mapping[str, np.ndarray],
        attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        for key, val in assets.items():
            self._ensure_dataset(key, val, attributes.get(key) if attributes else None)
            dset = self._f[key]
            cur = int(dset.shape[0])
            n = int(val.shape[0])
            if n == 0:
                continue
            dset.resize(cur + n, axis=0)
            dset[cur : cur + n] = val

    def update_file_attrs(self, file_attrs: Mapping[str, Any]) -> None:
        for a_k, a_v in file_attrs.items():
            self._f.attrs[a_k] = (
                json.dumps(a_v) if isinstance(a_v, dict) else ("None" if a_v is None else a_v)
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._f.close()
        finally:
            if self._tmp_path is not None:
                os.replace(self._tmp_path, self._target_path)
                self._tmp_path = None
            self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._f.close()
        finally:
            if self._tmp_path and os.path.exists(self._tmp_path):
                try:
                    os.remove(self._tmp_path)
                except Exception:
                    pass
            self._closed = True
