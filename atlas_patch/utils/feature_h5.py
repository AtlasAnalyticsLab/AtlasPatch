from __future__ import annotations

import json
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from atlas_patch.core.paths import patch_feature_dataset_key, slide_feature_dataset_key

REQUIRED_PATCH_FILE_ATTRS = (
    "patch_size_level0",
    "patch_size",
    "target_magnification",
)


@dataclass(frozen=True)
class PatchFeatureData:
    h5_path: Path
    feature_name: str
    dataset_key: str
    features: np.ndarray
    coords: np.ndarray
    patch_size_level0: int
    patch_size: int
    target_magnification: int

    @property
    def num_patches(self) -> int:
        return int(self.features.shape[0])


def _encode_attr_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if value is None:
        return "None"
    return value


def _write_attrs(target: Any, attrs: Mapping[str, Any] | None) -> None:
    if not attrs:
        return
    for key, value in attrs.items():
        target.attrs[key] = _encode_attr_value(value)


def _read_required_int_attrs(h5_path: Path, *sources: Mapping[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in REQUIRED_PATCH_FILE_ATTRS:
        for source in sources:
            if key not in source:
                continue
            values[key] = int(source[key])
            break
        else:
            raise ValueError(f"{h5_path} is missing required metadata '{key}'.")
    return values


def _coerce_embedding(embedding: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(embedding)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {vector.shape}")
    return vector


def load_patch_feature_data(
    h5_path: str | Path,
    feature_name: str,
    *,
    validate_shapes: bool = True,
) -> PatchFeatureData:
    """Load one AtlasPatch feature matrix and its aligned coordinate metadata."""
    path = Path(h5_path)
    dataset_key = patch_feature_dataset_key(feature_name)
    feature_key = dataset_key.split("/", 1)[1]

    with h5py.File(path, "r") as handle:
        coords_ds = handle.get("coords")
        if not isinstance(coords_ds, h5py.Dataset):
            raise ValueError(f"{path} is missing required dataset 'coords'.")

        features_ds = handle.get(dataset_key)
        if not isinstance(features_ds, h5py.Dataset):
            raise ValueError(f"{path} is missing required dataset '{dataset_key}'.")

        attrs = _read_required_int_attrs(path, handle.attrs, coords_ds.attrs)
        coords = np.asarray(coords_ds[()])
        features = np.asarray(features_ds[()])

    if validate_shapes:
        if features.ndim != 2:
            raise ValueError(f"{path} has invalid feature shape {features.shape}; expected 2-D.")
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"{path} has invalid coords shape {coords.shape}; expected at least 2 columns."
            )
        if coords.shape[0] != features.shape[0]:
            raise ValueError(
                f"{path} has mismatched features/coords lengths: "
                f"features {features.shape[0]} vs coords {coords.shape[0]}"
            )

    return PatchFeatureData(
        h5_path=path,
        feature_name=feature_key,
        dataset_key=dataset_key,
        features=np.asarray(features, dtype=np.float32),
        coords=np.asarray(coords, dtype=np.int64),
        patch_size_level0=attrs["patch_size_level0"],
        patch_size=attrs["patch_size"],
        target_magnification=attrs["target_magnification"],
    )


def append_slide_embedding(
    h5_path: str | Path,
    encoder_name: str,
    embedding: np.ndarray,
    *,
    attrs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    """Append or replace a slide-level embedding inside an existing AtlasPatch H5."""
    path = Path(h5_path)
    dataset_key = slide_feature_dataset_key(encoder_name)
    dataset_name = dataset_key.split("/", 1)[1]
    vector = _coerce_embedding(embedding, label=dataset_key)

    with h5py.File(path, "r+") as handle:
        group = handle.require_group("slide_features")
        if dataset_name in group:
            if not overwrite:
                raise ValueError(f"{path} already contains '{dataset_key}'.")
            del group[dataset_name]
        dset = group.create_dataset(dataset_name, data=vector, dtype=vector.dtype)
        merged_attrs = {"encoder_name": dataset_name}
        if attrs:
            merged_attrs.update(attrs)
        _write_attrs(dset, merged_attrs)
        handle.flush()

    return dataset_key


def write_patient_embedding_h5(
    output_path: str | Path,
    embedding: np.ndarray,
    *,
    attrs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a compact patient-level embedding H5 atomically."""
    path = Path(output_path)
    vector = _coerce_embedding(embedding, label=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Patient embedding already exists: {path}")

    tmp_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with h5py.File(tmp_path, "w") as handle:
            handle.create_dataset("features", data=vector, dtype=vector.dtype)
            _write_attrs(handle, attrs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()

    return path
