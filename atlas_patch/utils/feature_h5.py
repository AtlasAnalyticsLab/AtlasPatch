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
REQUIRED_SLIDE_EMBEDDING_ATTRS = (
    "source_patch_encoder",
    "num_patches",
    "patch_size",
    "patch_size_level0",
    "target_magnification",
)
REQUIRED_PATIENT_EMBEDDING_ATTRS = (
    "encoder_name",
    "num_slides",
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


@dataclass(frozen=True)
class PatchArtifactSummary:
    h5_path: Path
    num_patches: int
    patch_size_level0: int
    patch_size: int
    target_magnification: int
    wsi_path: str | None = None


@dataclass(frozen=True)
class SlideEmbeddingSummary:
    dataset_key: str
    embedding_dim: int
    source_patch_encoder: str
    num_patches: int
    patch_size: int
    patch_size_level0: int
    target_magnification: int


@dataclass(frozen=True)
class PatientEmbeddingSummary:
    embedding_dim: int
    encoder_name: str
    num_slides: int
    source_patch_encoder: str | None = None
    source_manifest: str | None = None
    source_slide_h5_paths: tuple[str, ...] = ()
    source_slide_patch_summaries: tuple[PatchArtifactSummary, ...] = ()


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


def coerce_1d_embedding(embedding: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(embedding)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {vector.shape}")
    return vector


def read_patch_artifact_summary(h5_path: str | Path) -> PatchArtifactSummary:
    path = Path(h5_path)
    with h5py.File(path, "r") as handle:
        coords_ds = handle.get("coords")
        if not isinstance(coords_ds, h5py.Dataset):
            raise ValueError(f"{path} is missing required dataset 'coords'.")

        num_patches_attr = handle.attrs.get("num_patches")
        if num_patches_attr is not None:
            num_patches = int(num_patches_attr)
        else:
            num_patches = int(coords_ds.shape[0])

        attrs = _read_required_int_attrs(path, handle.attrs, coords_ds.attrs)
        wsi_path = handle.attrs.get("wsi_path")

    return PatchArtifactSummary(
        h5_path=path,
        num_patches=num_patches,
        patch_size_level0=attrs["patch_size_level0"],
        patch_size=attrs["patch_size"],
        target_magnification=attrs["target_magnification"],
        wsi_path=str(wsi_path) if wsi_path is not None else None,
    )


def read_slide_embedding_summary(
    h5_path: str | Path,
    encoder_name: str,
) -> SlideEmbeddingSummary | None:
    path = Path(h5_path)
    dataset_key = slide_feature_dataset_key(encoder_name)

    with h5py.File(path, "r") as handle:
        dataset = handle.get(dataset_key)
        if dataset is None:
            return None
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{path} contains '{dataset_key}', but it is not an HDF5 dataset.")
        if dataset.ndim != 1:
            raise ValueError(f"{path} has invalid slide embedding shape {dataset.shape} for '{dataset_key}'.")

        attrs = _read_required_int_attrs(path, dataset.attrs)
        missing = [key for key in REQUIRED_SLIDE_EMBEDDING_ATTRS if key not in dataset.attrs]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{path} is missing required slide embedding metadata for '{dataset_key}': {joined}")

        return SlideEmbeddingSummary(
            dataset_key=dataset_key,
            embedding_dim=int(dataset.shape[0]),
            source_patch_encoder=str(dataset.attrs["source_patch_encoder"]).strip().lower(),
            num_patches=int(dataset.attrs["num_patches"]),
            patch_size=attrs["patch_size"],
            patch_size_level0=attrs["patch_size_level0"],
            target_magnification=attrs["target_magnification"],
        )


def _decode_json_list_attr(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must decode to a JSON list.")
    return tuple(str(item) for item in parsed)


def _decode_patient_source_slide_summaries(
    value: Any,
    *,
    label: str,
) -> tuple[PatchArtifactSummary, ...]:
    if value is None:
        return ()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must decode to a JSON list.")

    summaries: list[PatchArtifactSummary] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be JSON objects.")
        summaries.append(
            PatchArtifactSummary(
                h5_path=Path(str(item["h5_path"])).expanduser().resolve(),
                num_patches=int(item["num_patches"]),
                patch_size_level0=int(item["patch_size_level0"]),
                patch_size=int(item["patch_size"]),
                target_magnification=int(item["target_magnification"]),
                wsi_path=(str(item["wsi_path"]) if item.get("wsi_path") is not None else None),
            )
        )
    return tuple(summaries)


def read_patient_embedding_summary(h5_path: str | Path) -> PatientEmbeddingSummary:
    path = Path(h5_path)
    with h5py.File(path, "r") as handle:
        dataset = handle.get("features")
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{path} is missing required dataset 'features'.")
        if dataset.ndim != 1:
            raise ValueError(f"{path} has invalid patient embedding shape {dataset.shape}.")

        missing = [key for key in REQUIRED_PATIENT_EMBEDDING_ATTRS if key not in handle.attrs]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{path} is missing required patient embedding metadata: {joined}")

        return PatientEmbeddingSummary(
            embedding_dim=int(dataset.shape[0]),
            encoder_name=str(handle.attrs["encoder_name"]).strip().lower(),
            num_slides=int(handle.attrs["num_slides"]),
            source_patch_encoder=(
                str(handle.attrs["source_patch_encoder"]).strip().lower()
                if "source_patch_encoder" in handle.attrs
                else None
            ),
            source_manifest=(
                str(handle.attrs["source_manifest"])
                if "source_manifest" in handle.attrs
                else None
            ),
            source_slide_h5_paths=tuple(
                str(Path(item).expanduser().resolve())
                for item in _decode_json_list_attr(
                    handle.attrs.get("source_slide_h5_paths"),
                    label=f"{path} source_slide_h5_paths",
                )
            ),
            source_slide_patch_summaries=_decode_patient_source_slide_summaries(
                handle.attrs.get("source_slide_patch_summaries"),
                label=f"{path} source_slide_patch_summaries",
            ),
        )


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
    vector = coerce_1d_embedding(embedding, label=dataset_key)

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
    vector = coerce_1d_embedding(embedding, label=str(path))
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
