from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import torch

from slide_processor.core.config import ExtractionConfig, FeatureExtractionConfig, OutputConfig
from slide_processor.core.models import ExtractionResult
from slide_processor.core.paths import patch_lock_path
from slide_processor.core.wsi.iwsi import IWSI
from slide_processor.models.patch import build_default_registry
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry
from slide_processor.services.interfaces import FeatureEmbeddingService
from slide_processor.services.storage import H5PatchWriter
from slide_processor.utils import get_existing_features

logger = logging.getLogger("slide_processor.feature_embedding_service")


class PatchFeatureEmbeddingService(FeatureEmbeddingService):
    """Embeds patches into feature matrices and appends them to existing H5 files."""

    def __init__(
        self,
        extraction_cfg: ExtractionConfig,
        output_cfg: OutputConfig,
        feature_cfg: FeatureExtractionConfig,
        registry: PatchFeatureExtractorRegistry | None = None,
    ) -> None:
        self.cfg = extraction_cfg.validated()
        self.output_cfg = output_cfg.validated()
        self.feature_cfg = feature_cfg.validated()

        dev_str = self.feature_cfg.device
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Feature extraction requested on CUDA but unavailable; using CPU instead."
            )
            dev_str = "cpu"
        self.device = torch.device(dev_str)
        self.dtype = self._resolve_dtype()

        self.registry = registry or build_default_registry(
            device=self.device, num_workers=self.feature_cfg.num_workers, dtype=self.dtype
        )
        self.extractor_names: list[str] = [name.lower() for name in self.feature_cfg.extractors]
        self._feature_cache: dict[Path, tuple[int | None, set[str]]] = {}

    # --- helpers -------------------------------------------------------------------
    def _iter_patch_entries_coords(
        self, wsi: IWSI, result: ExtractionResult
    ) -> Iterable[tuple[int, int, int, int, int, np.ndarray | None]]:
        with h5py.File(result.h5_path, "r") as f:
            coords = f["coords"]
            for i in range(coords.shape[0]):
                x, y, rw, rh, lv = coords[i].tolist()
                patch_any = wsi.extract(
                    (int(x), int(y)), lv=int(lv), wh=(int(rw), int(rh)), mode="array"
                )
                if not isinstance(patch_any, np.ndarray):
                    continue
                patch = patch_any
                if patch.shape[0] != self.cfg.patch_size or patch.shape[1] != self.cfg.patch_size:
                    patch = cv2.resize(patch, (self.cfg.patch_size, self.cfg.patch_size))
                yield (int(x), int(y), int(rw), int(rh), int(lv), patch)

    def _acquire_feature_lock(self, slide) -> tuple[int | None, Path]:
        lock_path = patch_lock_path(slide, self.output_cfg, self.cfg)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"pid={os.getpid()},time={int(time.time())},slide={slide.path},phase=features"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, payload.encode())
            os.fsync(fd)
            return fd, lock_path
        except FileExistsError:
            return None, lock_path
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to create feature lock {lock_path}: {e}") from e

    @staticmethod
    def _release_feature_lock(fd: int | None, path: Path | None) -> None:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _feature_key(self, h5_path: Path) -> Path:
        return Path(h5_path).resolve()

    def _existing_features(self, h5_path: Path, expected_total: int | None = None) -> set[str]:
        """Get (and cache) complete feature datasets present in the file."""
        key = self._feature_key(h5_path)
        cached = self._feature_cache.get(key)
        if cached is not None:
            cached_total, cached_feats = cached
            if expected_total is None or cached_total == expected_total:
                return set(cached_feats)

        feats = get_existing_features(key, expected_total=expected_total)
        self._feature_cache[key] = (expected_total, set(feats))
        return set(feats)

    def _add_feature_to_cache(
        self, h5_path: Path, feature_name: str, *, expected_total: int
    ) -> None:
        key = self._feature_key(h5_path)
        existing_total, current = self._feature_cache.get(key, (expected_total, set()))
        # If total mismatches, prefer the provided expected_total since it reflects current file.
        self._feature_cache[key] = (expected_total, set(current) | {feature_name.lower()})

    def _feature_present(self, result: ExtractionResult, feature_name: str) -> bool:
        return feature_name.lower() in self._existing_features(
            result.h5_path, expected_total=result.num_patches
        )

    def _update_metadata_feature_sets(self, result: ExtractionResult) -> ExtractionResult:
        existing = sorted(
            self._existing_features(result.h5_path, expected_total=result.num_patches)
        )
        if existing:
            result.metadata["feature_sets"] = existing
        return result

    # --- public API ----------------------------------------------------------------
    def embed_features(self, result: ExtractionResult, *, wsi: IWSI) -> ExtractionResult:
        """Embed features for a single slide using the first configured extractor."""
        if not self.extractor_names:
            return result
        extractor = self.registry.create(self.extractor_names[0])
        try:
            return self._embed_with_extractor(result=result, wsi=wsi, extractor=extractor)
        finally:
            try:
                extractor.cleanup()
            except Exception:
                pass

    def _embed_with_extractor(
        self, *, result: ExtractionResult, wsi: IWSI, extractor
    ) -> ExtractionResult:
        """Embed features for a single slide with a provided extractor."""

        def entries_fn() -> Iterable[tuple[int, int, int, int, int, np.ndarray | None]]:
            return self._iter_patch_entries_coords(wsi, result)

        feature_names: list[str] = []
        lock_fd: int | None = None
        lock_path: Path | None = None
        lock_held = False
        try:
            lock_fd, lock_path = self._acquire_feature_lock(result.slide)
            if lock_fd is None:
                logger.info(
                    "Skipping feature embedding for %s (locked by another process).",
                    result.slide.path.name,
                )
                return self._update_metadata_feature_sets(result)

            lock_held = True

            if self._feature_present(result, extractor.name):
                logger.info(
                    "Skipping feature embedding for %s (feature '%s' already exists).",
                    result.slide.path.name,
                    extractor.name,
                )
                return self._update_metadata_feature_sets(result)

            feature_names.append(extractor.name)
            feature_attrs = {"name": extractor.name, "embedding_dim": extractor.embedding_dim}
            writer = H5PatchWriter(
                chunk_rows=self.cfg.write_batch,
                patch_size=self.cfg.patch_size,
                patch_size_level0=result.patch_size_level0 or 0,
                level0_mag=int(wsi.mag) if wsi.mag is not None else 0,
                target_mag=self.cfg.target_magnification,
                level0_wh=wsi.get_size(lv=0),
                overlap=max(
                    0, int(self.cfg.patch_size) - int(self.cfg.step_size or self.cfg.patch_size)
                ),
                slide_stem=result.slide.stem,
                wsi_path=str(wsi.path),
            )
            writer.append_features(
                output_path=result.h5_path,
                entries=entries_fn(),
                feature_name=extractor.name,
                feature_fn=lambda patches, ex=extractor: ex.extract_batch(
                    patches, batch_size=self.feature_cfg.batch_size
                ),
                feature_attrs=feature_attrs,
                feature_batch=self.feature_cfg.batch_size,
                expected_total=result.num_patches,
            )
            self._add_feature_to_cache(
                result.h5_path, extractor.name, expected_total=result.num_patches
            )
        finally:
            if lock_held:
                self._release_feature_lock(lock_fd, lock_path)

        existing_sets = result.metadata.get("feature_sets", [])
        if isinstance(existing_sets, list):
            combined = list(dict.fromkeys([*existing_sets, *feature_names]))
        else:
            combined = feature_names
        result.metadata["feature_sets"] = combined
        return self._update_metadata_feature_sets(result)

    def embed_all(
        self,
        results: list[ExtractionResult],
        *,
        wsi_loader,
        progress=None,
    ) -> list[tuple]:
        """Embed features per extractor across all slides, loading one model at a time."""
        failures: list[tuple] = []

        # Track which extractors are already present so we don't redo work (and so progress
        # accounts for previously completed features).
        pending: dict[Path, set[str]] = {}
        completed_units = 0
        for res in results:
            existing = self._existing_features(res.h5_path, expected_total=res.num_patches)
            missing = [name for name in self.extractor_names if name not in existing]
            if not missing:
                self._update_metadata_feature_sets(res)
            else:
                pending[res.h5_path] = set(missing)
            completed_units += len(self.extractor_names) - len(missing)

        if progress and completed_units:
            progress.update(completed_units)

        for name in self.extractor_names:
            try:
                extractor = self.registry.create(name)
            except Exception as e:  # noqa: BLE001
                for res in results:
                    missing_for_slide = pending.get(res.h5_path)
                    if missing_for_slide and name in missing_for_slide:
                        failures.append((res.slide, e))
                        if progress:
                            progress.update(1)
                continue

            try:
                for res in results:
                    missing_for_slide = pending.get(res.h5_path)
                    if not missing_for_slide or name not in missing_for_slide:
                        continue

                    wsi = None
                    try:
                        if not self._feature_present(res, extractor.name):
                            wsi = wsi_loader.open(res.slide)
                            self._embed_with_extractor(result=res, wsi=wsi, extractor=extractor)
                        self._update_metadata_feature_sets(res)
                    except Exception as e:  # noqa: BLE001
                        failures.append((res.slide, e))
                    finally:
                        if wsi is not None:
                            try:
                                wsi.cleanup()
                            except Exception:
                                pass
                    if progress:
                        progress.update(1)
            finally:
                try:
                    extractor.cleanup()
                except Exception:
                    pass
        return failures

    def _resolve_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = mapping.get(self.feature_cfg.precision, torch.float32)
        if self.device.type == "cpu" and dtype == torch.float16:
            logger.warning("float16 on CPU is unsupported in many ops; falling back to float32.")
            dtype = torch.float32
        return dtype
