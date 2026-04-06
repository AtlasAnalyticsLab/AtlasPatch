from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

from atlas_patch.core.config import ExtractionConfig, OutputConfig, SlideEncodingConfig
from atlas_patch.core.models import Slide, SlideEmbeddingResult
from atlas_patch.core.paths import slide_append_lock_path, slide_feature_dataset_key
from atlas_patch.models.slide import SlideEncoderRegistry, build_default_registry
from atlas_patch.utils.feature_h5 import (
    PatchArtifactSummary,
    SlideEmbeddingSummary,
    append_slide_embedding,
    read_patch_artifact_summary,
    read_slide_embedding_summary,
)


class SlideEmbeddingService:
    """Append slide-level embeddings into canonical AtlasPatch patch H5 files."""

    def __init__(
        self,
        extraction_cfg: ExtractionConfig,
        output_cfg: OutputConfig,
        slide_cfg: SlideEncodingConfig,
        *,
        registry: SlideEncoderRegistry | None = None,
    ) -> None:
        self.extraction_cfg = extraction_cfg.validated()
        self.output_cfg = output_cfg.validated()
        self.slide_cfg = slide_cfg.validated()
        self.registry = registry or build_default_registry(device=self.slide_cfg.device)

    def _acquire_lock(self, slide: Slide) -> tuple[int | None, Path]:
        lock_path = slide_append_lock_path(slide, self.output_cfg, self.extraction_cfg)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"pid={os.getpid()},time={int(time.time())},slide={slide.path},phase=slide-embedding"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, payload.encode())
            os.fsync(fd)
            return fd, lock_path
        except FileExistsError:
            return None, lock_path
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to create slide lock {lock_path}: {exc}") from exc

    @staticmethod
    def _release_lock(fd: int | None, lock_path: Path) -> None:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    @staticmethod
    def _cleanup_encoder(encoder) -> None:
        cleanup = getattr(encoder, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass

    @staticmethod
    def _build_result(
        *,
        slide: Slide,
        h5_path: Path,
        encoder_name: str,
        embedding_dim: int,
        source_patch_encoder: str,
        num_patches: int,
        patch_size: int,
        patch_size_level0: int,
        target_magnification: int,
        reused: bool,
    ) -> SlideEmbeddingResult:
        return SlideEmbeddingResult(
            slide=slide,
            h5_path=h5_path,
            encoder_name=encoder_name,
            dataset_key=slide_feature_dataset_key(encoder_name),
            embedding_dim=int(embedding_dim),
            num_patches=int(num_patches),
            source_patch_encoder=source_patch_encoder,
            metadata={
                "patch_size": int(patch_size),
                "patch_size_level0": int(patch_size_level0),
                "target_magnification": int(target_magnification),
                "reused": bool(reused),
            },
        )

    @staticmethod
    def _validate_reusable_embedding(
        *,
        encoder,
        h5_path: Path,
        summary: SlideEmbeddingSummary,
        patch_summary: PatchArtifactSummary,
    ) -> None:
        if summary.embedding_dim != encoder.embedding_dim:
            raise ValueError(
                f"{h5_path} has slide embedding dim {summary.embedding_dim} for '{summary.dataset_key}', "
                f"but '{encoder.name}' expects {encoder.embedding_dim}."
            )
        if summary.source_patch_encoder != encoder.required_patch_encoder:
            raise ValueError(
                f"{h5_path} was encoded from '{summary.source_patch_encoder}', "
                f"but '{encoder.name}' requires '{encoder.required_patch_encoder}'."
            )
        if summary.patch_size != encoder.required_patch_size:
            raise ValueError(
                f"{h5_path} records patch_size={summary.patch_size} for '{summary.dataset_key}', "
                f"but '{encoder.name}' requires {encoder.required_patch_size}."
            )
        if summary.num_patches != patch_summary.num_patches:
            raise ValueError(
                f"{h5_path} records {summary.num_patches} patches for '{summary.dataset_key}', "
                f"but the current patch artifact has {patch_summary.num_patches}."
            )
        if summary.patch_size_level0 != patch_summary.patch_size_level0:
            raise ValueError(
                f"{h5_path} records patch_size_level0={summary.patch_size_level0} for "
                f"'{summary.dataset_key}', but the current patch artifact has "
                f"{patch_summary.patch_size_level0}."
            )
        if summary.target_magnification != patch_summary.target_magnification:
            raise ValueError(
                f"{h5_path} records target_mag={summary.target_magnification} for "
                f"'{summary.dataset_key}', but the current patch artifact has "
                f"{patch_summary.target_magnification}."
            )

    def embed_all(
        self,
        artifacts: Iterable[tuple[Slide, Path]],
    ) -> tuple[list[SlideEmbeddingResult], list[tuple[Slide, Exception | str]]]:
        items = list(artifacts)
        results: list[SlideEmbeddingResult] = []
        failures: list[tuple[Slide, Exception | str]] = []
        if not items:
            return results, failures

        for encoder_name in self.slide_cfg.encoders:
            try:
                encoder = self.registry.create(encoder_name)
            except Exception as exc:  # noqa: BLE001
                for slide, _ in items:
                    failures.append((slide, RuntimeError(f"{encoder_name}: {exc}")))
                continue

            try:
                for slide, h5_path in items:
                    try:
                        patch_summary = read_patch_artifact_summary(h5_path)
                    except Exception as exc:  # noqa: BLE001
                        failures.append((slide, RuntimeError(f"{encoder_name}: {exc}")))
                        continue

                    overwrite_existing = not self.slide_cfg.skip_existing
                    if self.slide_cfg.skip_existing:
                        try:
                            existing = read_slide_embedding_summary(h5_path, encoder_name)
                        except Exception:
                            existing = None
                            overwrite_existing = True

                        if existing is not None:
                            try:
                                self._validate_reusable_embedding(
                                    encoder=encoder,
                                    h5_path=h5_path,
                                    summary=existing,
                                    patch_summary=patch_summary,
                                )
                            except Exception:
                                overwrite_existing = True
                            else:
                                results.append(
                                    self._build_result(
                                        slide=slide,
                                        h5_path=h5_path,
                                        encoder_name=encoder_name,
                                        embedding_dim=existing.embedding_dim,
                                        source_patch_encoder=existing.source_patch_encoder,
                                        num_patches=existing.num_patches,
                                        patch_size=existing.patch_size,
                                        patch_size_level0=existing.patch_size_level0,
                                        target_magnification=existing.target_magnification,
                                        reused=True,
                                    )
                                )
                                continue

                    lock_fd, lock_path = self._acquire_lock(slide)
                    if lock_fd is None:
                        failures.append(
                            (
                                slide,
                                RuntimeError(
                                    f"{encoder_name}: {h5_path} is locked by another process."
                                ),
                            )
                        )
                        continue

                    try:
                        if patch_summary.patch_size != encoder.required_patch_size:
                            raise ValueError(
                                f"{h5_path} has patch_size={patch_summary.patch_size}, "
                                f"but '{encoder_name}' requires {encoder.required_patch_size}."
                            )

                        embedding = encoder.encode_slide(h5_path)
                        append_slide_embedding(
                            h5_path,
                            encoder_name,
                            embedding,
                            attrs={
                                "source_patch_encoder": encoder.required_patch_encoder,
                                "num_patches": patch_summary.num_patches,
                                "patch_size": patch_summary.patch_size,
                                "patch_size_level0": patch_summary.patch_size_level0,
                                "target_magnification": patch_summary.target_magnification,
                            },
                            overwrite=overwrite_existing,
                        )
                        results.append(
                            self._build_result(
                                slide=slide,
                                h5_path=h5_path,
                                encoder_name=encoder_name,
                                embedding_dim=int(embedding.shape[0]),
                                source_patch_encoder=encoder.required_patch_encoder,
                                num_patches=patch_summary.num_patches,
                                patch_size=patch_summary.patch_size,
                                patch_size_level0=patch_summary.patch_size_level0,
                                target_magnification=patch_summary.target_magnification,
                                reused=False,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append((slide, RuntimeError(f"{encoder_name}: {exc}")))
                    finally:
                        self._release_lock(lock_fd, lock_path)
            finally:
                self._cleanup_encoder(encoder)

        return results, failures
