from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Iterable, Mapping

from atlas_patch.core.config import OutputConfig, PatientEncodingConfig
from atlas_patch.core.models import PatientCase, PatientEmbeddingResult, Slide
from atlas_patch.core.paths import patient_embedding_path, patient_lock_path, validate_output_stem
from atlas_patch.models.patient import PatientEncoderRegistry, build_default_registry
from atlas_patch.services._embedding_runtime import (
    acquire_exclusive_lock,
    cleanup_encoder,
    release_lock,
)
from atlas_patch.utils.feature_h5 import (
    PatchArtifactSummary,
    read_patch_artifact_summary,
    read_patient_embedding_summary,
    write_patient_embedding_h5,
)


def _parse_manifest_mpp(raw: str | None, *, manifest_path: Path, row_number: int) -> float | None:
    value = str(raw or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{manifest_path}:{row_number} has invalid mpp value {value!r}; expected a number."
        ) from exc


def load_patient_cases(manifest_path: str | Path) -> tuple[list[PatientCase], list[Slide]]:
    path = Path(manifest_path)
    manifest_dir = path.resolve().parent
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"case_id", "slide_path"}
        missing = [name for name in required if name not in fieldnames]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required column(s): {joined}")

        slides_by_path: dict[Path, Slide] = {}
        slide_stems: dict[str, Path] = {}
        cases_by_id: dict[str, list[Slide]] = {}
        seen_within_case: dict[str, set[Path]] = {}

        for row_number, row in enumerate(reader, start=2):
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"{path}:{row_number} has empty case_id.")
            validate_output_stem(case_id, field_name="case_id")

            slide_raw = str(row.get("slide_path") or "").strip()
            if not slide_raw:
                raise ValueError(f"{path}:{row_number} has empty slide_path.")
            slide_path = Path(slide_raw).expanduser()
            if not slide_path.is_absolute():
                slide_path = manifest_dir / slide_path
            if not slide_path.exists():
                raise FileNotFoundError(f"{path}:{row_number} references missing slide {slide_path}")
            if not slide_path.is_file():
                raise ValueError(f"{path}:{row_number} references non-file slide {slide_path}")
            resolved_path = slide_path.resolve()
            mpp = _parse_manifest_mpp(row.get("mpp"), manifest_path=path, row_number=row_number)

            existing = slides_by_path.get(resolved_path)
            if existing is None:
                existing_stem_path = slide_stems.get(slide_path.stem)
                if existing_stem_path is not None and existing_stem_path != resolved_path:
                    raise ValueError(
                        f"{path} includes multiple slides named '{slide_path.stem}', which would "
                        "collide in patches/<stem>.h5. Rename the slide files or split the run."
                    )
                existing = Slide(path=slide_path, mpp=mpp)
                slides_by_path[resolved_path] = existing
                slide_stems[slide_path.stem] = resolved_path
            elif existing.mpp != mpp:
                raise ValueError(
                    f"{path}:{row_number} assigns conflicting mpp overrides to {slide_path}."
                )

            case_slides = cases_by_id.setdefault(case_id, [])
            seen_paths = seen_within_case.setdefault(case_id, set())
            if resolved_path in seen_paths:
                raise ValueError(f"{path}:{row_number} repeats slide {slide_path} within case {case_id}.")
            seen_paths.add(resolved_path)
            case_slides.append(existing)

    cases = [PatientCase(case_id=case_id, slides=tuple(slides)) for case_id, slides in cases_by_id.items()]
    slides = list(slides_by_path.values())
    return cases, slides


class PatientEmbeddingService:
    """Write patient-level embeddings into separate case-scoped H5 files."""

    def __init__(
        self,
        output_cfg: OutputConfig,
        patient_cfg: PatientEncodingConfig,
        *,
        registry: PatientEncoderRegistry | None = None,
    ) -> None:
        self.output_cfg = output_cfg.validated()
        self.patient_cfg = patient_cfg.validated()
        self.registry = registry or build_default_registry(device=self.patient_cfg.device)

    @staticmethod
    def _build_source_slide_summaries(slide_h5_paths: list[Path]) -> tuple[PatchArtifactSummary, ...]:
        return tuple(read_patch_artifact_summary(path.resolve()) for path in slide_h5_paths)

    @staticmethod
    def _validate_reusable_embedding(
        *,
        output_path: Path,
        encoder,
        case: PatientCase,
        slide_h5_paths: list[Path],
    ) -> PatientEmbeddingResult | None:
        try:
            current_slide_summaries = PatientEmbeddingService._build_source_slide_summaries(
                slide_h5_paths
            )
            summary = read_patient_embedding_summary(output_path)
        except Exception:
            return None
        if summary.encoder_name != encoder.name:
            return None
        if summary.embedding_dim != encoder.embedding_dim:
            return None
        if summary.num_slides != case.num_slides:
            return None
        if summary.source_patch_encoder != encoder.required_patch_encoder:
            return None
        current_slide_h5s = tuple(str(path.resolve()) for path in slide_h5_paths)
        if summary.source_slide_h5_paths != current_slide_h5s:
            return None
        if summary.source_slide_patch_summaries != current_slide_summaries:
            return None
        return PatientEmbeddingResult(
            case_id=case.case_id,
            h5_path=output_path,
            encoder_name=encoder.name,
            embedding_dim=summary.embedding_dim,
            num_slides=summary.num_slides,
            source_patch_encoder=summary.source_patch_encoder,
            metadata={"reused": True},
        )

    def embed_all(
        self,
        cases: Iterable[PatientCase],
        *,
        slide_h5_by_path: Mapping[Path, Path],
    ) -> tuple[list[PatientEmbeddingResult], list[tuple[PatientCase, Exception | str]]]:
        ordered_cases = list(cases)
        results: list[PatientEmbeddingResult] = []
        failures: list[tuple[PatientCase, Exception | str]] = []
        if not ordered_cases:
            return results, failures

        for encoder_name in self.patient_cfg.encoders:
            try:
                encoder = self.registry.create(encoder_name)
            except Exception as exc:  # noqa: BLE001
                for case in ordered_cases:
                    failures.append((case, RuntimeError(f"{encoder_name}: {exc}")))
                continue

            try:
                for case in ordered_cases:
                    try:
                        slide_h5_paths = [
                            slide_h5_by_path[slide.path.resolve()].resolve()
                            for slide in case.slides
                        ]
                    except KeyError as exc:
                        failures.append(
                            (
                                case,
                                RuntimeError(
                                    f"{encoder_name}: missing canonical patch artifact for {exc.args[0]}"
                                ),
                            )
                        )
                        continue

                    output_path = patient_embedding_path(self.output_cfg, encoder_name, case.case_id)
                    if self.patient_cfg.skip_existing and output_path.exists():
                        reused = self._validate_reusable_embedding(
                            output_path=output_path,
                            encoder=encoder,
                            case=case,
                            slide_h5_paths=slide_h5_paths,
                        )
                        if reused is not None:
                            results.append(reused)
                            continue
                        overwrite = True
                    else:
                        overwrite = not self.patient_cfg.skip_existing

                    lock_fd, lock_path = acquire_exclusive_lock(
                        patient_lock_path(self.output_cfg, encoder_name, case.case_id),
                        payload=(
                            f"pid={os.getpid()},time={int(time.time())},"
                            f"case={case.case_id},phase=patient-embedding"
                        ),
                        label="patient",
                    )
                    if lock_fd is None:
                        failures.append(
                            (
                                case,
                                RuntimeError(
                                    f"{encoder_name}: {output_path} is locked by another process."
                                ),
                            )
                        )
                        continue

                    try:
                        embedding = encoder.encode_case(
                            slide_h5_paths,
                            metadata={
                                "case_id": case.case_id,
                                "manifest_path": self.patient_cfg.manifest_path.resolve(),
                            },
                        )
                        source_slide_summaries = self._build_source_slide_summaries(slide_h5_paths)
                        write_patient_embedding_h5(
                            output_path,
                            embedding,
                            attrs={
                                "encoder_name": encoder.name,
                                "num_slides": case.num_slides,
                                "source_patch_encoder": encoder.required_patch_encoder,
                                "source_manifest": str(self.patient_cfg.manifest_path.resolve()),
                                "source_slide_h5_paths": [
                                    str(path.resolve()) for path in slide_h5_paths
                                ],
                                "source_slide_patch_summaries": [
                                    {
                                        "h5_path": str(summary.h5_path.resolve()),
                                        "num_patches": summary.num_patches,
                                        "patch_size_level0": summary.patch_size_level0,
                                        "patch_size": summary.patch_size,
                                        "target_magnification": summary.target_magnification,
                                        "wsi_path": summary.wsi_path,
                                    }
                                    for summary in source_slide_summaries
                                ],
                            },
                            overwrite=overwrite,
                        )
                        results.append(
                            PatientEmbeddingResult(
                                case_id=case.case_id,
                                h5_path=output_path,
                                encoder_name=encoder.name,
                                embedding_dim=int(embedding.shape[0]),
                                num_slides=case.num_slides,
                                source_patch_encoder=encoder.required_patch_encoder,
                                metadata={"reused": False},
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append((case, RuntimeError(f"{encoder_name}: {exc}")))
                    finally:
                        release_lock(lock_fd, lock_path)
            finally:
                cleanup_encoder(encoder)

        return results, failures
