from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from atlas_patch.models.common import resolve_model_device
from atlas_patch.models.moozy_api import load_moozy_embedding, run_moozy_public_api
from atlas_patch.models.patient.base import PatientEncoder, PatientEncoderSpec
from atlas_patch.models.patient.registry import PatientEncoderRegistry
from atlas_patch.utils.feature_h5 import coerce_1d_embedding


class MOOZYPatientEncoder(PatientEncoder):
    spec = PatientEncoderSpec(
        name="moozy",
        embedding_dim=768,
        patch_encoder_name="lunit_vit_small_patch8_dino",
        patch_size=224,
    )

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = resolve_model_device(device)

    def encode_case(
        self,
        slide_h5_paths: Sequence[Path],
        metadata: Mapping[str, object] | None = None,
    ) -> np.ndarray:
        if not slide_h5_paths:
            raise ValueError("MOOZY patient encoding requires at least one slide H5.")

        with tempfile.TemporaryDirectory(prefix="atlaspatch_moozy_case_") as tmp_dir:
            output_path = Path(tmp_dir) / "case_embedding.h5"
            run_moozy_public_api(
                [str(path) for path in slide_h5_paths],
                str(output_path),
                device=self.device,
                mixed_precision=False,
            )
            embedding = load_moozy_embedding(output_path)

        vector = coerce_1d_embedding(embedding, label="MOOZY patient embedding")
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"MOOZY patient embedding dim mismatch: expected {self.embedding_dim}, got {vector.shape[0]}"
            )
        return np.asarray(vector, dtype=np.float32)


def register_moozy_patient_encoder(
    registry: PatientEncoderRegistry,
    *,
    device: str | torch.device = "cuda",
) -> None:
    registry.register(
        MOOZYPatientEncoder.spec,
        lambda: MOOZYPatientEncoder(device=device),
    )
