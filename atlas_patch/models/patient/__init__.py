from __future__ import annotations

from importlib import import_module

import torch

from atlas_patch.models.patient.base import PatientEncoder, PatientEncoderSpec
from atlas_patch.models.patient.registry import PatientEncoderRegistry

__all__ = [
    "PatientEncoder",
    "PatientEncoderRegistry",
    "PatientEncoderSpec",
    "build_default_registry",
]

_REGISTRARS: tuple[tuple[str, str], ...] = (
    ("atlas_patch.models.patient.moozy", "register_moozy_patient_encoder"),
)


def build_default_registry(
    *,
    device: str | torch.device = "cuda",
) -> PatientEncoderRegistry:
    """Factory that registers the built-in patient encoders."""
    registry = PatientEncoderRegistry()
    for module_name, registrar_name in _REGISTRARS:
        module = import_module(module_name)
        registrar = getattr(module, registrar_name)
        registrar(registry, device=device)
    return registry
