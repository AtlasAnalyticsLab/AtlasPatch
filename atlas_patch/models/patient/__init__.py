from atlas_patch.models.patient.base import PatientEncoder, PatientEncoderSpec
from atlas_patch.models.patient.registry import PatientEncoderRegistry

__all__ = [
    "PatientEncoder",
    "PatientEncoderRegistry",
    "PatientEncoderSpec",
    "build_default_registry",
]


def build_default_registry() -> PatientEncoderRegistry:
    """Return an empty patient-encoder registry for phased population."""
    return PatientEncoderRegistry()
