from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol

import torch
from PIL import Image

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger("atlas_patch.models.patch.custom")


class CustomEncoderLoader(Protocol):
    """Callable that builds a user-provided encoder for patch features.

    The callable is invoked lazily when AtlasPatch needs to construct the feature extractor.
    Implementers should move the model to the supplied device/dtype and return a
    CustomEncoderComponents bundle describing how to preprocess patches and run
    a forward pass.
    """

    def __call__(self, device: torch.device, dtype: torch.dtype) -> "CustomEncoderComponents": ...


@dataclass
class CustomEncoderComponents:
    """Bundle returned by a CustomEncoderLoader.

    Attributes:
        model: Torch module placed on the desired device and dtype; will be set to eval().
        preprocess: Callable that converts a PIL.Image patch into a model-ready tensor.
        forward_fn: Optional callable applied to each batch; when omitted, model(batch) is used.
    """

    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    forward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None


def register_custom_encoder(
    *,
    registry: PatchFeatureExtractorRegistry,
    name: str,
    embedding_dim: int,
    loader: CustomEncoderLoader,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int = 0,
    non_blocking: bool = False,
) -> None:
    """Register a user-defined encoder with minimal boilerplate.

    Args:
        registry: Target registry to receive the new extractor.
        name: Key used with --feature-extractors / FeatureExtractionConfig.extractors.
        embedding_dim: Dimension of the feature vector returned by forward_fn/model.
        loader: Callable that builds the encoder and returns CustomEncoderComponents.
        device: Device to place the model on.
        dtype: Torch dtype for model and inputs.
        num_workers: DataLoader workers used during patch embedding.
        non_blocking: Whether to perform non-blocking transfers to the device.
    """

    def _builder() -> PatchFeatureExtractor:
        components = loader(device, dtype)
        if not isinstance(components, CustomEncoderComponents):
            raise TypeError(
                f"Custom encoder loader for '{name}' must return CustomEncoderComponents, "
                f"got {type(components)}."
            )
        return PatchFeatureExtractor(
            name=name,
            model=components.model,
            embedding_dim=embedding_dim,
            preprocess=components.preprocess,
            device=device,
            dtype=dtype,
            num_workers=num_workers,
            non_blocking=non_blocking,
            forward_fn=components.forward_fn,
        )

    registry.register(name, _builder)


class CustomRegistryHook(Protocol):
    """Function signature expected from a custom feature plugin module."""

    def __call__(
        self,
        registry: PatchFeatureExtractorRegistry,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int,
    ) -> None: ...


def _import_module(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_feature_extractors_from_module(
    module_path: str | Path,
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int = 0,
) -> None:
    """Import a plugin module and invoke its registration hook.

    The module must expose a callable named ``register_feature_extractors`` with the
    signature:

        def register_feature_extractors(
            registry: PatchFeatureExtractorRegistry,
            device: torch.device,
            dtype: torch.dtype,
            num_workers: int,
        ) -> None:
            \"\"\"Register one or more feature extractors.\"\"\"

    Within that function, call :func:`register_custom_encoder` with your loader logic.
    """
    path = Path(module_path).expanduser().resolve()
    module = _import_module(path)
    hook = getattr(module, "register_feature_extractors", None)
    if not callable(hook):
        raise AttributeError(
            f"Custom encoder module {path} must define a callable "
            "'register_feature_extractors(registry, device, dtype, num_workers)'."
        )

    logger.info("Registering custom feature extractors from %s", path)
    hook(registry=registry, device=device, dtype=dtype, num_workers=num_workers)


__all__ = [
    "CustomEncoderComponents",
    "CustomEncoderLoader",
    "CustomRegistryHook",
    "register_custom_encoder",
    "register_feature_extractors_from_module",
]
