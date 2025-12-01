from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("slide_processor.models.patch")


class FeatureExtractor(ABC):
    """Base interface for patch-level feature extractors."""

    name: str
    embedding_dim: int

    @abstractmethod
    def extract_batch(
        self, patches: Sequence[np.ndarray], *, batch_size: int | None = None
    ) -> np.ndarray: ...

    @abstractmethod
    def cleanup(self) -> None:
        """Optional clean-up hook."""
        raise NotImplementedError


class PatchDataset(Dataset[torch.Tensor]):
    """Minimal dataset that applies a transform to cached patches (HWC or PIL)."""

    def __init__(self, patches: Sequence[np.ndarray | Image.Image], transform) -> None:
        self._patches = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self._patches[idx]
        pil_img = patch if isinstance(patch, Image.Image) else Image.fromarray(patch)
        return self.transform(pil_img)


class PatchFeatureExtractor(FeatureExtractor):
    """Shared DataLoader + forward loop for torch-based feature extractors."""

    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        embedding_dim: int,
        preprocess,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_workers: int = 0,
        non_blocking: bool = False,
        pin_memory: bool | None = None,
        forward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.name = name
        self.model = model.to(device=device, dtype=dtype).eval()
        self.embedding_dim = int(embedding_dim)
        self.preprocess = preprocess
        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))
        self.non_blocking = bool(non_blocking)
        self.pin_memory = pin_memory if pin_memory is not None else self.device.type == "cuda"
        self._forward_fn = forward_fn

    @torch.inference_mode()
    def extract_batch(
        self, patches: Sequence[np.ndarray], *, batch_size: int | None = None
    ) -> np.ndarray:
        if not patches:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        bs = min(len(patches), batch_size or len(patches))
        dataset = PatchDataset(patches, self.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        outputs: list[torch.Tensor] = []
        for batch in loader:
            batch = batch.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=self.non_blocking,
            )
            out = self._forward_fn(batch) if self._forward_fn else self.model(batch)
            if out.ndim > 2:
                out = torch.flatten(out, start_dim=1)
            outputs.append(out.detach())

        feats_t = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        feats = feats_t.cpu().to(dtype=torch.float32).numpy()
        return feats

    def cleanup(self) -> None:
        try:
            if hasattr(self.model, "cpu"):
                self.model.cpu()
        except Exception:
            pass


def _set_attr_path(obj: torch.nn.Module, attr_path: str, value: torch.nn.Module) -> None:
    """Set a nested attribute on a module given a dotted path."""
    parts = attr_path.split(".")
    target = obj
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], value)


def _resolve_torchvision_weights(weights_enum, *, model_name: str | None = None):
    """Pick a sensible pretrained weight entry for a torchvision weights enum.

    Prefer IMAGENET1K_V1 when available; fall back to the enum's DEFAULT or the
    first defined uppercase entry so models without V1 weights (e.g., ViT-H-14)
    still load.
    """
    preferred = getattr(weights_enum, "IMAGENET1K_V1", None)
    if preferred is not None:
        return preferred

    default = getattr(weights_enum, "DEFAULT", None)
    if default is not None:
        return default

    for attr in dir(weights_enum):
        if attr.isupper() and not attr.startswith("_"):
            return getattr(weights_enum, attr)

    raise AttributeError(f"{weights_enum.__name__} does not expose any pretrained weights")


def build_torchvision_extractor(
    *,
    name: str,
    model_ctor,
    weights_enum,
    head_attr: str,
    embedding_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int,
    non_blocking: bool = False,
) -> PatchFeatureExtractor:
    """Factory for torchvision backbones with a standard head replacement."""
    weights = _resolve_torchvision_weights(weights_enum, model_name=name)
    logger.debug(
        "Using torchvision weights %s for feature extractor '%s' (%s)",
        weights,
        name,
        weights_enum.__name__,
    )
    model = model_ctor(weights=weights)
    _set_attr_path(model, head_attr, torch.nn.Identity())
    preprocess = weights.transforms()
    return PatchFeatureExtractor(
        name=name,
        model=model,
        embedding_dim=embedding_dim,
        preprocess=preprocess,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
        non_blocking=non_blocking,
    )
