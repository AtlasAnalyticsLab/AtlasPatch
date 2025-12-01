from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from atlas_patch.models.patch.base import PatchFeatureExtractor
from atlas_patch.models.patch.registry import PatchFeatureExtractorRegistry

from timm.layers import to_2tuple

logger = logging.getLogger(__name__)

_CHECKPOINT_ID = "1_vgRF1QXa8sPCOpJ1S9BihwZhXQMOVJc"
_CHECKPOINT_NAME = "CHIEF_CTransPath.pth"
_EMB_DIM = 768


def _build_preprocess():
    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _download_checkpoint() -> Path:
    """Download CHIEF CTransPath weights to a torch hub cache directory."""
    cache_dir = Path(torch.hub.get_dir()) / "atlas_patch" / "chief"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / _CHECKPOINT_NAME

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        return checkpoint_path
    if checkpoint_path.exists() and checkpoint_path.stat().st_size == 0:
        checkpoint_path.unlink()

    try:
        import gdown
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "gdown is required to download CHIEF CTransPath weights automatically."
        ) from e

    url = f"https://drive.google.com/uc?id={_CHECKPOINT_ID}"
    logger.info("Downloading CHIEF CTransPath checkpoint to %s", checkpoint_path)
    downloaded = gdown.cached_download(url=url, path=str(checkpoint_path), quiet=False, fuzzy=True)
    if downloaded is None:
        raise RuntimeError("gdown did not return a downloaded checkpoint path.")
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        raise RuntimeError("Failed to download CHIEF CTransPath checkpoint from Google Drive.")

    return checkpoint_path


class ConvStem(nn.Module):
    """Convolutional stem used by the CHIEF CTransPath model."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool | None = None,
        strict_img_size: bool = True,
        output_fmt: str | None = None,
        **_: dict,
    ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if flatten is None:
            flatten = output_fmt != "NHWC"
        self.flatten = flatten
        self.output_fmt = output_fmt or ("NHWC" if not self.flatten else "NCHW")
        self.strict_img_size = strict_img_size

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False)
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        if self.strict_img_size:
            assert H == self.img_size[0] and W == self.img_size[1], (
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        x = self.proj(x)
        if self.output_fmt == "NHWC":
            x = x.permute(0, 2, 3, 1)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def _build_chief_ctranspath_model() -> nn.Module:
    import timm

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        pretrained=False,
    )
    model.head = nn.Identity()
    checkpoint_path = _download_checkpoint()
    td = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in td:
        raise RuntimeError(
            f"Unexpected checkpoint format at {checkpoint_path}; expected a 'model' key."
        )
    state_dict = {}
    for key, value in td["model"].items():
        if "relative_position_index" in key or "attn_mask" in key:
            continue
        if key.startswith("layers.0.downsample."):
            key = key.replace("layers.0.", "layers.1.")
        elif key.startswith("layers.1.downsample."):
            key = key.replace("layers.1.", "layers.2.")
        elif key.startswith("layers.2.downsample."):
            key = key.replace("layers.2.", "layers.3.")
        state_dict[key] = value

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            "CHIEF CTransPath checkpoint loaded with mismatches. Missing: %s; Unexpected: %s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    return model


class ChiefCTransPathEncoder(PatchFeatureExtractor):
    """CHIEF patch-level encoder (CTransPath) following the upstream usage recipe."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))

        model = _build_chief_ctranspath_model()
        model = model.to(device=self.device, dtype=self.dtype).eval()
        # timm>=1.x returns NHWC; pool over spatial dims to match upstream [B, 768] feature shape
        def _forward(inp: torch.Tensor) -> torch.Tensor:
            out = model(inp)
            if out.ndim == 4:
                out = out.mean(dim=(1, 2))
            elif out.ndim == 3:
                out = out.mean(dim=1)
            return out
        preprocess = _build_preprocess()

        super().__init__(
            name="chief-ctranspath",
            model=model,
            embedding_dim=_EMB_DIM,
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=_forward,
        )


def register_chief_ctranspath_model(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    registry.register(
        "chief-ctranspath",
        lambda: ChiefCTransPathEncoder(device=device, dtype=dtype, num_workers=num_workers),
    )
