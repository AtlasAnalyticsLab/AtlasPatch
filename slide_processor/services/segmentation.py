from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from slide_processor.core.config import SegmentationConfig
from slide_processor.core.models import Mask
from slide_processor.core.wsi.iwsi import IWSI
from slide_processor.services.interfaces import SegmentationService

logger = logging.getLogger("slide_processor.segmentation_service")


class _SAM2Predictor:
    """Lightweight wrapper around SAM2ImagePredictor with resizing helpers."""
    DEFAULT_MODEL_REPO = "AtlasAnalyticsLab/Atlas-Patch"
    DEFAULT_MODEL_FILENAME = "model.pth"

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg

        requested_dev = str(cfg.device).lower()
        dev_str = requested_dev
        if dev_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            dev_str = "cpu"
        self.device = torch.device(dev_str)
        self.input_size = 1024
        logger.info("SAM2 predictor device: %s (requested=%s)", self.device, requested_dev)

        self.checkpoint_path = self._resolve_checkpoint_path()
        self.predictor = self._build_predictor()

    def _resolve_checkpoint_path(self) -> Path:
        """Choose checkpoint: prefer explicit path, else download from Hugging Face."""
        if self.cfg.checkpoint_path is not None:
            return self.cfg.checkpoint_path

        repo_id = self.DEFAULT_MODEL_REPO
        filename = self.DEFAULT_MODEL_FILENAME
        try:
            logger.info("Downloading SAM2 checkpoint from Hugging Face: %s/%s", repo_id, filename)
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch checkpoint from {repo_id}: {exc}") from exc
        return Path(downloaded)

    def _build_predictor(self) -> SAM2ImagePredictor:
        try:
            conf = OmegaConf.load(str(self.cfg.config_path))
            model_cfg: Any = conf.get("model", conf)
            model = instantiate(model_cfg)
            predictor = SAM2ImagePredictor(model, mask_threshold=self.cfg.mask_threshold)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            predictor.model.load_state_dict(checkpoint["model"], strict=True)
            predictor.model.to(self.device).eval()
            return predictor
        except Exception as exc:
            raise RuntimeError(f"Failed to build SAM2 predictor: {exc}") from exc

    def _normalize_input(self, image: np.ndarray | Image.Image | torch.Tensor) -> np.ndarray:
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            return np.array(image, copy=True)
        if isinstance(image, torch.Tensor):
            arr = image.cpu().detach().numpy()
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if not arr.flags.writeable:
                arr = arr.copy()
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            return arr
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if not arr.flags.writeable:
                arr = arr.copy()
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            return arr
        raise TypeError(f"Unsupported input type: {type(image)}")

    def _resize_input_for_sam(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        original_shape: tuple[int, int] = (int(image.shape[0]), int(image.shape[1]))
        if original_shape[0] == self.input_size and original_shape[1] == self.input_size:
            return image, original_shape
        pil = Image.fromarray(image)
        resized = pil.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        return np.array(resized, copy=True), original_shape

    def _resize_mask(self, mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode="L")
        resized_pil = mask_pil.resize(
            (target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST
        )
        return np.asarray(resized_pil, dtype=np.float32) / 255.0

    @torch.inference_mode()
    def predict_image(
        self, image: np.ndarray | Image.Image | torch.Tensor, *, resize_to_input: bool = True
    ) -> np.ndarray:
        arr = self._normalize_input(image)
        arr_resized, orig_shape = self._resize_input_for_sam(arr)

        self.predictor.set_image(arr_resized)
        h, w = arr_resized.shape[:2]
        bbox = np.array([0, 0, w, h], dtype=np.float32)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
            return_logits=False,
        )
        mask = masks[0]
        if resize_to_input and mask.shape[:2] != orig_shape:
            mask = self._resize_mask(mask, orig_shape)
        return mask

    @torch.inference_mode()
    def predict_batch(
        self,
        images: Sequence[np.ndarray | Image.Image | torch.Tensor],
        *,
        resize_to_input: bool = True,
    ) -> list[np.ndarray]:
        if not images:
            raise ValueError("images must be a non-empty sequence")

        arrs_raw = [self._normalize_input(img) for img in images]
        arrs: list[np.ndarray] = []
        orig_shapes: list[tuple[int, int]] = []
        for im in arrs_raw:
            resized, orig_shape = self._resize_input_for_sam(im)
            arrs.append(resized)
            orig_shapes.append(orig_shape)

        self.predictor.set_image_batch(arrs)

        box_batch = []
        for im in arrs:
            h, w = im.shape[:2]
            box_batch.append(np.array([0, 0, float(w), float(h)], dtype=np.float32))

        masks_list, _, _ = self.predictor.predict_batch(
            box_batch=box_batch,
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )

        out_masks: list[np.ndarray] = []
        for i, masks_np in enumerate(masks_list):
            mask = masks_np[0]
            if resize_to_input and mask.shape[:2] != orig_shapes[i]:
                mask = self._resize_mask(mask, orig_shapes[i])
            out_masks.append(mask.astype(np.float32))
        return out_masks

    def close(self) -> None:
        """Release GPU memory held by the SAM2 model."""
        try:
            self.predictor.model.cpu()
        except Exception:
            pass
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class SAM2SegmentationService(SegmentationService):
    """Segmentation service that wraps SAM2 and standardizes mask output."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg.validated()
        self.predictor = _SAM2Predictor(self.cfg)

    def _prepare_thumbnail(self, wsi: IWSI):
        thumb = wsi.get_thumbnail_at_power(power=self.cfg.thumbnail_power, interpolation="optimise")
        if self.cfg.thumbnail_max:
            thumb.thumbnail((self.cfg.thumbnail_max, self.cfg.thumbnail_max))
        return thumb

    def segment_thumbnail(self, wsi: IWSI) -> Mask:
        thumb = self._prepare_thumbnail(wsi)
        mask_arr = self.predictor.predict_image(thumb, resize_to_input=True)
        return Mask(
            data=mask_arr.astype(np.float32),
            source_shape=(int(mask_arr.shape[0]), int(mask_arr.shape[1])),
        )

    def segment_batch(self, wsis: Sequence[IWSI]) -> list[Mask]:
        """Default batch segmenter; parallel thumbnail creation with threads."""
        max_workers = min(8, len(wsis), os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="thumb") as ex:
            thumbs = list(ex.map(self._prepare_thumbnail, wsis))

        masks = self.predictor.predict_batch(thumbs, resize_to_input=True)
        return [
            Mask(
                data=m.astype(np.float32),
                source_shape=(int(m.shape[0]), int(m.shape[1])),
            )
            for m in masks
        ]

    def close(self) -> None:
        """Free underlying SAM2 resources (notably GPU memory)."""
        try:
            self.predictor.close()
        except Exception:
            pass
