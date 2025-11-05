import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger("slide_processor.segmentation")


class SAM2SegmentationModel:
    """Wrapper for SAM 2.0 image segmentation model."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_file: Union[str, Path],
        device: str = "cuda",
        mask_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Initialize SAM2 model.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint.
            config_file: Path to SAM2 config file.
            device: 'cuda' or 'cpu'. Default: 'cuda'.
            mask_threshold: Mask binarization threshold. Default: 0.0.
            **kwargs: Additional predictor arguments.
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Require a YAML filesystem path for the SAM2 config
        cfg_arg = config_file if isinstance(config_file, str) else str(config_file)
        self.config_path = Path(cfg_arg)
        if self.config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                f"--config must be a YAML file path (*.yaml|*.yml), got: {self.config_path}"
            )

        dev_str = str(device).lower()
        if dev_str == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available; falling back to CPU. "
                "Use --device cpu to silence this warning."
            )
            dev_str = "cpu"
        self.device = torch.device(dev_str)
        self.mask_threshold = mask_threshold
        self.input_size: int = 1024

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.model = self._build_sam2()
        self.predictor = self._create_predictor(**kwargs)

    def _build_sam2(self) -> torch.nn.Module:
        """Build and load SAM2 model from YAML file."""
        try:
            conf = OmegaConf.load(str(self.config_path))
            # Configs in SAM2 typically have a top-level 'model' node
            model_cfg = conf.get("model", conf)
            model = instantiate(model_cfg)
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to build SAM2 model from YAML: {self.config_path}: {e}"
            ) from e

    def _create_predictor(self, **kwargs) -> SAM2ImagePredictor:
        """Create SAM2ImagePredictor."""
        try:
            predictor = SAM2ImagePredictor(
                self.model,
                mask_threshold=self.mask_threshold,
                **kwargs,
            )
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            predictor.model.load_state_dict(checkpoint["model"], strict=True)
            predictor.model.to(self.device).eval()
            return predictor

        except Exception as e:
            raise RuntimeError(f"Failed to build SAM2ImagePredictor model: {e}") from e

    def _normalize_input(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Normalize input image to numpy array.

        Args:
            image: Input image as numpy array, PIL Image, or tensor.

        Returns:
            Numpy array in RGB format with shape (H, W, 3).
        """
        if isinstance(image, Image.Image):
            # Convert PIL Image to writable numpy array
            if image.mode != "RGB":
                image = image.convert("RGB")
            # np.asarray(image) can be read-only; use np.array(copy=True)
            arr = np.array(image, copy=True)
            return arr
        elif isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            image = image.cpu().detach().numpy()
            # Handle different tensor shapes
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume values in [0, 1] range, convert to [0, 255]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            # Ensure writable and contiguous memory to avoid torchvision warnings
            if not image.flags.writeable:
                image = image.copy()
            if not image.flags.c_contiguous:
                image = np.ascontiguousarray(image)
            return image
        elif isinstance(image, np.ndarray):
            # Ensure proper format
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume values in [0, 1] range, convert to [0, 255]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
            # Ensure writable and contiguous memory
            if not image.flags.writeable:
                image = image.copy()
            if not image.flags.c_contiguous:
                image = np.ascontiguousarray(image)
            return image
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

    def _resize_input_for_sam(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Resize input image to fixed size (1024x1024) for SAM inference and
        return resized image along with the original (H, W) shape for later resizing back.
        """
        original_shape = image.shape[:2]
        if original_shape[0] == self.input_size and original_shape[1] == self.input_size:
            return image, original_shape

        pil = Image.fromarray(image)
        resized = pil.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        return np.array(resized, copy=True), original_shape

    def _resize_mask(self, mask: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Resize mask to target shape using nearest neighbor interpolation.

        Args:
            mask: Input mask with shape (H, W) and values in [0, 1].
            target_shape: Target shape as (height, width).

        Returns:
            Resized mask with shape target_shape and values in [0, 1].
        """
        # Convert to uint8 for PIL resizing
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode="L")

        # Resize using nearest neighbor (PIL uses width, height)
        resized_pil = mask_pil.resize(
            (target_shape[1], target_shape[0]),
            resample=Image.Resampling.NEAREST,
        )

        # Convert back to [0, 1] range
        return np.asarray(resized_pil, dtype=np.float32) / 255.0

    @torch.inference_mode()
    def predict_image(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        resize_to_input: bool = True,
    ) -> np.ndarray:
        """
        Predict segmentation mask for a single image.

        Args:
            image: Input image as numpy array, PIL Image, or tensor.
                   If numpy/tensor: should be (H, W, 3) in RGB format.
                   Values can be in [0, 255] (uint8) or [0, 1] (float).
            resize_to_input: If True, resize output mask to match input image dimensions.
                             Default: True.

        Returns:
            Binary segmentation mask as numpy array with shape (H, W) and values in [0, 1].
            If resize_to_input is True, mask shape matches input image.

        Raises:
            TypeError: If image type is not supported.
        """
        # Normalize input to numpy array
        image_array = self._normalize_input(image)
        image_array, original_shape = self._resize_input_for_sam(image_array)

        # Set the image in the predictor
        self.predictor.set_image(image_array)

        # Get image dimensions for bbox
        height, width = image_array.shape[:2]
        bbox = np.array([0, 0, width, height], dtype=np.float32)

        # Run inference with full image bbox
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
            return_logits=False,
        )

        mask = masks[0]

        # Resize mask to original input dimensions if requested
        if resize_to_input:
            mask = self._resize_mask(mask, original_shape)

        return mask

    @torch.inference_mode()
    def predict_batch(
        self,
        images: list[Union[np.ndarray, Image.Image, torch.Tensor]],
        *,
        resize_to_input: bool = True,
    ) -> list[np.ndarray]:
        """
        Predict segmentation masks for a batch of images.

        Args:
            images: List of images (numpy arrays, PIL Images, or tensors), each
                    expected to be RGB. Arrays should be HxWx3 with values in
                    [0, 255] (uint8) or [0, 1] (float).
            resize_to_input: If True, ensure output masks match input image size.

        Returns:
            List of binary masks (each HxW, float32 in [0, 1]) in the same
            order as input images.
        """
        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("images must be a non-empty list")

        # Normalize all inputs to numpy arrays
        np_images_raw: list[np.ndarray] = [self._normalize_input(img) for img in images]
        np_images: list[np.ndarray] = []
        orig_shapes: list[tuple[int, int]] = []
        for im in np_images_raw:
            resized, orig_shape = self._resize_input_for_sam(im)
            np_images.append(resized)
            orig_shapes.append(orig_shape)

        # Set batch into predictor
        self.predictor.set_image_batch(np_images)

        # Build full-image box prompts for each image
        box_batch = []
        for im in np_images:
            h, w = im.shape[:2]
            box = np.array([0, 0, float(w), float(h)], dtype=np.float32)
            box_batch.append(box)

        # Run batched prediction
        masks_list, _, _ = self.predictor.predict_batch(
            box_batch=box_batch,
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )

        out_masks: list[np.ndarray] = []
        for i, masks_np in enumerate(masks_list):
            mask = masks_np[0]  # take the single output mask
            if resize_to_input and mask.shape[:2] != orig_shapes[i]:
                mask = self._resize_mask(mask, orig_shapes[i])
            out_masks.append(mask.astype(np.float32))

        return out_masks

    def __repr__(self) -> str:
        return (
            f"SAM2SegmentationModel(device={self.device}, checkpoint={self.checkpoint_path.name})"
        )
