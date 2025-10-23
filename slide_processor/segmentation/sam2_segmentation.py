from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
        self.config_file = Path(config_file)
        self.device = torch.device(device)
        self.mask_threshold = mask_threshold

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        self.model = self._build_sam2()
        self.predictor = self._create_predictor(**kwargs)

    def _build_sam2(self) -> torch.nn.Module:
        """Build and load SAM2 model."""
        try:
            model = build_sam2(config_file=str(self.config_file), ckpt_path=None)

            return model
        except Exception as e:
            raise RuntimeError(f"Failed to build SAM2 model: {e}") from e

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
            # Convert PIL Image to numpy array
            if image.mode != "RGB":
                image = image.convert("RGB")
            return np.asarray(image)
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
            return image
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

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
        original_shape = image_array.shape[:2]

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
    def predict_batch(self):
        pass

    def __repr__(self) -> str:
        return (
            f"SAM2SegmentationModel(device={self.device}, checkpoint={self.checkpoint_path.name})"
        )
