from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import VisionDataset


def get_bounding_box(height: int, width: int) -> torch.Tensor:
    """Generate a full-image bounding box.

    Args:
        height: Image height.
        width: Image width.

    Returns:
        Bounding box tensor of shape (1, 4) with format [x_min, y_min, x_max, y_max].
    """
    return torch.tensor([[0, 0, width, height]], dtype=torch.float32)


class InferenceSegmentationDataset(VisionDataset):
    def __init__(
        self,
        images: Union[str, Path, List[Union[Image.Image, np.ndarray]]],
        image_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the inference dataset.

        Args:
            images: Either a directory path containing image files, or a list of images
                   (PIL Image or numpy array).
            image_transform: Optional transform to apply to images.

        Raises:
            ValueError: If images is empty or invalid type.
            NotADirectoryError: If images path is not a valid directory.
        """
        super().__init__(root=".")
        self.image_transform = image_transform
        self.image_files = []
        self.image_data = []
        self.data_source = None  # "files" or "memory"

        if isinstance(images, (str, Path)):
            # Load from directory
            self.data_source = "files"
            self.root = Path(images)
            if not self.root.is_dir():
                raise NotADirectoryError(f"images path must be a directory: {self.root}")

            self.image_files = sorted(
                list(self.root.glob("*.[jp][pn]g")) + list(self.root.glob("*.[JP][PN]G"))
            )

            if not self.image_files:
                raise ValueError(f"No images found in {self.root}")

        elif isinstance(images, list):
            # Load from memory
            if not images:
                raise ValueError("images list cannot be empty")

            self.data_source = "memory"
            self.image_data = images

            # Validate all items are PIL Image or numpy array
            for i, img in enumerate(self.image_data):
                if not isinstance(img, (Image.Image, np.ndarray)):
                    raise ValueError(
                        f"Image at index {i} must be PIL Image or numpy array, got {type(img)}"
                    )
        else:
            raise ValueError("images must be a directory path or list of PIL Images/numpy arrays")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        if self.data_source == "files":
            return len(self.image_files)
        else:
            return len(self.image_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single image sample.

        Args:
            index: Index of the sample.

        Returns:
            Tuple of (image_tensor, bbox_prompt) where:
            - image_tensor: Transformed image tensor of shape (C, H, W)
            - bbox_prompt: Full-image bounding box tensor of shape (1, 4)
        """
        if self.data_source == "files":
            image = self._load_image_from_file(self.image_files[index])
        else:
            image = self._convert_to_pil(self.image_data[index])

        if self.image_transform:
            image = self.image_transform(image)

        tensor_image: torch.Tensor = torch.as_tensor(image)
        h, w = tensor_image.shape[-2:]
        bbox_prompt = get_bounding_box(h, w)

        return tensor_image, bbox_prompt

    def _load_image_from_file(self, image_path: Path) -> Image.Image:
        """Load a single image from file.

        Args:
            image_path: Path to the image file.

        Returns:
            PIL Image in RGB format.
        """
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _convert_to_pil(self, img_data: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Convert image data to PIL Image.

        Args:
            img_data: PIL Image or numpy array.

        Returns:
            PIL Image in RGB format.
        """
        if isinstance(img_data, Image.Image):
            return img_data.convert("RGB")

        # Handle numpy array
        if img_data.dtype == np.uint8:
            # Assuming values in [0, 255]
            pil_img = Image.fromarray(img_data, mode="RGB")
        elif img_data.dtype in (np.float32, np.float64):
            # Assuming values in [0, 1]
            if img_data.max() <= 1.0:
                img_uint8 = (img_data * 255).astype(np.uint8)
            else:
                # Clip to [0, 255] if outside expected range
                img_uint8 = np.clip(img_data, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="RGB")
        else:
            raise ValueError(
                f"Unsupported numpy array dtype: {img_data.dtype}. "
                "Expected uint8 or float32/float64."
            )

        return pil_img.convert("RGB")
