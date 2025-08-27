import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Callable
from torchvision.datasets import VisionDataset
from PIL import ImageFile

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        x_min, x_max = 0, 0
        y_min, y_max = 0, 0
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = np.array([x_min, y_min, x_max, y_max])

    return bbox



class SegmentationDataset(VisionDataset):
    def __init__(self, data_root: str, image_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None) -> None:
        super().__init__(data_root)
        self.data_root = data_root
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.gt_path = Path(data_root) / 'masks'
        self.images_path = Path(data_root) / 'images'

        all_images = sorted(self.images_path.glob("*.[jp][pn]g"))
        all_masks = sorted(self.gt_path.glob("*.[jp][pn]g"))

        # Match only if both image and mask exist (by stem name)
        image_dict = {img.stem: img for img in all_images}
        mask_dict = {mask.stem: mask for mask in all_masks}

        common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
        self.images_files = [image_dict[k] for k in common_keys]
        self.gt_files = [mask_dict[k] for k in common_keys]

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        image_path = self.images_files[index]
        mask_path = self.gt_files[index]
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = Image.open(image_file).convert("RGB")
            mask = Image.open(mask_file).convert("1")

            # For bounding box prompt
            mask_1024 = mask.resize((1024, 1024))
            mask_array_1024 = np.array(mask_1024)
            bbox_prompt = get_bounding_box(mask_array_1024)

            if self.image_transform:
                image = self.image_transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            return (
                image,
                mask,
                bbox_prompt
            )

# class SegmentationDataset(VisionDataset):
#     """
#     The segmentation dataset handles images with corresponding masks and bounding box as prompts."""
    
#     def __init__(self, data_root: str, image_transform: Optional[Callable] = None,
#                  mask_transform: Optional[Callable] = None) -> None:
#         super().__init__(data_root)
#         self.data_root = data_root
#         self.image_transform = image_transform
#         self.mask_transform = mask_transform
#         self.gt_path = Path(self.data_root) / 'masks'
#         self.images_path = Path(self.data_root) / 'images'
#         self.gt_files = sorted(self.gt_path.glob("*.[jp][pn]g"))
#         self.images_files = sorted(self.images_path.glob("*.[jp][pn]g"))
#         self.images_files = [
#             file
#             for file in self.images_files
#             if os.path.isfile(os.path.join(self.images_path, os.path.basename(file)))
#         ]
#         self.gt_files = [
#             file
#             for file in self.gt_files
#             if os.path.isfile(os.path.join(self.gt_path, os.path.basename(file)))
#         ]

#     def __len__(self):
#         return len(self.gt_files)

#     def __getitem__(self, index):
#         image_path = self.images_files[index]
#         mask_path = self.gt_files[index]
#         with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
#             ImageFile.LOAD_TRUNCATED_IMAGES = True
#             image = Image.open(image_file)
#             mask = Image.open(mask_file)

#             image = image.convert("RGB")
#             mask = mask.convert("1")
            
#             mask_1024 = mask.resize((1024, 1024))
#             mask_array_1024 = np.array(mask_1024) 
#             bbox_prompt = get_bounding_box(mask_array_1024)

#             if self.image_transform:
#                 image = self.image_transform(image)
#             if self.mask_transform:
#                 mask = self.mask_transform(mask)
            
#             image_array = np.array(image)
#             mask_array = np.array(mask) 

#             return (
#                 image_array, 
#                 mask_array, 
#                 bbox_prompt
#                 # torch.tensor(bbox_prompt).float(),
#             )
"""
class SegmentationDataset(VisionDataset):
    
    The segmentation dataset handles images with corresponding masks and bounding box as prompts.
    
    def __init__(self, data_root: str, transforms: Optional[Callable] = None) -> None:
        super().__init__(data_root, transforms)
        self.data_root = data_root
        self.gt_path = Path(self.data_root) / 'masks'
        self.images_path = Path(self.data_root) / 'images'
        self.gt_files = sorted(self.gt_path.glob("*.jpg"))
        self.images_files = sorted(self.images_path.glob("*.jpg"))
        self.images_files = [
            file
            for file in self.images_files
            if os.path.isfile(os.path.join(self.images_path, os.path.basename(file)))
        ]
        self.gt_files = [
            file
            for file in self.gt_files
            if os.path.isfile(os.path.join(self.gt_path, os.path.basename(file)))
        ]

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        image_path = self.images_files[index]
        mask_path = self.gt_files[index]
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            mask = Image.open(mask_file)

            image = image.convert("RGB")
            mask = mask.convert("1")
            
            mask_1024 = mask.resize((1024, 1024))
            mask_array = np.array(mask_1024) 
            bbox_prompt = get_bounding_box(mask_array)

            if self.transforms:
                image = self.transforms(image)
                mask = self.transforms(mask)

            return (
                image, 
                mask, 
                torch.tensor(bbox_prompt).float(),
            )
"""       


