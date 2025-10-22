"""Datasets for segmentation tasks."""

from .inference_dataset import InferenceSegmentationDataset, get_bounding_box

__all__ = ["InferenceSegmentationDataset", "get_bounding_box"]
