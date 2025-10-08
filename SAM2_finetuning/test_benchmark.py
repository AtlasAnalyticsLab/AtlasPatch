from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             jaccard_score, precision_score, recall_score)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_NUMPY_EXTENSIONS = {".npy", ".npz"}
SUPPORTED_TORCH_EXTENSIONS = {".pt", ".pth"}

# Configure evaluation paths and options here.
DATASET_PATH = Path("/mnt/SDA/SegmentationProject/Segmentation_Dataset/Dataset/SegmentationDataset/test")
PREDICTIONS_PATH = Path("/mnt/SDA/SegmentationProject/Segmentation_Dataset/Clam_Masks/masks")
THRESHOLD = 0.5
RESIZE = 256  # Set to 0 to skip resizing.
LIMIT = None  # Optional limit on number of samples to evaluate.
_SUFFIX_TO_STRIP = "_thumbnail"


def load_ground_truth_mask(path: Path, resize_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    """Load a ground-truth mask, optionally resizing to the provided (height, width)."""
    with Image.open(path) as img:
        img = img.convert("L")
        if resize_hw:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.NEAREST)
        mask = np.array(img, dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def _ensure_2d(values: np.ndarray, resize_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if values.ndim == 2:
        return values.astype(np.float32)
    if values.ndim == 1 and resize_hw:
        expected = resize_hw[0] * resize_hw[1]
        if values.size == expected:
            return values.reshape(resize_hw).astype(np.float32)
    raise ValueError("Prediction masks must be convertible to 2D arrays.")


def _resize_array(values: np.ndarray, resize_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    values = _ensure_2d(values, resize_hw)
    if resize_hw is None or values.shape == resize_hw:
        return values.astype(np.float32)
    image = Image.fromarray(values.astype(np.float32))
    image = image.resize((resize_hw[1], resize_hw[0]), Image.NEAREST)
    return np.array(image).astype(np.float32)


def _load_from_numpy(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
        return np.asarray(data)
    data = np.load(path, allow_pickle=False)
    try:
        first_key = data.files[0]
        values = data[first_key]
    finally:
        data.close()
    return np.asarray(values)


def _load_from_torch(path: Path) -> np.ndarray:
    import torch

    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        tensor = None
        for value in data.values():
            if isinstance(value, torch.Tensor):
                tensor = value
                break
        if tensor is None:
            raise ValueError(f"Torch file {path} does not contain a tensor entry.")
    else:
        raise ValueError(f"Unsupported torch serialization format in {path}.")
    array = tensor.detach().cpu().numpy()
    return np.asarray(array)


def load_prediction_mask(path: Path, resize_hw: Optional[Tuple[int, int]], threshold: float) -> np.ndarray:
    """Load a predicted mask, normalise to [0, 1], resize, and binarise."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        with Image.open(path) as img:
            img = img.convert("L")
            if resize_hw:
                img = img.resize((resize_hw[1], resize_hw[0]), Image.NEAREST)
            values = np.array(img, dtype=np.float32)
    elif suffix in SUPPORTED_NUMPY_EXTENSIONS:
        values = _load_from_numpy(path).astype(np.float32)
        values = np.squeeze(values)
        values = _resize_array(values, resize_hw)
    elif suffix in SUPPORTED_TORCH_EXTENSIONS:
        values = _load_from_torch(path).astype(np.float32)
        values = np.squeeze(values)
        values = _resize_array(values, resize_hw)
    else:
        raise ValueError(f"Unsupported mask format: {path.suffix}")

    if values.ndim == 3:
        values = values.mean(axis=0)
    max_value = values.max() if values.size else 0.0
    if max_value > 1.0:
        scale = 255.0 if max_value <= 255.0 else max_value
        if scale > 0:
            values = values / scale
    values = np.nan_to_num(values, nan=0.0)
    binary_mask = (values >= threshold).astype(np.uint8)
    return binary_mask


def gather_prediction_files(predictions_dir: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in predictions_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS \
                and suffix not in SUPPORTED_NUMPY_EXTENSIONS \
                and suffix not in SUPPORTED_TORCH_EXTENSIONS:
            continue
        stem = path.stem
        if stem in files:
            raise ValueError(f"Duplicate prediction mask detected for '{stem}'.")
        files[stem] = path
    if not files:
        raise ValueError(f"No prediction masks found under {predictions_dir}.")
    return files


def _strip_suffix(stem: str) -> str:
    if _SUFFIX_TO_STRIP and stem.endswith(_SUFFIX_TO_STRIP):
        return stem[: -len(_SUFFIX_TO_STRIP)]
    return stem


def collect_mask_pairs(mask_dir: Path, predictions_dir: Path, resize_hw: Optional[Tuple[int, int]],
                       threshold: float, limit: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    prediction_map = gather_prediction_files(predictions_dir)
    gt_files = sorted(
        path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not gt_files:
        raise ValueError(f"No ground-truth masks found in {mask_dir}.")

    matched_gt, matched_pred = [], []
    for gt_path in gt_files:
        stem = gt_path.stem
        base_stem = _strip_suffix(stem)
        prediction_path = prediction_map.get(base_stem) or prediction_map.get(stem)
        if prediction_path is None:
            raise ValueError(f"Missing prediction for mask '{stem}'.")
        gt_mask = load_ground_truth_mask(gt_path, resize_hw)
        pred_mask = load_prediction_mask(prediction_path, resize_hw, threshold)
        matched_gt.append(gt_mask.reshape(-1))
        matched_pred.append(pred_mask.reshape(-1))
        if limit and len(matched_gt) >= limit:
            break

    all_gt = np.concatenate(matched_gt)
    all_pred = np.concatenate(matched_pred)
    return all_gt, all_pred


def evaluate_predictions(dataset_path: Path, predictions_path: Path, threshold: float,
                         resize: Optional[int], limit: Optional[int]) -> Tuple[np.ndarray, Dict[str, float]]:
    resize_hw = None if not resize or resize <= 0 else (resize, resize)
    mask_dir = dataset_path / "masks"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Expected 'masks' subdirectory under {dataset_path}.")
    all_gt, all_pred = collect_mask_pairs(mask_dir, predictions_path, resize_hw, threshold, limit)
    cm = confusion_matrix(all_gt, all_pred, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(all_gt, all_pred),
        "precision": precision_score(all_gt, all_pred, zero_division=0),
        "recall": recall_score(all_gt, all_pred, zero_division=0),
        "f1": f1_score(all_gt, all_pred, zero_division=0),
        "iou": jaccard_score(all_gt, all_pred, zero_division=0, average="binary"),
    }
    return cm, metrics


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions path not found: {PREDICTIONS_PATH}")

    cm, metrics = evaluate_predictions(
        dataset_path=DATASET_PATH,
        predictions_path=PREDICTIONS_PATH,
        threshold=THRESHOLD,
        resize=RESIZE,
        limit=LIMIT,
    )

    print("Confusion Matrix (rows=gt, cols=pred):")
    print(cm)
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
