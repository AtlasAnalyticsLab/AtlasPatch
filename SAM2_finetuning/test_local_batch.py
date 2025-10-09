import os
import glob
# import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from scripts.dataset import SegmentationDataset
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, jaccard_score
# from data.dataset import dataReadPip, loadedDataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from datetime import datetime


def compute_cm_metrics(predictor, threshold, test_dataloader, device='cuda', num_images=1000):
    predictor.model.eval()
    all_predictions, all_gt_masks = [], []
    image_count = 0

    with torch.no_grad():
        for image_batch, mask_batch, bbox_batch in tqdm(test_dataloader):
            # if image_count >= num_images:
            #     break

            images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
            masks_gt = mask_batch.to(device)
            bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

            predictor.set_image_batch(images)
            masks, _, _ = predictor.predict_batch(box_batch=bboxes, multimask_output=False)

            # Convert predictions and ground truth to binary masks
            prd_mask_bin = [(m > threshold).astype(float) for m in masks]
            
            # Flatten for metric computation
            # Loop through each predicted mask and corresponding ground truth
            for prd_mask, gt_mask in zip(prd_mask_bin, masks_gt):
                all_predictions.extend(prd_mask.reshape(-1))
                all_gt_masks.extend(gt_mask.cpu().numpy().reshape(-1))


            image_count += 1
            # if image_count >= num_images:
            #     break  # Stop processing after reaching the limit

    # Convert lists to numpy arrays
    all_predictions, all_gt_masks = np.array(all_predictions), np.array(all_gt_masks)

    # Compute metrics
    cm = confusion_matrix(all_gt_masks, all_predictions)
    accuracy = accuracy_score(all_gt_masks, all_predictions)
    precision = precision_score(all_gt_masks, all_predictions, zero_division=0)
    recall = recall_score(all_gt_masks, all_predictions, zero_division=0)
    f1 = f1_score(all_gt_masks, all_predictions, zero_division=0)
    iou = jaccard_score(all_gt_masks, all_predictions, zero_division=0, average='binary')

    return cm, accuracy, precision, recall, f1, iou



def evaluate_all_datasets(predictor, test_dataloader, threshold=0.5, device='cuda', num_images=1000):
    """
    Compute metrics for a single test dataset.

    Args:
        predictor: The SAM2ImagePredictor object.
        test_dataloader: The test dataloader.
        threshold: Threshold for binary classification.
        device: Device to run the model ('cuda' or 'cpu').
        num_images: Maximum number of images to process per dataset.

    Returns:
        Metrics computed for the dataset.
    """
    _, accuracy, precision, recall, f1, iou = compute_cm_metrics(
        predictor, threshold, test_dataloader, device, num_images
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }


def test_with_metrics(dataset_path, save_path, CHECK_POINT, Image_Size=256):
    """
    Run inference on a dataset and compute metrics.

    Args:
        dataset_path (str): Path to the dataset directory containing images and masks.
        save_path (str): Directory where results will be saved.
        CHECK_POINT (str): Path to the pre-trained model weights.

    Returns:
        Metrics computed for the dataset.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"size_{Image_Size}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Fetch image and mask paths
    image_dir = os.path.join(dataset_path, 'images')
    mask_dir = os.path.join(dataset_path, 'masks')
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.[jp][pn]g')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.[jp][pn]g')))

    if len(image_paths) == 0 or len(image_paths) != len(mask_paths):
        raise ValueError("Dataset structure invalid. Ensure images and masks are present and match.")

    # Define transformations
    images_transform = v2.Compose([
        v2.Resize(size=(Image_Size, Image_Size)),
        v2.ToTensor(),
    ])
    masks_transform = v2.Compose([
        v2.Resize(size=(Image_Size, Image_Size)),
        v2.ToTensor(),
    ])

    # Load dataset
    test_dataset = SegmentationDataset(data_root=dataset_path, image_transform=images_transform, mask_transform=masks_transform)
    cpu_count = os.cpu_count() or 1
    worker_count = min(8, cpu_count)
    loader_kwargs = {
        'batch_size': 2,
        'shuffle': False,
        'pin_memory': DEVICE.type == "cuda",
    }
    if worker_count > 0:
        loader_kwargs.update(
            num_workers=worker_count,
            persistent_workers=True,
            prefetch_factor=4,
        )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, **loader_kwargs)

    # Build model and load weights
    sam2_checkpoint_path = "/home/a_alagha/SegmentationProject/SlideProcessor/SAM2_finetuning/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"



    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint_path))

    predictor.model.to(DEVICE).eval()

    checkpoint = torch.load(CHECK_POINT, map_location=DEVICE)
    predictor.model.load_state_dict(checkpoint['model'], strict=False)

    # Compute metrics
    avg_metrics = evaluate_all_datasets(predictor, test_dataloader, threshold=0.5, device=DEVICE)

    print("Average Metrics Across Datasets:")
    for metric, value in avg_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    metrics_filepath = os.path.join(run_dir, "metrics.txt")
    with open(metrics_filepath, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Dataset: {dataset_path}\n")
        metrics_file.write(f"Checkpoint: {CHECK_POINT}\n")
        metrics_file.write(f"Image Size: {Image_Size}\n")
        metrics_file.write("Average Metrics Across Datasets:\n")
        for metric, value in avg_metrics.items():
            metrics_file.write(f"{metric.capitalize()}: {value:.4f}\n")

    print(f"Metrics saved to {metrics_filepath}")

    return avg_metrics

if __name__ == '__main__':
    Image_Size = 1024
    test_with_metrics(
        dataset_path='/data1/SegmentationDataset/test',
        save_path='/home/a_alagha/SegmentationProject/SlideProcessor/SAM2_finetuning/results',
        CHECK_POINT='/home/a_alagha/SegmentationProject/SlideProcessor/SAM2_finetuning/saved_models/experiment_sam2_layernorm5_20251009-015749/trained_sam2_layernorm5.pth',
        Image_Size=Image_Size
    )
