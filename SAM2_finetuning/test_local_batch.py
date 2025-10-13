import os
import glob
# import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from scripts.dataset import SegmentationDataset
from torchvision.transforms import v2
import torch.nn.functional as F
from PIL import Image
# from data.dataset import dataReadPip, loadedDataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from datetime import datetime


def _metrics_from_counts(tp, fp, fn, tn):
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        'confusion_matrix': np.array([[tn, fp], [fn, tp]], dtype=np.int64),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }


def compute_cm_metrics(predictor, threshold, test_dataloader, device='cuda', num_images=None):
    predictor.model.eval()
    apply_limit = num_images is not None
    processed = 0

    tp_resized = torch.zeros((), dtype=torch.long, device=device)
    fp_resized = torch.zeros((), dtype=torch.long, device=device)
    fn_resized = torch.zeros((), dtype=torch.long, device=device)
    tn_resized = torch.zeros((), dtype=torch.long, device=device)

    tp_orig = torch.zeros((), dtype=torch.long, device=device)
    fp_orig = torch.zeros((), dtype=torch.long, device=device)
    fn_orig = torch.zeros((), dtype=torch.long, device=device)
    tn_orig = torch.zeros((), dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            if apply_limit and processed >= num_images:
                break

            image_batch, mask_batch, bbox_batch, mask_paths = batch

            images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
            masks_gt = mask_batch.to(device)
            bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

            predictor.set_image_batch(images)
            masks, _, _ = predictor.predict_batch(box_batch=bboxes, multimask_output=False)

            for prd_mask, gt_mask, mask_path in zip(masks, masks_gt, mask_paths):
                if apply_limit and processed >= num_images:
                    break

                prd_tensor = torch.from_numpy(prd_mask).to(device=device)
                if prd_tensor.ndim > 2:
                    prd_tensor = prd_tensor.squeeze(0)

                gt_tensor = gt_mask.to(device=device)
                if gt_tensor.ndim > 2:
                    gt_tensor = gt_tensor.squeeze(0)

                pred_bin_resized = prd_tensor > threshold
                gt_bin_resized = gt_tensor > 0.5

                tp_resized += torch.count_nonzero(pred_bin_resized & gt_bin_resized)
                fp_resized += torch.count_nonzero(pred_bin_resized & ~gt_bin_resized)
                fn_resized += torch.count_nonzero(~pred_bin_resized & gt_bin_resized)
                tn_resized += torch.count_nonzero(~pred_bin_resized & ~gt_bin_resized)

                with Image.open(mask_path) as original_mask_img:
                    original_mask = np.array(original_mask_img.convert("L"), dtype=np.float32) / 255.0

                original_mask_tensor = torch.from_numpy(original_mask).to(device=device)
                if original_mask_tensor.ndim > 2:
                    original_mask_tensor = original_mask_tensor.squeeze(0)

                prd_for_interp = prd_tensor.unsqueeze(0).unsqueeze(0)
                pred_upscaled = F.interpolate(
                    prd_for_interp,
                    size=original_mask_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)

                pred_bin_orig = pred_upscaled > threshold
                gt_bin_orig = original_mask_tensor > 0.5

                tp_orig += torch.count_nonzero(pred_bin_orig & gt_bin_orig)
                fp_orig += torch.count_nonzero(pred_bin_orig & ~gt_bin_orig)
                fn_orig += torch.count_nonzero(~pred_bin_orig & gt_bin_orig)
                tn_orig += torch.count_nonzero(~pred_bin_orig & ~gt_bin_orig)

                processed += 1

            if apply_limit and processed >= num_images:
                break

    resized_metrics = _metrics_from_counts(tp_resized.item(), fp_resized.item(), fn_resized.item(), tn_resized.item())
    original_metrics = _metrics_from_counts(tp_orig.item(), fp_orig.item(), fn_orig.item(), tn_orig.item())

    return {
        'resized': resized_metrics,
        'original': original_metrics,
    }



def evaluate_all_datasets(predictor, test_dataloader, threshold=0.5, device='cuda', num_images=None):
    """
    Compute metrics for a single test dataset.

    Args:
        predictor: The SAM2ImagePredictor object.
        test_dataloader: The test dataloader.
        threshold: Threshold for binary classification.
        device: Device to run the model ('cuda' or 'cpu').
        num_images: Optional limit on number of samples to evaluate.

    Returns:
        Metrics computed for the dataset.
    """
    return compute_cm_metrics(
        predictor, threshold, test_dataloader, device, num_images=num_images
    )


def test_with_metrics(
    dataset_path,
    save_path,
    CHECK_POINT,
    Image_Size=256,
    num_samples=None,
    save_outputs=True,
):
    """
    Run inference on a dataset and compute metrics.

    Args:
        dataset_path (str): Path to the dataset directory containing images and masks.
        save_path (str): Directory where results will be saved.
        CHECK_POINT (str): Path to the pre-trained model weights.
        num_samples (int, optional): Limit on the number of samples to evaluate. None means all samples.
        save_outputs (bool, optional): When True (default), metrics are written to a timestamped
            subfolder under save_path. Set to False to skip writing files.

    Returns:
        Metrics computed for the dataset.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    run_dir = save_path
    if save_outputs:
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
    test_dataset = SegmentationDataset(
        data_root=dataset_path,
        image_transform=images_transform,
        mask_transform=masks_transform,
        return_mask_path=True,
    )
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
    metrics_by_scale = evaluate_all_datasets(
        predictor,
        test_dataloader,
        threshold=0.5,
        device=DEVICE,
        num_images=num_samples,
    )

    print("Evaluation Metrics:")
    for scale, metrics in metrics_by_scale.items():
        print(f"{scale.capitalize()} Resolution:")
        for metric_name, value in metrics.items():
            if metric_name == 'confusion_matrix':
                print(f"  {metric_name.replace('_', ' ').capitalize()}:\n{value}")
            else:
                print(f"  {metric_name.capitalize()}: {value:.4f}")

    if save_outputs:
        metrics_filepath = os.path.join(run_dir, "metrics.txt")
        with open(metrics_filepath, "w", encoding="utf-8") as metrics_file:
            metrics_file.write(f"Dataset: {dataset_path}\n")
            metrics_file.write(f"Checkpoint: {CHECK_POINT}\n")
            metrics_file.write(f"Image Size: {Image_Size}\n")
            metrics_file.write(f"Sample Limit: {num_samples if num_samples is not None else 'All'}\n")
            metrics_file.write("Evaluation Metrics:\n")
            for scale, metrics in metrics_by_scale.items():
                metrics_file.write(f"{scale.capitalize()} Resolution:\n")
                for metric_name, value in metrics.items():
                    if metric_name == 'confusion_matrix':
                        metrics_file.write("Confusion Matrix:\n")
                        metrics_file.write(f"{value}\n")
                    else:
                        metrics_file.write(f"{metric_name.capitalize()}: {value:.4f}\n")

        print(f"Metrics saved to {metrics_filepath}")

    return metrics_by_scale

if __name__ == '__main__':
    Image_Size = 1024
    test_with_metrics(
        dataset_path='/data1/SegmentationDataset/test',
        save_path='/home/a_alagha/SegmentationProject/SlideProcessor/SAM2_finetuning/results',
        CHECK_POINT='/home/a_alagha/SegmentationProject/SlideProcessor/SAM2_finetuning/saved_models/experiment_sam2_layernorm5_20251009-015749/trained_sam2_layernorm5.pth',
        Image_Size=Image_Size,
        num_samples=None  # Set to an integer to limit number of samples (e.g., 50
    )
