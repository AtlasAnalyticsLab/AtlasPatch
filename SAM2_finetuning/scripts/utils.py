import torch
import monai
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from pathlib import Path
import torchvision.transforms.functional as TF
from PIL import Image


# class DiceBCELoss(nn.Module):
#     """
#     Loss combined of Binary Cross Entropy loss and Dice loss.
#     """
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(DiceBCELoss, self).__init__()
#         self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
#         self.dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

#     def forward(self, prob_masks, target):
#         # prob_masks: predicted logits
#         # target: ground truth binary segmentation mask (0.0 or 1.0, float type)

#         # Calculate pixel-wise binary cross-entropy loss
#         bce_loss = self.bce_with_logits_loss(prob_masks, target)
#         dice_loss = self.dice_loss_fn(prob_masks, target)

#         return (0.2*bce_loss + 0.8*dice_loss)

class DiceBCELoss(nn.Module):
    """
    Loss combined of Binary Cross Entropy loss and Dice loss.
    """
    def __init__(self, gamma=0.9):
        super(DiceBCELoss, self).__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.gamma = gamma

    def forward(self, prob_masks, target):
        # prob_masks: predicted logits
        # target: ground truth binary segmentation mask (0.0 or 1.0, float type)

        # Calculate pixel-wise binary cross-entropy loss
        bce_loss = self.bce_with_logits_loss(prob_masks, target)
        dice_loss = self.dice_loss_fn(prob_masks, target)

        return (bce_loss + self.gamma*dice_loss)
    

class EarlyStopping:
        def __init__(self, patience=5, delta=0, verbose=False):
            """
            Args:
                patience (int): How many epochs to wait after last improvement.
                delta (float): Minimum change in monitored metric to qualify as an improvement.
                verbose (bool): Whether to print messages about early stopping.
            """
            self.patience = patience
            self.delta = delta
            self.verbose = verbose
            self.best_loss = float('inf')
            self.epochs_without_improvement = 0
            self.should_stop = False

        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.verbose:
                    print(f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs.")

            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
    

def plot_transformed_mask(image_paths, transform, n=2, seed=52):
    """Plots a series of random images from image_paths with their transforms.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 2.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    #n = min(n, len(image_paths))
    #random_image_paths = random.sample(image_paths, k=n)
    random_image_paths = random.choice(image_paths)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            #f = f.convert("1")
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()



def SaveModel(
    model,
    optimizer,
    epochs,
    file_name: str,
    save_dir: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    losses: Optional[Dict[int, Dict[str, float]]] = None,
):
    """
    This functions saves your models. 
    Please enter your file_name as string.
    
    """
    
    models_path = Path(save_dir)

    # If the folder doesn't exist, make one... 
    if not models_path.is_dir():
        print(f"Did not find {models_path} directory, creating one...")
        models_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{models_path} directory exists.")
    
    name = f"{file_name}.pth"  #give it a name
    model_path = models_path / name 

    torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'epoch': epochs,
            }, model_path)

    summary_path = models_path / f"{file_name}_hyperparameters.txt"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("Training Hyperparameters Summary\n")
        summary_file.write(f"Saved Epoch: {epochs}\n")
        summary_file.write("\n")
        if hyperparams:
            for key in sorted(hyperparams.keys()):
                summary_file.write(f"{key}: {hyperparams[key]}\n")
        summary_file.write("\nFinal Epoch Losses\n")
        if losses:
            for epoch_idx in sorted(losses.keys()):
                epoch_info = losses[epoch_idx]
                summary_file.write(
                    f"Epoch {epoch_idx}: train_loss={epoch_info.get('train_loss')}, val_loss={epoch_info.get('val_loss')}\n"
                )
    

def convert_checkpoint(sam_checkpoint_path: str,
                       cracksam_checkpoint_path: str,
                       saving_path: str):
    """
    Convert crack model checkpoint to sam checkpoint format for convenient inference
    """
    sam_ckpt = torch.load(sam_checkpoint_path)
    medsam_ckpt = torch.load(cracksam_checkpoint_path)
    sam_keys = sam_ckpt.keys()
    for key in sam_keys:
        sam_ckpt[key] = medsam_ckpt["model"][key]

    torch.save(sam_ckpt, saving_path)
