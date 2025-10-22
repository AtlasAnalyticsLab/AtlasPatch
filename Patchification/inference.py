import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide-python not installed. WSI file support (.svs, .ndpi) will be unavailable.")

WSI_EXTENSIONS = frozenset(['.svs', '.ndpi', '.tif', '.tiff', '.scn', '.mrxs', '.vms', '.vmu', '.bif'])

class SAM2Inferencer:
    """Optimized SAM2 inference for whole slide image thumbnails."""

    def __init__(
        self,
        model_path: str,
        input_size: int,
        base_checkpoint: str,
        base_model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.input_size = input_size

        self._bbox_cache = np.array([0, 0, input_size, input_size], dtype=np.float32) #cache bbox for reuse

        print(f"Loading SAM2 model...")
        print(f"  - Config: {base_model_cfg}")
        print(f"  - Base checkpoint: {base_checkpoint}")
        print(f"  - Fine-tuned checkpoint: {model_path}")
        print(f"  - Input size: {input_size}x{input_size}")
        print(f"  - Device: {device}")

        self.predictor = SAM2ImagePredictor(
            build_sam2(base_model_cfg, base_checkpoint, device=device)
        )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        self.predictor.model.load_state_dict(state_dict, strict=False)
        self.predictor.model.eval()

        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        print("---Model loaded---\n")

    @staticmethod
    def is_wsi_file(image_path: str) -> bool:
        return Path(image_path).suffix.lower() in WSI_EXTENSIONS

    def get_wsi_thumbnail(self, wsi_path: str) -> np.ndarray:
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("openslide-python required for WSI support")

        slide = openslide.OpenSlide(wsi_path)
        level_count = slide.level_count
        level_dimensions = slide.level_dimensions

        # Best Level search :
        dims_array = np.array(level_dimensions)
        smaller_dims = dims_array.min(axis=1)
        best_level = np.argmin(np.abs(smaller_dims - self.input_size))

        print(f"WSI: {Path(wsi_path).name} | Level {best_level}/{level_count-1} | Size: {level_dimensions[best_level]}")

        thumbnail = slide.read_region((0, 0), best_level, level_dimensions[best_level])
        slide.close()

        #RGBA -> RGB
        return np.asarray(thumbnail)[:, :, :3]

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized image preprocessing.

        Returns:
            (original_image, resized_image) as numpy arrays
        """
        if self.is_wsi_file(image_path) and OPENSLIDE_AVAILABLE:
            original_image = self.get_wsi_thumbnail(image_path)
            image_pil = Image.fromarray(original_image, mode='RGB')
        else:
            image_pil = Image.open(image_path).convert('RGB')
            original_image = np.asarray(image_pil)

        # Resize 
        resized_pil = image_pil.resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.BILINEAR
        )
        resized_image = np.asarray(resized_pil)

        return original_image, resized_image

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized inference with torch.inference_mode.

        Returns:
            Binary mask (HxW, [0-1])
        """
        self.predictor.set_image(image)

        # Use cached bbox
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=self._bbox_cache,
            multimask_output=False,
            return_logits=False,
        )

        return masks[0]

    def upscale_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Optimized mask upscaling using PIL with pre-conversion.

        Args:
            mask: Input mask (HxW, [0-1])
            target_shape: (height, width)

        Returns:
            Upscaled mask (HxW, [0-1])
        """
        # Convert once and reuse
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        upscaled_pil = mask_pil.resize(
            (target_shape[1], target_shape[0]),  # PIL uses (width, height)
            resample=Image.Resampling.NEAREST
        )
        return np.asarray(upscaled_pil, dtype=np.float32) / 255.0

    def save_visualization(self, image: np.ndarray, mask: np.ndarray, save_path: str):
        """
        Optimized visualization saving without displaying.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(image)
        # Optimized overlay creation
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        mask_rgba[mask > 0.5] = [0, 1, 0, 0.5]
        axes[2].imshow(mask_rgba)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
        plt.close(fig)

    def infer_single_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True,
        save_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized single image inference.

        Returns:
            (original_image, upscaled_mask)
        """
        print(f"Processing: {Path(image_path).name}")

        # Preprocessing
        original_image, resized_image = self.preprocess_image(image_path)

        # Inference
        mask_resized = self.predict(resized_image)

        # Upscale mask
        mask_upscaled = self.upscale_mask(mask_resized, original_image.shape[:2])

        print(f"  Input: {original_image.shape} | Model: {resized_image.shape} | Mask: {mask_upscaled.shape}")

        # Save outputs
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = Path(image_path).stem

            if save_mask:
                mask_path = os.path.join(output_dir, f"{stem}_mask.png")
                Image.fromarray((mask_upscaled * 255).astype(np.uint8), mode='L').save(
                    mask_path, optimize=True
                )
                print(f"  Saved mask: {mask_path}")

            if save_visualization:
                viz_path = os.path.join(output_dir, f"{stem}_result.png")
                self.save_visualization(original_image, mask_upscaled, viz_path)
                print(f"  Saved visualization: {viz_path}")

        return original_image, mask_upscaled

    def infer_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True,
        save_mask: bool = True,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.svs', '.ndpi', '.scn', '.mrxs')
    ):
        """
        Optimized batch inference.
        """
        # Fast file collection using set for deduplication
        folder = Path(folder_path)
        image_files = set()
        for ext in extensions:
            image_files.update(folder.glob(f"*{ext}"))
            image_files.update(folder.glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)

        if not image_files:
            print(f"No images found in {folder_path}")
            return

        print(f"Found {len(image_files)} images\n")

        # Process batch
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")
            self.infer_single_image(
                str(image_path),
                output_dir=output_dir,
                save_visualization=save_visualization,
                save_mask=save_mask
            )

        if output_dir:
            print(f"\nAll results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized SAM2 Inference for WSI Thumbnails",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_path', type=str, help='Single image path')
    input_group.add_argument('--folder_path', type=str, help='Folder path for batch processing')

    parser.add_argument('--model_name', type=str, required=True, help='Model filename')
    parser.add_argument('--input_size', type=int, required=True, help='Input size (256/512/1024)')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--base_checkpoint', type=str, required=True, help='Base checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='Config path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--save_visualization', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--save_mask', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Initialize inferencer
    inferencer = SAM2Inferencer(
        model_path=model_path,
        input_size=args.input_size,
        base_model_cfg=args.config,
        base_checkpoint=args.base_checkpoint,
        device=args.device
    )

    # Run inference
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image not found at {args.image_path}")
            sys.exit(1)
        inferencer.infer_single_image(
            args.image_path,
            output_dir=args.output_dir,
            save_visualization=args.save_visualization,
            save_mask=args.save_mask
        )
    else:
        if not os.path.exists(args.folder_path):
            print(f"Error: Folder not found at {args.folder_path}")
            sys.exit(1)
        inferencer.infer_folder(
            args.folder_path,
            output_dir=args.output_dir,
            save_visualization=args.save_visualization,
            save_mask=args.save_mask
        )


if __name__ == "__main__":
    main()
