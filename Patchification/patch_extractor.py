import os
import sys
import time
import cv2
import h5py
import numpy as np
import openslide
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import glob
from PIL import Image
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import SAM2Inferencer


class ContourChecker:
    
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        raise NotImplementedError


class isInContourV3_Easy(ContourChecker):
    """
    Four-point easy check - ANY of 4 corners inside tissue.
    """

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift)
            ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) >= 0:
                return True
        return False


class isInContourV3_Hard(ContourChecker):
    """
    Four-point hard check - ALL 4 corners inside tissue.
    """
    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift)
            ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) < 0:
                return False
        return True


def isBlackPatch(patch, rgbThresh=40):
    """Check if patch is mostly black."""
    return np.all(patch < rgbThresh)


def isWhitePatch(patch, satThresh=5):
    """Check if patch is mostly white."""
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return np.all(patch_hsv[:, :, 1] < satThresh)


class WSIPatchifier:
    """
    WSI patchification with contour-based tissue detection.
    """

    def __init__(
        self,
        patch_size: int = 256,
        step_size: int = 256,
        contour_fn: str = 'four_pt',
        white_thresh: int = 15,
        black_thresh: int = 50,
        tissue_area_thresh: float = 0.01,
        use_padding: bool = True
    ):
        """
        Args:
            patch_size: Patch size at level 0
            step_size: Step size (= patch_size for no overlap)
            contour_fn: 'four_pt' or 'four_pt_hard'
            white_thresh: White patch threshold
            black_thresh: Black patch threshold
            tissue_area_thresh: Min tissue area as % of image (0.0-100.0)
            use_padding: Allow patches at boundaries
        """
        self.patch_size = patch_size
        self.step_size = step_size
        self.contour_fn = contour_fn
        self.white_thresh = white_thresh
        self.black_thresh = black_thresh
        self.tissue_area_thresh = tissue_area_thresh
        self.use_padding = use_padding

        print(f"\n{'='*70}")
        print(f"WSI Patchifier Initialized")
        print(f"{'='*70}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Step size: {step_size}")
        print(f"  Contour function: {contour_fn}")
        print(f"  White/Black filtering: {white_thresh}/{black_thresh}")
        print(f"  Tissue area threshold: {tissue_area_thresh}% of image")
        print(f"{'='*70}\n")

    def mask_to_contours(
        self,
        mask: np.ndarray,
        filter_params: Dict = None
    ) -> Tuple[List, List]:
        """
        Convert binary mask to OpenCV contours.

        Args:
            mask: Binary mask (H x W) with values 0-1
            filter_params: Filtering parameters

        Returns:
            (tissue_contours, holes_contours)
        """
        print(f"  Converting mask to contours...")

        if filter_params is None:
            filter_params = {
                'a_t': 100,      # Min tissue area, can be played with 
                'a_h': 16,       # Min hole area
                'max_n_holes': 10
            }

        # Convert to uint8 binary
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Find contours
        contours, hierarchy = cv2.findContours(
            mask_uint8,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE
        )

        if hierarchy is None:
            print(f"  No contours found!")
            return [], []

        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # Calculate minimum area threshold based on percentage of image
        image_area = mask.shape[0] * mask.shape[1]
        min_area_threshold = (self.tissue_area_thresh / 100.0) * image_area

        # Use the larger of: user percentage or filter_params['a_t']
        effective_min_area = max(min_area_threshold, filter_params['a_t'])

        # Separate tissue and hole contours
        foreground_contours = []
        hole_contours = []

        for idx, cont in enumerate(contours):
            # Tissue contours (no parent)
            if hierarchy[idx][1] == -1:
                area = cv2.contourArea(cont)
                if area >= effective_min_area:
                    foreground_contours.append(cont)

            # Hole contours (has parent)
            else:
                area = cv2.contourArea(cont)
                if area >= filter_params['a_h']:
                    hole_contours.append(cont)

        # Limit holes
        if len(hole_contours) > filter_params['max_n_holes']:
            hole_contours = sorted(hole_contours, key=cv2.contourArea, reverse=True)
            hole_contours = hole_contours[:filter_params['max_n_holes']]

        # Group holes by parent
        holes_by_contour = []
        for _ in foreground_contours:
            holes_by_contour.append([])

        # assign all holes to first contour
        if len(foreground_contours) > 0 and len(hole_contours) > 0:
            holes_by_contour[0] = hole_contours

        print(f"  Minimum tissue area: {effective_min_area:.0f} pixels² ({self.tissue_area_thresh}% of {image_area} pixels)")
        print(f"  Found {len(foreground_contours)} tissue contours")
        print(f"  Found {len(hole_contours)} holes")

        return foreground_contours, holes_by_contour

    def _is_in_holes(self, holes: List, pt: Tuple[int, int]) -> bool:
        """Check if point is inside any hole."""
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2), False) > 0:
                return True
        return False

    def _is_in_contours(
        self,
        cont_check_fn,
        pt: Tuple[int, int],
        holes: List
    ) -> bool:
        """Check if point is in tissue and not in holes."""
        return cont_check_fn(pt) and not self._is_in_holes(holes, pt)

    def _get_patch_generator(
        self,
        wsi,
        contour,
        holes,
        cont_idx: int,
        wsi_name: str
    ):
        """
        Patch generator implementation.

        Yields patches within a tissue contour.
        """
        # Get contour bounding box
        start_x, start_y, w, h = cv2.boundingRect(contour)

        print(f"  Contour {cont_idx}: BBox=({start_x}, {start_y}, {w}, {h}), Area={cv2.contourArea(contour):.0f}")

        # Create contour checker
        if self.contour_fn == 'four_pt':
            cont_check_fn = isInContourV3_Easy(
                contour=contour,
                patch_size=self.patch_size,
                center_shift=0.5
            )
        elif self.contour_fn == 'four_pt_hard':
            cont_check_fn = isInContourV3_Hard(
                contour=contour,
                patch_size=self.patch_size,
                center_shift=0.5
            )
        else:
            raise ValueError(f"Unknown contour_fn: {self.contour_fn}")

        # Get WSI dimensions
        img_w, img_h = wsi.dimensions

        # Calculate iteration bounds
        if self.use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - self.patch_size)
            stop_x = min(start_x + w, img_w - self.patch_size)

        # Iterate over patch locations
        count = 0
        for y in range(start_y, stop_y, self.step_size):
            for x in range(start_x, stop_x, self.step_size):
                # Check if patch is in tissue and not in hole
                if not self._is_in_contours(cont_check_fn, (x, y), holes):
                    continue

                count += 1

                # Read patch from WSI
                patch_pil = wsi.read_region((x, y), 0, (self.patch_size, self.patch_size))
                patch_pil = patch_pil.convert('RGB')
                patch = np.array(patch_pil)

                # Filter white/black patches
                if isBlackPatch(patch, rgbThresh=self.black_thresh) or \
                   isWhitePatch(patch, satThresh=self.white_thresh):
                    continue

                # Yield patch info
                patch_info = {
                    'x': x,
                    'y': y,
                    'cont_idx': cont_idx,
                    'patch': patch,
                    'name': wsi_name
                }

                yield patch_info

        print(f"    Patches extracted from contour {cont_idx}: {count}")

    def create_patches_hdf5(
        self,
        wsi_path: str,
        tissue_contours: List,
        holes_contours: List,
        output_path: str,
        save_image_patches: bool = False
    ) -> Tuple[str, float]:
        """
        Extract patches and save to HDF5.

        Args:
            wsi_path: Path to WSI
            tissue_contours: List of tissue contours
            holes_contours: List of hole lists (one per tissue contour)
            output_path: Output HDF5 path
            save_image_patches: Whether to save individual patch images

        Returns:
            (Path to HDF5 file, Image saving time in seconds)
        """
        print(f"\n  Creating patches and saving to HDF5 (batch writing)...")

        wsi = openslide.OpenSlide(wsi_path)
        wsi_name = Path(wsi_path).stem

        # Create patches directory if saving images
        patches_img_dir = None
        if save_image_patches:
            patches_img_dir = os.path.join(os.path.dirname(output_path), 'patch_images')
            os.makedirs(patches_img_dir, exist_ok=True)
            print(f"  Saving individual patch images to: {patches_img_dir}")

        # Batch writing setup 
        batch_size = 512
        h5_file = None
        patch_buffer = []
        coord_buffer = []
        total_patches = 0

        def flush_buffer():
            """Write buffered patches to HDF5"""
            nonlocal h5_file, total_patches
            if len(patch_buffer) == 0:
                return

            patches_array = np.array(patch_buffer, dtype=np.uint8)
            coords_array = np.array(coord_buffer, dtype=np.int32)

            if h5_file is None:
                # Create file with first batch (resizable datasets)
                h5_file = h5py.File(output_path, 'w')
                h5_file.create_dataset(
                    'imgs',
                    data=patches_array,
                    maxshape=(None, self.patch_size, self.patch_size, 3),
                    chunks=(1, self.patch_size, self.patch_size, 3),
                    dtype=np.uint8
                )
                h5_file.create_dataset(
                    'coords',
                    data=coords_array,
                    maxshape=(None, 2),
                    chunks=(1, 2),
                    dtype=np.int32
                )
                # Metadata
                h5_file.attrs['patch_size'] = self.patch_size
                h5_file.attrs['wsi_path'] = wsi_path
            else:
                # Resize and append
                h5_file['imgs'].resize(total_patches + len(patches_array), axis=0)
                h5_file['imgs'][-len(patches_array):] = patches_array

                h5_file['coords'].resize(total_patches + len(coords_array), axis=0)
                h5_file['coords'][-len(coords_array):] = coords_array

            total_patches += len(patches_array)
            patch_buffer.clear()
            coord_buffer.clear()

        # Process each tissue contour
        for idx, (cont, holes) in enumerate(zip(tissue_contours, holes_contours)):
            patch_gen = self._get_patch_generator(
                wsi=wsi,
                contour=cont,
                holes=holes,
                cont_idx=idx,
                wsi_name=wsi_name
            )

            for patch_info in patch_gen:
                patch_buffer.append(patch_info['patch'])
                coord_buffer.append([patch_info['x'], patch_info['y']])

                # Flush when batch is full
                if len(patch_buffer) >= batch_size:
                    flush_buffer()

        # Flush remaining patches
        flush_buffer()

        wsi.close()

        img_save_time = 0.0

        if h5_file is not None:
            h5_file.attrs['num_patches'] = total_patches
            h5_file.close()
            print(f"  Saved {total_patches} patches to {output_path}")

            # Save individual patch images AFTER patchification timing (if requested)
            if save_image_patches and total_patches > 0:
                print(f"\n  Saving {total_patches} individual patch images...")
                img_save_start = time.time()

                # Re-read from HDF5 and save as images
                with h5py.File(output_path, 'r') as f:
                    imgs = f['imgs']
                    coords = f['coords']
                    for i in range(total_patches):
                        patch = imgs[i]
                        x, y = coords[i]
                        patch_filename = f"{wsi_name}_x{x}_y{y}.png"
                        patch_path = os.path.join(patches_img_dir, patch_filename)
                        Image.fromarray(patch).save(patch_path)

                img_save_time = time.time() - img_save_start
                print(f"  Saved {total_patches} patch images to {patches_img_dir}")
                print(f"  Image saving time: {img_save_time:.3f}s")

            return output_path, img_save_time
        else:
            print(f"  No patches extracted!")
            return None, 0.0


def visualize_random_patches(hdf5_path: str, output_dir: str, n_patches: int = 10):
    """
    Visualize random patches from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory for visualization
        n_patches: Number of random patches to show
    """
    print(f"\n  Visualizing {n_patches} random patches from HDF5...")

    with h5py.File(hdf5_path, 'r') as f:
        total_patches = f['coords'].shape[0]

        # Select random indices
        if total_patches < n_patches:
            n_patches = total_patches

        random_indices = np.random.choice(total_patches, n_patches, replace=False)
        random_indices = sorted(random_indices)

        # Read patches
        patches = []
        coords = []
        for idx in random_indices:
            patches.append(f['imgs'][idx])
            coords.append(f['coords'][idx])

        patches = np.array(patches)
        coords = np.array(coords)

    # Create visualization grid
    n_cols = 5
    n_rows = (n_patches + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_patches > 1 else [axes]

    for idx, (patch, coord, ax) in enumerate(zip(patches, coords, axes)):
        ax.imshow(patch)
        ax.set_title(f'Patch {random_indices[idx]}\nCoord: ({coord[0]}, {coord[1]})', fontsize=9)
        ax.axis('off')

    # Hide extra subplots
    for idx in range(n_patches, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'random_patches_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved random patches visualization: {output_path}")
    return output_path


def visualize_patches_on_thumbnail(
    hdf5_path: str,
    wsi_path: str,
    thumbnail_image: np.ndarray,
    output_dir: str,
    patch_size: int = 256
):
    """
    Visualize patch locations overlaid on WSI thumbnail.

    Args:
        hdf5_path: Path to HDF5 file
        wsi_path: Path to WSI
        thumbnail_image: Thumbnail image
        output_dir: Output directory
        patch_size: Patch size at level 0
    """
    print(f"\n  Creating patch overlay visualization on thumbnail...")

    # Read coordinates from HDF5
    with h5py.File(hdf5_path, 'r') as f:
        coords = f['coords'][:]

    # Get WSI dimensions and calculate downsample
    wsi = openslide.OpenSlide(wsi_path)
    wsi_dims = wsi.dimensions
    wsi.close()

    downsample_x = wsi_dims[0] / thumbnail_image.shape[1]
    downsample_y = wsi_dims[1] / thumbnail_image.shape[0]

    # Scale coordinates to thumbnail space
    coords_thumb = coords.astype(np.float32)
    coords_thumb[:, 0] = coords_thumb[:, 0] / downsample_x
    coords_thumb[:, 1] = coords_thumb[:, 1] / downsample_y
    patch_size_thumb_x = patch_size / downsample_x
    patch_size_thumb_y = patch_size / downsample_y

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(thumbnail_image)

    # Draw all patch rectangles
    for coord in coords_thumb:
        rect = mpatches.Rectangle(
            (coord[0], coord[1]),
            patch_size_thumb_x,
            patch_size_thumb_y,
            linewidth=0.5,
            edgecolor='lime',
            facecolor='none',
            alpha=0.6
        )
        ax.add_patch(rect)

    ax.set_title(f'Patch Locations on WSI Thumbnail\n{len(coords)} patches extracted', fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'patches_on_thumbnail.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Saved patch overlay visualization: {output_path}")
    return output_path


def find_wsi_files(folder_path: str) -> List[str]:
    """
    Find all WSI files in a folder.

    Args:
        folder_path: Path to folder containing WSI files

    Returns:
        List of WSI file paths
    """
    wsi_extensions = ['*.svs', '*.ndpi', '*.tiff', '*.tif', '*.mrxs', '*.scn', '*.vms', '*.vmu', '*.bif']
    wsi_files = []

    for ext in wsi_extensions:
        wsi_files.extend(glob.glob(os.path.join(folder_path, ext)))

    return sorted(wsi_files)


def run_batch_processing(
    wsi_files: List[str],
    output_root_dir: str,
    patch_size: int = 256,
    step_size: int = 256,
    tissue_area_thresh: float = 0.01,
    contour_fn: str = 'four_pt',
    white_thresh: int = 15,
    black_thresh: int = 50,
    model_path: str = '/home/o_a38510/Segmentation_Project/sam2_bs2_inp1024_lr5e-4.pth',
    base_checkpoint: str = '/home/o_a38510/Segmentation_Project/sam2.1_hiera_tiny.pt',
    config: str = 'configs/sam2.1/sam2.1_hiera_t.yaml',
    input_size: int = 1024,
    device: str = 'cuda',
    save_image_patches: bool = False
):
    """
    Process multiple WSI files and generate summary statistics.

    Args:
        wsi_files: List of WSI file paths
        output_root_dir: Root output directory
        ... (other parameters same as run_sam2_patchification_pipeline)

    Returns:
        Dictionary with batch results and statistics
    """
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(wsi_files)} WSI FILES")
    print(f"{'='*70}")
    print(f"  Output directory: {output_root_dir}")
    print(f"{'='*70}\n")

    os.makedirs(output_root_dir, exist_ok=True)

    all_results = []
    batch_start = time.time()

    for idx, wsi_path in enumerate(wsi_files, 1):
        wsi_name = Path(wsi_path).stem
        print(f"\n{'#'*70}")
        print(f"# Processing WSI {idx}/{len(wsi_files)}: {wsi_name}")
        print(f"{'#'*70}\n")

        # Create subfolder for this WSI
        wsi_output_dir = os.path.join(output_root_dir, wsi_name)
        os.makedirs(wsi_output_dir, exist_ok=True)

        try:
            # Run pipeline for this WSI
            result = run_sam2_patchification_pipeline(
                wsi_path=wsi_path,
                output_dir=wsi_output_dir,
                patch_size=patch_size,
                step_size=step_size,
                tissue_area_thresh=tissue_area_thresh,
                contour_fn=contour_fn,
                white_thresh=white_thresh,
                black_thresh=black_thresh,
                model_path=model_path,
                base_checkpoint=base_checkpoint,
                config=config,
                input_size=input_size,
                device=device,
                save_image_patches=save_image_patches
            )

            result['wsi_name'] = wsi_name
            result['wsi_path'] = wsi_path
            result['status'] = 'success'
            all_results.append(result)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {wsi_name}: {str(e)}")
            all_results.append({
                'wsi_name': wsi_name,
                'wsi_path': wsi_path,
                'status': 'failed',
                'error': str(e),
                'timing': {'total': 0},
                'num_patches': 0
            })

    batch_end = time.time()
    batch_total_time = batch_end - batch_start

    # Generate summary report
    print(f"\n\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}\n")

    # Per-WSI statistics table
    print(f"{'WSI Name':<40} {'Status':<10} {'Inf(s)':<10} {'Cont(s)':<10} {'Patch(s)':<10} {'Total(s)':<10} {'#Patches':<10}")
    print(f"{'-'*70}")

    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']

    for result in all_results:
        name = result['wsi_name'][:38]
        status = result['status']

        if status == 'success':
            timing = result['timing']
            inf_time = timing.get('inference', 0)
            cont_time = timing.get('contour_conversion', 0)
            patch_time = timing.get('patchification', 0)
            total_time = timing.get('total', 0)
            num_patches = result['num_patches']

            print(f"{name:<40} {status:<10} {inf_time:<10.2f} {cont_time:<10.2f} {patch_time:<10.2f} {total_time:<10.2f} {num_patches:<10}")
        else:
            print(f"{name:<40} {status:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print(f"{'-'*70}\n")

    # Overall statistics
    if successful:
        total_patches = sum(r['num_patches'] for r in successful)
        avg_inference = np.mean([r['timing']['inference'] for r in successful])
        avg_contour = np.mean([r['timing']['contour_conversion'] for r in successful])
        avg_patch = np.mean([r['timing']['patchification'] for r in successful])
        avg_total = np.mean([r['timing']['total'] for r in successful])

        print(f"SUMMARY STATISTICS:")
        print(f"  Total WSIs processed:     {len(wsi_files)}")
        print(f"  Successful:               {len(successful)}")
        print(f"  Failed:                   {len(failed)}")
        print(f"  Total patches extracted:  {total_patches:,}")
        print(f"\nAVERAGE TIMING (successful WSIs):")
        print(f"  Inference:                {avg_inference:.2f}s")
        print(f"  Contour conversion:       {avg_contour:.2f}s")
        print(f"  Patchification:           {avg_patch:.2f}s")
        print(f"  Total per WSI:            {avg_total:.2f}s")
        print(f"\nBATCH TOTALS:")
        print(f"  Total batch time:         {batch_total_time:.2f}s ({batch_total_time/60:.2f} min)")
        print(f"  Output directory:         {output_root_dir}")
    else:
        print(f"SUMMARY: All {len(wsi_files)} WSIs failed to process.")

    print(f"{'='*70}\n")

    if failed:
        print(f"FAILED WSIs:")
        for result in failed:
            print(f"  - {result['wsi_name']}: {result.get('error', 'Unknown error')}")
        print()

    return {
        'all_results': all_results,
        'successful': successful,
        'failed': failed,
        'batch_time': batch_total_time,
        'total_patches': sum(r['num_patches'] for r in successful) if successful else 0
    }


def run_sam2_patchification_pipeline(
    wsi_path: str,
    output_dir: str,
    patch_size: int = 256,
    step_size: int = 256,
    tissue_area_thresh: float = 0.01,
    contour_fn: str = 'four_pt',
    white_thresh: int = 15,
    black_thresh: int = 50,
    model_path: str = '/home/o_a38510/Segmentation_Project/sam2_bs2_inp1024_lr5e-4.pth',
    base_checkpoint: str = '/home/o_a38510/Segmentation_Project/sam2.1_hiera_tiny.pt',
    config: str = 'configs/sam2.1/sam2.1_hiera_t.yaml',
    input_size: int = 1024,
    device: str = 'cuda',
    save_image_patches: bool = False
):
    """
    Complete pipeline: SAM2 segmentation + patchification.

    Args:
        wsi_path: Path to WSI
        output_dir: Output directory
        patch_size: Patch size at level 0
        step_size: Step size
        tissue_threshold: Not used (kept for compatibility)
        contour_fn: Contour checking function
        white_thresh: White patch threshold
        black_thresh: Black patch threshold
        model_path: SAM2 model path
        base_checkpoint: SAM2 base checkpoint
        config: SAM2 config
        input_size: SAM2 input size
        device: Device
        save_image_patches: Whether to save individual patch images

    Returns:
        Dictionary with results and timing
    """
    stem = Path(wsi_path).stem

    # Create WSI-specific subfolder
    wsi_output_dir = os.path.join(output_dir, stem)
    os.makedirs(wsi_output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAM2 PATCHIFICATION PIPELINE")
    print(f"{'='*70}")
    print(f"  WSI: {Path(wsi_path).name}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Output: {wsi_output_dir}")
    print(f"{'='*70}")

    timing = {}

    # Step 1: SAM2 Tissue Segmentation
    print(f"\n[STEP 1/3] Running SAM2 tissue segmentation...")
    t_inf_start = time.time()

    inferencer = SAM2Inferencer(
        model_path=model_path,
        input_size=input_size,
        base_model_cfg=config,
        base_checkpoint=base_checkpoint,
        device=device
    )

    thumbnail_image, thumbnail_mask = inferencer.infer_single_image(
        wsi_path,
        output_dir=wsi_output_dir,
        save_visualization=True,
        save_mask=True
    )

    t_inf_end = time.time()
    timing['inference'] = t_inf_end - t_inf_start

    print(f"  Inference time: {timing['inference']:.3f}s")
    print(f"  Mask shape: {thumbnail_mask.shape}")

    # Step 2: Convert mask to contours
    print(f"\n[STEP 2/3] Converting mask to contours...")
    t_contour_start = time.time()

    patchifier = WSIPatchifier(
        patch_size=patch_size,
        step_size=step_size,
        contour_fn=contour_fn,
        white_thresh=white_thresh,
        black_thresh=black_thresh,
        tissue_area_thresh=tissue_area_thresh
    )

    tissue_contours, holes_contours = patchifier.mask_to_contours(thumbnail_mask)

    # Scale contours to WSI level 0
    wsi = openslide.OpenSlide(wsi_path)
    wsi_dims = wsi.dimensions
    wsi.close()

    scale_x = wsi_dims[0] / thumbnail_mask.shape[1]
    scale_y = wsi_dims[1] / thumbnail_mask.shape[0]

    # Scale contours
    tissue_contours_scaled = []
    for cont in tissue_contours:
        cont_scaled = cont.astype(np.float32)
        cont_scaled[:, :, 0] *= scale_x
        cont_scaled[:, :, 1] *= scale_y
        tissue_contours_scaled.append(cont_scaled.astype(np.int32))

    holes_contours_scaled = []
    for holes in holes_contours:
        holes_scaled = []
        for hole in holes:
            hole_scaled = hole.astype(np.float32)
            hole_scaled[:, :, 0] *= scale_x
            hole_scaled[:, :, 1] *= scale_y
            holes_scaled.append(hole_scaled.astype(np.int32))
        holes_contours_scaled.append(holes_scaled)

    t_contour_end = time.time()
    timing['contour_conversion'] = t_contour_end - t_contour_start

    print(f"  Contour conversion time: {timing['contour_conversion']:.3f}s")

    # Step 3: Patch extraction
    print(f"\n[STEP 3/3] Extracting patches...")
    t_patch_start = time.time()

    output_h5 = os.path.join(wsi_output_dir, f"{stem}.h5")

    hdf5_path, img_save_time = patchifier.create_patches_hdf5(
        wsi_path=wsi_path,
        tissue_contours=tissue_contours_scaled,
        holes_contours=holes_contours_scaled,
        output_path=output_h5,
        save_image_patches=save_image_patches
    )

    t_patch_end = time.time()
    # Subtract image saving time to get pure HDF5 patchification time
    timing['patchification'] = (t_patch_end - t_patch_start) - img_save_time
    timing['image_saving'] = img_save_time
    timing['total'] = t_patch_end - t_inf_start

    print(f"  Patchification time: {timing['patchification']:.3f}s")

    # Get patch count
    num_patches = 0
    if hdf5_path and os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            num_patches = f['coords'].shape[0]

    # Step 4: Visualizations
    print(f"\n[STEP 4/4] Creating visualizations...")
    vis_paths = {}

    if hdf5_path and os.path.exists(hdf5_path) and num_patches > 0:
        # Visualize random patches from HDF5
        vis_paths['random_patches'] = visualize_random_patches(
            hdf5_path=hdf5_path,
            output_dir=wsi_output_dir,
            n_patches=10
        )

        # Visualize patches on thumbnail
        vis_paths['patches_on_thumbnail'] = visualize_patches_on_thumbnail(
            hdf5_path=hdf5_path,
            wsi_path=wsi_path,
            thumbnail_image=thumbnail_image,
            output_dir=wsi_output_dir,
            patch_size=patch_size
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Inference time:         {timing['inference']:8.3f}s")
    print(f"  Contour conversion:     {timing['contour_conversion']:8.3f}s")

    if save_image_patches and timing['image_saving'] > 0:
        # Show patchification separate from image saving
        print(f"  Patchification (HDF5):  {timing['patchification']:8.3f}s")
        print(f"  Image saving (PNG):     {timing['image_saving']:8.3f}s")
    else:
        # Just show patchification
        print(f"  Patchification:         {timing['patchification']:8.3f}s")

    print(f"  {'─'*66}")
    print(f"  Total time:             {timing['total']:8.3f}s")
    print(f"{'='*70}")
    print(f"  Patches extracted:      {num_patches}")
    print(f"  Output file:            {hdf5_path}")
    if vis_paths:
        print(f"\n  Visualizations:")
        for name, path in vis_paths.items():
            print(f"    - {name}: {Path(path).name}")
    print(f"{'='*70}\n")

    return {
        'num_patches': num_patches,
        'timing': timing,
        'output_file': hdf5_path,
        'visualizations': vis_paths
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM2 WSI Patchification Pipeline\n\n"
                    "Single WSI:  --wsi_path <file> --output_dir <dir>\n"
                    "Folder ALL:  --wsi_path <folder> --all --output_dir <dir>\n"
                    "Folder N:    --wsi_path <folder> --num_samples N --output_dir <dir>",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument('--wsi_path', type=str, required=True,
                        help='Path to WSI file or folder containing WSI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (subfolders created per WSI in batch mode)')

    # Batch processing options
    parser.add_argument('--all', action='store_true',
                        help='Process all WSI files in the folder')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Random sample N WSI files from folder')

    # Patchification
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--step_size', type=int, default=256)
    parser.add_argument('--tissue_area_thresh', type=float, default=0.001,
                        help='Minimum tissue area as percentage of image (0.0-100.0)')
    parser.add_argument('--contour_fn', type=str, default='four_pt',
                        choices=['four_pt', 'four_pt_hard'])
    parser.add_argument('--white_thresh', type=int, default=15)
    parser.add_argument('--black_thresh', type=int, default=50)
    parser.add_argument('--save_image_patches', action='store_true',
                        help='Save individual patch images as PNG files')

    # SAM2 model
    parser.add_argument('--model_path', type=str,
                        default='/home/common-data/SegmentationModels/Latest_Models/sam2_bs2_inp1024_lr5e-4.pth')
    parser.add_argument('--base_checkpoint', type=str,
                        default='/home/o_a38510/Segmentation_Project/sam2.1_hiera_tiny.pt')
    parser.add_argument('--config', type=str,
                        default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Determine if processing single file or batch
    if os.path.isfile(args.wsi_path):
        # Single WSI processing
        if args.all or args.num_samples:
            print("WARNING: --all and --num_samples are ignored when processing a single file")

        results = run_sam2_patchification_pipeline(
            wsi_path=args.wsi_path,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            step_size=args.step_size,
            tissue_area_thresh=args.tissue_area_thresh,
            contour_fn=args.contour_fn,
            white_thresh=args.white_thresh,
            black_thresh=args.black_thresh,
            model_path=args.model_path,
            base_checkpoint=args.base_checkpoint,
            config=args.config,
            input_size=args.input_size,
            device=args.device,
            save_image_patches=args.save_image_patches
        )
        print("Pipeline complete!")

    elif os.path.isdir(args.wsi_path):
        # Batch processing from folder
        if not args.all and args.num_samples is None:
            print("ERROR: For folder input, specify --all or --num_samples N")
            sys.exit(1)

        # Find all WSI files
        wsi_files = find_wsi_files(args.wsi_path)

        if len(wsi_files) == 0:
            print(f"ERROR: No WSI files found in {args.wsi_path}")
            sys.exit(1)

        print(f"Found {len(wsi_files)} WSI files in {args.wsi_path}")

        # Select files to process
        if args.all:
            selected_files = wsi_files
            print(f"Processing ALL {len(selected_files)} files")
        else:
            if args.num_samples > len(wsi_files):
                print(f"WARNING: Requested {args.num_samples} samples but only {len(wsi_files)} files available")
                selected_files = wsi_files
            else:
                selected_files = random.sample(wsi_files, args.num_samples)
                print(f"Randomly selected {len(selected_files)} files from {len(wsi_files)}")

        # Run batch processing
        batch_results = run_batch_processing(
            wsi_files=selected_files,
            output_root_dir=args.output_dir,
            patch_size=args.patch_size,
            step_size=args.step_size,
            tissue_area_thresh=args.tissue_area_thresh,
            contour_fn=args.contour_fn,
            white_thresh=args.white_thresh,
            black_thresh=args.black_thresh,
            model_path=args.model_path,
            base_checkpoint=args.base_checkpoint,
            config=args.config,
            input_size=args.input_size,
            device=args.device,
            save_image_patches=args.save_image_patches
        )
        print("Batch processing complete!")

    else:
        print(f"ERROR: Path not found: {args.wsi_path}")
        sys.exit(1)