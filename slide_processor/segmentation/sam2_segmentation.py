from pathlib import Path
from typing import Union

import torch
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
            predictor.model.to(self.device).eval()
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            predictor.model.load_state_dict(checkpoint["model"], strict=True)
            return predictor

        except Exception as e:
            raise RuntimeError(f"Failed to build SAM2ImagePredictor model: {e}") from e

    def predict_image(self):
        pass

    def predict_batch(self):
        pass

    def __repr__(self) -> str:
        return (
            f"SAM2SegmentationModel("
            f"device={self.device}, "
            f"checkpoint={self.checkpoint_path.name})"
        )
