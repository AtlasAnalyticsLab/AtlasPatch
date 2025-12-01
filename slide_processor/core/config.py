from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _ensure_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _ensure_non_negative(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _ensure_fraction(value: float, name: str) -> float:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def _validate_device(device: str) -> str:
    dev = device.strip().lower()
    if dev == "cpu":
        return dev
    if dev == "cuda" or dev.startswith("cuda:"):
        if dev.startswith("cuda:"):
            suffix = dev.split("cuda:", 1)[1]
            if suffix and not suffix.isdigit():
                raise ValueError(
                    f"Invalid CUDA device specification '{device}'. Use 'cuda' or 'cuda:<index>'."
                )
        return dev
    raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:<index>', got {device}")


@dataclass
class SegmentationConfig:
    checkpoint_path: Path | None
    config_path: Path
    device: str = "cuda"
    thumbnail_power: float = 1.25
    thumbnail_max: int = 1024
    batch_size: int = 1
    mask_threshold: float = 0.0

    def validated(self) -> SegmentationConfig:
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"SAM2 config not found: {self.config_path}")
        self.device = _validate_device(str(self.device))
        _ensure_positive(self.thumbnail_max, "thumbnail_max")
        _ensure_positive(self.batch_size, "segmentation batch_size")
        return self


@dataclass
class ExtractionConfig:
    patch_size: int
    target_magnification: int
    step_size: int | None = None
    workers: int | None = None
    max_open_slides: int | None = None
    tissue_threshold: float = 0.01
    white_threshold: int = 15
    black_threshold: int = 50
    fast_mode: bool = True
    write_batch: int = 8192

    def validated(self) -> ExtractionConfig:
        _ensure_positive(self.patch_size, "patch_size")
        _ensure_positive(self.target_magnification, "target_magnification")
        if self.step_size is None:
            self.step_size = self.patch_size
        _ensure_positive(self.step_size, "step_size")
        _ensure_fraction(self.tissue_threshold, "tissue_threshold")
        _ensure_positive(self.white_threshold, "white_threshold")
        _ensure_positive(self.black_threshold, "black_threshold")
        _ensure_positive(self.write_batch, "write_batch")
        if self.workers is not None:
            _ensure_positive(self.workers, "workers")
        if self.max_open_slides is None:
            self.max_open_slides = 200
        _ensure_positive(self.max_open_slides, "max_open_slides")
        return self


@dataclass
class FeatureExtractionConfig:
    extractors: list[str]
    batch_size: int = 32
    device: str = "cuda"
    num_workers: int = 4
    precision: str = "float32"

    def validated(self) -> FeatureExtractionConfig:
        if not self.extractors:
            raise ValueError("At least one feature extractor must be provided.")
        _ensure_positive(self.batch_size, "feature batch_size")
        _ensure_non_negative(self.num_workers, "feature num_workers")
        self.device = _validate_device(str(self.device))
        allowed_prec = {"float32", "float16", "bfloat16"}
        prec = str(self.precision).lower()
        if prec not in allowed_prec:
            raise ValueError(
                f"precision must be one of {sorted(allowed_prec)}, got {self.precision}"
            )
        self.precision = prec
        return self


@dataclass
class OutputConfig:
    output_root: Path
    save_images: bool = False
    visualize_grids: bool = False
    visualize_mask: bool = False
    visualize_contours: bool = False
    skip_existing: bool = True

    def validated(self) -> OutputConfig:
        self.output_root.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class ProcessingConfig:
    input_path: Path
    recursive: bool = False
    mpp_csv: Path | None = None

    def validated(self) -> ProcessingConfig:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        if self.mpp_csv is not None and not self.mpp_csv.exists():
            raise FileNotFoundError(f"MPP CSV not found: {self.mpp_csv}")
        return self


@dataclass
class VisualizationConfig:
    thumbnail_size: int = 1024

    def validated(self) -> VisualizationConfig:
        _ensure_positive(self.thumbnail_size, "thumbnail_size")
        return self


@dataclass
class AppConfig:
    processing: ProcessingConfig
    segmentation: SegmentationConfig
    extraction: ExtractionConfig
    output: OutputConfig
    features: FeatureExtractionConfig | None = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    device: str = "cuda"

    def validated(self) -> AppConfig:
        self.processing = self.processing.validated()
        self.segmentation = self.segmentation.validated()
        self.extraction = self.extraction.validated()
        self.output = self.output.validated()
        if self.features is not None:
            self.features = self.features.validated()
        self.visualization = self.visualization.validated()
        return self
