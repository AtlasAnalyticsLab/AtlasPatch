import logging
from pathlib import Path

import click

logger = logging.getLogger("slide_processor.utils")


def validate_path(ctx, param, value):
    """Validate that file/directory path exists."""
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Path does not exist: {value}")
    return str(path.absolute())


def validate_positive_int(ctx, param, value):
    """Validate that value is a positive integer."""
    if value is not None and value <= 0:
        raise click.BadParameter(f"{param.name} must be positive, got {value}")
    return value


def get_wsi_files(path: str, *, recursive: bool = False) -> list[str]:
    """Get list of WSI files from path (file or directory).

    Supported formats:
    - OpenSlide: .svs, .tif, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .dcm
    - Image: .png, .jpg, .jpeg, .bmp, .webp, .gif
    """
    supported_exts = {
        ".svs",
        ".tif",
        ".tiff",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".bif",
        ".biff",
        ".dcm",
        ".dicom",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".gif",
    }
    path_obj = Path(path)

    if path_obj.is_file():
        if path_obj.suffix.lower() not in supported_exts:
            logger.warning(f"File may not be a supported WSI format: {path_obj.name}")
        return [str(path_obj)]

    # Directory: collect all supported files
    files_set: set[Path] = set()
    if recursive:
        # Recursive search using rglob
        for ext in supported_exts:
            files_set.update(path_obj.rglob(f"*{ext}"))
            files_set.update(path_obj.rglob(f"*{ext.upper()}"))
    else:
        # Non-recursive (current directory only)
        for ext in supported_exts:
            files_set.update(path_obj.glob(f"*{ext}"))
            files_set.update(path_obj.glob(f"*{ext.upper()}"))

    files = sorted(files_set)
    if not files:
        raise click.ClickException(
            f"No WSI files found in directory: {path}\n"
            f"Supported formats: SVS, TIF, TIFF, NDPI, PNG, JPG, etc."
        )

    return [str(f) for f in files]
