import logging
from pathlib import Path
from typing import Dict, Optional

import click

logger = logging.getLogger("atlas_patch.utils")


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


def load_mpp_csv(csv_path: str) -> Dict[str, float]:
    """Load MPP values from CSV file.

    Expected CSV format with columns: wsi, mpp
    - wsi: filename (with or without full path)
    - mpp: microns per pixel value (float)

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing WSI names and their MPP values.

    Returns
    -------
    dict
        Mapping of WSI stem to MPP value.

    Raises
    ------
    click.ClickException
        If CSV file is invalid or missing required columns.
    """
    try:
        import csv
    except ImportError as e:
        raise click.ClickException(f"CSV module required: {e}") from e

    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise click.ClickException(f"MPP CSV file not found: {csv_path}")

    mpp_dict: Dict[str, float] = {}

    try:
        with open(csv_path_obj, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate headers
            if (
                reader.fieldnames is None
                or "wsi" not in reader.fieldnames
                or "mpp" not in reader.fieldnames
            ):
                raise click.ClickException(
                    f"CSV must contain 'wsi' and 'mpp' columns. Found: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
                wsi_name = row.get("wsi", "").strip()
                mpp_str = row.get("mpp", "").strip()

                if not wsi_name:
                    continue

                if not mpp_str:
                    continue

                try:
                    mpp_value = float(mpp_str)
                    if mpp_value <= 0:
                        logger.warning(
                            f"Row {row_num}: MPP value must be positive for {wsi_name}, got {mpp_value}, skipping"
                        )
                        continue
                except ValueError:
                    logger.warning(
                        f"Row {row_num}: Invalid MPP value '{mpp_str}' for {wsi_name}, skipping"
                    )
                    continue

                # Use stem (filename without extension) as key for flexibility
                wsi_stem = Path(wsi_name).stem
                mpp_dict[wsi_stem] = mpp_value

        if not mpp_dict:
            raise click.ClickException(f"No valid MPP entries found in CSV: {csv_path}")

        return mpp_dict

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error reading CSV file: {e}") from e


def get_mpp_for_wsi(wsi_path: str, mpp_dict: Optional[Dict[str, float]]) -> Optional[float]:
    """Get MPP value for a specific WSI from the loaded dictionary.

    Parameters
    ----------
    wsi_path : str
        Path to WSI file.
    mpp_dict : dict or None
        Dictionary mapping WSI stems to MPP values (from load_mpp_csv).

    Returns
    -------
    float or None
        MPP value if found, None otherwise.
    """
    if mpp_dict is None:
        return None

    wsi_stem = Path(wsi_path).stem
    mpp = mpp_dict.get(wsi_stem)

    return mpp
