from __future__ import annotations

from pathlib import Path
from typing import Sequence

import click
import h5py


def parse_named_list(raw: str, *, choices: list[str], item_label: str) -> list[str]:
    """Normalize, validate, and deduplicate a list of named registry entries."""
    parts = [p.strip().lower() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        raise click.BadParameter(f"At least one {item_label} name is required.")
    unknown = [p for p in parts if p not in choices]
    if unknown:
        raise click.BadParameter(
            f"Unknown {item_label}(s): {', '.join(unknown)}. Available: {', '.join(choices)}"
        )
    seen: set[str] = set()
    dupes: list[str] = []
    unique_parts: list[str] = []
    for p in parts:
        if p in seen:
            dupes.append(p)
            continue
        seen.add(p)
        unique_parts.append(p)
    if dupes:
        raise click.BadParameter(
            f"Duplicate {item_label}(s) specified: {', '.join(sorted(set(dupes)))}. "
            f"Provide each {item_label} once."
        )
    return unique_parts


def get_existing_features(h5_path: str | Path, *, expected_total: int | None = None) -> set[str]:
    """Return feature dataset names already present in an H5 file.

    When expected_total is provided, only datasets whose first dimension matches
    expected_total are returned (to skip partial/incomplete embeddings).
    """
    path = Path(h5_path)
    try:
        with h5py.File(path, "r") as f:
            if "features" not in f:
                return set()
            names: set[str] = set()
            for name, ds in f["features"].items():
                if expected_total is not None:
                    try:
                        if int(ds.shape[0]) != int(expected_total):
                            continue
                    except Exception:
                        continue
                names.add(str(name).lower())
            return names
    except FileNotFoundError:
        return set()
    except Exception:
        # If the file is unreadable, treat as missing so it can be regenerated.
        return set()


def missing_features(
    h5_path: str | Path, required: Sequence[str], *, expected_total: int | None = None
) -> list[str]:
    """Return the list of required feature names that are absent or incomplete in the H5."""
    existing = get_existing_features(h5_path, expected_total=expected_total)
    required_norm = [r.lower() for r in required]
    return [name for name in required_norm if name not in existing]
