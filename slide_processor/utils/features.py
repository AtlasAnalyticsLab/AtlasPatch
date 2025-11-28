from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import click
import h5py


def parse_feature_list(raw: str, *, choices: list[str]) -> list[str]:
    """Normalize, validate, and deduplicate a feature extractor list."""
    parts = [p.strip().lower() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        raise click.BadParameter("At least one feature extractor name is required.")
    unknown = [p for p in parts if p not in choices]
    if unknown:
        raise click.BadParameter(
            f"Unknown extractor(s): {', '.join(unknown)}. Available: {', '.join(choices)}"
        )
    seen = set()
    dupes = []
    unique_parts: list[str] = []
    for p in parts:
        if p in seen:
            dupes.append(p)
            continue
        seen.add(p)
        unique_parts.append(p)
    if dupes:
        raise click.BadParameter(
            f"Duplicate extractor(s) specified: {', '.join(sorted(set(dupes)))}. "
            "Provide each extractor at most once."
        )
    return unique_parts


def parse_feature_sets_attr(raw: Any) -> dict[str, dict[str, Any]]:
    """Parse the feature_sets file attribute into a dict."""
    if isinstance(raw, (bytes, str)):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


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
