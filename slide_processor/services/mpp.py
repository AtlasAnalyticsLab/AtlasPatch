from __future__ import annotations

from pathlib import Path

from slide_processor.core.models import Slide
from slide_processor.utils.params import get_mpp_for_wsi, load_mpp_csv


class CSVMPPResolver:
    """Resolve MPP values from an optional CSV mapping."""

    def __init__(self, csv_path: Path | None) -> None:
        self._mpp_map: dict[str, float] | None = None
        if csv_path is not None:
            self._mpp_map = load_mpp_csv(str(csv_path))

    def resolve(self, slide: Slide) -> float | None:
        return get_mpp_for_wsi(str(slide.path), self._mpp_map)
