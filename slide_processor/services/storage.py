from __future__ import annotations

import concurrent.futures as _fut
import os
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from slide_processor.utils.h5 import H5AppendWriter


class H5PatchWriter:
    """Handles HDF5 writing (and optional image export) for patch coordinates."""

    def __init__(
        self,
        *,
        chunk_rows: int,
        patch_size: int,
        patch_size_level0: int,
        level0_mag: int,
        target_mag: int,
        level0_wh: tuple[int, int],
        overlap: int,
        slide_stem: str,
        wsi_path: str,
    ) -> None:
        self.chunk_rows = max(1, int(chunk_rows))
        self.patch_size = int(patch_size)
        self.patch_size_level0 = int(patch_size_level0)
        self.level0_mag = int(level0_mag)
        self.target_mag = int(target_mag)
        self.level0_wh = level0_wh
        self.overlap = int(overlap)
        self.slide_stem = slide_stem
        self.wsi_path = wsi_path

    def _seed_writer(self, output_path: Path) -> H5AppendWriter:
        writer = H5AppendWriter(str(output_path), chunk_rows=self.chunk_rows)
        empty_coords = np.empty((0, 2), dtype=np.int32)
        empty_coords_ext = np.empty((0, 5), dtype=np.int32)
        level0_width, level0_height = self.level0_wh
        coords_metadata: dict[str, int | str] = {
            "description": "(x, y) coordinates at level 0",
            "patch_size": self.patch_size,
            "patch_size_level0": self.patch_size_level0,
            "level0_magnification": self.level0_mag,
            "target_magnification": self.target_mag,
            "overlap": self.overlap,
            "name": self.slide_stem,
            "savetodir": str(output_path.resolve().parent),
            "level0_width": int(level0_width),
            "level0_height": int(level0_height),
        }
        dset_attrs: dict[str, dict[str, int | str]] = {
            "coords": coords_metadata,
            "coords_ext": {"description": "(x, y, w, h, level) at extraction"},
        }
        writer.append(
            {"coords": empty_coords, "coords_ext": empty_coords_ext}, attributes=dset_attrs
        )
        writer.update_file_attrs(
            {
                "patch_size": coords_metadata["patch_size"],
                "patch_size_level0": coords_metadata["patch_size_level0"],
                "level0_magnification": coords_metadata["level0_magnification"],
                "target_magnification": coords_metadata["target_magnification"],
                "overlap": coords_metadata["overlap"],
                "level0_width": coords_metadata["level0_width"],
                "level0_height": coords_metadata["level0_height"],
                "wsi_path": self.wsi_path,
            }
        )
        return writer

    def write_coords(
        self,
        output_path: Path,
        entries: Iterable[tuple[int, int, int, int, int, np.ndarray | None]],
        *,
        batch: int,
        collect_coords: bool = False,
    ) -> tuple[int, np.ndarray | None]:
        writer = self._seed_writer(output_path)
        total = 0
        buf_xy: list[tuple[int, int]] = []
        buf_ext: list[tuple[int, int, int, int, int]] = []
        coords_viz: list[tuple[int, int]] | None = [] if collect_coords else None

        try:
            for x, y, rw, rh, lv, _ in entries:
                buf_xy.append((x, y))
                buf_ext.append((x, y, int(rw), int(rh), int(lv)))
                if coords_viz is not None:
                    coords_viz.append((x, y))
                if len(buf_xy) >= batch:
                    coords = np.asarray(buf_xy, dtype=np.int32)
                    coords_ext = np.asarray(buf_ext, dtype=np.int32)
                    writer.append({"coords": coords, "coords_ext": coords_ext})
                    total += int(coords.shape[0])
                    buf_xy.clear()
                    buf_ext.clear()

            if buf_xy:
                coords = np.asarray(buf_xy, dtype=np.int32)
                coords_ext = np.asarray(buf_ext, dtype=np.int32)
                writer.append({"coords": coords, "coords_ext": coords_ext})
                total += int(coords.shape[0])

            writer.update_file_attrs({"num_patches": int(total)})
            writer.close()
        except Exception:
            try:
                writer.abort()
            finally:
                pass
            raise

        coords_arr = np.asarray(coords_viz, dtype=np.int32) if coords_viz is not None else None
        return int(total), coords_arr

    def write_coords_and_images(
        self,
        output_path: Path,
        entries: Iterable[tuple[int, int, int, int, int, np.ndarray | None]],
        image_dir: Path,
        *,
        batch: int,
        collect_coords: bool,
    ) -> tuple[int, np.ndarray | None]:
        writer = self._seed_writer(output_path)
        stem = self.slide_stem

        max_workers = max(2, min(8, os.cpu_count() or 4))
        max_pending = max_workers * 4
        futures: deque[_fut.Future[None]] = deque()
        executor = _fut.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="patch-img")

        def _submit_image(x: int, y: int, patch_arr: np.ndarray) -> None:
            out_name = f"{stem}_x{x}_y{y}.png"
            fut = executor.submit(self._save_patch_image, patch_arr.copy(), image_dir / out_name)
            futures.append(fut)
            if len(futures) >= max_pending:
                futures.popleft().result()

        try:
            total, coords_viz = self._write_coords_stream(
                writer=writer,
                entries=entries,
                batch=batch,
                on_patch=_submit_image,
                collect_coords=collect_coords,
            )
            while futures:
                futures.popleft().result()
            return total, coords_viz
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            while futures:
                try:
                    futures.popleft().result()
                except Exception:
                    pass

    @staticmethod
    def _save_patch_image(patch_arr: np.ndarray, out_path: Path) -> None:
        Image.fromarray(patch_arr).save(str(out_path))

    def _write_coords_stream(
        self,
        *,
        writer: H5AppendWriter,
        entries: Iterable[tuple[int, int, int, int, int, np.ndarray | None]],
        batch: int,
        on_patch: Callable[[int, int, np.ndarray], None] | None = None,
        collect_coords: bool = False,
    ) -> tuple[int, np.ndarray | None]:
        total = 0
        buf_xy: list[tuple[int, int]] = []
        buf_ext: list[tuple[int, int, int, int, int]] = []
        coords_viz: list[tuple[int, int]] | None = [] if collect_coords else None

        try:
            for x, y, rw, rh, lv, patch in entries:
                buf_xy.append((x, y))
                buf_ext.append((x, y, int(rw), int(rh), int(lv)))
                if coords_viz is not None:
                    coords_viz.append((x, y))
                if len(buf_xy) >= batch:
                    coords = np.asarray(buf_xy, dtype=np.int32)
                    coords_ext = np.asarray(buf_ext, dtype=np.int32)
                    writer.append({"coords": coords, "coords_ext": coords_ext})
                    total += int(coords.shape[0])
                    buf_xy.clear()
                    buf_ext.clear()

                if on_patch is not None and patch is not None:
                    on_patch(int(x), int(y), patch)

            if buf_xy:
                coords = np.asarray(buf_xy, dtype=np.int32)
                coords_ext = np.asarray(buf_ext, dtype=np.int32)
                writer.append({"coords": coords, "coords_ext": coords_ext})
                total += int(coords.shape[0])

            writer.update_file_attrs({"num_patches": int(total)})
            writer.close()
        except Exception:
            try:
                writer.abort()
            finally:
                pass
            raise

        coords_arr = np.asarray(coords_viz, dtype=np.int32) if coords_viz is not None else None
        return int(total), coords_arr
