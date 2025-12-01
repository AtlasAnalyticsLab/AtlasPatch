from __future__ import annotations

import concurrent.futures as _fut
import os
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image

from atlas_patch.utils.h5 import H5AppendWriter


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
        extra_file_attrs: Mapping[str, Any] | None = None,
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
        self.extra_file_attrs = dict(extra_file_attrs) if extra_file_attrs else {}
        self._passport_dtype = np.dtype("S128")

    def _seed_writer(self, output_path: Path) -> H5AppendWriter:
        writer = H5AppendWriter(str(output_path), chunk_rows=self.chunk_rows)
        empty_coords = np.empty((0, 5), dtype=np.int32)
        empty_passports = np.empty((0,), dtype=self._passport_dtype)
        level0_width, level0_height = self.level0_wh
        writer.append({"coords": empty_coords, "passports": empty_passports})
        file_attrs = {
            "patch_size": self.patch_size,
            "patch_size_level0": self.patch_size_level0,
            "level0_magnification": self.level0_mag,
            "target_magnification": self.target_mag,
            "overlap": self.overlap,
            "level0_width": int(level0_width),
            "level0_height": int(level0_height),
            "wsi_path": self.wsi_path,
            "passport_format": "{stem}__x{X}_y{Y}_rw{RW}_rh{RH}_lv{LV}_mag{MAG}_tmag{TMAG}",
            "passport_version": 1,
        }
        if self.extra_file_attrs:
            file_attrs.update(self.extra_file_attrs)
        writer.update_file_attrs(file_attrs)
        return writer

    @staticmethod
    def _schedule_image_save(
        executor: _fut.ThreadPoolExecutor | None,
        futures: deque[_fut.Future[None]],
        max_pending: int | None,
        stem: str,
        image_dir: Path | None,
        x: int,
        y: int,
        patch_arr: np.ndarray,
    ) -> None:
        if executor is None or max_pending is None or image_dir is None:
            return
        out_name = f"{stem}_x{x}_y{y}.png"
        fut = executor.submit(
            H5PatchWriter._save_patch_image, patch_arr.copy(), image_dir / out_name
        )
        futures.append(fut)
        if len(futures) >= max_pending:
            futures.popleft().result()

    @staticmethod
    def _drain_futures(
        executor: _fut.ThreadPoolExecutor | None, futures: deque[_fut.Future[None]]
    ) -> None:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        while futures:
            try:
                futures.popleft().result()
            except Exception:
                pass

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
        buf_coords: list[tuple[int, int, int, int, int]] = []
        coords_viz: list[tuple[int, int]] | None = [] if collect_coords else None
        buf_passports: list[str] = []

        try:
            for x, y, rw, rh, lv, _ in entries:
                buf_coords.append((int(x), int(y), int(rw), int(rh), int(lv)))
                buf_passports.append(self._passport(int(x), int(y), int(rw), int(rh), int(lv)))
                if coords_viz is not None:
                    coords_viz.append((int(x), int(y)))
                if len(buf_coords) >= batch:
                    coords = np.asarray(buf_coords, dtype=np.int32)
                    passports = np.asarray(buf_passports, dtype=self._passport_dtype)
                    writer.append({"coords": coords, "passports": passports})
                    total += int(coords.shape[0])
                    buf_coords.clear()
                    buf_passports.clear()

            if buf_coords:
                coords = np.asarray(buf_coords, dtype=np.int32)
                passports = np.asarray(buf_passports, dtype=self._passport_dtype)
                writer.append({"coords": coords, "passports": passports})
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
                on_patch=lambda x, y, patch: self._schedule_image_save(
                    executor=executor,
                    futures=futures,
                    max_pending=max_pending,
                    stem=stem,
                    image_dir=image_dir,
                    x=x,
                    y=y,
                    patch_arr=patch,
                ),
                collect_coords=collect_coords,
            )
            while futures:
                futures.popleft().result()
            return total, coords_viz
        finally:
            self._drain_futures(executor, futures)

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
        buf_coords: list[tuple[int, int, int, int, int]] = []
        coords_viz: list[tuple[int, int]] | None = [] if collect_coords else None
        buf_passports: list[str] = []

        try:
            for x, y, rw, rh, lv, patch in entries:
                buf_coords.append((int(x), int(y), int(rw), int(rh), int(lv)))
                buf_passports.append(self._passport(int(x), int(y), int(rw), int(rh), int(lv)))
                if coords_viz is not None:
                    coords_viz.append((int(x), int(y)))
                if len(buf_coords) >= batch:
                    coords = np.asarray(buf_coords, dtype=np.int32)
                    passports = np.asarray(buf_passports, dtype=self._passport_dtype)
                    writer.append({"coords": coords, "passports": passports})
                    total += int(coords.shape[0])
                    buf_coords.clear()
                    buf_passports.clear()

                if on_patch is not None and patch is not None:
                    on_patch(int(x), int(y), patch)

            if buf_coords:
                coords = np.asarray(buf_coords, dtype=np.int32)
                passports = np.asarray(buf_passports, dtype=self._passport_dtype)
                writer.append({"coords": coords, "passports": passports})
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

    def append_features(
        self,
        *,
        output_path: Path,
        entries: Iterable[tuple[int, int, int, int, int, np.ndarray | None]],
        feature_name: str,
        feature_fn: Callable[[Sequence[np.ndarray]], np.ndarray],
        feature_attrs: Mapping[str, int | str],
        feature_batch: int,
        expected_total: int | None = None,
    ) -> int:
        """Append a single feature dataset to an existing H5 using a patch iterator."""
        batch_size = max(1, int(feature_batch))
        total_written = 0
        dataset = None
        tmp_name = f"__tmp_{feature_name}"

        with h5py.File(output_path, "a") as f:
            grp = f.require_group("features")
            if feature_name in grp:
                raise ValueError(
                    f"Feature dataset '{feature_name}' already exists in {output_path}."
                )
            if tmp_name in grp:
                del grp[tmp_name]
            buf: list[np.ndarray] = []

            moved = False
            try:
                for _x, _y, _rw, _rh, _lv, patch in entries:
                    if patch is None:
                        continue
                    buf.append(patch)
                    if len(buf) >= batch_size:
                        dataset, total_written = self._append_feature_batch(
                            grp=grp,
                            dataset=dataset,
                            dataset_name=tmp_name,
                            feature_name=feature_name,
                            feature_attrs=feature_attrs,
                            batch_size=batch_size,
                            buf=buf,
                            feature_fn=feature_fn,
                            total_written=total_written,
                        )

                if buf:
                    dataset, total_written = self._append_feature_batch(
                        grp=grp,
                        dataset=dataset,
                        dataset_name=tmp_name,
                        feature_name=feature_name,
                        feature_attrs=feature_attrs,
                        batch_size=batch_size,
                        buf=buf,
                        feature_fn=feature_fn,
                        total_written=total_written,
                    )

                if dataset is None:
                    emb_dim = int(feature_attrs.get("embedding_dim", 0))
                    if emb_dim <= 0:
                        raise ValueError(
                            f"Feature extractor '{feature_name}' missing valid embedding_dim to create dataset."
                        )
                    dataset = grp.create_dataset(
                        tmp_name,
                        shape=(0, emb_dim),
                        maxshape=(None, emb_dim),
                        chunks=(batch_size, emb_dim),
                        dtype=np.float32,
                    )

                if expected_total is not None and total_written != int(expected_total):
                    raise ValueError(
                        f"Feature rows written ({total_written}) do not match expected coords ({expected_total})"
                    )

                grp.move(tmp_name, feature_name)
                moved = True
            except Exception:
                if tmp_name in grp:
                    del grp[tmp_name]
                elif moved and feature_name in grp:
                    del grp[feature_name]
                raise

        return int(total_written)

    @staticmethod
    def _append_feature_batch(
        *,
        grp,
        dataset,
        dataset_name: str,
        feature_name: str,
        feature_attrs: Mapping[str, int | str],
        batch_size: int,
        buf: list[np.ndarray],
        feature_fn: Callable[[Sequence[np.ndarray]], np.ndarray],
        total_written: int,
    ):
        if not buf:
            return dataset, total_written

        feats_arr = feature_fn(buf)
        arr = np.asarray(feats_arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"Feature extractor '{feature_name}' must return a 2D array, got shape {arr.shape}"
            )
        if arr.shape[0] != len(buf):
            raise ValueError(
                f"Feature extractor '{feature_name}' returned {arr.shape[0]} rows for batch of size {len(buf)}."
            )
        if dataset is None:
            feat_dim = int(arr.shape[1])
            dataset = grp.create_dataset(
                dataset_name,
                shape=(0, feat_dim),
                maxshape=(None, feat_dim),
                chunks=(batch_size, feat_dim),
                dtype=np.float32,
            )
        elif dataset.shape[1] != arr.shape[1]:
            raise ValueError(
                f"Feature dim mismatch for '{feature_name}': existing {dataset.shape[1]}, new {arr.shape[1]}"
            )

        start = int(total_written)
        end = start + arr.shape[0]
        dataset.resize((end, dataset.shape[1]))
        dataset[start:end, :] = arr
        total_written = end
        buf.clear()
        return dataset, total_written

    def _passport(self, x: int, y: int, rw: int, rh: int, lv: int) -> str:
        mag_val = self.level0_mag if self.level0_mag else "na"
        tgt_val = self.target_mag if self.target_mag else "na"
        return f"{self.slide_stem}__x{x}_y{y}_rw{rw}_rh{rh}_lv{lv}_mag{mag_val}_tmag{tgt_val}"
