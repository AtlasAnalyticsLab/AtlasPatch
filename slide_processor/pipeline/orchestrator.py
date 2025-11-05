from __future__ import annotations

import concurrent.futures as _fut
import logging
import multiprocessing as _mp
from pathlib import Path

import numpy as _np

from slide_processor.pipeline.patchify import PatchifyParams, SegmentParams, segment_and_patchify
from slide_processor.utils.params import get_mpp_for_wsi
from slide_processor.utils.progress import ProgressReporter
from slide_processor.wsi import WSIFactory

logger = logging.getLogger("slide_processor.pipeline")


def pack_mask(mask_arr):
    """Pack a predicted mask to compact uint8 for IPC."""
    if mask_arr is None:
        return None
    if isinstance(mask_arr, _np.ndarray):
        return (mask_arr > 0.5).astype(_np.uint8) if mask_arr.dtype != _np.uint8 else mask_arr
    return None


def build_wsi_task(
    *,
    file_path: str,
    mask_arr,
    output_dir: Path,
    patch_params: PatchifyParams,
    effective_step_size: int,
    save_images: bool,
    fast_mode: bool,
    write_batch: int,
    visualize_grids: bool,
    visualize_mask: bool,
    visualize_contours: bool,
    device: str,
    patch_size: int,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    target_mag: int,
    mpp: float | None = None,
):
    """Build a serializable task payload for per-WSI worker processing."""
    return {
        "wsi_path": file_path,
        "mask": pack_mask(mask_arr),
        "output_dir": str(output_dir),
        "mpp": mpp,
        "patch": {
            "patch_size": int(patch_params.patch_size),
            "step_size": int(effective_step_size),
            "target_mag": int(patch_params.target_magnification),
            "tissue_thresh": float(patch_params.tissue_area_thresh),
            "white_thresh": int(patch_params.white_thresh),
            "black_thresh": int(patch_params.black_thresh),
        },
        "opts": {
            "save_images": bool(save_images),
            "fast_mode": bool(fast_mode),
            "write_batch": int(write_batch),
            "visualize_grids": bool(visualize_grids),
            "visualize_mask": bool(visualize_mask),
            "visualize_contours": bool(visualize_contours),
            "device": device,
            "patch_size": int(patch_size),
            "step_size": int(effective_step_size),
            "tissue_thresh": float(tissue_thresh),
            "white_thresh": int(white_thresh),
            "black_thresh": int(black_thresh),
            "target_mag": int(target_mag),
        },
    }


def run_wsi_tasks(
    tasks: list[dict], *, max_workers: int, reporter: ProgressReporter | None, pbar, verbose: bool
) -> tuple[int, int]:
    """Execute per-WSI tasks in a process pool and aggregate results."""
    successful, failed = 0, 0
    ctx = _mp.get_context("spawn")
    executor = _fut.ProcessPoolExecutor(max_workers=max(1, int(max_workers)), mp_context=ctx)

    with executor as ex:
        fut_map = {ex.submit(process_wsi_worker, t): t["wsi_path"] for t in tasks}
        for fut in _fut.as_completed(fut_map):
            fpath = fut_map[fut]
            try:
                ok, msg = fut.result()
            except Exception as e:
                ok, msg = False, str(e)
            if ok:
                successful += 1
                if verbose:
                    logger.info(f"Saved patches to: {msg}")
                elif reporter:
                    reporter.update(success=True)
            else:
                failed += 1
                if verbose:
                    logger.error(f"Failed to process {Path(fpath).name}: {msg}")
                elif reporter:
                    reporter.update(success=False)
            if reporter and pbar:
                reporter.update_progress_bar(pbar)
    return successful, failed


def _visualize_result(
    result_h5: str,
    wsi,
    output_path: Path,
    patch_size: int,
    step_size: int,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    fast_mode: bool,
    save_images: bool,
    target_mag: int,
    *,
    mask,
    do_grids: bool,
    do_mask: bool,
    do_contours: bool,
) -> None:
    """Visualize outputs on thumbnail based on requested types."""
    from slide_processor.visualization import (
        visualize_contours_on_thumbnail,
        visualize_mask_on_thumbnail,
        visualize_patches_on_thumbnail,
    )

    vis_output_dir = output_path / "visualization"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    cli_args_dict = {
        "patch_size": patch_size,
        "step_size": step_size,
        "thumbnail_size": 1024,
        "device": device,
        "tissue_thresh": tissue_thresh,
        "white_thresh": white_thresh,
        "black_thresh": black_thresh,
        "fast_mode": fast_mode,
        "save_images": save_images,
        "target_mag": target_mag,
    }
    try:
        if do_grids:
            visualize_patches_on_thumbnail(
                hdf5_path=result_h5,
                wsi=wsi,
                output_dir=str(vis_output_dir),
                cli_args=cli_args_dict,
            )
        if do_mask and mask is not None:
            visualize_mask_on_thumbnail(
                mask=mask,
                wsi=wsi,
                output_dir=str(vis_output_dir),
            )
        if do_contours and mask is not None:
            # Compute contours once for visualization
            from slide_processor.utils.contours import mask_to_contours as _mask_to_contours
            from slide_processor.utils.contours import scale_contours as _scale_contours

            # mask is in thumbnail space; scale contours to level-0
            W0, H0 = wsi.get_size(lv=0)
            mh, mw = mask.shape[:2]
            sx = W0 / float(mw)
            sy = H0 / float(mh)
            tcs_t, hcs_t = _mask_to_contours(mask, tissue_area_thresh=float(tissue_thresh))
            tcs = _scale_contours(tcs_t, sx, sy)
            hcs = [_scale_contours(hs, sx, sy) for hs in hcs_t]

            visualize_contours_on_thumbnail(
                tissue_contours=tcs,
                holes_contours=hcs,
                wsi=wsi,
                output_dir=str(vis_output_dir),
            )
    except Exception as e:
        logger.warning(f"Visualization failed for {Path(wsi.path).name}: {e}")


def process_wsi_worker(task: dict) -> tuple[bool, str]:
    """Worker: perform patchification only (segmentation already done in main process).
    Returns (ok, message). On success, message is HDF5 path; on error, message is error str.
    """
    try:
        from pathlib import Path as _Path

        import numpy as _np

        from slide_processor.patch_extractor.patch_extractor import (
            PatchExtractor as _PatchExtractor,
        )
        from slide_processor.utils.contours import mask_to_contours as _mask_to_contours
        from slide_processor.utils.contours import scale_contours as _scale_contours
        from slide_processor.visualization import visualize_patches_on_thumbnail as _viz
        from slide_processor.wsi import WSIFactory as _WSIFactory

        # Load WSI for patchification only
        mpp_value = task.get("mpp")
        wsi = _WSIFactory.load(task["wsi_path"], mpp=mpp_value)
        try:
            # Get mask from task (already computed in main process)
            mask = task["mask"]
            if mask is not None and isinstance(mask, _np.ndarray) and mask.dtype != _np.uint8:
                mask = (mask > 0.5).astype(_np.uint8)

            # Setup patchification dimensions
            W, H = wsi.get_size(lv=0)
            ht, wt = mask.shape[:2]

            # Extract contours from pre-computed mask
            tissue_contours_t, holes_contours_t = _mask_to_contours(
                mask, tissue_area_thresh=float(task["patch"]["tissue_thresh"])
            )

            # Scale contours to level 0
            sx = W / float(wt)
            sy = H / float(ht)
            tissue_contours = _scale_contours(tissue_contours_t, sx, sy)
            holes_contours = [_scale_contours(hs, sx, sy) for hs in holes_contours_t]

            # Create patch extractor
            extractor = _PatchExtractor(
                patch_size=int(task["patch"]["patch_size"]),
                step_size=int(task["patch"]["step_size"]),
                target_mag=int(task["patch"]["target_mag"]),
                white_thresh=int(task["patch"]["white_thresh"]),
                black_thresh=int(task["patch"]["black_thresh"]),
            )

            # Setup output paths
            stem = _Path(wsi.path).stem
            patches_root = _Path(task["output_dir"]) / "patches"
            patches_root.mkdir(parents=True, exist_ok=True)
            out_h5 = str(patches_root / f"{stem}.h5")

            img_dir = (
                str(_Path(task["output_dir"]) / "images" / stem)
                if bool(task["opts"]["save_images"])
                else None
            )

            # Extract patches to HDF5
            result_path = extractor.extract_to_h5(
                wsi,
                tissue_contours,
                holes_contours,
                out_h5,
                image_output_dir=img_dir,
                fast_mode=bool(task["opts"]["fast_mode"]),
                batch=int(task["opts"]["write_batch"]),
            )

            if not result_path:
                return False, "No patches extracted"

            # Visualize if requested
            if (
                bool(task["opts"].get("visualize_grids"))
                or bool(task["opts"].get("visualize_mask"))
                or bool(task["opts"].get("visualize_contours"))
            ):
                from slide_processor.visualization import (
                    visualize_contours_on_thumbnail as _viz_contours,
                )
                from slide_processor.visualization import (
                    visualize_mask_on_thumbnail as _viz_mask,
                )

                cli_args_dict = {
                    "patch_size": int(task["opts"]["patch_size"]),
                    "step_size": int(task["opts"]["step_size"]),
                    "thumbnail_size": 1024,
                    "device": task["opts"]["device"],
                    "tissue_thresh": float(task["opts"]["tissue_thresh"]),
                    "white_thresh": int(task["opts"]["white_thresh"]),
                    "black_thresh": int(task["opts"]["black_thresh"]),
                    "fast_mode": bool(task["opts"]["fast_mode"]),
                    "save_images": bool(task["opts"]["save_images"]),
                    "target_mag": int(task["opts"]["target_mag"]),
                }
                out_dir = str(_Path(task["output_dir"]) / "visualization")
                if bool(task["opts"].get("visualize_grids")):
                    _viz(
                        hdf5_path=result_path,
                        wsi=wsi,
                        output_dir=out_dir,
                        cli_args=cli_args_dict,
                    )
                if bool(task["opts"].get("visualize_mask")) and mask is not None:
                    _viz_mask(mask=mask, wsi=wsi, output_dir=out_dir)
                if bool(task["opts"].get("visualize_contours")) and mask is not None:
                    _viz_contours(
                        tissue_contours=tissue_contours,
                        holes_contours=holes_contours,
                        wsi=wsi,
                        output_dir=out_dir,
                    )

            return True, result_path
        finally:
            try:
                wsi.cleanup()
            except Exception:
                pass
    except Exception as e:
        return False, str(e)


def process_files_batch(
    batch_files: list[str],
    masks,
    seg_params: SegmentParams,
    patch_params: PatchifyParams,
    output_path: Path,
    save_images: bool,
    fast_mode: bool,
    thumb_max: int,
    write_batch: int,
    visualize_grids: bool,
    visualize_mask: bool,
    visualize_contours: bool,
    patch_size: int,
    effective_step_size: int,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    target_mag: int,
    verbose: bool,
    reporter: ProgressReporter | None = None,
    pbar=None,
    wsi_workers: int = 1,
    mpp_dict=None,
    wsis: list | None = None,
) -> tuple[int, int]:
    """Process a batch of files. Returns (successful, failed) counts."""
    successful, failed = 0, 0

    # Parallel per-WSI processing if requested
    if max(1, int(wsi_workers)) > 1 and len(batch_files) > 1:
        tasks = [
            build_wsi_task(
                file_path=f,
                mask_arr=m,
                output_dir=output_path,
                patch_params=patch_params,
                effective_step_size=effective_step_size,
                save_images=save_images,
                fast_mode=fast_mode,
                write_batch=write_batch,
                visualize_grids=visualize_grids,
                visualize_mask=visualize_mask,
                visualize_contours=visualize_contours,
                device=device,
                patch_size=patch_size,
                tissue_thresh=tissue_thresh,
                white_thresh=white_thresh,
                black_thresh=black_thresh,
                target_mag=target_mag,
                mpp=get_mpp_for_wsi(f, mpp_dict),
            )
            for f, m in zip(batch_files, masks)
        ]
        s_delta, f_delta = run_wsi_tasks(
            tasks,
            max_workers=max(1, int(wsi_workers)),
            reporter=reporter,
            pbar=pbar,
            verbose=verbose,
        )
        successful += s_delta
        failed += f_delta
        return successful, failed

    for idx, (f, m) in enumerate(zip(batch_files, masks)):
        try:
            if wsis is not None:
                wsi = wsis[idx]
            else:
                mpp_value = get_mpp_for_wsi(f, mpp_dict)
                wsi = WSIFactory.load(f, mpp=mpp_value)
            try:
                result_h5 = segment_and_patchify(
                    wsi=wsi,
                    output_dir=str(output_path),
                    seg=seg_params,
                    patch=patch_params,
                    save_images=save_images,
                    fast_mode=fast_mode,
                    thumb_max=thumb_max,
                    mask_override=m if m is not None else None,
                    write_batch=write_batch,
                )

                if result_h5:
                    successful += 1
                    if verbose:
                        logger.info(f"Saved patches to: {result_h5}")
                    elif reporter:
                        reporter.update(success=True)
                    if visualize_grids or visualize_mask or visualize_contours:
                        _visualize_result(
                            result_h5,
                            wsi,
                            output_path,
                            patch_size,
                            effective_step_size,
                            device,
                            tissue_thresh,
                            white_thresh,
                            black_thresh,
                            fast_mode,
                            save_images,
                            target_mag,
                            mask=m,
                            do_grids=visualize_grids,
                            do_mask=visualize_mask,
                            do_contours=visualize_contours,
                        )
                        if verbose:
                            logger.info(f"Visualization saved to: {result_h5}")
                else:
                    failed += 1
                    if verbose:
                        logger.warning(f"No patches extracted from {Path(f).name}")
                    elif reporter:
                        reporter.update(success=False)
            finally:
                try:
                    wsi.cleanup()
                except Exception:
                    pass
        except Exception as e:
            failed += 1
            if verbose:
                logger.error(f"Failed to process {Path(f).name}: {e}")
                raise
            elif reporter:
                reporter.update(success=False)
        finally:
            if reporter and pbar:
                reporter.update_progress_bar(pbar)

    return successful, failed


def process_files_pipeline(
    *,
    wsi_files: list[str],
    sam2_model,
    output_path: Path,
    seg_params: SegmentParams,
    patch_params: PatchifyParams,
    save_images: bool,
    fast_mode: bool,
    thumb_max: int,
    write_batch: int,
    visualize_grids: bool,
    visualize_mask: bool,
    visualize_contours: bool,
    patch_size: int,
    effective_step_size: int,
    device: str,
    tissue_thresh: float,
    white_thresh: int,
    black_thresh: int,
    target_mag: int,
    verbose: bool,
    reporter: ProgressReporter | None = None,
    pbar=None,
    wsi_workers: int = 1,
    seg_batch_size: int = 1,
    mpp_dict=None,
) -> tuple[int, int]:
    """Pipeline GPU segmentation with CPU patchification using a bounded queue."""
    import queue as _q
    import threading as _th

    successful, failed = 0, 0

    qsize = max(2, 4 * max(1, int(wsi_workers)))
    q: _q.Queue[tuple[str, _np.ndarray] | None] = _q.Queue(maxsize=qsize)

    # Prepare common opts to embed in each worker task
    def _make_task(file_path: str, mask_arr):
        return build_wsi_task(
            file_path=file_path,
            mask_arr=mask_arr,
            output_dir=output_path,
            patch_params=patch_params,
            effective_step_size=effective_step_size,
            save_images=save_images,
            fast_mode=fast_mode,
            write_batch=write_batch,
            visualize_grids=visualize_grids,
            visualize_mask=visualize_mask,
            visualize_contours=visualize_contours,
            device=device,
            patch_size=patch_size,
            tissue_thresh=tissue_thresh,
            white_thresh=white_thresh,
            black_thresh=black_thresh,
            target_mag=target_mag,
            mpp=get_mpp_for_wsi(file_path, mpp_dict),
        )

    # Producer: segment thumbnails and enqueue results
    def _producer(files: list[str]) -> None:
        pending: list[str] = []
        try:
            for fp in files:
                pending.append(fp)
                should_flush = seg_batch_size > 1 and len(pending) >= seg_batch_size
                if not should_flush and fp != files[-1]:
                    continue

                # Process batch
                batch_files = pending
                pending = []
                thumbs = []
                for f in batch_files:
                    mpp_value = get_mpp_for_wsi(f, mpp_dict)
                    wsi_tmp = WSIFactory.load(f, mpp=mpp_value)
                    try:
                        thumbs.append(
                            wsi_tmp.get_thumbnail_at_power(power=1.25, interpolation="optimise")
                        )
                    finally:
                        try:
                            wsi_tmp.cleanup()
                        except Exception:
                            pass
                masks = sam2_model.predict_batch(thumbs, resize_to_input=True)
                # Enqueue items (blocking if queue is full)
                for f, m in zip(batch_files, masks):
                    q.put((f, m))

                # Release GPU cache after batch
                if str(device).lower() == "cuda":
                    try:
                        import torch as _torch

                        _torch.cuda.empty_cache()
                    except Exception:
                        pass

            # Flush any remaining
            if pending:
                thumbs = []
                for f in pending:
                    mpp_value = get_mpp_for_wsi(f, mpp_dict)
                    wsi_tmp = WSIFactory.load(f, mpp=mpp_value)
                    try:
                        thumbs.append(
                            wsi_tmp.get_thumbnail_at_power(power=1.25, interpolation="optimise")
                        )
                    finally:
                        try:
                            wsi_tmp.cleanup()
                        except Exception:
                            pass
                masks = sam2_model.predict_batch(thumbs, resize_to_input=True)
                for f, m in zip(pending, masks):
                    q.put((f, m))
                if str(device).lower() == "cuda":
                    try:
                        import torch as _torch

                        _torch.cuda.empty_cache()
                    except Exception:
                        pass
        finally:
            # Signal completion
            q.put(None)

    # Start producer thread
    prod = _th.Thread(
        target=_producer, args=(wsi_files,), name="segmentation-producer", daemon=True
    )
    prod.start()

    # Submit tasks to ProcessPool as they arrive and collect results incrementally
    ctx = _mp.get_context("spawn")
    executor = _fut.ProcessPoolExecutor(max_workers=max(1, int(wsi_workers)), mp_context=ctx)

    fut_map: dict[_fut.Future, str] = {}
    producer_done = False

    try:
        with executor as ex:
            while True:
                # Drain queue (non-blocking) while also polling futures
                if not producer_done:
                    try:
                        item = q.get(timeout=0.1)
                        if item is None:
                            producer_done = True
                        else:
                            fpath, mask = item
                            fut = ex.submit(process_wsi_worker, _make_task(fpath, mask))
                            fut_map[fut] = fpath
                    except Exception:
                        # Timeout or queue.Empty: fall through to poll futures
                        pass

                # Poll for completed futures and update progress
                if fut_map:
                    done_now = [f for f in list(fut_map.keys()) if f.done()]
                    for f in done_now:
                        fpath = fut_map.pop(f)
                        try:
                            ok, msg = f.result()
                        except Exception as e:
                            ok, msg = False, str(e)
                        if ok:
                            successful += 1
                            if verbose:
                                logger.info(f"Saved patches to: {msg}")
                            elif reporter:
                                reporter.update(success=True)
                        else:
                            failed += 1
                            if verbose:
                                logger.error(f"Failed to process {Path(fpath).name}: {msg}")
                            elif reporter:
                                reporter.update(success=False)
                        if reporter and pbar:
                            reporter.update_progress_bar(pbar)

                # Exit when producer finished and no pending futures remain
                if producer_done and not fut_map:
                    break
    finally:
        try:
            # Ensure producer terminates
            prod.join(timeout=1.0)
        except Exception:
            pass

    return successful, failed
