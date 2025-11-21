from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from tqdm import tqdm

from slide_processor.core.config import AppConfig
from slide_processor.core.models import ExtractionResult, Slide
from slide_processor.core.paths import find_existing_patch, patch_lock_path
from slide_processor.core.wsi.iwsi import IWSI
from slide_processor.orchestration.parallel import (
    ExtractionTask,
    InflightTracker,
    PatchExtractionExecutor,
)
from slide_processor.services.interfaces import (
    ExtractionService,
    MPPResolver,
    SegmentationService,
    VisualizationService,
    WSILoader,
)
from slide_processor.utils.params import get_wsi_files

logger = logging.getLogger("slide_processor.runner")


def _chunked(items: Sequence[Slide], size: int) -> Iterable[Sequence[Slide]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class ProcessingRunner:
    """High-level orchestration of WSI segmentation, patch extraction, and visualization."""

    def __init__(
        self,
        config: AppConfig,
        segmentation: SegmentationService,
        extractor: ExtractionService,
        visualizer: VisualizationService | None,
        mpp_resolver: MPPResolver,
        wsi_loader: WSILoader,
        *,
        show_progress: bool = False,
    ) -> None:
        self.config = config.validated()
        self.segmentation = segmentation
        self.extractor = extractor
        self.visualizer = visualizer
        self.mpp_resolver = mpp_resolver
        self.wsi_loader = wsi_loader
        self.show_progress = show_progress

    def discover_slides(self) -> list[Slide]:
        files = get_wsi_files(
            str(self.config.processing.input_path), recursive=self.config.processing.recursive
        )
        slides: list[Slide] = []
        for f in files:
            slide = Slide(path=Path(f))
            slides.append(slide)
        return slides

    def _should_skip(self, slide: Slide) -> bool:
        if not self.config.output.skip_existing:
            return False
        exists = find_existing_patch(slide, self.config.output, self.config.extraction)
        return exists is not None

    def _acquire_lock(self, slide: Slide) -> tuple[int | None, Path]:
        """Attempt to acquire a per-slide lock file. Returns (fd, path) or (None, path if held elsewhere)."""
        lock_path = patch_lock_path(slide, self.config.output, self.config.extraction)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"pid={os.getpid()},time={int(time.time())},slide={slide.path}"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, payload.encode())
            os.fsync(fd)
            return fd, lock_path
        except FileExistsError:
            return None, lock_path
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to create lock {lock_path}: {e}") from e

    @staticmethod
    def _release_lock(fd: int | None, path: Path) -> None:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _attach_mpp(self, slides: list[Slide]) -> list[Slide]:
        resolved: list[Slide] = []
        for s in slides:
            mpp_value = self.mpp_resolver.resolve(s)
            resolved.append(Slide(path=s.path, mpp=mpp_value, backend=s.backend))
        return resolved

    def _resolve_patch_workers(self) -> int:
        workers_cfg = self.config.extraction.workers
        if workers_cfg is not None:
            return max(1, int(workers_cfg))
        return max(1, int(os.cpu_count() or 4))

    def _resolve_max_open_slides(self, patch_workers: int, batch_size: int) -> int:
        cfg_val = self.config.extraction.max_open_slides
        if cfg_val is None:
            raise ValueError("max_open_slides must be defined")
        return max(1, int(cfg_val))

    def run(self) -> tuple[list[ExtractionResult], list[Tuple[Slide, Exception | str]]]:
        slides = self.discover_slides()
        slides = self._attach_mpp(slides)

        if not slides:
            logger.warning("No slides found to process.")
            return [], []

        results: list[ExtractionResult] = []
        failures: list[Tuple[Slide, Exception | str]] = []

        progress = tqdm(total=len(slides), disable=not self.show_progress, desc="Processing slides")
        progress_bar = progress if self.show_progress else None
        patch_workers = self._resolve_patch_workers()
        batch_size = max(1, self.config.segmentation.batch_size)
        max_open_slides = self._resolve_max_open_slides(patch_workers, batch_size=batch_size)
        with PatchExtractionExecutor(
            extractor=self.extractor,
            visualizer=self.visualizer,
            release_lock=self._release_lock,
            max_workers=patch_workers,
        ) as executor:
            tracker = InflightTracker(results=results, failures=failures, progress=progress_bar)

            for batch in _chunked(slides, batch_size):
                # Ensure capacity for new openings before starting next batch.
                allow_inflight = max(0, max_open_slides - batch_size)
                tracker.wait_until_at_most(limit=allow_inflight if allow_inflight > 0 else 0)

                opened: list[tuple[Slide, IWSI, int | None, Path]] = []

                # Skip check and open WSIs once with lock acquisition
                for slide in batch:
                    if self._should_skip(slide):
                        logger.info("Skipping %s (already processed).", slide.path.name)
                        if progress_bar:
                            progress_bar.update(1)
                        continue
                    fd, lock_path = self._acquire_lock(slide)
                    if fd is None:
                        logger.info("Skipping %s (locked by another process).", slide.path.name)
                        if progress_bar:
                            progress_bar.update(1)
                        continue
                    try:
                        opened.append((slide, self.wsi_loader.open(slide), fd, lock_path))
                    except Exception as e:  # noqa: BLE001
                        failures.append((slide, e))
                        logger.error("Failed to open %s: %s", slide.path.name, e)
                        self._release_lock(fd, lock_path)
                        if progress_bar:
                            progress_bar.update(1)

                if not opened:
                    continue

                submitted_wsis: set[IWSI] = set()
                masks = None
                try:
                    wsis_only = [w for _, w, _, _ in opened]
                    masks = (
                        self.segmentation.segment_batch(wsis_only)
                        if len(wsis_only) > 1
                        else [self.segmentation.segment_thumbnail(wsis_only[0])]
                    )
                except Exception as e:  # noqa: BLE001
                    for slide, wsi, fd, path in opened:
                        failures.append((slide, e))
                        logger.error("Segmentation failed for %s: %s", slide.path.name, e)
                        try:
                            wsi.cleanup()
                        except Exception:
                            pass
                        self._release_lock(fd, path)
                        if progress_bar:
                            progress_bar.update(1)
                else:
                    for (slide, wsi, lock_fd, lock_path), mask in zip(opened, masks):
                        if self._should_skip(slide):
                            logger.info("Skipping %s (already processed).", slide.path.name)
                            try:
                                wsi.cleanup()
                            except Exception:
                                pass
                            self._release_lock(lock_fd, lock_path)
                            if progress_bar:
                                progress_bar.update(1)
                            continue
                        task = ExtractionTask(
                            slide=slide,
                            wsi=wsi,
                            mask=mask.data,
                            lock_fd=lock_fd,
                            lock_path=lock_path,
                        )
                        fut = executor.submit(task)
                        tracker.add(fut, slide)
                        submitted_wsis.add(wsi)
                finally:
                    # Clean up WSIs that were not submitted (e.g., skipped or segmentation failed).
                    for slide, wsi, lock_fd, lock_path in opened:
                        if wsi in submitted_wsis:
                            continue
                        try:
                            wsi.cleanup()
                        except Exception:
                            pass
                        self._release_lock(lock_fd, lock_path)

                # Respect global cap on open slides after submissions.
                tracker.wait_until_at_most(limit=max_open_slides)

            # Drain any remaining in-flight tasks.
            tracker.wait_until_at_most(limit=0)

        if self.show_progress:
            progress.close()
        return results, failures
