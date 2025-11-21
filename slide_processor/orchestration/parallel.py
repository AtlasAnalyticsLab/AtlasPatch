from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

from slide_processor.core.models import ExtractionResult, Slide
from slide_processor.core.wsi.iwsi import IWSI
from slide_processor.services.interfaces import ExtractionService, VisualizationService

logger = logging.getLogger("slide_processor.parallel")


@dataclass
class ExtractionTask:
    """Represents a single slide extraction unit."""

    slide: Slide
    wsi: IWSI
    mask: np.ndarray
    lock_fd: int | None
    lock_path: Path


class InflightTracker:
    """Thread-safe tracker for in-flight extraction futures."""

    def __init__(
        self,
        *,
        results: list[ExtractionResult],
        failures: list[Tuple[Slide, Exception | str]],
        progress,
    ) -> None:
        self._results = results
        self._failures = failures
        self._progress = progress
        self._inflight: dict[Future[ExtractionResult], Slide] = {}
        self._lock = threading.Lock()

    def add(self, fut: Future[ExtractionResult], slide: Slide) -> None:
        with self._lock:
            self._inflight[fut] = slide
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut: Future[ExtractionResult]) -> None:
        slide_done: Slide | None = None
        with self._lock:
            slide_done = self._inflight.pop(fut, None)
        if slide_done is None:
            return
        try:
            res = fut.result()
            self._results.append(res)
            logger.info(
                "Processed %s -> %s (patches=%s)",
                res.slide.path.name,
                res.h5_path,
                res.num_patches,
            )
        except Exception as e:  # noqa: BLE001
            self._failures.append((slide_done, e))
            logger.error("Failed to process %s: %s", slide_done.path.name, e)
        finally:
            if self._progress:
                self._progress.update(1)

    def count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def wait_until_at_most(self, limit: int) -> None:
        """Block until in-flight tasks are <= limit."""
        limit = max(0, int(limit))
        while True:
            with self._lock:
                count = len(self._inflight)
                futs = list(self._inflight.keys())
            if count == 0 or count <= limit:
                return
            wait(futs, return_when=FIRST_COMPLETED)


class PatchExtractionExecutor:
    """Runs per-slide patch extraction concurrently."""

    def __init__(
        self,
        *,
        extractor: ExtractionService,
        visualizer: VisualizationService | None,
        release_lock: Callable[[int | None, Path], None],
        max_workers: int | None = None,
    ) -> None:
        self.extractor = extractor
        self.visualizer = visualizer
        self.release_lock = release_lock
        self.max_workers = self._resolve_workers(max_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="patch-extract"
        )

    @staticmethod
    def _resolve_workers(requested: int | None) -> int:
        if requested is not None:
            return max(1, int(requested))
        return max(1, int(os.cpu_count() or 4))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def execute(
        self,
        tasks: Sequence[ExtractionTask],
        *,
        progress=None,
    ) -> Tuple[list[ExtractionResult], list[Tuple[Slide, Exception]]]:
        """Submit tasks and collect completed results/failures."""
        if not tasks:
            return [], []

        futures = {self._executor.submit(self._run_task, task): task for task in tasks}
        results: list[ExtractionResult] = []
        failures: list[Tuple[Slide, Exception]] = []

        for fut in as_completed(futures):
            task = futures[fut]
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
            except Exception as e:  # noqa: BLE001
                failures.append((task.slide, e))
            finally:
                if progress is not None:
                    progress.update(1)
        return results, failures

    def submit(self, task: ExtractionTask):
        """Submit a single extraction task; caller manages completion."""
        return self._executor.submit(self._run_task, task)

    def _run_task(self, task: ExtractionTask) -> ExtractionResult:
        try:
            result = self.extractor.extract(task.wsi, task.mask, slide=task.slide)
            if self.visualizer:
                self.visualizer.visualize(result, wsi=task.wsi, mask=task.mask)
            return result
        finally:
            try:
                task.wsi.cleanup()
            except Exception:
                pass
            self.release_lock(task.lock_fd, task.lock_path)

    def __enter__(self) -> PatchExtractionExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
