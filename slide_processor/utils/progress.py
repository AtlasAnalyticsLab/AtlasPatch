"""Progress reporting utilities for batch processing."""

import os
import sys
import time
from contextlib import contextmanager

import click


class ProgressReporter:
    """Tracks and reports progress for batch file processing."""

    def __init__(self, total_files: int):
        """Initialize progress reporter.

        Args:
            total_files: Total number of files to process
        """
        self.total_files = total_files
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.monotonic()

    def update(self, success: bool = True) -> None:
        """Update progress counters.

        Args:
            success: Whether the last item processed was successful
        """
        self.processed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1

    @staticmethod
    def _fmt_duration(s: float) -> str:
        """Format duration in seconds to HH:MM:SS format.

        Args:
            s: Duration in seconds

        Returns:
            Formatted duration string
        """
        s = max(0.0, float(s))
        m, sec = divmod(int(s + 0.5), 60)
        h, min_ = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{min_:02d}:{sec:02d}"
        return f"{min_:02d}:{sec:02d}"

    def get_status_line(self) -> str:
        """Get current status line for progress bar.

        Returns:
            Status string with progress, timing, and counters
        """
        left = max(0, self.total_files - self.processed)
        now = time.monotonic()

        if self.processed > 0:
            elapsed = now - self.start_time
            avg = elapsed / self.processed
            eta = avg * left
            avg_str = f"{avg:.2f}s/it"
            eta_str = self._fmt_duration(eta)
            elapsed_str = self._fmt_duration(elapsed)
        else:
            avg_str = "– s/it"
            eta_str = "--:--"
            elapsed_str = "00:00"

        return (
            f"{self.processed}/{self.total_files} "
            f"[{elapsed_str}<{eta_str}, {avg_str}]  "
            f"S:{self.successful} F:{self.failed}"
        )

    @contextmanager
    def progress_bar(self, label: str = "Processing"):
        """Context manager for click progress bar.

        Args:
            label: Label to display above the progress bar

        Yields:
            Click progress bar object
        """
        interactive = sys.stderr.isatty()
        stream = sys.stderr if interactive else open(os.devnull, "w")

        try:
            with click.progressbar(
                length=self.total_files,
                label=f"{label}  {self.get_status_line()}",
                file=stream,
            ) as pbar:
                pbar.reporter = self  # type: ignore
                yield pbar
        finally:
            if not interactive:
                try:
                    stream.close()
                except Exception:
                    pass

    def update_progress_bar(self, pbar) -> None:  # type: ignore
        """Update progress bar with current status.

        Args:
            pbar: Click progress bar object
        """
        pbar.label = f"Processing WSI files  {self.get_status_line()}"
        pbar.update(1)
