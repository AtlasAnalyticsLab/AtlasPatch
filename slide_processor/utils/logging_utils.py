from __future__ import annotations

import logging


class SuppressEmbeddingLogs(logging.Filter):
    """Filter out noisy embedding logs from upstream libraries."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        msg = record.getMessage()
        if "Computing image embeddings for the provided images" in msg:
            return False
        if "Image embeddings computed" in msg:
            return False
        return True


def install_embedding_log_filter() -> None:
    """Attach the embedding log filter to the root logger."""
    logging.getLogger().addFilter(SuppressEmbeddingLogs())


def configure_logging(verbose: bool) -> None:
    """Set global logging levels."""
    root = logging.getLogger()
    target = logging.getLogger("slide_processor")
    level = logging.DEBUG if verbose else logging.WARNING

    root.setLevel(level)
    target.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root.addHandler(handler)

    for handler in root.handlers:
        handler.setLevel(level)
