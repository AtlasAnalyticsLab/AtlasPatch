from __future__ import annotations

import os
from pathlib import Path


def acquire_exclusive_lock(
    lock_path: Path,
    *,
    payload: str,
    label: str,
) -> tuple[int | None, Path]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, payload.encode())
        os.fsync(fd)
        return fd, lock_path
    except FileExistsError:
        return None, lock_path
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to create {label} lock {lock_path}: {exc}") from exc


def release_lock(fd: int | None, lock_path: Path) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except Exception:
            pass
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def cleanup_encoder(encoder: object) -> None:
    cleanup = getattr(encoder, "cleanup", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception:
            pass
