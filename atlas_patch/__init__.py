"""AtlasPatch package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.1.0"
__all__ = ["core", "services", "__version__"]


def __getattr__(name: str) -> Any:
    if name in {"core", "services"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
