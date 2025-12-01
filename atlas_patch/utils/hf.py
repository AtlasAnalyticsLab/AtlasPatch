from __future__ import annotations

import importlib.util
from types import ModuleType

from huggingface_hub import hf_hub_download


def import_module_from_hf(repo_id: str, filename: str) -> ModuleType:
    """Download a Python file from HuggingFace Hub and import it as a module."""
    path = hf_hub_download(repo_id, filename=filename)
    spec = importlib.util.spec_from_file_location(f"{repo_id.replace('/', '_')}_{filename}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {repo_id}/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
