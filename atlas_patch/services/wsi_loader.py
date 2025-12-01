from __future__ import annotations

from atlas_patch.core.models import Slide
from atlas_patch.core.wsi import WSIFactory
from atlas_patch.services.interfaces import WSILoader


class DefaultWSILoader(WSILoader):
    """Concrete WSI loader that delegates to the factory."""

    def open(self, slide: Slide):
        return WSIFactory.load(str(slide.path), mpp=slide.mpp, backend=slide.backend)
