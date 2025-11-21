from __future__ import annotations

from slide_processor.core.models import Slide
from slide_processor.services.interfaces import WSILoader
from slide_processor.core.wsi import WSIFactory


class DefaultWSILoader(WSILoader):
    """Concrete WSI loader that delegates to the factory."""

    def open(self, slide: Slide):
        return WSIFactory.load(str(slide.path), mpp=slide.mpp, backend=slide.backend)
