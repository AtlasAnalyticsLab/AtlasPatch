from __future__ import annotations

import logging

import torch
from transformers import CLIPModel, CLIPProcessor

from slide_processor.models.patch.base import PatchFeatureExtractor
from slide_processor.models.patch.registry import PatchFeatureExtractorRegistry

logger = logging.getLogger(__name__)

_QUILT_SPECS = [
    ("quilt_b_32", "wisdomik/QuiltNet-B-32"),
    ("quilt_b_16", "wisdomik/QuiltNet-B-16"),
    ("quilt_b_16_pmb", "wisdomik/QuiltNet-B-16-PMB"),
]


class QuiltNet(PatchFeatureExtractor):
    """QuiltNet CLIP-style patch encoder (image tower only).

    Paper: "Quilt-1M: One Million Image-Text Pairs for Histopathology"
    https://quilt1m.github.io/
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        num_workers: int = 0,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_workers = max(0, int(num_workers))
        self.model_id = model_id

        load_mode = "transformers"
        forward_fn = None

        if model_id.lower().endswith("quiltnet-b-16-pmb"):
            import open_clip

            load_mode = "open_clip"
            model_name = f"hf-hub:{model_id}"
            model, processor = open_clip.create_model_from_pretrained(model_name=model_name)

            def _forward(x, m=model):
                return m.encode_image(x)

            forward_fn = _forward
        else:
            model = CLIPModel.from_pretrained(model_id)
            processor = CLIPProcessor.from_pretrained(model_id)

            def _forward(x, m=model):
                return m.get_image_features(pixel_values=x)

            forward_fn = _forward
        if load_mode == "transformers" and not hasattr(model, "get_image_features"):
            raise RuntimeError(
                f"Loaded model '{model_id}' does not expose get_image_features; cannot be used for patch embeddings."
            )
        if load_mode == "open_clip" and not hasattr(model, "encode_image"):
            raise RuntimeError(
                f"Loaded model '{model_id}' does not expose encode_image; cannot be used for patch embeddings."
            )

        model = model.to(device=self.device, dtype=self.dtype).eval()
        emb_dim = 512

        preprocess = self._build_preprocess(processor, load_mode)

        super().__init__(
            name=name,
            model=model,
            embedding_dim=int(emb_dim),
            preprocess=preprocess,
            device=self.device,
            dtype=self.dtype,
            num_workers=self.num_workers,
            non_blocking=True,
            forward_fn=forward_fn,
        )

    @staticmethod
    def _build_preprocess(processor, load_mode: str):
        if load_mode == "open_clip":
            return processor

        def _preprocess(pil_img):
            inputs = processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return _preprocess


def register_quilt_models(
    registry: PatchFeatureExtractorRegistry,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
) -> None:
    for name, model_id in _QUILT_SPECS:
        registry.register(
            name,
            lambda n=name, mid=model_id: QuiltNet(
                name=n, model_id=mid, device=device, dtype=dtype, num_workers=num_workers
            ),
        )
