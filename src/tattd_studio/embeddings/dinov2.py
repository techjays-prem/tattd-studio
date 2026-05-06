"""Visual embedding client (DINOv2-ViT-B14) for Two-Stage Matcher rerank.

Stage 2 of the Two-Stage Matcher reranks Stage 1's recall set via the
*visual embedding* of the chosen Candidate Design's image. This module
exposes a small protocol so tests use a deterministic stub and
production wires the real DINOv2 inference path.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Protocol

VISUAL_EMBEDDING_DIM = 768


class VisualEmbeddingClient(Protocol):
    """Embeds an image (referenced by URI) into the visual embedding space."""

    def embed_image(self, image_uri: str) -> list[float]: ...


class DeterministicVisualEmbeddingClient:
    """Hash-based stub used in CI tests.

    Mirrors the Knowledge Corpus deterministic stub: SHA256 of the URI
    seeds a stretched 768-dim unit vector. Two URIs that differ produce
    effectively orthogonal vectors; the same URI re-embeds identically.
    Good enough to verify the rerank wiring without GPU inference.
    """

    def __init__(self, dim: int = VISUAL_EMBEDDING_DIM) -> None:
        self._dim = dim

    def embed_image(self, image_uri: str) -> list[float]:
        digest = hashlib.sha256(image_uri.encode("utf-8")).digest()
        out: list[float] = []
        counter = 0
        while len(out) < self._dim:
            block = hashlib.sha256(struct.pack(">I", counter) + digest).digest()
            for i in range(0, len(block), 4):
                if len(out) >= self._dim:
                    break
                value = struct.unpack(">I", block[i : i + 4])[0]
                out.append((value / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        norm = sum(x * x for x in out) ** 0.5
        return [x / norm for x in out] if norm > 0 else out


def build_dinov2_visual_embedding_client(
    *,
    model_id: str = "facebook/dinov2-base",
) -> VisualEmbeddingClient:
    """Construct a real DINOv2-backed visual embedding client.

    Lazy-imports torch and transformers; only callers wiring the live
    rerank path pay that cost. Reads the image from a local ``file://``
    URI or fetches it from a remote URL.
    """
    import io
    from urllib.parse import urlparse

    import requests
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if os.environ.get("TATTD_DINOV2_DISABLED") == "1":
        raise RuntimeError("DINOv2 live client disabled via env")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    # Inference-only mode for the loaded weights.
    model.training = False

    class _DINOv2Client:
        def embed_image(self, image_uri: str) -> list[float]:
            parsed = urlparse(image_uri)
            if parsed.scheme == "file":
                path = parsed.path
                if not path:
                    raise ValueError(f"empty file path in URI: {image_uri}")
                img = Image.open(path).convert("RGB")
            elif parsed.scheme in ("http", "https"):
                r = requests.get(image_uri, timeout=10)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
            else:
                img = Image.open(image_uri).convert("RGB")

            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)
                cls_vec = cls_vec / cls_vec.norm()
            return cls_vec.tolist()

    return _DINOv2Client()  # type: ignore[return-value]
