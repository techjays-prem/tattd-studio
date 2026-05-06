"""Metrics for the Comparison Matrix.

Per IMPLEMENTATION_PLAN.md the matrix scores entries on:

- FID — Fréchet Inception Distance over the entry's images vs a reference
  set (production: against the Artist Portfolio Index of the LoRA's
  training artist; baseline: against the full Artist Portfolio Index).
- CLIP score — semantic alignment between prompt and image.
- GEval — DeepEval-driven LLM-judge rubric for tattoo-specific quality.
- Style adherence via multimodal embedding cosine — image-side embedding
  cosine vs the Intent's text embedding.

This module exposes a small ``ComparisonMetrics`` envelope and a
``score_entry`` runner that computes the four metrics for one
``(prompt, image_uri)`` pair using injectable embedders/judges. Real
FID and CLIP need extra deps; the deterministic CI path produces
reproducible numbers via the embedder-cosine proxy plus a fixed-rubric
GEval stub.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from tattd_studio.knowledge.embedding import TextEmbeddingClient


@dataclass(frozen=True)
class ComparisonMetrics:
    """One row's metric envelope."""

    fid_proxy: float
    clip_proxy: float
    geval: float
    style_adherence: float

    @property
    def aggregate(self) -> float:
        """Equal-weighted aggregate of the four metrics in [0, 1]."""
        return (
            self.fid_proxy + self.clip_proxy + self.geval + self.style_adherence
        ) / 4.0


def score_entry(
    *,
    prompt: str,
    image_uri: str,
    embedder: TextEmbeddingClient,
    geval_score: float | None = None,
) -> ComparisonMetrics:
    """Score one (prompt, image_uri) pair.

    The deterministic baseline uses:

    - ``fid_proxy``: a hash-derived [0, 1] number that's stable per
      (prompt, image_uri); proxy for the FID a real run would compute
      against a reference distribution.
    - ``clip_proxy``: cosine of embedder(prompt) vs embedder(image_uri)
      rescaled to [0, 1]. Proxy for a real CLIP score; live runs swap
      in a real CLIP model via the same shape.
    - ``geval``: defaults to a fixed 0.65 baseline when no DeepEval
      ``GEval`` instance is wired in. Live runs pass the score directly.
    - ``style_adherence``: cosine of embedder(prompt) vs
      embedder(image_uri) rescaled to [0, 1]. Same shape as ``clip_proxy``
      with the live wire-up landing a real multimodal embedding.
    """
    cosine = _rescaled_cosine(
        embedder.embed(prompt), embedder.embed(image_uri)
    )
    return ComparisonMetrics(
        fid_proxy=_stable_unit(prompt + "|fid|" + image_uri),
        clip_proxy=cosine,
        geval=geval_score if geval_score is not None else 0.65,
        style_adherence=cosine,
    )


def _rescaled_cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))


def _stable_unit(seed: str) -> float:
    """Deterministic [0, 1] number from a seed string."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    value = struct.unpack(">I", digest[:4])[0]
    return value / 0xFFFFFFFF
