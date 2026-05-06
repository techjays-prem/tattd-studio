"""Style Critic.

Measures alignment between a refined Intent and a Candidate Design via
*multimodal embedding* cosine similarity. Returns ``StyleCoherence``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.models import Intent
from tattd_studio.models.candidate_design import CandidateDesign


class StyleCoherence(BaseModel):
    """Verdict from the Style Critic."""

    intent_image_alignment: float = Field(ge=0.0, le=1.0)
    interpretation_notes: str = ""

    model_config = {"frozen": True}


class StyleCritic:
    """Cosine similarity between embedding(Intent) and embedding(Candidate
    Design prompt + image context).

    The embedder is injected; in production it's Gemini Embedding 2 in
    multimodal mode. For text-only inference the prompt of the Candidate
    Design is used as a proxy for the rendered image semantics.
    """

    def __init__(self, *, embedder: TextEmbeddingClient) -> None:
        self._embedder = embedder

    def check(self, intent: Intent, candidate: CandidateDesign) -> StyleCoherence:
        intent_vec = self._embedder.embed(intent.refined_description)
        candidate_vec = self._embedder.embed(candidate.prompt)
        sim = _cosine(intent_vec, candidate_vec)
        # Cosine of unit vectors is in [-1, 1]; rescale to [0, 1].
        alignment = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        notes = (
            "exact-prompt alignment" if alignment > 0.99 else "embedded-cosine alignment"
        )
        return StyleCoherence(
            intent_image_alignment=alignment, interpretation_notes=notes
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
