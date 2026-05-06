"""Custom DeepEval ``BaseMetric`` for Candidate Design originality.

Originality is the inverse of near-duplicate similarity against the
Famous Tattoos Corpus and Artist Portfolio Index. A Candidate Design
that does not match anything in the reference corpora gets a high
originality score; one that closely matches an indexed reference
(e.g., a Sailor Jerry flash design) gets a low score and would be
flagged by the Plagiarism Critic at a slightly lower threshold.

This metric is the *quantitative* counterpart of the Plagiarism Critic
in Tier 1 Generation evaluation: the Critic gates Routing on a
binary flag, the metric scores the gradient.
"""

from __future__ import annotations

import math

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.vectordb import VectorStore


class OriginalityMetric(BaseMetric):
    """Originality = 1 - (max cosine similarity against any reference).

    Reads the Candidate Design's prompt from ``test_case.actual_output``
    and queries the configured Vector Store collection for the nearest
    reference. The threshold defaults to 0.5: scores at or above pass.
    """

    def __init__(
        self,
        *,
        store: VectorStore,
        collection: str,
        embedder: TextEmbeddingClient,
        threshold: float = 0.5,
    ) -> None:
        self._store = store
        self._collection = collection
        self._embedder = embedder
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        prompt = test_case.actual_output or ""
        if not prompt.strip():
            self.score = 0.0
            self.success = False
            self.reason = "empty Candidate Design prompt"
            return self.score

        vec = self._embedder.embed(prompt)
        hits = self._store.query_named(
            collection=self._collection,
            vector_name="multimodal-1024",
            query_vector=vec[:1024],
            limit=1,
        )
        if not hits:
            self.score = 1.0
        else:
            # Normalize cosine ([-1, 1]) → similarity ([0, 1]).
            sim = max(0.0, min(1.0, (hits[0].score + 1.0) / 2.0))
            self.score = max(0.0, 1.0 - sim)

        self.success = self.score >= self.threshold
        self.reason = (
            f"originality {self.score:.3f} "
            f"({'≥' if self.success else '<'} threshold {self.threshold})"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "OriginalityMetric"


def faithfulness_score(
    intent_text: str,
    candidate_prompt: str,
    *,
    embedder: TextEmbeddingClient,
) -> float:
    """Quantitative faithfulness in [0, 1].

    Cosine similarity between the Intent text and the Candidate Design
    prompt, rescaled to [0, 1]. Mirrors the Style Critic's scoring shape
    so the two metrics agree on direction.
    """
    a = embedder.embed(intent_text)
    b = embedder.embed(candidate_prompt)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    cos = dot / (na * nb)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))
