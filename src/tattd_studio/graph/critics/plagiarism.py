"""Plagiarism Critic.

Scores a Candidate Design's *multimodal embedding* against the Famous
Tattoos Corpus and (in slice #8) the Artist Portfolio Index. Returns
``PlagiarismCheck`` with the top-matching reference and similarity.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.models.candidate_design import CandidateDesign
from tattd_studio.vectordb import VectorStore


class PlagiarismCheck(BaseModel):
    """Verdict from the Plagiarism Critic for one Candidate Design."""

    flagged: bool
    top_match_artist: str = ""
    top_match_similarity: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)
    corpus_hit: str = ""  # which corpus the top match came from

    model_config = {"frozen": True}


class PlagiarismCritic:
    """Embeds the Candidate Design's prompt + image_uri context and queries
    the Vector Store against the Famous Tattoos Corpus.

    The embedder is injected (deterministic stub for tests; Gemini
    Embedding 2 in production), as is the threshold (loaded from
    ``evals/calibrated_thresholds.toml`` by the Studio at boot).
    """

    def __init__(
        self,
        *,
        store: VectorStore,
        collection: str,
        embedder: TextEmbeddingClient,
        threshold: float = 0.85,
        embed_query_for: Callable[[CandidateDesign], str] | None = None,
    ) -> None:
        self._store = store
        self._collection = collection
        self._embedder = embedder
        self._threshold = threshold
        self._embed_query_for = embed_query_for or (lambda cd: cd.prompt)

    def check(self, candidate: CandidateDesign) -> PlagiarismCheck:
        query = self._embed_query_for(candidate)
        vec = self._embedder.embed(query)
        hits = self._store.query_named(
            collection=self._collection,
            vector_name="multimodal-1024",
            query_vector=vec[:1024],
            limit=1,
        )
        if not hits:
            return PlagiarismCheck(
                flagged=False,
                top_match_artist="",
                top_match_similarity=0.0,
                threshold_used=self._threshold,
                corpus_hit=self._collection,
            )
        top = hits[0]
        sim = max(0.0, min(1.0, top.score))
        return PlagiarismCheck(
            flagged=sim >= self._threshold,
            top_match_artist=top.payload.get("artist", ""),
            top_match_similarity=sim,
            threshold_used=self._threshold,
            corpus_hit=self._collection,
        )
