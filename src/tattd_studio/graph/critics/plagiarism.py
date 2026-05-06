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
        collection: str | None = None,
        collections: list[str] | None = None,
        embedder: TextEmbeddingClient,
        threshold: float = 0.85,
        embed_query_for: Callable[[CandidateDesign], str] | None = None,
    ) -> None:
        if collection is not None and collections is not None:
            raise ValueError("pass either `collection` or `collections`, not both")
        if collection is None and not collections:
            raise ValueError("at least one collection is required")
        self._store = store
        # Slice #7 took a single ``collection``; slice #8 broadens to a
        # list. Both signatures supported for backwards-compat.
        self._collections: list[str] = (
            collections if collections is not None else [collection]  # type: ignore[list-item]
        )
        self._embedder = embedder
        self._threshold = threshold
        self._embed_query_for = embed_query_for or (lambda cd: cd.prompt)

    def check(self, candidate: CandidateDesign) -> PlagiarismCheck:
        query = self._embed_query_for(candidate)
        vec = self._embedder.embed(query)
        # Query each configured corpus and keep the best match.
        best_hit = None
        best_corpus = ""
        for coll in self._collections:
            hits = self._store.query_named(
                collection=coll,
                vector_name="multimodal-1024",
                query_vector=vec[:1024],
                limit=1,
            )
            if hits and (best_hit is None or hits[0].score > best_hit.score):
                best_hit = hits[0]
                best_corpus = coll

        if best_hit is None:
            return PlagiarismCheck(
                flagged=False,
                top_match_artist="",
                top_match_similarity=0.0,
                threshold_used=self._threshold,
                corpus_hit="",
            )
        sim = max(0.0, min(1.0, best_hit.score))
        return PlagiarismCheck(
            flagged=sim >= self._threshold,
            top_match_artist=best_hit.payload.get("artist", ""),
            top_match_similarity=sim,
            threshold_used=self._threshold,
            corpus_hit=best_corpus,
        )
