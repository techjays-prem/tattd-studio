"""Two-Stage Matcher.

Stage 1: multimodal embedding cosine recall against the Artist Portfolio
Index (typically top-50 to widen the recall net).
Stage 2: visual embedding rerank against the chosen Candidate Design's
image (DINOv2 cosine reranks the Stage 1 set down to top-K).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tattd_studio.embeddings import VisualEmbeddingClient
from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.models.candidate_design import CandidateDesign
from tattd_studio.vectordb import VectorStore


@dataclass(frozen=True)
class RankedArtist:
    """One ranked artist surfaced by the Two-Stage Matcher."""

    artist_slug: str
    display_name: str
    portfolio_url: str
    primary_style: str
    stage1_score: float  # multimodal embedding cosine
    stage2_score: float  # visual embedding cosine after rerank


class TwoStageMatcher:
    def __init__(
        self,
        *,
        store: VectorStore,
        collection: str,
        text_embedder: TextEmbeddingClient,
        visual_embedder: VisualEmbeddingClient,
        stage1_limit: int = 50,
        artist_visual_vectors: dict[str, list[float]] | None = None,
    ) -> None:
        self._store = store
        self._collection = collection
        self._text_embedder = text_embedder
        self._visual_embedder = visual_embedder
        self._stage1_limit = stage1_limit
        # Optional cache: pre-embedded visual vectors per artist_slug,
        # used in Stage 2 when querying the named-vector slot would
        # require a separate round-trip.
        self._artist_visual_vectors = artist_visual_vectors or {}

    def find_artists(
        self, chosen_design: CandidateDesign, k: int = 5
    ) -> list[RankedArtist]:
        # Stage 1: multimodal embedding recall.
        text_vec = self._text_embedder.embed(chosen_design.prompt)
        recall = self._store.query_named(
            collection=self._collection,
            vector_name="multimodal-1024",
            query_vector=text_vec[:1024],
            limit=self._stage1_limit,
        )

        # Stage 2: visual embedding rerank.
        chosen_visual = self._visual_embedder.embed_image(chosen_design.image_uri)

        ranked: list[RankedArtist] = []
        for hit in recall:
            payload = hit.payload
            artist_slug = payload.get("artist_slug", "")
            artist_visual = self._artist_visual_vectors.get(artist_slug)
            if artist_visual is None:
                # Fall back to embedding the artist's portfolio URL on
                # demand. The deterministic stub treats this as a stable
                # per-URL signature; the live DINOv2 path would actually
                # fetch the artist's first portfolio image.
                artist_visual = self._visual_embedder.embed_image(
                    payload.get("portfolio_url", "")
                )
            stage2 = _cosine(chosen_visual, artist_visual)
            ranked.append(
                RankedArtist(
                    artist_slug=artist_slug,
                    display_name=payload.get("display_name", ""),
                    portfolio_url=payload.get("portfolio_url", ""),
                    primary_style=payload.get("primary_style", ""),
                    stage1_score=hit.score,
                    stage2_score=stage2,
                )
            )

        ranked.sort(key=lambda r: r.stage2_score, reverse=True)
        return ranked[:k]


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
