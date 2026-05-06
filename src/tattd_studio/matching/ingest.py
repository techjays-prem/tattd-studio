"""Artist Portfolio Index ingest pipeline.

Each artist record is embedded twice:

- ``multimodal-1024`` slot via the *multimodal embedding* (text mode of
  Gemini Embedding 2 in production) — drives Stage 1 recall
- ``visual`` slot via the *visual embedding* (DINOv2-ViT-B14 in
  production) — drives Stage 2 rerank

The ``multimodal-3072`` slot is filled with the multimodal vector zero-
padded to 3,072 so the named-vector schema stays consistent across
collections.
"""

from __future__ import annotations

from collections.abc import Iterable

from tattd_studio.embeddings import VisualEmbeddingClient
from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.matching.artists import ArtistRecord
from tattd_studio.vectordb import (
    MULTIMODAL_EMBEDDING_DIM_1024,
    VISUAL_EMBEDDING_DIM,
    VectorStore,
)

ARTIST_PORTFOLIO_INDEX_ALIAS = "artist_portfolio_index"


def ingest_artist_portfolio_index(
    *,
    store: VectorStore,
    collection: str,
    artists: Iterable[ArtistRecord],
    text_embedder: TextEmbeddingClient,
    visual_embedder: VisualEmbeddingClient,
) -> int:
    """Embed each artist twice and upsert into ``collection``."""
    n = 0
    for i, artist in enumerate(artists):
        text_vec = text_embedder.embed(artist.style_text())
        visual_vec = visual_embedder.embed_image(artist.portfolio_url)
        vectors = {
            "multimodal-1024": text_vec[:MULTIMODAL_EMBEDDING_DIM_1024],
            "multimodal-3072": _pad(text_vec, 3072),
            "visual": visual_vec[:VISUAL_EMBEDDING_DIM],
        }
        store.upsert_point(
            collection=collection,
            point_id=i + 1,
            vectors=vectors,
            payload=artist.to_payload(),
        )
        n += 1
    return n


def _pad(vec: list[float], target: int) -> list[float]:
    if len(vec) >= target:
        return list(vec[:target])
    return list(vec) + [0.0] * (target - len(vec))
