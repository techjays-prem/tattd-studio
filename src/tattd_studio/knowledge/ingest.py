"""Knowledge Corpus ingest pipeline.

Reads the per-area markdown files under ``data/knowledge/``, parses them
into Chunks, embeds each via the supplied TextEmbeddingClient, and upserts
into the Vector Store collection that backs the Knowledge Retriever.

The collection is rebuilt behind the alias ``KNOWLEDGE_CORPUS_ALIAS`` via
``tattd_studio.vectordb.reindex.rebuild`` so readers querying the alias
never see an empty index during a refresh.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tattd_studio.knowledge.chunks import Chunk, parse_chunks_from_markdown
from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.vectordb import (
    MULTIMODAL_EMBEDDING_DIM_1024,
    VISUAL_EMBEDDING_DIM,
    VectorStore,
)

KNOWLEDGE_CORPUS_ALIAS = "knowledge_corpus"


def load_chunks_from_dir(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        chunks.extend(parse_chunks_from_markdown(path.read_text()))
    return chunks


def ingest_corpus(
    *,
    store: VectorStore,
    collection: str,
    chunks: Iterable[Chunk],
    embedder: TextEmbeddingClient,
) -> int:
    """Embed each chunk and upsert into ``collection``. Returns the count."""
    n = 0
    for i, chunk in enumerate(chunks):
        body_vec = embedder.embed(chunk.body)
        # Knowledge Corpus is text-only at ingest time; the visual slot
        # is populated with a zero-vector so the named-vector schema
        # stays consistent with image-bearing collections.
        vectors = {
            "multimodal-1024": body_vec[:MULTIMODAL_EMBEDDING_DIM_1024],
            "multimodal-3072": _pad(body_vec, 3072),
            "visual": [0.0] * VISUAL_EMBEDDING_DIM,
        }
        store.upsert_point(
            collection=collection,
            point_id=i + 1,
            vectors=vectors,
            payload=chunk.to_payload(),
        )
        n += 1
    return n


def _pad(vec: list[float], target: int) -> list[float]:
    if len(vec) >= target:
        return list(vec[:target])
    out = list(vec) + [0.0] * (target - len(vec))
    return out
