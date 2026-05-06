"""Knowledge Retriever — RAG read path over the Knowledge Corpus collection.

Queries are embedded via the same TextEmbeddingClient used at ingest time,
then routed through the Vector Store's named-vector query against
``multimodal-1024``. The retrieved chunk records carry their full
Provenance metadata so the Consultation node can cite sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from tattd_studio.knowledge.embedding import TextEmbeddingClient
from tattd_studio.vectordb import VectorStore

KNOWLEDGE_VECTOR_SLOT = "multimodal-1024"


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by the Knowledge Retriever, with score + Provenance."""

    chunk_id: str
    title: str
    body: str
    source_url: str
    area: str
    score: float


class KnowledgeRetriever:
    """Retrieves Knowledge Corpus chunks for a Consultation query."""

    def __init__(
        self,
        *,
        store: VectorStore,
        collection: str,
        embedder: TextEmbeddingClient,
    ) -> None:
        self._store = store
        self._collection = collection
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        vec = self._embedder.embed(query)
        hits = self._store.query_named(
            collection=self._collection,
            vector_name=KNOWLEDGE_VECTOR_SLOT,
            query_vector=vec,
            limit=k,
        )
        return [
            RetrievedChunk(
                chunk_id=h.payload.get("chunk_id", ""),
                title=h.payload.get("title", ""),
                body=h.payload.get("body", ""),
                source_url=h.payload.get("source_url", ""),
                area=h.payload.get("area", ""),
                score=h.score,
            )
            for h in hits
        ]
