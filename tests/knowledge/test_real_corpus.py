"""End-to-end test against the real Knowledge Corpus markdown files.

Loads every chunk from `data/knowledge/`, ingests with the deterministic
embedding stub, and queries to confirm:

- The corpus parses cleanly across all area files
- Each chunk has all Provenance fields
- Retrieval returns the expected chunk for an exact-body query, with
  the source field populated
"""

from __future__ import annotations

from pathlib import Path

from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    KnowledgeRetriever,
    ingest_corpus,
)
from tattd_studio.knowledge.ingest import load_chunks_from_dir
from tattd_studio.vectordb import VectorStore

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"


def test_real_corpus_loads_all_provenance_fields() -> None:
    chunks = load_chunks_from_dir(KNOWLEDGE_DIR)
    # Spec target ~150 chunks, distributed across five areas.
    assert 100 <= len(chunks) <= 200, f"unexpected corpus size: {len(chunks)}"

    by_area: dict[str, int] = {}
    for c in chunks:
        by_area[c.area] = by_area.get(c.area, 0) + 1
        # Provenance frontmatter must be populated.
        assert c.source_url
        assert c.curator
        assert c.capture_date
        assert c.permission

    assert set(by_area) == {"taxonomy", "placement", "aftercare", "ip", "cultural"}


def test_real_corpus_round_trips_through_ingest_and_retrieve() -> None:
    chunks = load_chunks_from_dir(KNOWLEDGE_DIR)

    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus_test")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    n = ingest_corpus(
        store=store,
        collection="knowledge_corpus_test",
        chunks=chunks,
        embedder=embedder,
    )
    assert n == len(chunks)

    retriever = KnowledgeRetriever(
        store=store,
        collection="knowledge_corpus_test",
        embedder=embedder,
    )

    # Pick a stable, representative chunk to retrieve.
    target = next(c for c in chunks if c.chunk_id == "aftercare-saniderm")
    hits = retriever.retrieve(target.body, k=1)
    assert hits[0].chunk_id == "aftercare-saniderm"
    assert hits[0].source_url == target.source_url
    assert hits[0].area == "aftercare"
