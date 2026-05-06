"""Integration test: ingest a small corpus and query it via the Knowledge
Retriever. Uses the deterministic embedding stub so no API keys are
needed in CI.
"""

from __future__ import annotations

import textwrap

from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    KnowledgeRetriever,
    ingest_corpus,
    parse_chunks_from_markdown,
)
from tattd_studio.vectordb import VectorStore

CORPUS = textwrap.dedent(
    """\
    <!-- CHUNK -->
    ---
    chunk_id: aftercare-saniderm
    area: aftercare
    title: Saniderm second-skin aftercare
    source_url: https://example.com/aftercare/saniderm
    curator: tattd-studio-dev
    capture_date: 2026-05-06
    synthetic: true
    permission: synthetic-content-tattd-studio-poc
    ---

    Saniderm is a breathable second-skin bandage worn for the first 3-5 days
    post-tattoo. It traps plasma, prevents scabbing, and reduces healing time.

    <!-- CHUNK -->
    ---
    chunk_id: ip-portrait-likeness
    area: ip
    title: Portrait likeness and right of publicity
    source_url: https://example.com/ip/portrait-likeness
    curator: tattd-studio-dev
    capture_date: 2026-05-06
    synthetic: true
    permission: synthetic-content-tattd-studio-poc
    ---

    Tattooing a recognizable celebrity portrait without consent risks a right-
    of-publicity claim in many US states. Studios should obtain written
    permission or use clearly fan-art-style derivatives.

    <!-- CHUNK -->
    ---
    chunk_id: cultural-japanese-irezumi
    area: cultural
    title: Japanese irezumi cultural notes
    source_url: https://example.com/cultural/irezumi
    curator: tattd-studio-dev
    capture_date: 2026-05-06
    synthetic: true
    permission: synthetic-content-tattd-studio-poc
    ---

    Traditional Japanese irezumi has historic associations with the yakuza,
    and full-body suits remain socially fraught in Japan despite a thriving
    international community.
    """
)


def test_ingest_then_retrieve_returns_chunk_with_source() -> None:
    chunks = parse_chunks_from_markdown(CORPUS)
    assert len(chunks) == 3

    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus_test")
    embedder = DeterministicTextEmbeddingClient(dim=1024)

    n = ingest_corpus(
        store=store,
        collection="knowledge_corpus_test",
        chunks=chunks,
        embedder=embedder,
    )
    assert n == 3

    retriever = KnowledgeRetriever(
        store=store,
        collection="knowledge_corpus_test",
        embedder=embedder,
    )

    # Query using the exact body of the aftercare chunk; the deterministic
    # stub guarantees that this query embedding equals that chunk's body
    # embedding, so it should be the top hit.
    target = next(c for c in chunks if c.chunk_id == "aftercare-saniderm")
    hits = retriever.retrieve(target.body, k=3)

    assert len(hits) >= 1
    top = hits[0]
    assert top.chunk_id == "aftercare-saniderm"
    assert top.source_url == "https://example.com/aftercare/saniderm"
    assert top.area == "aftercare"
    assert "Saniderm" in top.body


def test_retrieve_rejects_empty_query() -> None:
    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus_test")
    retriever = KnowledgeRetriever(
        store=store,
        collection="knowledge_corpus_test",
        embedder=DeterministicTextEmbeddingClient(dim=1024),
    )
    import pytest

    with pytest.raises(ValueError):
        retriever.retrieve("", k=3)
