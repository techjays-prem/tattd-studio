"""Test the alias-swap rebuild script."""

from __future__ import annotations

import datetime as dt

from tattd_studio.vectordb import (
    MULTIMODAL_EMBEDDING_DIM_1024,
    MULTIMODAL_EMBEDDING_DIM_3072,
    VISUAL_EMBEDDING_DIM,
    VectorStore,
)
from tattd_studio.vectordb.reindex import rebuild


def _vec(probe: float) -> dict[str, list[float]]:
    return {
        "multimodal-3072": [probe] * MULTIMODAL_EMBEDDING_DIM_3072,
        "multimodal-1024": [probe] * MULTIMODAL_EMBEDDING_DIM_1024,
        "visual": [probe] * VISUAL_EMBEDDING_DIM,
    }


def test_rebuild_creates_timestamped_collection_and_swaps_alias() -> None:
    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus_seed")
    store.upsert_point("knowledge_corpus_seed", 1, _vec(0.0), {"gen": "seed"})
    store.set_alias("knowledge_corpus", "knowledge_corpus_seed")

    populated: list[str] = []

    def populate(s: VectorStore, name: str) -> None:
        s.upsert_point(name, 1, _vec(0.0), {"gen": "fresh"})
        populated.append(name)

    new_name = rebuild(
        store=store,
        alias="knowledge_corpus",
        populate=populate,
        timestamp=dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=dt.UTC),
    )

    assert new_name == "knowledge_corpus_20260506T120000"
    assert populated == [new_name]

    hits = store.query_named(
        "knowledge_corpus",
        "multimodal-3072",
        _vec(0.0)["multimodal-3072"],
        limit=1,
    )
    assert hits[0].payload["gen"] == "fresh"
    assert store.collection_exists("knowledge_corpus_seed") is False
