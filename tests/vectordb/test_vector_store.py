"""Behavioral tests for the Vector Store module.

Drives RED→GREEN against the public interface only. No test reaches into
`qdrant-client` types directly; everything goes through `VectorStore` so
the tests survive an internal driver swap.
"""

from __future__ import annotations

from tattd_studio.vectordb import (
    MULTIMODAL_EMBEDDING_DIM_1024,
    MULTIMODAL_EMBEDDING_DIM_3072,
    VISUAL_EMBEDDING_DIM,
    VectorStore,
)


def _store() -> VectorStore:
    return VectorStore(location=":memory:")


def test_create_collection_declares_three_named_vector_slots() -> None:
    store = _store()
    store.create_collection("artist_portfolio_index")

    schema = store.describe_collection("artist_portfolio_index")

    assert schema.named_vector_dims == {
        "multimodal-3072": MULTIMODAL_EMBEDDING_DIM_3072,
        "multimodal-1024": MULTIMODAL_EMBEDDING_DIM_1024,
        "visual": VISUAL_EMBEDDING_DIM,
    }
    assert schema.int8_quantization_enabled is True


def test_round_trip_upsert_and_named_vector_query() -> None:
    store = _store()
    store.create_collection("artist_portfolio_index")

    point_vectors = {
        "multimodal-3072": [0.1] * MULTIMODAL_EMBEDDING_DIM_3072,
        "multimodal-1024": [0.1] * MULTIMODAL_EMBEDDING_DIM_1024,
        "visual": [0.1] * VISUAL_EMBEDDING_DIM,
    }
    payload = {"artist_slug": "kira-inks", "style": "fineline"}
    store.upsert_point(
        collection="artist_portfolio_index",
        point_id=1,
        vectors=point_vectors,
        payload=payload,
    )

    hits = store.query_named(
        collection="artist_portfolio_index",
        vector_name="multimodal-3072",
        query_vector=point_vectors["multimodal-3072"],
        limit=1,
    )

    assert len(hits) == 1
    assert hits[0].point_id == 1
    assert hits[0].payload["artist_slug"] == "kira-inks"
    assert hits[0].payload["style"] == "fineline"


def test_query_named_filters_by_payload() -> None:
    store = _store()
    store.create_collection("artist_portfolio_index")

    base_vec = [0.1] * MULTIMODAL_EMBEDDING_DIM_3072

    def _vectors(probe: float) -> dict[str, list[float]]:
        return {
            "multimodal-3072": [probe] * MULTIMODAL_EMBEDDING_DIM_3072,
            "multimodal-1024": [probe] * MULTIMODAL_EMBEDDING_DIM_1024,
            "visual": [probe] * VISUAL_EMBEDDING_DIM,
        }

    store.upsert_point(
        "artist_portfolio_index", 1, _vectors(0.1), {"style": "fineline"}
    )
    store.upsert_point(
        "artist_portfolio_index", 2, _vectors(0.1), {"style": "traditional"}
    )

    hits = store.query_named(
        collection="artist_portfolio_index",
        vector_name="multimodal-3072",
        query_vector=base_vec,
        limit=10,
        payload_filter={"style": "traditional"},
    )

    assert [h.point_id for h in hits] == [2]


def test_alias_swap_rebuild_atomically_replaces_target() -> None:
    store = _store()

    # Initial collection populated and pointed at by the alias.
    store.create_collection("artist_portfolio_index_v1")
    vec = {
        "multimodal-3072": [0.0] * MULTIMODAL_EMBEDDING_DIM_3072,
        "multimodal-1024": [0.0] * MULTIMODAL_EMBEDDING_DIM_1024,
        "visual": [0.0] * VISUAL_EMBEDDING_DIM,
    }
    store.upsert_point(
        "artist_portfolio_index_v1", 1, vec, {"generation": "v1"}
    )
    store.set_alias("artist_portfolio_index", "artist_portfolio_index_v1")

    # Build callback creates the new generation with fresh content.
    def build_v2(name: str) -> None:
        store.create_collection(name)
        store.upsert_point(name, 1, vec, {"generation": "v2"})

    store.alias_swap_rebuild(
        alias="artist_portfolio_index",
        new_collection="artist_portfolio_index_v2",
        build_fn=build_v2,
    )

    # Alias-routed query now hits the v2 collection.
    hits = store.query_named(
        collection="artist_portfolio_index",
        vector_name="multimodal-3072",
        query_vector=vec["multimodal-3072"],
        limit=1,
    )
    assert hits[0].payload["generation"] == "v2"

    # Old collection is gone.
    assert store.collection_exists("artist_portfolio_index_v1") is False
    assert store.collection_exists("artist_portfolio_index_v2") is True
