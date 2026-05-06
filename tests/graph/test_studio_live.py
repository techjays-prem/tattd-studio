"""Gated live end-to-end test: drives the Studio graph against the real
Gemini providers (Embedding 2 + Nano Banana 2 / Pro + Gemini Pro VLM
judge). Skipped by default so CI never makes paid API calls.

To run:

    RUN_LIVE_STUDIO_TESTS=1 GEMINI_API_KEY=... uv run pytest tests/graph/test_studio_live.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

RUN_LIVE = os.environ.get("RUN_LIVE_STUDIO_TESTS") == "1"
HAS_KEY = bool(os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(
    not (RUN_LIVE and HAS_KEY),
    reason="set RUN_LIVE_STUDIO_TESTS=1 and GEMINI_API_KEY to enable",
)
def test_live_studio_surfaces_candidate_designs() -> None:
    from tattd_studio.generation import (
        GenerationClient,
        build_gemini_generate_fn,
    )
    from tattd_studio.graph.critics import (
        AnatomyCritic,
        PlacementContext,
        build_gemini_anatomy_judge,
    )
    from tattd_studio.graph.studio import build_studio_graph
    from tattd_studio.knowledge import (
        KnowledgeRetriever,
        build_gemini_text_embedding_client,
        ingest_corpus,
    )
    from tattd_studio.knowledge.ingest import load_chunks_from_dir
    from tattd_studio.models import Intent
    from tattd_studio.vectordb import VectorStore

    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus_live")
    embedder = build_gemini_text_embedding_client()
    chunks = load_chunks_from_dir(
        Path(__file__).resolve().parents[2] / "data" / "knowledge"
    )
    ingest_corpus(
        store=store,
        collection="knowledge_corpus_live",
        chunks=chunks,
        embedder=embedder,
    )
    retriever = KnowledgeRetriever(
        store=store, collection="knowledge_corpus_live", embedder=embedder
    )

    source_model_id = os.environ.get(
        "TATTD_GENERATION_SOURCE_MODEL_ID", "gemini-nano-banana-2"
    )
    graph = build_studio_graph(
        retriever=retriever,
        generation_client=GenerationClient(
            source_model_id=source_model_id,
            generate_fn=build_gemini_generate_fn(source_model_id),
        ),
        anatomy_critic=AnatomyCritic(judge_fn=build_gemini_anatomy_judge()),
    )

    state = graph.invoke(
        {
            "intent": Intent(
                refined_description=(
                    "fineline minimalist mountain on inner forearm, ~3 inches"
                )
            ),
            "placement_context": PlacementContext(
                body_part="inner forearm", size_inches=3.0
            ),
            "n_candidates": 1,
            "knowledge_chunks": [],
            "candidate_designs": [],
            "anatomy_checks": [],
            "metadata": {},
        }
    )

    assert len(state["candidate_designs"]) == 1
    assert state["candidate_designs"][0].image_uri.startswith("file://")
    assert len(state["anatomy_checks"]) == 1
    assert state["metadata"]["latency_seconds"] >= 0.0
