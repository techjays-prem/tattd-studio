"""End-to-end test for the minimal Studio graph (slice #6).

Drives the compiled LangGraph from a fixture Intent through Consultation,
Generation, and Anatomy Critic with all dependencies injected as stubs so
no API calls happen in CI.
"""

from __future__ import annotations

import datetime as dt

from tattd_studio.generation import GenerationClient
from tattd_studio.graph.critics import AnatomyCheck, AnatomyCritic, PlacementContext
from tattd_studio.graph.studio import build_studio_graph
from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    KnowledgeRetriever,
    ingest_corpus,
    parse_chunks_from_markdown,
)
from tattd_studio.models import Intent
from tattd_studio.vectordb import VectorStore

CORPUS = """\
<!-- CHUNK -->
---
chunk_id: placement-inner-forearm
area: placement
title: Inner forearm
source_url: https://example.com/p/inner-forearm
curator: tattd-studio-dev
capture_date: 2026-05-06
synthetic: true
permission: synthetic-content-tattd-studio-poc
---

The inner forearm holds detail well and is forgiving for fineline work.
"""


def _build_test_studio() -> object:
    # Vector Store + Knowledge Retriever
    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus_test")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    chunks = parse_chunks_from_markdown(CORPUS)
    ingest_corpus(
        store=store,
        collection="knowledge_corpus_test",
        chunks=chunks,
        embedder=embedder,
    )
    retriever = KnowledgeRetriever(
        store=store,
        collection="knowledge_corpus_test",
        embedder=embedder,
    )

    # Generation Client with a fake provider call.
    def fake_generate(prompt: str, n: int) -> list[str]:
        return [f"file:///tmp/cd-{i}.png" for i in range(n)]

    generation_client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=fake_generate,
        clock=lambda: dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )

    # Anatomy Critic with a deterministic judge.
    def judge(cd, ctx) -> AnatomyCheck:
        return AnatomyCheck(
            placement_valid=True, issues=[], confidence=0.92
        )

    critic = AnatomyCritic(judge_fn=judge)

    return build_studio_graph(
        retriever=retriever,
        generation_client=generation_client,
        anatomy_critic=critic,
    )


def test_studio_graph_runs_consultation_generation_anatomy_to_completion() -> None:
    graph = _build_test_studio()

    final_state = graph.invoke(
        {
            "intent": Intent(
                refined_description=(
                    "fineline minimalist mountain on inner forearm, ~3 inches"
                )
            ),
            "placement_context": PlacementContext(
                body_part="inner forearm", size_inches=3.0
            ),
            "n_candidates": 2,
            "knowledge_chunks": [],
            "candidate_designs": [],
            "anatomy_checks": [],
            "plagiarism_checks": [],
            "style_checks": [],
            "quality_checks": [],
            "routing_decisions": [],
            "refinement_attempts": 0,
            "chosen_design_index": 0,
            "matched_artists": [],
            "metadata": {},
        }
    )

    # Consultation populated retrieved chunks.
    assert len(final_state["knowledge_chunks"]) >= 1
    assert final_state["knowledge_chunks"][0].source_url
    # Generation produced N candidates.
    assert len(final_state["candidate_designs"]) == 2
    assert final_state["candidate_designs"][0].image_uri.startswith("file://")
    # Anatomy Critic emitted one verdict per candidate.
    assert len(final_state["anatomy_checks"]) == 2
    assert all(c.placement_valid is True for c in final_state["anatomy_checks"])
    # Trace metadata recorded latency.
    assert "latency_seconds" in final_state["metadata"]
    assert final_state["metadata"]["latency_seconds"] >= 0.0


def test_studio_graph_does_not_carry_image_bytes_in_state() -> None:
    """State carries Candidate Designs by URI, never raw bytes — per
    IMPLEMENTATION_PLAN.md → 'Multimodal state in LangGraph'."""
    graph = _build_test_studio()
    final_state = graph.invoke(
        {
            "intent": Intent(refined_description="x"),
            "placement_context": PlacementContext(body_part="inner forearm", size_inches=2.0),
            "n_candidates": 1,
            "knowledge_chunks": [],
            "candidate_designs": [],
            "anatomy_checks": [],
            "plagiarism_checks": [],
            "style_checks": [],
            "quality_checks": [],
            "routing_decisions": [],
            "refinement_attempts": 0,
            "chosen_design_index": 0,
            "matched_artists": [],
            "metadata": {},
        }
    )
    for cd in final_state["candidate_designs"]:
        # CandidateDesign envelope carries image_uri, not bytes.
        assert isinstance(cd.image_uri, str)
        assert not hasattr(cd, "image_bytes")
