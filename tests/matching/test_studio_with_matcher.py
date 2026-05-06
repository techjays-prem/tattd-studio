"""Studio graph end-to-end including the Two-Stage Matcher terminal node."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from tattd_studio.embeddings import DeterministicVisualEmbeddingClient
from tattd_studio.generation import GenerationClient
from tattd_studio.graph.critics import (
    AnatomyCheck,
    AnatomyCritic,
    PlacementContext,
    PlagiarismCritic,
    QualityCritic,
    QualityScore,
    StyleCritic,
)
from tattd_studio.graph.routing import RoutingThresholds
from tattd_studio.graph.studio import build_studio_graph
from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    KnowledgeRetriever,
    ingest_corpus,
    parse_chunks_from_markdown,
)
from tattd_studio.matching import (
    TwoStageMatcher,
    ingest_artist_portfolio_index,
    load_artist_records,
)
from tattd_studio.models import Intent
from tattd_studio.vectordb import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTISTS_JSONL = REPO_ROOT / "data" / "artists" / "artists.jsonl"
THRESHOLDS_PATH = REPO_ROOT / "evals" / "calibrated_thresholds.toml"

KNOWLEDGE_CORPUS = """\
<!-- CHUNK -->
---
chunk_id: kc-fixture
area: placement
title: fixture
source_url: https://example.com/kc/fixture
curator: tattd-studio-dev
capture_date: 2026-05-06
synthetic: true
permission: synthetic-content-tattd-studio-poc
---

Fineline placement on inner forearm holds detail well.
"""

FAMOUS_CORPUS = """\
<!-- CHUNK -->
---
chunk_id: ft-fixture
area: famous
title: fixture
artist: Fixture Famous Artist
source_url: https://example.com/ft/fixture
curator: tattd-studio-dev
capture_date: 2026-05-06
synthetic: true
permission: synthetic-content-tattd-studio-poc
---

Original ornamental composition with no near-duplicate signal.
"""


def test_studio_graph_routes_through_two_stage_matcher() -> None:
    store = VectorStore(location=":memory:")
    text_embedder = DeterministicTextEmbeddingClient(dim=1024)
    visual_embedder = DeterministicVisualEmbeddingClient(dim=768)

    # Knowledge Corpus.
    store.create_collection("knowledge_corpus")
    ingest_corpus(
        store=store,
        collection="knowledge_corpus",
        chunks=parse_chunks_from_markdown(KNOWLEDGE_CORPUS),
        embedder=text_embedder,
    )
    retriever = KnowledgeRetriever(
        store=store, collection="knowledge_corpus", embedder=text_embedder
    )

    # Famous Tattoos Corpus.
    store.create_collection("famous_tattoos_corpus")
    ingest_corpus(
        store=store,
        collection="famous_tattoos_corpus",
        chunks=parse_chunks_from_markdown(FAMOUS_CORPUS),
        embedder=text_embedder,
    )

    # Artist Portfolio Index.
    store.create_collection("artist_portfolio_index")
    ingest_artist_portfolio_index(
        store=store,
        collection="artist_portfolio_index",
        artists=load_artist_records(ARTISTS_JSONL),
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
    )

    def fake_generate(prompt: str, n: int) -> list[str]:
        return [f"file:///tmp/cd-{i}.png" for i in range(n)]

    generation_client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=fake_generate,
        clock=lambda: dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )

    plagiarism = PlagiarismCritic(
        store=store,
        collections=["artist_portfolio_index", "famous_tattoos_corpus"],
        embedder=text_embedder,
        threshold=0.999,  # high → no false flags from the deterministic stub
    )

    matcher = TwoStageMatcher(
        store=store,
        collection="artist_portfolio_index",
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        stage1_limit=20,
    )

    graph = build_studio_graph(
        retriever=retriever,
        generation_client=generation_client,
        anatomy_critic=AnatomyCritic(
            judge_fn=lambda cd, ctx: AnatomyCheck(
                placement_valid=True, issues=[], confidence=0.9
            )
        ),
        plagiarism_critic=plagiarism,
        style_critic=StyleCritic(embedder=text_embedder),
        quality_critic=QualityCritic(
            judge_fn=lambda cd: QualityScore(
                composition=0.8, linework=0.8, balance=0.8, originality=0.8, notes="ok"
            )
        ),
        thresholds=RoutingThresholds.from_toml(THRESHOLDS_PATH),
        matcher=matcher,
        matched_artists_k=5,
    )

    final_state = graph.invoke(
        {
            "intent": Intent(
                refined_description="fineline minimalist mountain on inner forearm"
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

    assert len(final_state["matched_artists"]) == 5
    for ranked in final_state["matched_artists"]:
        assert ranked.artist_slug
        assert ranked.portfolio_url.startswith("https://")
    assert final_state["metadata"]["matched_artist_count"] == 5
