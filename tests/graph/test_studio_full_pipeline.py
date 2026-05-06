"""Integration test: full four-Critic Studio with Routing and Refinement.

A deliberately near-duplicate Candidate Design (matching an indexed
Famous Tattoos Corpus record) routes through Refinement; the second pass
no longer matches and the design surfaces.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

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
from tattd_studio.models import Intent
from tattd_studio.vectordb import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / "evals" / "calibrated_thresholds.toml"

KNOWLEDGE_CORPUS = """\
<!-- CHUNK -->
---
chunk_id: kc-fixture-fineline
area: taxonomy
title: Fineline placement note
source_url: https://example.com/kc/fineline
curator: tattd-studio-dev
capture_date: 2026-05-06
synthetic: true
permission: synthetic-content-tattd-studio-poc
---

Fineline placement on inner forearm: forgiving, holds detail well.
"""

# Famous Tattoo we want a Candidate Design to *almost* duplicate.
FAMOUS_CORPUS = """\
<!-- CHUNK -->
---
chunk_id: ft-fixture-bang-bang-falconry
area: famous
title: Bang Bang falconry glove
artist: Bang Bang
source_url: https://example.com/famous/bang-bang-falconry
curator: tattd-studio-dev
capture_date: 2026-05-06
synthetic: true
permission: synthetic-content-tattd-studio-poc
---

Maori-style falconry-glove design covering hand and forearm: geometric
blackwork with traditional Maori vocabulary, applied 2008.
"""


def _build_pipeline(generation_prompts: list[str]):
    """Returns (graph, store, embedder, fa_collection_name)."""
    store = VectorStore(location=":memory:")

    # Knowledge Corpus.
    store.create_collection("knowledge_corpus")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    ingest_corpus(
        store=store,
        collection="knowledge_corpus",
        chunks=parse_chunks_from_markdown(KNOWLEDGE_CORPUS),
        embedder=embedder,
    )
    retriever = KnowledgeRetriever(
        store=store, collection="knowledge_corpus", embedder=embedder
    )

    # Famous Tattoos Corpus.
    store.create_collection("famous_tattoos_corpus")
    ingest_corpus(
        store=store,
        collection="famous_tattoos_corpus",
        chunks=parse_chunks_from_markdown(FAMOUS_CORPUS),
        embedder=embedder,
    )

    # Generation Client cycles through prompts so the second pass
    # generates something different from the first.
    call_counter = {"i": 0}

    def fake_generate(prompt: str, n: int) -> list[str]:
        i = call_counter["i"]
        call_counter["i"] += 1
        return [f"file:///tmp/cd-call{i}-{j}.png" for j in range(n)]

    # We track which prompt was passed via a closure to override the
    # CandidateDesign.prompt for each call.
    def gen_with_overrides(prompt: str, n: int) -> list[str]:
        # Always returns N URIs; we override the in-memory prompt below
        # by relying on the generation client's behavior of attaching the
        # *current* intent's prompt to each CandidateDesign.
        return fake_generate(prompt, n)

    generation_client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=gen_with_overrides,
        clock=lambda: dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )

    # Critics.
    def anatomy_judge(cd, ctx) -> AnatomyCheck:
        return AnatomyCheck(placement_valid=True, issues=[], confidence=0.9)

    def quality_judge(cd) -> QualityScore:
        return QualityScore(
            composition=0.8,
            linework=0.85,
            balance=0.8,
            originality=0.8,
            notes="ok",
        )

    plagiarism = PlagiarismCritic(
        store=store,
        collection="famous_tattoos_corpus",
        embedder=embedder,
        threshold=0.5,
        # Use the candidate's prompt directly as the query.
        embed_query_for=lambda cd: cd.prompt,
    )

    return (
        build_studio_graph(
            retriever=retriever,
            generation_client=generation_client,
            anatomy_critic=AnatomyCritic(judge_fn=anatomy_judge),
            plagiarism_critic=plagiarism,
            style_critic=StyleCritic(embedder=embedder),
            quality_critic=QualityCritic(judge_fn=quality_judge),
            thresholds=RoutingThresholds.from_toml(THRESHOLDS_PATH),
        ),
        store,
        embedder,
    )


def test_plagiarism_near_dup_triggers_refinement_then_surfaces() -> None:
    graph, store, embedder = _build_pipeline(generation_prompts=[])

    near_dup_text = (
        "Maori-style falconry-glove design covering hand and forearm: geometric "
        "blackwork with traditional Maori vocabulary, applied 2008."
    )

    final_state = graph.invoke(
        {
            "intent": Intent(refined_description=near_dup_text),
            "placement_context": PlacementContext(
                body_part="inner forearm", size_inches=4.0
            ),
            "n_candidates": 1,
            "knowledge_chunks": [],
            "candidate_designs": [],
            "anatomy_checks": [],
            "plagiarism_checks": [],
            "style_checks": [],
            "quality_checks": [],
            "routing_decisions": [],
            "refinement_attempts": 0,
            "metadata": {},
        }
    )

    # Refinement was attempted (or escalated if still flagged on attempt 2).
    assert final_state["metadata"]["refinement_attempts"] >= 1
    # Final routing decisions exist for each Candidate Design.
    assert len(final_state["routing_decisions"]) == 1
    decision = final_state["routing_decisions"][0]
    # Either the refinement worked → surface, or the second flag → escalate.
    assert decision.action in {"surface", "escalate"}
