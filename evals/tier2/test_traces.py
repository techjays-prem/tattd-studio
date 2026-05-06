"""Tier 2 trace eval — records latency and (when available) token cost
for one end-to-end Studio run.

This tier targets agent-level metrics rather than component metrics. The
default CI run uses the in-memory stubs already exercised by
`tests/graph/test_studio_graph.py`, so the eval is reproducible without
provider credentials. The committed report under `evals/results/`
captures the latest measured latency.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "evals" / "results" / "studio_traces_latest.md"

CORPUS = """\
<!-- CHUNK -->
---
chunk_id: trace-eval-fixture
area: placement
title: Trace eval fixture
source_url: https://example.com/trace-eval-fixture
curator: tattd-studio-dev
capture_date: 2026-05-06
synthetic: true
permission: synthetic-content-tattd-studio-poc
---

Inner forearm fineline placement is forgiving for a 3-inch design.
"""


def _build_studio() -> object:
    store = VectorStore(location=":memory:")
    store.create_collection("kc_trace_eval")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    chunks = parse_chunks_from_markdown(CORPUS)
    ingest_corpus(
        store=store, collection="kc_trace_eval", chunks=chunks, embedder=embedder
    )
    retriever = KnowledgeRetriever(
        store=store, collection="kc_trace_eval", embedder=embedder
    )

    def fake_generate(prompt: str, n: int) -> list[str]:
        return [f"file:///tmp/trace-cd-{i}.png" for i in range(n)]

    generation_client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=fake_generate,
        clock=lambda: dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )

    def judge(cd, ctx) -> AnatomyCheck:
        return AnatomyCheck(placement_valid=True, issues=[], confidence=0.9)

    return build_studio_graph(
        retriever=retriever,
        generation_client=generation_client,
        anatomy_critic=AnatomyCritic(judge_fn=judge),
    )


def _emit_report(latency: float, candidates: int, chunks: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Studio End-to-End Trace — Eval Report",
                "",
                f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
                "- Mode: stubbed (deterministic embedder + fake generate_fn + fixed judge)",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Latency (s) | {latency:.4f} |",
                f"| Candidate Designs surfaced | {candidates} |",
                f"| Knowledge chunks retrieved | {chunks} |",
                "| Token cost | n/a (stubbed) |",
                "",
            ]
        )
    )


def test_studio_trace_records_latency() -> None:
    graph = _build_studio()
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
            "n_candidates": 4,
            "knowledge_chunks": [],
            "candidate_designs": [],
            "anatomy_checks": [],
            "metadata": {},
        }
    )
    latency = state["metadata"]["latency_seconds"]
    candidates = state["metadata"]["candidate_count"]
    chunks = state["metadata"]["chunk_count"]

    _emit_report(latency, candidates, chunks)

    # Sanity bounds: a stub run should take well under a second.
    assert 0.0 <= latency < 5.0
    assert candidates == 4
    assert chunks >= 1
