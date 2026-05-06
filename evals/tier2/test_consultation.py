"""Tier 2 — Consultation multi-turn quality.

Per the implementation plan's Eval Surface, this layer measures
quality + faithfulness + knowledge retention across multi-turn
Consultation sessions using DeepEval's ``AnswerRelevancy``,
``Faithfulness``, ``ConversationRelevancy``, and ``KnowledgeRetention``
primitives.

Default CI run computes deterministic IR-style metrics over the
conversation traces fixture (``data/eval/conversation_traces.jsonl``):
intent-substring coverage and grounding-area match. The DeepEval LLM
primitives run only under ``RUN_LIVE_CONSULTATION_EVAL=1``.

The Eval Report is committed at
``evals/results/consultation_tier2_latest.md``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from tattd_studio.consultation import ConsultationSession
from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    KnowledgeRetriever,
    ingest_corpus,
)
from tattd_studio.knowledge.ingest import load_chunks_from_dir
from tattd_studio.vectordb import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = REPO_ROOT / "data" / "eval" / "conversation_traces.jsonl"
KNOWLEDGE_DIR = REPO_ROOT / "data" / "knowledge"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "consultation_tier2_latest.md"


def _build_retriever() -> KnowledgeRetriever:
    chunks = load_chunks_from_dir(KNOWLEDGE_DIR)
    store = VectorStore(location=":memory:")
    store.create_collection("kc_consultation_eval")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    ingest_corpus(
        store=store,
        collection="kc_consultation_eval",
        chunks=chunks,
        embedder=embedder,
    )
    return KnowledgeRetriever(
        store=store, collection="kc_consultation_eval", embedder=embedder
    )


def _run_trace(retriever: KnowledgeRetriever, turns: list[str]) -> ConsultationSession:
    session = ConsultationSession(retriever=retriever)
    for turn in turns:
        session.advance(turn)
    return session


def _intent_coverage(session: ConsultationSession, expected_substrings: list[str]) -> float:
    """Fraction of expected substrings present in the final Intent."""
    if not expected_substrings:
        return 1.0
    if session.intent is None:
        return 0.0
    text = session.intent.refined_description.lower()
    hits = sum(1 for s in expected_substrings if s.lower() in text)
    return hits / len(expected_substrings)


def _area_match(session: ConsultationSession, expected_areas: list[str]) -> float:
    """Whether the session's grounding chunks cover the expected areas (lenient)."""
    if not expected_areas:
        return 1.0
    seen = {chunk.area for turn in session.history for chunk in turn.grounding_chunks}
    hits = sum(1 for a in expected_areas if a in seen)
    return hits / len(expected_areas)


def _maybe_run_judge_metrics(
    session: ConsultationSession, traces: list[dict]
) -> dict[str, float] | None:
    """Live mode: DeepEval Answer / Faithfulness / Conversation / Retention metrics."""
    if os.environ.get("RUN_LIVE_CONSULTATION_EVAL") != "1":
        return None
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ConversationRelevancyMetric,
        FaithfulnessMetric,
        KnowledgeRetentionMetric,
    )
    from deepeval.test_case import LLMTestCase

    answer = AnswerRelevancyMetric()
    faith = FaithfulnessMetric()
    convo = ConversationRelevancyMetric()
    retention = KnowledgeRetentionMetric()

    answer_scores: list[float] = []
    faith_scores: list[float] = []
    convo_scores: list[float] = []
    retention_scores: list[float] = []

    for trace in traces:
        last_turn = session.history[-1]
        case = LLMTestCase(
            input=trace["turns"][-1],
            actual_output=last_turn.studio_response,
            retrieval_context=session.retrieval_context,
        )
        answer.measure(case)
        faith.measure(case)
        convo.measure(case)
        retention.measure(case)
        answer_scores.append(float(answer.score or 0.0))
        faith_scores.append(float(faith.score or 0.0))
        convo_scores.append(float(convo.score or 0.0))
        retention_scores.append(float(retention.score or 0.0))

    n = len(traces) or 1
    return {
        "AnswerRelevancy": sum(answer_scores) / n,
        "Faithfulness": sum(faith_scores) / n,
        "ConversationRelevancy": sum(convo_scores) / n,
        "KnowledgeRetention": sum(retention_scores) / n,
    }


def _emit_report(
    *,
    n_traces: int,
    mean_intent_coverage: float,
    mean_area_match: float,
    judge_scores: dict[str, float] | None,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Consultation Tier 2 — Eval Report",
        "",
        f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
        f"- Conversation traces: `data/eval/conversation_traces.jsonl` ({n_traces} traces)",
        "- Mode: " + ("live" if judge_scores is not None else "ci"),
        "",
        "## Deterministic metrics (CI baseline)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Intent-substring coverage | {mean_intent_coverage:.3f} |",
        f"| Grounding-area match | {mean_area_match:.3f} |",
        "",
        "**Intent-substring coverage** — fraction of the trace's expected",
        "substrings (e.g. style and placement keywords) present in the final",
        "Intent's refined_description after all turns.",
        "",
        "**Grounding-area match** — fraction of expected Knowledge Corpus",
        "areas (taxonomy / placement / aftercare / ip / cultural) covered",
        "by the chunks retrieved across the session.",
        "",
        "## DeepEval primitives (live only)",
        "",
    ]
    if judge_scores:
        lines.extend(
            [
                "| Metric | Mean score |",
                "|---|---|",
                *[f"| {k} | {v:.3f} |" for k, v in judge_scores.items()],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Set `RUN_LIVE_CONSULTATION_EVAL=1` to run DeepEval's",
                "`AnswerRelevancy`, `Faithfulness`, `ConversationRelevancy`,",
                "and `KnowledgeRetention` metrics with an LLM judge. CI",
                "skips them by default.",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines))


def test_consultation_multi_turn_metrics() -> None:
    traces = [
        json.loads(line)
        for line in TRACES_PATH.read_text().splitlines()
        if line.strip()
    ]
    retriever = _build_retriever()

    intent_coverages: list[float] = []
    area_matches: list[float] = []
    last_session: ConsultationSession | None = None

    for trace in traces:
        session = _run_trace(retriever, trace["turns"])
        intent_coverages.append(
            _intent_coverage(session, trace["expected_intent_substrings"])
        )
        area_matches.append(_area_match(session, trace["expected_grounding_areas"]))
        last_session = session

    mean_intent = sum(intent_coverages) / len(intent_coverages)
    mean_area = sum(area_matches) / len(area_matches)
    judge_scores = (
        _maybe_run_judge_metrics(last_session, traces) if last_session else None
    )

    _emit_report(
        n_traces=len(traces),
        mean_intent_coverage=mean_intent,
        mean_area_match=mean_area,
        judge_scores=judge_scores,
    )

    assert 0.0 <= mean_intent <= 1.0
    assert 0.0 <= mean_area <= 1.0
    assert mean_intent >= 0.7, f"intent coverage too low: {mean_intent}"
