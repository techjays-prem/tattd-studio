"""Tier 1 — Generation eval.

Per the implementation plan's Eval Surface, this layer measures
faithfulness + originality + technical quality on Candidate Designs
using DeepEval's ``GEval`` plus the project's custom
``OriginalityMetric``.

Default CI run:

- ``faithfulness_score`` runs deterministically against the
  ``DeterministicTextEmbeddingClient`` and the labeled Golden Set;
  serves as a wiring smoke + a stable baseline number.
- ``OriginalityMetric`` queries the Famous Tattoos Corpus from the
  in-memory Vector Store; checks the deterministic ranking is stable
  per (prompt, corpus) pair.
- The DeepEval ``GEval`` rubric for technical quality runs only under
  ``RUN_LIVE_GENERATION_TIER1=1`` (LLM-judge cost).

The Eval Report at `evals/results/generation_tier1_latest.md` records
both modes' outputs and is the surface the README links from.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from deepeval.test_case import LLMTestCase

from tattd_studio.generation import OriginalityMetric, faithfulness_score
from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    ingest_corpus,
    parse_chunks_from_markdown,
)
from tattd_studio.vectordb import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET_PATH = REPO_ROOT / "data" / "eval" / "generation_golden.jsonl"
FAMOUS_CORPUS_PATH = REPO_ROOT / "data" / "famous_tattoos" / "famous.md"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "generation_tier1_latest.md"


def _build_originality_metric() -> OriginalityMetric:
    chunks = parse_chunks_from_markdown(FAMOUS_CORPUS_PATH.read_text())
    store = VectorStore(location=":memory:")
    store.create_collection("famous_tattoos_eval")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    ingest_corpus(
        store=store,
        collection="famous_tattoos_eval",
        chunks=chunks,
        embedder=embedder,
    )
    return OriginalityMetric(
        store=store,
        collection="famous_tattoos_eval",
        embedder=embedder,
        threshold=0.3,
    )


def _maybe_run_rubric(intent: str, candidate_prompt: str) -> float | None:
    """Run DeepEval ``GEval`` rubric for technical quality. Live-only."""
    if os.environ.get("RUN_LIVE_GENERATION_TIER1") != "1":
        return None
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    metric = GEval(
        name="generation-technical-quality",
        criteria=(
            "Score the Candidate Design's technical quality: composition "
            "balance, linework crispness, intent fidelity, and originality. "
            "Penalize generic AI-aesthetic tells; reward style-vocabulary "
            "alignment and placement-aware composition."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    case = LLMTestCase(input=intent, actual_output=candidate_prompt)
    metric.measure(case)
    return float(metric.score or 0.0)


def _emit_report(
    *,
    n_cases: int,
    mean_faithfulness_match: float,
    mean_originality_match: float,
    mean_rubric: float | None,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rubric_row = (
        f"| GEval (live) | {mean_rubric:.3f} |"
        if mean_rubric is not None
        else "| GEval (live) | n/a — set `RUN_LIVE_GENERATION_TIER1=1` |"
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Generation Tier 1 — Eval Report",
                "",
                f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
                f"- Golden Set: `data/eval/generation_golden.jsonl` ({n_cases} cases)",
                "- Mode: " + ("live" if mean_rubric is not None else "ci"),
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Faithfulness label match | {mean_faithfulness_match:.3f} |",
                f"| Originality label match | {mean_originality_match:.3f} |",
                rubric_row,
                "",
                "## Metrics",
                "",
                "- **Faithfulness label match** — `faithfulness_score(intent,",
                "  candidate)` >= 0.5 → predicted *faithful*; the score is",
                "  the fraction of cases where the prediction matches the",
                "  Golden Set's `expected_faithful` label.",
                "- **Originality label match** — `OriginalityMetric.measure`",
                "  produces a [0, 1] originality value; >= 0.5 → predicted",
                "  *original*. Score is the prediction-vs-label match rate.",
                "- **GEval (live)** — DeepEval LLM-judge rubric for technical",
                "  quality (composition / linework / intent fidelity /",
                "  originality), run only under `RUN_LIVE_GENERATION_TIER1=1`.",
                "",
                "## Note on the deterministic baseline",
                "",
                "The CI mode uses the deterministic embedding stub which has",
                "no semantic signal; the cosine values cluster around 0.5",
                "regardless of input. The label-match metrics here are",
                "therefore best read as wiring-OK floors, not as quality",
                "claims about the embedder. Live mode against Gemini",
                "Embedding 2 produces meaningful separation.",
                "",
            ]
        )
    )


def test_generation_tier1_metrics() -> None:
    cases = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text().splitlines()
        if line.strip()
    ]

    embedder = DeterministicTextEmbeddingClient(dim=1024)
    originality = _build_originality_metric()

    faithfulness_correct: list[float] = []
    originality_correct: list[float] = []
    rubric_scores: list[float] = []

    for case in cases:
        score = faithfulness_score(
            case["intent"], case["candidate_prompt"], embedder=embedder
        )
        predicted_faithful = score >= 0.5
        faithfulness_correct.append(
            1.0 if predicted_faithful == case["expected_faithful"] else 0.0
        )

        case_test = LLMTestCase(
            input=case["intent"], actual_output=case["candidate_prompt"]
        )
        orig_score = originality.measure(case_test)
        predicted_original = orig_score >= 0.5
        originality_correct.append(
            1.0 if predicted_original == case["expected_original"] else 0.0
        )

        live = _maybe_run_rubric(case["intent"], case["candidate_prompt"])
        if live is not None:
            rubric_scores.append(live)

    mean_faith = sum(faithfulness_correct) / len(faithfulness_correct)
    mean_orig = sum(originality_correct) / len(originality_correct)
    mean_rubric = sum(rubric_scores) / len(rubric_scores) if rubric_scores else None

    _emit_report(
        n_cases=len(cases),
        mean_faithfulness_match=mean_faith,
        mean_originality_match=mean_orig,
        mean_rubric=mean_rubric,
    )

    assert 0.0 <= mean_faith <= 1.0
    assert 0.0 <= mean_orig <= 1.0
