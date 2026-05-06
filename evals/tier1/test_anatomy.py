"""Tier 1 eval — Anatomy Critic precision and recall on the Golden Set.

Pytest-discoverable per the Eval Harness contract. Default CI run uses the
deterministic ``heuristic_anatomy_judge`` so the eval is reproducible
without API credentials. To run against the live Gemini Pro VLM judge:

    RUN_LIVE_ANATOMY_EVAL=1 GEMINI_API_KEY=... uv run pytest evals/tier1/test_anatomy.py

The eval emits a markdown Eval Report alongside the assertions so the
latest committed report under ``evals/results/`` always reflects the most
recent run.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tomllib
from pathlib import Path

from tattd_studio.graph.critics import (
    AnatomyCritic,
    PlacementContext,
    build_gemini_anatomy_judge,
    heuristic_anatomy_judge,
)
from tattd_studio.models import CandidateDesign

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET_PATH = REPO_ROOT / "data" / "eval" / "anatomy_cases.jsonl"
THRESHOLDS_PATH = REPO_ROOT / "evals" / "calibrated_thresholds.toml"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "anatomy_critic_latest.md"


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_SET_PATH.read_text().splitlines() if line.strip()]


def _load_thresholds() -> dict:
    with THRESHOLDS_PATH.open("rb") as f:
        return tomllib.load(f)["anatomy_critic"]


def _candidate_for(case: dict) -> CandidateDesign:
    return CandidateDesign(
        image_uri=f"file:///tmp/{case['id']}.png",
        prompt=case["prompt"],
        source_model_id="eval-fixture",
        metadata={"prompt_template_version": "v1.0"},
        created_at=dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )


def _score(judge_label: str, judge_fn) -> dict:
    cases = _load_cases()
    critic = AnatomyCritic(judge_fn=judge_fn)
    tp = fp = tn = fn = 0
    for case in cases:
        verdict = critic.check(
            _candidate_for(case),
            PlacementContext(
                body_part=case["body_part"],
                size_inches=case["size_inches"],
            ),
        )
        # Positive class = invalid placement.
        actual_invalid = not verdict.placement_valid
        expected_invalid = not case["expected_placement_valid"]
        if actual_invalid and expected_invalid:
            tp += 1
        elif actual_invalid and not expected_invalid:
            fp += 1
        elif not actual_invalid and not expected_invalid:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "judge": judge_label,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "n": len(cases),
    }


def _emit_report(metrics: dict, thresholds: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Anatomy Critic — Eval Report",
        "",
        f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
        f"- Golden Set: `data/eval/anatomy_cases.jsonl` ({metrics['n']} cases)",
        f"- Judge: `{metrics['judge']}`",
        "",
        "| Metric | Value | Threshold | Status |",
        "|---|---|---|---|",
        f"| Precision | {metrics['precision']:.3f} | {thresholds['precision_min']:.2f} | "
        f"{'PASS' if metrics['precision'] >= thresholds['precision_min'] else 'FAIL'} |",
        f"| Recall | {metrics['recall']:.3f} | {thresholds['recall_min']:.2f} | "
        f"{'PASS' if metrics['recall'] >= thresholds['recall_min'] else 'FAIL'} |",
        "",
        "## Confusion matrix",
        "",
        "Positive class: invalid placement (the verdict the Critic must catch).",
        "",
        "| | Predicted invalid | Predicted valid |",
        "|---|---|---|",
        f"| **Actually invalid** | TP={metrics['tp']} | FN={metrics['fn']} |",
        f"| **Actually valid** | FP={metrics['fp']} | TN={metrics['tn']} |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def test_anatomy_critic_meets_calibrated_thresholds() -> None:
    """Default CI eval: heuristic baseline must clear committed thresholds."""
    thresholds = _load_thresholds()
    metrics = _score("heuristic-baseline-v1", heuristic_anatomy_judge)
    _emit_report(metrics, thresholds)

    assert metrics["precision"] >= thresholds["precision_min"], (
        f"precision {metrics['precision']:.3f} < {thresholds['precision_min']}"
    )
    assert metrics["recall"] >= thresholds["recall_min"], (
        f"recall {metrics['recall']:.3f} < {thresholds['recall_min']}"
    )


def test_live_anatomy_critic_against_gemini_pro() -> None:
    """Gated: scores the live VLM judge on the same Golden Set."""
    if os.environ.get("RUN_LIVE_ANATOMY_EVAL") != "1":
        import pytest

        pytest.skip("set RUN_LIVE_ANATOMY_EVAL=1 to enable")
    thresholds = _load_thresholds()
    metrics = _score("gemini-pro-vlm", build_gemini_anatomy_judge())
    _emit_report(metrics, thresholds)
    assert metrics["precision"] >= thresholds["precision_min"]
    assert metrics["recall"] >= thresholds["recall_min"]
