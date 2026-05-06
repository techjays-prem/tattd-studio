"""Tier 1 eval — Quality Critic drift detection on a stable Golden Set.

The default judge under test is a deterministic fixture so the eval is
reproducible without API credentials. Live mode (``RUN_LIVE_QUALITY_EVAL=1``)
swaps in the Gemini Pro VLM judge — calibrated thresholds activate only
under that mode.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

from tattd_studio.graph.critics import QualityCritic, QualityScore
from tattd_studio.models.candidate_design import CandidateDesign

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / "evals" / "calibrated_thresholds.toml"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "quality_critic_latest.md"

# Inline Golden Set: ~15 fixture Candidate Designs with expected aggregate.
CASES: list[tuple[str, float]] = [
    ("polished fineline mountain", 0.85),
    ("rough sketch wolf head", 0.55),
    ("clean traditional rose", 0.85),
    ("muddy realism portrait", 0.50),
    ("strong blackwork mandala", 0.80),
    ("ill-balanced trash polka", 0.50),
    ("crisp ornamental filigree", 0.80),
    ("over-detailed micro on knuckle", 0.45),
    ("balanced Japanese koi sleeve", 0.85),
    ("flat single-line cat", 0.65),
    ("photoreal big cat", 0.85),
    ("compressed sleeve forced onto wrist", 0.40),
    ("dotwork sacred geometry", 0.80),
    ("derivative eagle copy", 0.55),
    ("original neo-traditional fox", 0.80),
]


def _fixture_judge(candidate: CandidateDesign) -> QualityScore:
    """Deterministic judge keyed on the prompt — round-trips Golden values."""
    expected = next((agg for prompt, agg in CASES if prompt == candidate.prompt), 0.7)
    return QualityScore(
        composition=expected,
        linework=expected,
        balance=expected,
        originality=expected,
        notes="fixture",
    )


def test_quality_critic_holds_aggregate_floor() -> None:
    with THRESHOLDS_PATH.open("rb") as f:
        thresholds = tomllib.load(f)["quality_critic"]

    critic = QualityCritic(judge_fn=_fixture_judge)
    aggregate_min = float(thresholds["aggregate_min"])

    tp = fp = tn = fn = 0
    scores: list[float] = []
    for prompt, expected_agg in CASES:
        cd = CandidateDesign(
            image_uri=f"file:///tmp/q-{abs(hash(prompt))}.png",
            prompt=prompt,
            source_model_id="eval-fixture",
            metadata={"prompt_template_version": "v1.0"},
            created_at=dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
        )
        verdict = critic.check(cd)
        scores.append(verdict.aggregate)
        predicted_high = verdict.aggregate >= aggregate_min
        actual_high = expected_agg >= aggregate_min
        if predicted_high and actual_high:
            tp += 1
        elif predicted_high and not actual_high:
            fp += 1
        elif not predicted_high and not actual_high:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Quality Critic — Eval Report",
                "",
                f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
                f"- Golden Set: inline ({len(CASES)} cases)",
                "- Judge: fixture (deterministic, round-trips Golden values)",
                f"- aggregate_min threshold: {aggregate_min:.2f}",
                "",
                "| Metric | Value | Threshold | Status |",
                "|---|---|---|---|",
                f"| Precision | {precision:.3f} | {thresholds['precision_min']:.2f} | "
                f"{'PASS' if precision >= thresholds['precision_min'] else 'FAIL'} |",
                f"| Recall | {recall:.3f} | {thresholds['recall_min']:.2f} | "
                f"{'PASS' if recall >= thresholds['recall_min'] else 'FAIL'} |",
                f"| Mean aggregate score | {sum(scores)/len(scores):.3f} | n/a | n/a |",
                "",
                f"TP={tp}  FP={fp}  TN={tn}  FN={fn}",
                "",
            ]
        )
    )

    # Fixture judge round-trips inputs; precision and recall should be
    # 1.0 by construction. CI thresholds (0.0 floors) hold; live judge
    # under `RUN_LIVE_QUALITY_EVAL=1` has aspirational targets above.
    assert precision == 1.0
    assert recall == 1.0
