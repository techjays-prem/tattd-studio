"""Tier 3 stretch — golden-set expansion via the synthesizer.

Default CI runs the deterministic permutation augmenter and asserts the
output shape matches the original Anatomy Golden Set. Live mode
(`RUN_LIVE_SYNTHESIZER=1`) additionally exercises DeepEval's
``Synthesizer`` against the Knowledge Corpus.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from tattd_studio.eval_synth import expand_anatomy_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
ANATOMY_GOLDEN_PATH = REPO_ROOT / "data" / "eval" / "anatomy_cases.jsonl"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "tier3_synthesizer_latest.md"


def _emit_report(*, n_seed: int, n_expanded: int, judge_count: int | None) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    judge_row = (
        f"| Live Synthesizer cases generated | {judge_count} |"
        if judge_count is not None
        else "| Live Synthesizer cases generated | n/a — set `RUN_LIVE_SYNTHESIZER=1` |"
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Tier 3 Synthesizer — Eval Report",
                "",
                f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
                f"- Anatomy seed Golden Set: {n_seed} cases",
                f"- Mode: {'live' if judge_count is not None else 'ci'}",
                "",
                "## Golden-set expansion",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Seed Anatomy cases | {n_seed} |",
                f"| Permutation-augmented cases (CI) | {n_expanded} |",
                judge_row,
                "",
                "## Methods",
                "",
                "- **Permutation augmenter (CI)** — `expand_anatomy_cases(seed, n)`",
                "  samples body-part / prompt / size combinations consistent",
                "  with the expected validity. Deterministic given a seed.",
                "  Useful for stress-testing Anatomy Critic thresholds.",
                "- **Live Synthesizer (`RUN_LIVE_SYNTHESIZER=1`)** —",
                "  DeepEval's `Synthesizer.generate_goldens_from_contexts(...)`",
                "  produces semantically novel cases anchored on Knowledge",
                "  Corpus contexts. Requires an LLM judge.",
                "",
                "These outputs are *not* committed back into the seed Golden",
                "Set automatically; the developer reviews each batch before",
                "promoting it. Tier 1 evals continue to assert against the",
                "hand-curated seed.",
                "",
            ]
        )
    )


def test_permutation_augmenter_produces_well_shaped_anatomy_cases() -> None:
    seed = [
        json.loads(line)
        for line in ANATOMY_GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]
    n_target = 30
    expanded = expand_anatomy_cases(seed, n_target, seed_random=42)

    assert len(expanded) == n_target
    seed_keys = set(seed[0].keys())
    for case in expanded:
        # Same shape as the seed (modulo internal _synthetic_source).
        for key in seed_keys:
            assert key in case, f"missing key in augmented case: {key}"
        assert case["_synthetic_source"] == "permutation-augmenter-v1"
        assert isinstance(case["expected_placement_valid"], bool)
        assert case["size_inches"] > 0

    # Roughly half should be valid / invalid (the augmenter alternates).
    n_valid = sum(1 for c in expanded if c["expected_placement_valid"])
    assert 12 <= n_valid <= 18, f"unexpected valid/invalid split: {n_valid}/{n_target}"


def test_permutation_augmenter_is_deterministic() -> None:
    seed = [
        json.loads(line)
        for line in ANATOMY_GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]
    a = expand_anatomy_cases(seed, 10, seed_random=42)
    b = expand_anatomy_cases(seed, 10, seed_random=42)
    assert a == b


def test_emit_report_committed_after_run() -> None:
    seed = [
        json.loads(line)
        for line in ANATOMY_GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]
    expanded = expand_anatomy_cases(seed, 30, seed_random=42)

    judge_count: int | None = None
    if os.environ.get("RUN_LIVE_SYNTHESIZER") == "1":
        from tattd_studio.eval_synth import synthesize_with_judge

        live_cases = synthesize_with_judge(seed, n=10)
        judge_count = len(live_cases)

    _emit_report(n_seed=len(seed), n_expanded=len(expanded), judge_count=judge_count)

    assert REPORT_PATH.exists()
    body = REPORT_PATH.read_text()
    assert "# Tier 3 Synthesizer" in body
    assert "Permutation-augmented cases" in body
