"""Tier 3 stretch — drift detection.

Loads ``evals/results/history/baselines.toml`` and compares each
committed baseline against the matching live Eval Report under
``evals/results/``. Fails the CI build when any monitored metric
regresses beyond its declared tolerance.

The companion scheduled workflow at ``.github/workflows/drift.yml``
runs this same gate on a daily cron against ``main`` so silent decay
of upstream provider quality is caught even without PR activity.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from tattd_studio.drift import DriftBaselines, compute_drift

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_PATH = REPO_ROOT / "evals" / "results" / "history" / "baselines.toml"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "drift_latest.md"


def _emit_report(checked: int, regressions_lines: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Drift Detection — Eval Report",
                "",
                f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
                "- Baselines file: `evals/results/history/baselines.toml`",
                f"- Metrics checked: {checked}",
                f"- Regressions: {len(regressions_lines)}",
                "",
                "## Status",
                "",
                "✅ All monitored metrics within tolerance."
                if not regressions_lines
                else "🚨 Regressions detected:",
                "",
                *regressions_lines,
                "",
                "## How to bump a baseline",
                "",
                "Update the matching `[[metric]]` row in",
                "`evals/results/history/baselines.toml` in the same PR that",
                "produces the new Eval Report. The PR description should",
                "explain why the bar moved.",
                "",
            ]
        )
    )


def test_no_regression_against_committed_baselines() -> None:
    baselines = DriftBaselines.from_toml(BASELINES_PATH)
    report = compute_drift(baselines, REPO_ROOT)

    lines: list[str] = []
    for r in report.regressions:
        lines.append(
            f"- `{r.eval_path}` :: **{r.metric_label}** — "
            f"baseline {r.baseline:.3f}, current {r.current:.3f} "
            f"(delta {r.delta:+.3f}, tolerance ±{r.tolerance:.3f})"
        )

    _emit_report(report.checked_count, lines)

    assert report.checked_count > 0, "no baselines were matched against live reports"
    assert report.is_clean, (
        f"{len(report.regressions)} metric regression(s) detected — "
        "see `evals/results/drift_latest.md`"
    )
