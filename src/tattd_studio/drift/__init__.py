"""Drift detection.

Per the implementation plan's Eval Surface, drift detection is the
second Tier 3 stretch: a scheduled CI run that compares current Eval
Reports to historical baselines and flags regressions over time.

This module exposes:

- ``DriftBaselines`` — loads committed baseline metric values from
  ``evals/results/history/baselines.toml``.
- ``parse_metric(report_path, metric_label)`` — extracts a numeric
  value from a markdown report's metric table.
- ``compute_drift(baselines, current)`` — returns the list of metrics
  that regressed beyond their configured tolerance.

The CI gate at ``evals/tier3/test_drift.py`` runs this against every
committed Eval Report and fails the build on regression. The companion
scheduled workflow at ``.github/workflows/drift.yml`` runs the same
gate on a daily cron against ``main`` so silent decay of upstream
provider quality is caught even without PR activity.
"""

from tattd_studio.drift.detector import (
    DriftBaselines,
    DriftReport,
    Regression,
    compute_drift,
    parse_metric,
)

__all__ = [
    "DriftBaselines",
    "DriftReport",
    "Regression",
    "compute_drift",
    "parse_metric",
]
