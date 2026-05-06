"""Drift detector — parses Eval Reports and flags regressions vs baseline."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Matches a markdown table row of the shape:
#   `| Precision | 0.967 | ...`
# capturing the label and the first numeric value.
_TABLE_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([0-9]+\.?[0-9]*)\s*\|"
)


@dataclass(frozen=True)
class Regression:
    """One metric whose current value regressed from the baseline."""

    eval_path: str
    metric_label: str
    baseline: float
    current: float
    delta: float  # current − baseline (negative for regressions)
    tolerance: float


@dataclass(frozen=True)
class DriftReport:
    """Aggregate drift across all monitored metrics."""

    regressions: list[Regression]
    checked_count: int

    @property
    def is_clean(self) -> bool:
        return not self.regressions


@dataclass(frozen=True)
class _BaselineMetric:
    """One baseline entry parsed from the TOML."""

    eval_path: str
    metric_label: str
    baseline_value: float
    tolerance: float
    direction: str  # "higher_is_better" or "lower_is_better"


@dataclass(frozen=True)
class DriftBaselines:
    """All committed baseline metric values."""

    metrics: list[_BaselineMetric]

    @classmethod
    def from_toml(cls, path: Path) -> DriftBaselines:
        with path.open("rb") as f:
            data = tomllib.load(f)
        out: list[_BaselineMetric] = []
        for entry in data.get("metric", []):
            out.append(
                _BaselineMetric(
                    eval_path=entry["eval_path"],
                    metric_label=entry["metric_label"],
                    baseline_value=float(entry["baseline_value"]),
                    tolerance=float(entry["tolerance"]),
                    direction=entry.get("direction", "higher_is_better"),
                )
            )
        return cls(metrics=out)


def parse_metric(report_path: Path, metric_label: str) -> float | None:
    """Extract a numeric value for ``metric_label`` from a markdown report.

    Returns ``None`` when the label is absent or the value is non-numeric.
    Matches case-insensitively on the row label.
    """
    text = report_path.read_text()
    needle = metric_label.lower().strip()
    for line in text.splitlines():
        match = _TABLE_ROW_RE.match(line)
        if not match:
            continue
        label = match.group(1).lower().strip()
        if label == needle:
            try:
                return float(match.group(2))
            except ValueError:
                return None
    return None


def compute_drift(
    baselines: DriftBaselines, repo_root: Path
) -> DriftReport:
    """Compare current Eval Reports against committed baselines.

    Each baseline entry names an ``eval_path`` (relative to the repo
    root), a ``metric_label``, the ``baseline_value`` we expect, a
    ``tolerance`` (allowed delta), and a ``direction`` (whether higher
    or lower numbers are better). A regression is logged when the
    current value falls below the baseline by more than the tolerance,
    direction-aware.
    """
    regressions: list[Regression] = []
    checked = 0
    for entry in baselines.metrics:
        report_path = repo_root / entry.eval_path
        if not report_path.exists():
            continue
        current = parse_metric(report_path, entry.metric_label)
        if current is None:
            continue
        checked += 1
        delta = current - entry.baseline_value
        if entry.direction == "higher_is_better":
            if delta < -entry.tolerance:
                regressions.append(
                    Regression(
                        eval_path=entry.eval_path,
                        metric_label=entry.metric_label,
                        baseline=entry.baseline_value,
                        current=current,
                        delta=delta,
                        tolerance=entry.tolerance,
                    )
                )
        else:
            # lower_is_better
            if delta > entry.tolerance:
                regressions.append(
                    Regression(
                        eval_path=entry.eval_path,
                        metric_label=entry.metric_label,
                        baseline=entry.baseline_value,
                        current=current,
                        delta=delta,
                        tolerance=entry.tolerance,
                    )
                )
    return DriftReport(regressions=regressions, checked_count=checked)
