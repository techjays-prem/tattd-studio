"""Comparison Matrix runner.

Runs each Comparison Matrix column against the shared prompt set and
produces a ``ComparisonMatrix`` row-per-column table. Each entry has a
``image_uri_for(prompt)`` callable so the runner stays agnostic to
whether a column is the Generation Client, a deterministic stub, a
LoRA-adapted FLUX run, or a third-party endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tattd_studio.comparison_matrix.metrics import ComparisonMetrics, score_entry
from tattd_studio.knowledge.embedding import TextEmbeddingClient


@dataclass(frozen=True)
class ComparisonEntry:
    """One column in the Comparison Matrix."""

    label: str
    """Human-readable column label (e.g. 'FLUX.2-dev + LoRA Artifact (kira)')."""

    image_uri_for: Callable[[str], str]
    """Returns the URI of the image this column generates for a given prompt."""


@dataclass(frozen=True)
class ComparisonRow:
    """One row of the matrix: one prompt, one column, one metric envelope."""

    prompt_id: str
    prompt: str
    column_label: str
    image_uri: str
    metrics: ComparisonMetrics


@dataclass(frozen=True)
class ComparisonMatrix:
    """The full table after running every prompt through every column."""

    rows: list[ComparisonRow]

    def by_column(self) -> dict[str, list[ComparisonRow]]:
        groups: dict[str, list[ComparisonRow]] = {}
        for row in self.rows:
            groups.setdefault(row.column_label, []).append(row)
        return groups

    def column_aggregates(self) -> dict[str, float]:
        """Mean aggregate metric per column label."""
        out: dict[str, float] = {}
        for label, rows in self.by_column().items():
            values = [r.metrics.aggregate for r in rows]
            out[label] = sum(values) / len(values) if values else 0.0
        return out


def load_comparison_prompts(jsonl_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]


def run_matrix(
    *,
    prompts: list[dict],
    entries: list[ComparisonEntry],
    embedder: TextEmbeddingClient,
) -> ComparisonMatrix:
    rows: list[ComparisonRow] = []
    for case in prompts:
        for entry in entries:
            uri = entry.image_uri_for(case["prompt"])
            metrics = score_entry(
                prompt=case["prompt"], image_uri=uri, embedder=embedder
            )
            rows.append(
                ComparisonRow(
                    prompt_id=case["id"],
                    prompt=case["prompt"],
                    column_label=entry.label,
                    image_uri=uri,
                    metrics=metrics,
                )
            )
    return ComparisonMatrix(rows=rows)
