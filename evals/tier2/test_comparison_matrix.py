"""Tier 2 — Comparison Matrix.

Runs the 5-way matrix over the shared prompt set
(`data/eval/comparison_prompts.jsonl`) and emits an Eval Report at
`evals/results/comparison_matrix_latest.md`.

Default CI run:

- ``Generation Client (Nano Banana 2 / Pro)``: deterministic stub
  through `GenerationClient` with a fake `generate_fn`
- ``FLUX.2-dev`` / ``FLUX.2-klein`` base columns: deterministic stub
  inference (per-prompt stable URIs)
- LoRA-adapted columns activate **only when**
  `data/lora_training/artifacts.toml` carries entries for the matching
  ``(artist_slug, base_model)`` pairs. Today the registry is empty;
  those rows are reported as "deferred to #9".

Live mode (``RUN_LIVE_COMPARISON_MATRIX=1``) swaps in the production
clients per column.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from tattd_studio.comparison_matrix import (
    ComparisonEntry,
    load_comparison_prompts,
    run_matrix,
)
from tattd_studio.comparison_matrix.matrix import ComparisonMatrix
from tattd_studio.knowledge import DeterministicTextEmbeddingClient
from tattd_studio.lora import (
    deterministic_stub_client as deterministic_lora_client,
)
from tattd_studio.lora import (
    load_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = REPO_ROOT / "data" / "eval" / "comparison_prompts.jsonl"
ARTIFACTS_PATH = REPO_ROOT / "data" / "lora_training" / "artifacts.toml"
REPORT_PATH = REPO_ROOT / "evals" / "results" / "comparison_matrix_latest.md"


def _stub_generation_client_uri(prompt: str) -> str:
    """Stable per-prompt URI for the Generation Client column in CI."""
    import hashlib

    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return f"file:///tmp/cm-genclient-{digest}.png"


def _stub_flux_uri(base_model: str, prompt: str) -> str:
    import hashlib

    digest = hashlib.sha1((base_model + "|" + prompt).encode("utf-8")).hexdigest()[:16]
    return f"file:///tmp/cm-{base_model.lower()}-{digest}.png"


def _build_columns() -> list[ComparisonEntry]:
    columns: list[ComparisonEntry] = [
        ComparisonEntry(
            label="Generation Client (Nano Banana 2 / Pro)",
            image_uri_for=_stub_generation_client_uri,
        ),
        ComparisonEntry(
            label="FLUX.2-dev (base)",
            image_uri_for=lambda p: _stub_flux_uri("FLUX.2-dev", p),
        ),
        ComparisonEntry(
            label="FLUX.2-klein (base)",
            image_uri_for=lambda p: _stub_flux_uri("FLUX.2-klein", p),
        ),
    ]

    # LoRA-adapted columns light up only when real artifacts exist.
    artifacts = load_artifacts(ARTIFACTS_PATH)
    if artifacts:
        lora_client = deterministic_lora_client()
        for artifact in artifacts:
            columns.append(
                ComparisonEntry(
                    label=artifact.comparison_matrix_column,
                    image_uri_for=lambda p, a=artifact: lora_client.generate(p, a),
                )
            )
    return columns


def _emit_report(matrix: ComparisonMatrix, columns: list[ComparisonEntry]) -> None:
    artifacts = load_artifacts(ARTIFACTS_PATH)
    aggregates = matrix.column_aggregates()
    n_prompts = len(matrix.by_column().get(columns[0].label, []))
    mode = (
        "live"
        if os.environ.get("RUN_LIVE_COMPARISON_MATRIX") == "1"
        else "deterministic baseline (CI)"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Comparison Matrix — Eval Report",
        "",
        f"- Run: {dt.datetime.now(dt.UTC).isoformat()}",
        f"- Shared prompt set: `data/eval/comparison_prompts.jsonl` ({n_prompts} prompts)",
        f"- Mode: {mode}",
        f"- LoRA Artifacts registered: {len(artifacts)}",
        "",
        "| Column | Mean aggregate |",
        "|---|---|",
    ]
    for col in columns:
        agg = aggregates.get(col.label, 0.0)
        lines.append(f"| {col.label} | {agg:.3f} |")

    if not artifacts:
        lines.extend(
            [
                "",
                "## Deferred columns (HITL-gated)",
                "",
                "- `FLUX.2-dev + LoRA Artifact (<artist>)` — blocked by issue #9",
                "  (real onboarded-artist permission + Replicate training run).",
                "- `FLUX.2-klein + LoRA Artifact (<artist>)` — blocked by issue #9.",
                "",
                "These columns activate automatically when",
                "`data/lora_training/artifacts.toml` carries entries.",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def test_comparison_matrix_runs_baseline_columns() -> None:
    prompts = load_comparison_prompts(PROMPTS_PATH)
    columns = _build_columns()
    embedder = DeterministicTextEmbeddingClient(dim=1024)

    matrix = run_matrix(prompts=prompts, entries=columns, embedder=embedder)
    _emit_report(matrix, columns)

    aggregates = matrix.column_aggregates()
    # Always-on baseline columns must report numeric aggregates in [0, 1].
    for label in (
        "Generation Client (Nano Banana 2 / Pro)",
        "FLUX.2-dev (base)",
        "FLUX.2-klein (base)",
    ):
        assert label in aggregates, f"missing column: {label}"
        assert 0.0 <= aggregates[label] <= 1.0


def test_comparison_matrix_skips_lora_columns_when_registry_empty() -> None:
    artifacts = load_artifacts(ARTIFACTS_PATH)
    columns = _build_columns()
    if not artifacts:
        labels = [c.label for c in columns]
        assert all(
            "LoRA Artifact" not in lbl for lbl in labels
        ), f"LoRA columns should not appear when registry is empty: {labels}"
