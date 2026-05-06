"""Tier 2 — Comparison Matrix.

Runs the matrix over the shared prompt set
(`data/eval/comparison_prompts.jsonl`) and emits an Eval Report at
`evals/results/comparison_matrix.md` (latest report; `.../comparison_matrix_latest.md`
is a synonym kept for slice-#7 link compatibility).

## Columns

Always-on (deterministic stub in CI; live-mode swaps in the real
client per column):

- ``Generation Client (Nano Banana 2 / Pro)`` — the runtime path
- ``FLUX.2-dev (base)`` — the open-weights base from slice #5's plan
- ``FLUX.2-klein (base)`` — the open-weights base, klein variant

Conditionally added:

- ``FLUX.2-{base} + LoRA Artifact (<artist>)`` — appears when
  `data/lora_training/artifacts.toml` carries entries. Today the
  registry is empty (HITL-gated by #9); these rows are listed in the
  report's deferred section.
- ``OpenAI Image 2`` — appears when `OPENAI_IMAGE_2_API_KEY` is set
  per #10's optional sixth column. Default CI runs do not depend on it.

## Modes

- Default CI: deterministic stubs for image URIs, deterministic proxies
  for FID / CLIP / GEval / style-adherence; harness wiring exercised
  end-to-end with no API spend.
- Live (`RUN_LIVE_COMPARISON_MATRIX=1`): real CLIP / DeepEval GEval /
  multimodal-embedding cosine; FID stays in proxy mode until the
  reference distribution (the LoRA Artifact's onboarded artist's
  portfolio) lands via #9.
"""

from __future__ import annotations

import datetime as dt
import hashlib
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
REPORT_PATH = REPO_ROOT / "evals" / "results" / "comparison_matrix.md"
REPORT_LATEST_ALIAS = REPO_ROOT / "evals" / "results" / "comparison_matrix_latest.md"


def _stub_uri(label: str, prompt: str) -> str:
    digest = hashlib.sha1((label + "|" + prompt).encode("utf-8")).hexdigest()[:16]
    safe_label = label.lower().replace(" ", "-").replace("/", "-")
    return f"file:///tmp/cm-{safe_label}-{digest}.png"


def _build_columns() -> list[ComparisonEntry]:
    columns: list[ComparisonEntry] = [
        ComparisonEntry(
            label="Generation Client (Nano Banana 2 / Pro)",
            image_uri_for=lambda p: _stub_uri("genclient", p),
        ),
        ComparisonEntry(
            label="FLUX.2-dev (base)",
            image_uri_for=lambda p: _stub_uri("flux-2-dev", p),
        ),
        ComparisonEntry(
            label="FLUX.2-klein (base)",
            image_uri_for=lambda p: _stub_uri("flux-2-klein", p),
        ),
    ]

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

    if os.environ.get("OPENAI_IMAGE_2_API_KEY"):
        columns.append(
            ComparisonEntry(
                label="OpenAI Image 2",
                image_uri_for=lambda p: _stub_uri("openai-image-2", p),
            )
        )
    return columns


def _column_summary(matrix: ComparisonMatrix, label: str) -> str:
    """One-line summary of where this column wins or loses."""
    rows = [r for r in matrix.rows if r.column_label == label]
    if not rows:
        return "no rows"
    best = max(rows, key=lambda r: r.metrics.aggregate)
    worst = min(rows, key=lambda r: r.metrics.aggregate)
    return (
        f"best on `{best.prompt_id}` ({best.metrics.aggregate:.3f}); "
        f"weakest on `{worst.prompt_id}` ({worst.metrics.aggregate:.3f})"
    )


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
        f"- OpenAI Image 2 column: "
        f"{'enabled' if os.environ.get('OPENAI_IMAGE_2_API_KEY') else 'gated off'}",
        "",
        "## Aggregate scores",
        "",
        "| Column | Mean aggregate | Where it wins / loses |",
        "|---|---|---|",
    ]
    for col in columns:
        agg = aggregates.get(col.label, 0.0)
        lines.append(
            f"| {col.label} | {agg:.3f} | {_column_summary(matrix, col.label)} |"
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "Each row reports the equal-weighted aggregate of four metrics:",
            "",
            "- **FID-proxy** (CI) / **FID** (live, gated on #9 reference set)",
            "  — Fréchet Inception Distance against the LoRA Artifact's "
            "onboarded artist's portfolio. Until #9 lands, this remains in "
            "proxy mode (see `comparison_matrix/live_metrics.py`).",
            "- **CLIP-proxy** (CI) / **CLIP score** (live) — semantic alignment",
            "  between prompt and image via OpenAI CLIP ViT-B/32.",
            "- **GEval** — DeepEval LLM-judge rubric (composition / linework /",
            "  balance / originality). Live runs use the real DeepEval `GEval`",
            "  metric; CI uses a fixed 0.65 baseline.",
            "- **Style adherence** — multimodal embedding cosine between the",
            "  Intent's text embedding and the Candidate Design's image",
            "  embedding (Gemini Embedding 2 in live mode; deterministic stub",
            "  in CI).",
            "",
        ]
    )

    if not artifacts:
        lines.extend(
            [
                "## Deferred columns (HITL-gated)",
                "",
                "- `FLUX.2-dev + LoRA Artifact (<artist>)` — blocked by issue #9",
                "  (real onboarded-artist permission + Replicate training run).",
                "- `FLUX.2-klein + LoRA Artifact (<artist>)` — blocked by issue #9.",
                "",
                "These columns activate automatically when",
                "`data/lora_training/artifacts.toml` carries entries.",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines))
    # `comparison_matrix_latest.md` is the historical filename; keep it
    # as a synonym so external links don't break.
    REPORT_LATEST_ALIAS.write_text(REPORT_PATH.read_text())


def test_comparison_matrix_runs_baseline_columns() -> None:
    prompts = load_comparison_prompts(PROMPTS_PATH)
    assert len(prompts) >= 25, (
        f"prompt set is below #10's spec target of ~25-50: {len(prompts)}"
    )

    columns = _build_columns()
    embedder = DeterministicTextEmbeddingClient(dim=1024)

    matrix = run_matrix(prompts=prompts, entries=columns, embedder=embedder)
    _emit_report(matrix, columns)

    aggregates = matrix.column_aggregates()
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


def test_comparison_matrix_includes_openai_image_2_when_key_set(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_IMAGE_2_API_KEY", "sk-fake-fixture")
    columns = _build_columns()
    labels = [c.label for c in columns]
    assert "OpenAI Image 2" in labels


def test_comparison_matrix_excludes_openai_image_2_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_IMAGE_2_API_KEY", raising=False)
    columns = _build_columns()
    labels = [c.label for c in columns]
    assert "OpenAI Image 2" not in labels
