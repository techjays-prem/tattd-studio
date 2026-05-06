# Comparison Matrix — Eval Report

- Run: 2026-05-06T12:09:43.965658+00:00
- Shared prompt set: `data/eval/comparison_prompts.jsonl` (30 prompts)
- Mode: deterministic baseline (CI)
- LoRA Artifacts registered: 0
- OpenAI Image 2 column: gated off

## Aggregate scores

| Column | Mean aggregate | Where it wins / loses |
|---|---|---|
| Generation Client (Nano Banana 2 / Pro) | 0.533 | best on `cm16` (0.652); weakest on `cm22` (0.442) |
| FLUX.2-dev (base) | 0.545 | best on `cm09` (0.665); weakest on `cm28` (0.427) |
| FLUX.2-klein (base) | 0.525 | best on `cm05` (0.654); weakest on `cm09` (0.428) |

## Metrics

Each row reports the equal-weighted aggregate of four metrics:

- **FID-proxy** (CI) / **FID** (live, gated on #9 reference set)
  — Fréchet Inception Distance against the LoRA Artifact's onboarded artist's portfolio. Until #9 lands, this remains in proxy mode (see `comparison_matrix/live_metrics.py`).
- **CLIP-proxy** (CI) / **CLIP score** (live) — semantic alignment
  between prompt and image via OpenAI CLIP ViT-B/32.
- **GEval** — DeepEval LLM-judge rubric (composition / linework /
  balance / originality). Live runs use the real DeepEval `GEval`
  metric; CI uses a fixed 0.65 baseline.
- **Style adherence** — multimodal embedding cosine between the
  Intent's text embedding and the Candidate Design's image
  embedding (Gemini Embedding 2 in live mode; deterministic stub
  in CI).

## Deferred columns (HITL-gated)

- `FLUX.2-dev + LoRA Artifact (<artist>)` — blocked by issue #9
  (real onboarded-artist permission + Replicate training run).
- `FLUX.2-klein + LoRA Artifact (<artist>)` — blocked by issue #9.

These columns activate automatically when
`data/lora_training/artifacts.toml` carries entries.
