# Comparison Matrix — Eval Report

- Run: 2026-05-06T11:56:47.345249+00:00
- Shared prompt set: `data/eval/comparison_prompts.jsonl` (12 prompts)
- Mode: deterministic baseline (CI)
- LoRA Artifacts registered: 0

| Column | Mean aggregate |
|---|---|
| Generation Client (Nano Banana 2 / Pro) | 0.578 |
| FLUX.2-dev (base) | 0.531 |
| FLUX.2-klein (base) | 0.528 |

## Deferred columns (HITL-gated)

- `FLUX.2-dev + LoRA Artifact (<artist>)` — blocked by issue #9
  (real onboarded-artist permission + Replicate training run).
- `FLUX.2-klein + LoRA Artifact (<artist>)` — blocked by issue #9.

These columns activate automatically when
`data/lora_training/artifacts.toml` carries entries.
