# Generation Tier 1 — Eval Report

- Run: 2026-05-06T12:23:06.538923+00:00
- Golden Set: `data/eval/generation_golden.jsonl` (15 cases)
- Mode: ci

| Metric | Value |
|---|---|
| Faithfulness label match | 0.467 |
| Originality label match | 0.133 |
| GEval (live) | n/a — set `RUN_LIVE_GENERATION_TIER1=1` |

## Metrics

- **Faithfulness label match** — `faithfulness_score(intent,
  candidate)` >= 0.5 → predicted *faithful*; the score is
  the fraction of cases where the prediction matches the
  Golden Set's `expected_faithful` label.
- **Originality label match** — `OriginalityMetric.measure`
  produces a [0, 1] originality value; >= 0.5 → predicted
  *original*. Score is the prediction-vs-label match rate.
- **GEval (live)** — DeepEval LLM-judge rubric for technical
  quality (composition / linework / intent fidelity /
  originality), run only under `RUN_LIVE_GENERATION_TIER1=1`.

## Note on the deterministic baseline

The CI mode uses the deterministic embedding stub which has
no semantic signal; the cosine values cluster around 0.5
regardless of input. The label-match metrics here are
therefore best read as wiring-OK floors, not as quality
claims about the embedder. Live mode against Gemini
Embedding 2 produces meaningful separation.
