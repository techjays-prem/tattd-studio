# Tier 3 Synthesizer — Eval Report

- Run: 2026-05-06T12:40:59.466497+00:00
- Anatomy seed Golden Set: 60 cases
- Mode: ci

## Golden-set expansion

| Metric | Value |
|---|---|
| Seed Anatomy cases | 60 |
| Permutation-augmented cases (CI) | 30 |
| Live Synthesizer cases generated | n/a — set `RUN_LIVE_SYNTHESIZER=1` |

## Methods

- **Permutation augmenter (CI)** — `expand_anatomy_cases(seed, n)`
  samples body-part / prompt / size combinations consistent
  with the expected validity. Deterministic given a seed.
  Useful for stress-testing Anatomy Critic thresholds.
- **Live Synthesizer (`RUN_LIVE_SYNTHESIZER=1`)** —
  DeepEval's `Synthesizer.generate_goldens_from_contexts(...)`
  produces semantically novel cases anchored on Knowledge
  Corpus contexts. Requires an LLM judge.

These outputs are *not* committed back into the seed Golden
Set automatically; the developer reviews each batch before
promoting it. Tier 1 evals continue to assert against the
hand-curated seed.
