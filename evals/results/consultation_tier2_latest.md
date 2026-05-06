# Consultation Tier 2 — Eval Report

- Run: 2026-05-06T12:40:59.257424+00:00
- Conversation traces: `data/eval/conversation_traces.jsonl` (8 traces)
- Mode: ci

## Deterministic metrics (CI baseline)

| Metric | Value |
|---|---|
| Intent-substring coverage | 1.000 |
| Grounding-area match | 0.938 |

**Intent-substring coverage** — fraction of the trace's expected
substrings (e.g. style and placement keywords) present in the final
Intent's refined_description after all turns.

**Grounding-area match** — fraction of expected Knowledge Corpus
areas (taxonomy / placement / aftercare / ip / cultural) covered
by the chunks retrieved across the session.

## DeepEval primitives (live only)

Set `RUN_LIVE_CONSULTATION_EVAL=1` to run DeepEval's
`AnswerRelevancy`, `Faithfulness`, `ConversationRelevancy`,
and `KnowledgeRetention` metrics with an LLM judge. CI
skips them by default.
