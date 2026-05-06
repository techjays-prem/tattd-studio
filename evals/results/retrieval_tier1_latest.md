# Knowledge Retriever Tier 1 — Eval Report

- Run: 2026-05-06T12:14:28.285244+00:00
- Golden Set: `data/eval/retrieval_knowledge_golden.jsonl` (20 queries)
- Mode: ci
- Embedder: DeterministicTextEmbeddingClient(dim=1024)
- Top-k: 5

## Information retrieval metrics

| Metric | Value |
|---|---|
| recall@5 | 0.050 |
| MRR | 0.050 |
| area_recall@5 | 0.500 |

Area recall measures whether the retrieved set contains at
least one chunk from the query's expected area (`taxonomy`,
`placement`, `aftercare`, `ip`, or `cultural`); useful as a
lenient sanity floor when the deterministic embedder cannot
distinguish chunks semantically.

## DeepEval contextual primitives

The plan's Tier 1 Retrieval surface specifies DeepEval's
`ContextualRelevancy`, `ContextualRecall`, and
`ContextualPrecision`. These metrics require an LLM judge
and run only under `RUN_LIVE_RETRIEVAL_EVAL=1`. The runner
is wired (see `_run_deepeval_contextual` below); CI does
not invoke it.
