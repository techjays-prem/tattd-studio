# Multimodal Embedding 3-Way Benchmark — Eval Report

- Run: 2026-05-06T12:23:07.220385+00:00
- Golden Set: `data/eval/retrieval_golden.jsonl` (20 queries)
- Mode: deterministic baseline (CI)

| Embedder | recall@5 | MRR | NDCG@5 |
|---|---|---|---|
| `gemini-embedding-2` | 0.117 | 0.092 | 0.098 |
| `multimodalembedding@001` | 0.117 | 0.092 | 0.098 |
| `siglip-2` | 0.117 | 0.092 | 0.098 |

*Deterministic baseline note: the CI stub has no semantic signal so all three rows track each other within the noise floor of the hash function. Live runs against Gemini Embedding 2 vs `multimodalembedding@001` vs SigLIP 2 will produce meaningful separation.*