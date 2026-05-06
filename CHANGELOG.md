# Changelog

All notable changes shipped to this artifact, organized by slice. The
implementation plan numbers each slice as one GitHub issue (#1–#11);
where extra rigor work landed beyond the issue scope, it's listed under
"Beyond plan slices".

## Unreleased

- _Awaiting #9 (HITL: real onboarded-artist permission for the LoRA
  Artifact). When that lands, #10 (Comparison Matrix LoRA-adapted
  columns) and #11's LoRA row in the Eval Surface table activate
  automatically; no eval-runner code changes required._

## Plan slices

### Slice #1 — Repo + CI + Eval Harness shell ([PR #13](https://github.com/techjays-prem/tattd-studio/pull/13))

- `pyproject.toml` with `uv` build backend, Python 3.12+, LangGraph
  pinned to **1.0.10** with API-churn rationale comment
- Repo layout established
- `.env.example`, `.gitignore`, `.github/workflows/ci.yml`
- Tracer-bullet pytest unit + DeepEval Tier 1 eval

### Slice #2 — Vector Store wrapper ([PR #14](https://github.com/techjays-prem/tattd-studio/pull/14))

- `tattd_studio.vectordb.VectorStore` with named-vector schema
  (`multimodal-3072` / `multimodal-1024` / `visual`) + int8 scalar
  quantization
- `upsert_point`, `query_named` (with payload filter), `alias_swap_rebuild`
- `vectordb/reindex.py` — alias-swap rebuild script
- `infra/docker-compose.yml` — Qdrant v1.13.4 with snapshots configured
- 4 vectordb tests passing against `:memory:` Qdrant

### Slice #5 — Generation Client wrapper ([PR #15](https://github.com/techjays-prem/tattd-studio/pull/15))

- `GenerationClient.generate(intent, n) → list[CandidateDesign]`
- Bounded retry on `TransientGenerationError`, in-memory cache keyed
  on `(prompt, source_model_id)`, `PROMPT_TEMPLATE_VERSION = "v1.0"`
  in metadata
- `build_gemini_generate_fn(...)` for production
- 4 mocked tests + 1 gated live test

### Slice #4 — Anatomy Critic ([PR #16](https://github.com/techjays-prem/tattd-studio/pull/16))

- `AnatomyCheck` + `AnatomyCritic` + `PlacementContext`
- `heuristic_anatomy_judge` (CI baseline) + `build_gemini_anatomy_judge`
- Golden Set: 30 valid + 30 invalid placements
- Tier 1 eval: P=0.967, R=0.967

### Slice #3 — Knowledge Corpus + Knowledge Retriever ([PR #17](https://github.com/techjays-prem/tattd-studio/pull/17))

- 136 chunks across 5 areas (taxonomy 71, placement 25, aftercare 13,
  ip 12, cultural 15)
- `parse_chunks_from_markdown` (chunk-marker delimited, unambiguous
  vs YAML frontmatter)
- `KnowledgeRetriever.retrieve(query, k) → list[RetrievedChunk]`
- `DeterministicTextEmbeddingClient` (CI) + `build_gemini_text_embedding_client`
- DATA_PROVENANCE.md seeded

### Slice #6 — Minimal Studio graph ([PR #18](https://github.com/techjays-prem/tattd-studio/pull/18))

- LangGraph `StateGraph[StudioState]`: trace_start → consultation →
  generation → anatomy_critic → trace_end → END
- Candidate Designs by URI only (no bytes in state — verified)
- `python -m tattd_studio.main` Gradio entry
- Tier 2 trace eval recording latency

### Slice #7 — 3 more Critics + Routing + Refinement ([PR #19](https://github.com/techjays-prem/tattd-studio/pull/19))

- `PlagiarismCritic`, `StyleCritic`, `QualityCritic` Pydantic verdicts
- `routing.evaluate(verdicts, thresholds, attempts) → RoutingDecision`
- LangGraph parallel Critic topology + Refinement loop (one retry
  before escalation)
- 50-record Famous Tattoos Corpus
- 3 new Tier 1 evals + DATA_PROVENANCE.md updated
- **Critical fix**: parallel nodes return delta dicts, not full state,
  to avoid LangGraph's concurrent-write rejection

### Slice #8 — Two-Stage Matcher ([PR #20](https://github.com/techjays-prem/tattd-studio/pull/20))

- `TwoStageMatcher.find_artists(chosen_design, k)` — multimodal recall
  → DINOv2 visual rerank
- 35-record Artist Portfolio Index (synthetic style-coverage; real
  artists deferred to #9 per ethics)
- Plagiarism Critic extended to consult both corpora
- Tier 2 multimodal embedding 3-way benchmark
- Studio graph routes through matcher in a terminal node

### Slice #11 — Deploy artifacts + ARCHITECTURE + README polish ([PR #21](https://github.com/techjays-prem/tattd-studio/pull/21))

- Multi-stage uv `Dockerfile` on Python 3.12
- `modal.py` — Modal deploy config with `tattd-gemini` secret
- `infra/docker-compose.yml` — Qdrant + Studio together
- ARCHITECTURE.md with diagram + module map + decision log + risk
  inventory
- README populated: hero, JD-discrepancy framing, Run-in-5-minutes,
  Eval surface table, "what's next at production scale", out of scope

### Slice #9/#10/#11-residual scaffolding ([PR #22](https://github.com/techjays-prem/tattd-studio/pull/22))

- `infra/train_lora.yaml` ai-toolkit recipe
- `data/lora_training/permission/README.md` schema
- `data/lora_training/artifacts.toml` (empty registry)
- `tattd_studio.lora` (registry + Replicate live + deterministic stub)
- `tattd_studio.comparison_matrix` framework
- 12-prompt shared set + first matrix Eval Report

## Beyond plan slices

### Comparison Matrix rigor ([PR #23](https://github.com/techjays-prem/tattd-studio/pull/23))

- Prompt set 12 → 30
- `comparison_matrix/live_metrics.py`: real CLIP (open_clip ViT-B/32),
  FID (torchmetrics), DeepEval `GEval`
- `OpenAI Image 2` sixth column gated on `OPENAI_IMAGE_2_API_KEY`
- Spec-conformant filename `comparison_matrix.md` (`comparison_matrix_latest.md`
  preserved as alias)
- Per-column "where it wins / loses" summaries

### Tier 1 Retrieval ([PR #24](https://github.com/techjays-prem/tattd-studio/pull/24))

- `data/eval/retrieval_knowledge_golden.jsonl` — 20 queries
- `evals/tier1/test_retrieval.py` — recall@5 / MRR / area_recall@5
  (CI) + DeepEval `ContextualRelevancy / ContextualRecall /
  ContextualPrecision` (live)

### Tier 1 Generation ([PR #25](https://github.com/techjays-prem/tattd-studio/pull/25))

- `tattd_studio.generation.OriginalityMetric` — DeepEval `BaseMetric`
  over Famous Tattoos Corpus (gradient counterpart of Plagiarism Critic)
- `faithfulness_score(intent, candidate)` — Style-Critic-shaped cosine
- 15-case Golden Set including deliberate near-duplicates of Famous
  Tattoos entries
- `GEval` rubric for technical quality (live)

### Tier 2 Consultation multi-turn ([PR #26](https://github.com/techjays-prem/tattd-studio/pull/26))

- `tattd_studio.consultation.ConsultationSession.advance(user_message)`
- `ConversationTurn` records grounding chunks + intent_after_turn
- `refine_intent` handles `intent=None` for first turn
- 8 conversation traces, intent-substring coverage 1.000, grounding-area
  match 0.938
- DeepEval `AnswerRelevancy / Faithfulness / ConversationRelevancy /
  KnowledgeRetention` live-only

### Tier 3 Synthesizer ([PR #27](https://github.com/techjays-prem/tattd-studio/pull/27))

- `tattd_studio.eval_synth` — permutation augmenter (CI baseline) +
  DeepEval Synthesizer wrapper (live)
- CI workflow extended to run `evals/tier3`

### Tier 3 Drift detection ([PR #28](https://github.com/techjays-prem/tattd-studio/pull/28))

- `tattd_studio.drift` — `DriftBaselines.from_toml`, `parse_metric`,
  direction-aware `compute_drift`
- `evals/results/history/baselines.toml` — 6 monitored metrics
- `evals/tier3/test_drift.py` — CI gate (clean on first run)
- `.github/workflows/drift.yml` — daily 09:00 UTC cron + `workflow_dispatch`

## Snapshot drill, CHANGELOG, HITL helpers (this PR)

- `infra/scripts/snapshot.sh` — Qdrant snapshot take / list / restore
  (closes the snapshot-drill gap from IMPLEMENTATION_PLAN.md → Risks)
- `infra/scripts/validate_permission.py` — schema validator for the
  per-artist permission marker that slice #9 requires
- `CHANGELOG.md` — this file
- README "Operations" section linking the new scripts and documenting
  the snapshot-restore drill

## Plan eval-surface coverage

Every named entry in IMPLEMENTATION_PLAN.md → Eval Surface table is
shipped:

- **Tier 1**: Anatomy ✅ Plagiarism ✅ Style ✅ Quality ✅ Retrieval ✅ Generation ✅
- **Tier 2**: multimodal embedding benchmark ✅ Comparison Matrix ✅
  Consultation (multi-turn) ✅ End-to-end Studio trace ✅
- **Tier 3**: Synthetic test-set generator ✅ Drift detection ✅
