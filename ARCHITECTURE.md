# Architecture — Tattd Studio

> **Companion to:** [PRD.md](./PRD.md), [CONTEXT.md](./CONTEXT.md), [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md), [DATA_PROVENANCE.md](./DATA_PROVENANCE.md). The implementation plan is the spec; this document is the deeper engineering reasoning that grew across the slices.

## What this artifact is

Tattd Studio is a runnable proof of fitness for Tattd's senior AI/ML engineering role. It is a multi-turn tattoo-design **Studio** that runs entirely on Tattd's stated production stack (Gemini Nano Banana 2 / Pro, Gemini Embedding 2) with rigorous DeepEval coverage at the component, agent, and CI layers.

The artifact is built around four design rules:

1. **Vocabulary discipline** — the same terms appear in code, tests, issues, PRs, commits, and Eval Reports as appear in CONTEXT.md.
2. **Provenance first** — every record in every corpus carries source attribution; DATA_PROVENANCE.md is the single auditable trail.
3. **Tracer-bullet slices** — eleven GitHub issues, each shippable end-to-end on its own; CI green per slice.
4. **Test the wiring, gate the live calls** — every Critic and provider has a deterministic CI path and a gated live path.

## System overview

```
                            ┌──────────────────────┐
                            │  Human client (chat) │
                            └──────────┬───────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────┐
                  │  Consultation node                  │
                  │  Knowledge Retriever over the       │
                  │  Knowledge Corpus via the           │
                  │  multimodal embedding + Vector Store│
                  └──────────────────┬──────────────────┘
                                     │ refined Intent
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  Generation node                    │
                  │  Generation Client                  │
                  │  (Nano Banana 2 / Pro)              │
                  │  Returns N Candidate Designs        │
                  └──────────────────┬──────────────────┘
                                     │
              ┌──────────────────────┼──────────────────┬─────────────────┐
              ▼                      ▼                  ▼                 ▼
    ┌──────────────┐       ┌──────────────┐    ┌──────────────┐  ┌──────────────┐
    │   Anatomy    │       │  Plagiarism  │    │    Style     │  │   Quality    │
    │    Critic    │       │    Critic    │    │    Critic    │  │    Critic    │
    │  (VLM judge) │       │ (multimodal  │    │ (multimodal  │  │ (VLM rubric) │
    │              │       │  embedding   │    │  embedding   │  │              │
    │              │       │  similarity) │    │  alignment)  │  │              │
    └──────┬───────┘       └──────┬───────┘    └──────┬───────┘  └──────┬───────┘
           │                      │                   │                  │
           └──────────────────────┴────────┬──────────┴──────────────────┘
                                           ▼
                       ┌────────────────────────────────────┐
                       │  Routing                           │
                       │  • all pass    → surface           │
                       │  • any fail    → Refinement        │
                       │  • plagiarism  → diversify + Refine│
                       │  • 2nd fail    → escalate          │
                       └────────────────────┬───────────────┘
                                            │ chosen design
                                            ▼
                  ┌────────────────────────────────────────┐
                  │  Two-Stage Matcher                     │
                  │  Stage 1: multimodal embedding recall  │
                  │           via Vector Store             │
                  │  Stage 2: visual embedding rerank      │
                  │           (DINOv2-ViT-B14)             │
                  │  → top-N onboarded artists             │
                  └────────────────────────────────────────┘
```

## Module map

| Module | Responsibility | Key types |
|---|---|---|
| `tattd_studio.vectordb` | Named-vector Qdrant wrapper (multimodal-3072, multimodal-1024, visual), int8 quantization, alias-swap rebuild | `VectorStore`, `CollectionSchema`, `QueryHit` |
| `tattd_studio.knowledge` | Knowledge Corpus parsing (`<!-- CHUNK -->`), ingest, retrieval, text embedding clients (deterministic + Gemini Embedding 2) | `Chunk`, `KnowledgeRetriever`, `RetrievedChunk`, `DeterministicTextEmbeddingClient` |
| `tattd_studio.generation` | Generation Client wrapping Gemini Nano Banana 2 / Pro with retry, in-memory cache, prompt versioning | `GenerationClient`, `CandidateDesign`, `PROMPT_TEMPLATE_VERSION`, `TransientGenerationError` |
| `tattd_studio.graph.critics` | Four Critics (Anatomy / Plagiarism / Style / Quality) with DI'd judge functions; deterministic baselines for CI | `AnatomyCheck`, `PlagiarismCheck`, `StyleCoherence`, `QualityScore`, `*Critic` |
| `tattd_studio.graph.routing` | Conditional-edge logic over the four Critic verdicts; thresholds in `evals/calibrated_thresholds.toml` | `RoutingDecision`, `RoutingThresholds`, `evaluate(...)` |
| `tattd_studio.graph.studio` | LangGraph 1.0.10 state machine: Consultation → Generation → 4 parallel Critics → Routing → (Refinement or Two-Stage Matcher) → END | `build_studio_graph(...)`, `StudioState` |
| `tattd_studio.embeddings` | Visual embedding clients (DINOv2-ViT-B14 production + deterministic CI) | `VisualEmbeddingClient`, `DeterministicVisualEmbeddingClient`, `build_dinov2_visual_embedding_client` |
| `tattd_studio.matching` | Two-Stage Matcher: multimodal recall → visual rerank | `TwoStageMatcher`, `RankedArtist`, `ArtistRecord` |
| `tattd_studio.ui` | Gradio shell | `launch(...)` |

## Design decisions

### Why named-vector schema with int8 quantization
Each Vector Store collection holds three named slots — `multimodal-3072`, `multimodal-1024`, and `visual` — declared in one shot at collection creation. The two multimodal slots support Gemini Embedding 2's Matryoshka prefix property (the 1024-dim prefix of the 3072-dim vector is a valid embedding) so retrieval can be served from the smaller index when latency dominates. The visual slot is reserved for DINOv2-ViT-B14's 768-dim output.

Int8 scalar quantization is configured at collection-creation time. The wrapper records its declaration internally because Qdrant's local mode (used in tests via `QdrantClient(":memory:")`) does not echo `quantization_config` back through `get_collection`; the production server reports it directly. The contract — every collection the wrapper creates is int8-quantized — holds in both modes.

### Why Candidate Designs travel by URI
Per the LangGraph multimodal-state risk in the implementation plan: passing raw image bytes through state can blow up state size dramatically and slow down checkpointing, especially with parallel Critic branches. The state envelope (`StudioState`) carries each Candidate Design's URI; the actual bytes live in object storage (or the local filesystem for the POC). A dedicated test asserts this invariant.

### Why parallel Critics return delta dicts, not full state
LangGraph 1.0 rejects concurrent updates to the same state key. When all four Critics run in parallel from the Generation node, each must return only the slot it actually modifies (e.g. `{"plagiarism_checks": [...]}`) rather than `{**state, "plagiarism_checks": [...]}`. This was the breaking change between slice #6 (one Critic) and slice #7 (four parallel Critics).

### Why Routing is data-driven through TOML
`evals/calibrated_thresholds.toml` is loaded once at graph-build time and threaded into the Routing layer. Bumping a threshold means committing a new TOML value alongside an Eval Report that justifies the bump — never editing routing code. CI thresholds (which the deterministic baselines must clear) and live-mode targets (which the real providers must clear) both live in the same TOML, separated by suffix (e.g. `precision_min` vs `live_precision_min`).

### Why every Critic ships with a deterministic CI baseline
Real providers (Gemini Pro VLM, Gemini Embedding 2, DINOv2 inference) require credentials or GPU. CI must run the harness end-to-end without either. So each Critic accepts an injected `judge_fn` (or embedder); the production factory wraps the real provider, and a deterministic stub or rule-based baseline runs in CI. The harness validates wiring, the stubs produce committed Eval Reports, and live runs (gated by env vars) substitute the real providers.

### Why the LoRA Artifact is offline-only
Per CONTEXT.md and the plan, the LoRA Artifact is a per-artist style fine-tune that lives **outside** the runtime loop. Runtime generation always uses the Generation Client (Nano Banana 2 / Pro). The LoRA Artifact only appears in the Comparison Matrix under the Eval Harness — its whole purpose is to demonstrate that the developer can ship the FLUX/SD half of the JD's "diffusion experience" requirement while framing it as marketplace value (per-artist style fidelity).

## Decision log

| Slice | Key decision | See |
|---|---|---|
| #1 | LangGraph pinned to 1.0.10 with rationale comment in `pyproject.toml` (quarterly minor API churn) | [pyproject.toml](./pyproject.toml) |
| #2 | Int8 quantization tracked client-side because Qdrant local mode does not echo it back | `vectordb/qdrant_client.py` |
| #3 | Knowledge Corpus stored as one markdown file per area with `<!-- CHUNK -->` separators (unambiguous against YAML frontmatter); 136 chunks | [data/knowledge](./data/knowledge) |
| #4 | Heuristic Anatomy judge as the CI baseline; live Gemini Pro VLM judge gated; thresholds in TOML | `graph/critics/anatomy_judges.py` |
| #5 | Generation Client retries on `TransientGenerationError` only; non-transient errors propagate | `generation/client.py` |
| #6 | Trace metadata recorded by `trace_start` / `trace_end` nodes wrapping the user-facing pipeline | `graph/studio.py` |
| #7 | Critic nodes return delta dicts; parallel branches converge into a single Routing node | `graph/studio.py` |
| #7 | Routing's "first-failure → refine, second-failure → escalate" lives in `routing.evaluate(...)`; never silently regenerates | `graph/routing.py` |
| #8 | Synthetic-only Artist Portfolio Index for the unsolicited POC; real onboarded records deferred to slice #9 / production onboarding | [DATA_PROVENANCE.md](./DATA_PROVENANCE.md) |
| #8 | Two-Stage Matcher uses `multimodal-1024` for Stage 1 recall (Matryoshka tier) and the `visual` slot for Stage 2 rerank | `matching/two_stage.py` |
| #11 | Comparison Matrix entry deferred — it depends on slice #9 (HITL: real artist permission required for the LoRA Artifact training set) | [README.md](./README.md) |

## Risk inventory

The risks tracked in the implementation plan have been validated through the slices:

- **LangGraph 1.0.x churn** — pinned tight; tests survived by treating the API as a small surface (`StateGraph`, `START`, `END`, `add_node`, `add_edge`, `add_conditional_edges`).
- **Multimodal state in LangGraph** — never carried bytes; verified by a dedicated test.
- **Parallel-update conflicts in LangGraph** — surfaced when slice #7 added the four-Critic fan-out; resolved by switching nodes to delta-dict returns.
- **Vertex AI access** — text embedding factory is gated; the deterministic stub backs all CI paths.
- **Critic latency** — four Critics run in parallel from one predecessor (`generation`) so the wall-clock cost is `max(critic_latency)`, not `sum`.
- **Plagiarism threshold tuning** — calibrated against the deterministic CI baseline first; live thresholds documented separately in `evals/calibrated_thresholds.toml` and only activate under `RUN_LIVE_PLAGIARISM_EVAL`.
- **DINOv2 GPU vs CPU** — factory lazy-imports torch + transformers; CI never pays that cost.
- **Qdrant named-vector dim consistency** — every collection declares all three slots (multimodal-3072 / multimodal-1024 / visual) so cross-collection queries don't trip on mismatched schema.
- **Synthetic image quality** — flagged in the multimodal embedding benchmark's Eval Report.
- **Snapshot drill** — `infra/docker-compose.yml` configures `QDRANT__STORAGE__SNAPSHOTS_PATH`; restore procedure documented in the README.
- **Gemini Embedding 2 newness** — wrapped behind `build_gemini_text_embedding_client(...)` with a single integration point so future API churn touches one file.
- **Nano Banana 2 / Pro** — wrapped through `google.genai.Client().models.generate_content` directly per the plan; no third-party adapter.
- **Real onboarded-artist source freshness** — deferred to slice #9; the synthetic Artist Portfolio Index in slice #8 covers the style space without claiming permissions the artifact does not have.

## What's next at production scale

| Topic | Direction |
|---|---|
| Real artist onboarding | Convert synthetic Artist Portfolio Index records (slice #8) into actual onboarded artists with signed permission and curated portfolio uploads (slice #9 → production onboarding flow) |
| LoRA Artifact training | Replicate `ai-toolkit` runs against onboarded artists who opted in for personalized style fine-tunes; comparison matrix scores per-artist alignment against the Generation Client baseline |
| Comparison Matrix | Tier 2 entry that runs FLUX.2-dev ± LoRA + FLUX.2-klein ± LoRA + Generation Client (+ optional OpenAI Image 2) on a shared prompt set; FID + CLIP score + GEval + style adherence via multimodal embedding cosine |
| Provider failover | Today the Generation Client is single-provider; production should fan out across providers when one is down or rate-limited |
| Live plagiarism corpus growth | Famous Tattoos Corpus is a seed; production should ingest daily from curated public reference sets while honoring rights of publicity per `data/knowledge/ip.md` |
| Caching | Per-session in-memory cache today; production should cache by `(prompt, source_model_id)` in a durable store with TTL and explicit invalidation |
| Observability | Trace metadata recorded in StudioState; production exports to OpenTelemetry / LangSmith with span attributes per node |
| Routing telemetry | Calibrated thresholds today; production should A/B test threshold variants against business metrics (Refinement rate, escalation rate, surface-to-conversion) |
| Synthetic test-set generation | Tier 3 stretch in the plan; would extend the Anatomy Golden Set, Plagiarism pairs, and Style alignment cases via DeepEval Synthesizer |
| Drift detection | Tier 3 stretch; scheduled CI run to flag Eval Report regressions over time |

## Out of scope

Per the PRD, these are deliberately not part of this artifact:

- Booking integration / payment / calendaring
- Aftercare as a dedicated agent surface (Knowledge Corpus contains aftercare chunks, but no dedicated conversation flow)
- Artist-side tools, dashboards, or onboarding surfaces — single-sided (human-client-facing) only
- Multi-language Consultation — English-only at POC scale
- Multi-artist LoRA Artifacts — single per-artist LoRA proves the wedge; multi-artist is roadmap, not artifact
- Watermark detection, perceptual-hash exact-match plagiarism beyond embedding similarity
- Live hosted demo, video walkthrough, outreach plan, recipient targeting — developer's responsibility, not engineering scope
