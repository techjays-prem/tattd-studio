# Implementation Plan — Tattd Studio

> **Companion to:** [PRD.md](./PRD.md) (product requirement) and [CONTEXT.md](./CONTEXT.md) (vocabulary). This document is the architectural / decision-log spec; the GitHub issues at [#1–#11](https://github.com/techjays-prem/tattd-studio/issues) are the executable tracer bullets.

---

## Context

**Why this exists.** Tattd (tattd.ai) — Brooklyn-based AI tattoo design + artist booking marketplace — has an open senior AI/ML engineering role whose JD insists on production-shipped diffusion + embedding/RAG experience and explicitly rules out generic data-science backgrounds.

The developer is building an unsolicited POC to demonstrate fitness for the role within a 2–4 month solo timeline. The artifact must:

- Run on Tattd's actual production stack (Gemini Nano Banana 2 / Pro, Gemini Embedding 2)
- Address the JD's "FLUX/SD" language by including open-weights work, framed for marketplace value (per-artist **LoRA Artifact**) rather than as a generic checkbox
- Demonstrate production rigor through DeepEval coverage at component, agent, and CI layers
- Engage Tattd's specific concerns (style fidelity, plagiarism, IP) directly through data **Provenance** discipline and **Plagiarism Critic** posture

---

## Architecture

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

The **LoRA Artifact** lives outside this loop — it appears only in the **Comparison Matrix** under the **Eval Harness**. Runtime generation always uses the **Generation Client**.

---

## Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.12+ | Modern type hints, perf |
| Package mgmt | uv | Replaces poetry / pip-tools |
| Orchestration | LangGraph (1.0.x pinned) | StateGraph + conditional edges; pin tight per API churn risk |
| Generation primary | Gemini Nano Banana 2 / Pro | Tattd's production model |
| Comparison Matrix entries | FLUX.2-dev (± LoRA Artifact), FLUX.2-klein (± LoRA Artifact), Generation Client; optional OpenAI Image 2 sixth column | Unified 5-way / 6-way offline eval |
| Multimodal embedding | Gemini Embedding 2 (3,072-dim with Matryoshka) | Both Knowledge Retriever and Stage 1 of Two-Stage Matcher |
| Visual embedding | DINOv2-ViT-B14 | Stage 2 rerank only |
| Vector Store | Qdrant | Named vectors, int8 quantization, alias-swap rebuild, snapshot drill |
| LoRA training | ai-toolkit (ostris) on Replicate | One per-artist LoRA Artifact, FLUX.2-dev + FLUX.2-klein bases |
| Eval framework | DeepEval | Tier 1 (component) + Tier 2 (agent) + CI; Tier 3 (synthetic, drift) stretch |
| UX shell | Gradio | Chat + Candidate Design grid + ranked-artist gallery |
| Deploy artifacts | Dockerfile + Modal config | Reviewer runs locally or deploys themselves |
| Tests | pytest | Mocked unit + gated live integration |
| Evals | DeepEval as pytest-discoverable functions | Wired into CI |
| CI | GitHub Actions | Runs both tests and evals on every push |
| Repo license | MIT | |

---

## Repo Structure

```
tattd-studio/
├── README.md                    # Hero narrative + "Run in 5 minutes" + eval results table
├── ARCHITECTURE.md              # Final deeper reasoning (issue #11)
├── DATA_PROVENANCE.md           # Aggregated per-record source attribution (built across #3, #7, #8, #9, finalized #11)
├── PRD.md                       # Product requirement
├── CONTEXT.md                   # Project glossary (vocabulary source of truth)
├── IMPLEMENTATION_PLAN.md       # This file
├── pyproject.toml               # uv-managed, LangGraph pinned with rationale
├── .env.example                 # GEMINI_API_KEY, VERTEX_PROJECT_ID, REPLICATE_API_TOKEN, HF_TOKEN
├── Dockerfile
├── modal.py                     # Modal deploy config
├── .github/workflows/ci.yml     # pytest + Eval Harness on every push
├── src/tattd_studio/
│   ├── __init__.py
│   ├── main.py                  # Gradio entrypoint
│   ├── graph/                   # LangGraph state machine
│   │   ├── state.py             # StudioState TypedDict
│   │   ├── consultation.py      # Consultation node + Knowledge Retriever wiring
│   │   ├── generation.py        # Generation Client invocation
│   │   ├── critics/
│   │   │   ├── anatomy.py       # → AnatomyCheck
│   │   │   ├── plagiarism.py    # → PlagiarismCheck
│   │   │   ├── style.py         # → StyleCoherence
│   │   │   └── quality.py       # → QualityScore
│   │   ├── matching.py          # Two-Stage Matcher
│   │   └── routing.py           # Conditional edge logic
│   ├── models/                  # Pydantic schemas (all critic outputs + Candidate Design envelope)
│   ├── embeddings/
│   │   ├── gemini.py            # multimodal embedding client (Gemini Embedding 2)
│   │   ├── dinov2.py            # visual embedding client (DINOv2)
│   │   └── benchmark.py         # 3-way multimodal embedding benchmark runner
│   ├── vectordb/
│   │   ├── qdrant_client.py     # Vector Store: named-vector schema, int8 quant
│   │   └── reindex.py           # Alias-swap rebuild script
│   ├── knowledge/
│   │   ├── chunks/              # ~150 Knowledge Corpus markdown chunks
│   │   └── ingest.py            # Chunk → multimodal embedding → Vector Store
│   ├── generation/              # Generation Client wrapper
│   └── ui/
│       └── gradio_app.py        # Chat + Candidate Design grid + match gallery
├── data/
│   ├── artists/                 # Curated real (10–20) + synthetic (20–30)
│   ├── famous_tattoos/          # ~50 iconic / celebrity tattoos
│   ├── knowledge/               # Source markdown for the Knowledge Corpus
│   ├── lora_training/           # ~25–30 images for the per-artist LoRA Artifact + permission marker
│   └── eval/                    # Golden Sets (jsonl)
│       ├── retrieval_golden.jsonl
│       ├── plagiarism_pairs.jsonl
│       ├── anatomy_cases.jsonl
│       ├── conversation_traces.jsonl
│       └── comparison_prompts.jsonl
├── evals/
│   ├── tier1/                   # Component evals (one per Critic + retrieval + generation)
│   ├── tier2/                   # Agent-level (conversation, traces, embedding benchmark, Comparison Matrix)
│   ├── tier3/                   # Stretch (synthetic data, drift)
│   ├── calibrated_thresholds.toml
│   └── results/                 # Committed Eval Reports (markdown)
├── infra/
│   ├── docker-compose.yml       # Qdrant + Studio for local dev
│   └── train_lora.yaml          # ai-toolkit config for the LoRA Artifact
└── tests/                       # Unit tests separate from evals
```

---

## Build Sequence

Tracer bullets filed as GitHub issues. Slice number == issue number.

| # | Title | Type | Blocked by |
|---|-------|------|------------|
| [#1](https://github.com/techjays-prem/tattd-studio/issues/1) | Repo + CI skeleton + Eval Harness shell | AFK | — |
| [#2](https://github.com/techjays-prem/tattd-studio/issues/2) | Vector Store wrapper | AFK | #1 |
| [#3](https://github.com/techjays-prem/tattd-studio/issues/3) | Knowledge Corpus ingest + Knowledge Retriever happy path | AFK | #2 |
| [#4](https://github.com/techjays-prem/tattd-studio/issues/4) | Anatomy Critic end-to-end | AFK | #1 |
| [#5](https://github.com/techjays-prem/tattd-studio/issues/5) | Generation Client wrapper | AFK | #1 |
| [#6](https://github.com/techjays-prem/tattd-studio/issues/6) | Minimal LangGraph Studio: Consultation → Generation → Anatomy Critic → end | AFK | #3, #4, #5 |
| [#7](https://github.com/techjays-prem/tattd-studio/issues/7) | Plagiarism Critic + Style Critic + Quality Critic + Routing | AFK | #6 |
| [#8](https://github.com/techjays-prem/tattd-studio/issues/8) | Artist Portfolio Index + Two-Stage Matcher + multimodal embedding benchmark | AFK | #2, #7 |
| [#9](https://github.com/techjays-prem/tattd-studio/issues/9) | LoRA Artifact training run on Replicate | HITL | #1 |
| [#10](https://github.com/techjays-prem/tattd-studio/issues/10) | Comparison Matrix | AFK | #5, #9 |
| [#11](https://github.com/techjays-prem/tattd-studio/issues/11) | Deploy artifacts + documentation polish | AFK | #6, #7, #8, #10 |

**Parallelizable paths:**

- After #1: #2, #4, #5, #9 can all run in parallel.
- After #5: #6 starts whenever #3 also finishes.
- After #5 + #9: #10 can run in parallel with #6 / #7 / #8.
- #11 is the convergence; all other slices land first.

---

## Eval Surface

| Layer | Tier | What it tests | DeepEval primitives |
|-------|------|---------------|---------------------|
| Anatomy Critic | T1 | precision / recall on labeled valid + invalid placements | custom `BaseMetric` subclass |
| Plagiarism Critic | T1 | AUC + threshold calibration on near-dup vs original pairs | custom `BaseMetric` |
| Style Critic | T1 | correlation with human ratings on (Intent, Candidate Design) pairs | custom `BaseMetric` |
| Quality Critic | T1 | drift detection on stable golden set | `GEval` |
| Retrieval | T1 | recall@k, MRR, NDCG | `ContextualRelevancy`, `ContextualRecall`, `ContextualPrecision` |
| Generation | T1 | faithfulness + originality + technical quality on Candidate Designs | `GEval` + custom `OriginalityMetric` |
| multimodal embedding benchmark | T2 | 3-way: Gemini Embedding 2 vs `multimodalembedding@001` vs SigLIP 2 on retrieval Golden Set | recall@k, MRR, NDCG |
| Comparison Matrix | T2 | 5-way (or 6-way): FLUX.2-dev ± LoRA Artifact, FLUX.2-klein ± LoRA Artifact, Generation Client, optional OpenAI Image 2 | FID + CLIP score + `GEval` + style-adherence-via-multimodal-embedding-cosine |
| Consultation (multi-turn) | T2 | quality, faithfulness, knowledge retention | `AnswerRelevancy`, `Faithfulness`, `ConversationRelevancy`, `KnowledgeRetention` |
| End-to-end Studio | T2 | goal completion, latency, cost-per-session | trace evaluation |
| Synthetic test-set generator | T3 (stretch) | golden set expansion | DeepEval Synthesizer |
| Drift detection | T3 (stretch) | regression over time | scheduled CI |
| **CI integration** | required | pytest + Eval Harness on every push | GitHub Actions |

---

## Verification (end-to-end)

After issue #11 lands, the system passes the following:

1. **Local startup.** `git clone` → `cp .env.example .env` → fill keys → `uv sync` → `python -m tattd_studio.main` → Gradio at `localhost:7860` shows chat.
2. **Happy path.** Type *"I want a fineline minimalist mountain on my inner forearm, ~3 inches"* → Consultation refines → 4 Candidate Designs returned → 4 Critics each return Pydantic verdicts visible in UI → choose one → top 5 onboarded artists shown with name + portfolio link.
3. **Critic failure path.** Request something that should fail anatomy (e.g., *"huge realism portrait on knuckle"*) → Anatomy Critic flags → Routing triggers Refinement → re-generates with adjusted prompt.
4. **Plagiarism path.** Generate a deliberately near-duplicate of an indexed artist (test fixture) → Plagiarism Critic flags with `top_match_artist` and `top_match_similarity` populated → Routing diversifies and re-generates once → second flag escalates.
5. **Eval suite.** `pytest evals/` runs all Tier 1 + Tier 2 evals, generates a report, all metrics within calibrated thresholds. Output committed as `evals/results/latest.md`.
6. **CI green.** `git push` → GitHub Actions runs pytest + Eval Harness → results visible in PR comments.
7. **Comparison Matrix artifact.** `evals/tier2/test_model_comparison.py` produces a 5-way table comparing the four Comparison Matrix entries against the Generation Client on the shared prompt set — table committed in repo.
8. **Provenance complete.** Every artist record, every Knowledge Corpus chunk, every Famous Tattoos Corpus entry, every LoRA training image has a source field documented in DATA_PROVENANCE.md.
9. **Deploy dry-run.** `modal deploy modal.py` succeeds without actual deploy (or with auth-gated deploy if user chooses).

---

## Risks & Gotchas

- **LangGraph 1.0.x churn.** Pin tight; budget half-day per upgrade. Document pin rationale in `pyproject.toml` comment.
- **Multimodal state in LangGraph.** Don't pass raw image bytes through graph state. Store images by URI in object storage (or local FS for POC); pass URIs through state.
- **Vertex AI access.** Confirm GCP project has Vertex AI API enabled before issue #2 starts implementation. Without it, Gemini Embedding 2 is unavailable; fall back to SigLIP 2 as primary, document the deviation.
- **Critic latency.** 4 Critics in series is 5–15 s. LangGraph runs independent branches in parallel — wire as parallel nodes from the start.
- **Plagiarism threshold tuning** is itself an eval problem. Calibrate against known near-dup + known-original pairs; expect 1–2 days to land defensible thresholds. Thresholds live in `evals/calibrated_thresholds.toml` so Routing reads them at runtime.
- **DINOv2 GPU vs CPU.** Inference is fine on CPU at this scale (~50–200 ms per image). GPU only matters at high throughput.
- **LoRA training iterations.** Plan for 2–3 retrains to dial in style fidelity. ai-toolkit's `content_or_style: balanced` + `caption_dropout_rate: 0.05` are sane starting points.
- **Qdrant named-vector dim consistency.** Per-name dim is fixed. For Matryoshka tiers, declare `multimodal-1024` and `multimodal-3072` as separate named vectors on the same point; slice 3072→1024 client-side (Gemini Matryoshka guarantees prefix correctness).
- **Synthetic image quality** may show seams in the embedding space. Flag in the multimodal embedding benchmark Eval Report.
- **Real onboarded-artist source freshness.** Self-promoted public accounts can vanish. Cache locally with Provenance metadata and source-snapshot date.
- **Snapshot drill.** Set Qdrant snapshot config + a documented restore procedure in the README from day one. Removes the "we lost the index" failure mode.
- **Gemini Embedding 2 is brand-new (April 22, 2026).** Expect undocumented edges; budget half a day for first-time wiring.
- **Nano Banana 2 / Pro is not in any framework's indexed docs.** Wrap `google.genai.Client().models.generate_content` directly in a node; no third-party adapter exists.

---

## Out of Scope

- Booking integration / payment / calendaring
- Aftercare as a dedicated agent surface (Knowledge Corpus contains aftercare chunks, but no dedicated conversation flow)
- Artist-side tools, dashboards, or onboarding surfaces — single-sided (human-client-facing) only
- Multi-language Consultation — English-only at POC scale
- Multi-artist LoRA Artifacts — single per-artist LoRA proves the wedge; multi-artist is roadmap, not artifact
- Watermark detection, perceptual-hash exact-match plagiarism beyond embedding similarity (mention in README "what's next")
- Live hosted demo, video walkthrough, outreach plan, recipient targeting — developer's responsibility, not engineering scope

---

## Workflow Constraints

- **No AI attribution** in commits, PRs, or code.
- **Protected `main`** — every change ships via feature branch + PR. Force-pushes blocked, deletions blocked.
- **CONTEXT.md vocabulary verbatim** in code, issues, PRs, commits, eval reports. Synonyms listed under each term's `_Avoid_` are banned.
- **DATA_PROVENANCE.md is append-only across slices** until issue #11 finalizes the structure.

---

## Handoff to /tdd

The next session begins with the user invoking `mattpocock-skills:tdd` against whichever GitHub issue they pick (#1 first by topology). The TDD loop drives red-green-refactor against that issue's acceptance criteria, with this plan, the PRD, and CONTEXT.md as the standing references.
