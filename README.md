# Tattd Studio

A multi-turn tattoo-design **Studio** built on Tattd's stated production stack (Gemini Nano Banana 2 / Pro, Gemini Embedding 2) with rigorous DeepEval coverage at the component, agent, and CI layers.

This repo is the unsolicited proof of fitness for Tattd's senior AI/ML engineering role. The job description requires production-shipped experience with diffusion models, embedding pipelines, vector databases, and retrieval-augmented generation; it explicitly excludes generic data-science backgrounds. Every choice in this artifact tracks back to that bar:

- **Real production stack.** Gemini Nano Banana 2 / Pro for generation; Gemini Embedding 2 (Matryoshka) for retrieval; Qdrant with named-vector schema and int8 quantization for the Vector Store; DINOv2-ViT-B14 for the Two-Stage Matcher's visual rerank.
- **Multi-turn agentic flow under DeepEval.** LangGraph 1.0.10 state machine: Consultation → Generation → four parallel Critics (Anatomy / Plagiarism / Style / Quality) → Routing (with Refinement loop) → Two-Stage Matcher → ranked artists.
- **JD-discrepancy framing.** The JD names "FLUX/SD" experience as a hard requirement. This artifact honors that **as a Comparison Matrix** — FLUX.2-dev / FLUX.2-klein with per-artist LoRA Artifacts vs the Generation Client — rather than as a generic checkbox. The LoRA Artifact never participates in runtime generation; it only appears under the Eval Harness, framed as marketplace value (per-artist style fidelity).
- **Provenance discipline.** Every record in every corpus carries source attribution; [DATA_PROVENANCE.md](./DATA_PROVENANCE.md) is the single auditable trail.

## Run in 5 minutes

```bash
git clone https://github.com/techjays-prem/tattd-studio.git
cd tattd-studio
cp .env.example .env   # fill: GEMINI_API_KEY (required), VERTEX_PROJECT_ID, REPLICATE_API_TOKEN, HF_TOKEN
uv sync --extra ui --extra dev
python -m tattd_studio.main
# Gradio UI at http://localhost:7860
```

What you can do:

- Type a tattoo description ("fineline minimalist mountain on inner forearm, ~3 inches")
- The Studio runs Consultation → Generation → 4 parallel Critics → Routing → Two-Stage Matcher
- Surfaced Candidate Designs render with each Critic's verdict overlaid; the Two-Stage Matcher returns top-N onboarded artists with portfolio links

Run the test suite (no API keys required):

```bash
uv run ruff check .
uv run pytest tests/ evals/tier1 evals/tier2
```

Run the live integration tests (consumes API quota):

```bash
RUN_LIVE_GENERATION_TESTS=1 \
RUN_LIVE_ANATOMY_EVAL=1 \
RUN_LIVE_STUDIO_TESTS=1 \
RUN_LIVE_EMBEDDING_BENCHMARK=1 \
GEMINI_API_KEY=... \
uv run pytest
```

## Deploy

**Docker** — `Dockerfile` at the repo root builds a self-contained image:

```bash
docker build -t tattd-studio .
docker run -p 7860:7860 -e GEMINI_API_KEY=... tattd-studio
```

**Docker Compose** — `infra/docker-compose.yml` brings up Qdrant + the Studio together for local dev with a real persistent Vector Store (instead of `:memory:`):

```bash
GEMINI_API_KEY=... docker compose -f infra/docker-compose.yml up
```

**Modal** — `modal.py` at the repo root is the auth-gated cloud deploy:

```bash
modal token new                                # one-time
modal secret create tattd-gemini GEMINI_API_KEY=...
modal deploy modal.py
```

The reviewer can deploy this themselves; the developer doesn't operate any service.

## Eval surface

Tiered DeepEval suite, every entry committed under `evals/results/` so reviewers see metrics without rerunning evaluations:

| Layer | Tier | Eval Report |
|---|---|---|
| Anatomy Critic — placement validity precision/recall on a 60-case Golden Set | T1 | [`anatomy_critic_latest.md`](./evals/results/anatomy_critic_latest.md) |
| Plagiarism Critic — near-dup AUC on 20 pairs | T1 | [`plagiarism_critic_latest.md`](./evals/results/plagiarism_critic_latest.md) |
| Style Critic — Intent → Candidate Design alignment on 15 pairs | T1 | [`style_critic_latest.md`](./evals/results/style_critic_latest.md) |
| Quality Critic — composition / linework / balance / originality drift | T1 | [`quality_critic_latest.md`](./evals/results/quality_critic_latest.md) |
| Knowledge Retriever — recall@k / MRR / area-recall + DeepEval `ContextualRelevancy / ContextualRecall / ContextualPrecision` (live only) | T1 | [`retrieval_tier1_latest.md`](./evals/results/retrieval_tier1_latest.md) |
| Generation — faithfulness + originality (custom `OriginalityMetric` over the Famous Tattoos Corpus) + DeepEval `GEval` rubric (live only) | T1 | [`generation_tier1_latest.md`](./evals/results/generation_tier1_latest.md) |
| End-to-end Studio trace — latency + candidate / chunk counts | T2 | [`studio_traces_latest.md`](./evals/results/studio_traces_latest.md) |
| Consultation (multi-turn) — intent-substring coverage + grounding-area match + DeepEval `AnswerRelevancy / Faithfulness / ConversationRelevancy / KnowledgeRetention` (live only) | T2 | [`consultation_tier2_latest.md`](./evals/results/consultation_tier2_latest.md) |
| Multimodal embedding 3-way benchmark — Gemini Embedding 2 vs `multimodalembedding@001` vs SigLIP 2 on 20-query retrieval Golden Set | T2 | [`embedding_benchmark.md`](./evals/results/embedding_benchmark.md) |
| Comparison Matrix — FLUX.2-dev ± LoRA Artifact, FLUX.2-klein ± LoRA Artifact, Generation Client (+ optional OpenAI Image 2 sixth column) | T2 | [`comparison_matrix.md`](./evals/results/comparison_matrix.md) (30 prompts; LoRA-adapted columns deferred to [#9](https://github.com/techjays-prem/tattd-studio/issues/9) — light up automatically when `data/lora_training/artifacts.toml` carries real entries; `OpenAI Image 2` column activated by `OPENAI_IMAGE_2_API_KEY`) |

Default CI uses deterministic baselines (no API keys). Live thresholds activate under env-gated runs (`RUN_LIVE_*=1` + `GEMINI_API_KEY`).

## What's next at production scale

See [ARCHITECTURE.md → What's next at production scale](./ARCHITECTURE.md#whats-next-at-production-scale) for the full table. Short version:

- Real artist onboarding flow → swaps the synthetic Artist Portfolio Index for real records with signed permission
- LoRA Artifact training run on Replicate → unblocks the Comparison Matrix
- Provider failover, durable caching, OpenTelemetry traces, Routing-threshold A/B tests
- Tier 3: synthetic test-set generation (DeepEval Synthesizer) + drift detection in scheduled CI

## Out of scope

Per the PRD:

- Booking integration / payment / calendaring
- Aftercare as a dedicated agent surface
- Artist-side tools, dashboards, or onboarding surfaces — single-sided (human-client-facing) only
- Multi-language Consultation — English-only at POC scale
- Multi-artist LoRA Artifacts — single per-artist LoRA proves the wedge
- Watermark detection / perceptual-hash exact-match plagiarism beyond embedding similarity
- Live hosted demo, video walkthrough, outreach — developer's responsibility, not engineering scope

## Documents

- [PRD.md](./PRD.md) — product requirement
- [CONTEXT.md](./CONTEXT.md) — project glossary; vocabulary used in code, issues, PRs, and commits
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) — architectural / decision-log spec
- [ARCHITECTURE.md](./ARCHITECTURE.md) — deeper engineering reasoning, decision log, risk inventory
- [DATA_PROVENANCE.md](./DATA_PROVENANCE.md) — single auditable trail across all corpora

## License

MIT — see [LICENSE](./LICENSE).
