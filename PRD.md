# PRD — Tattd Studio POC

> Author: Prem Nath · Date: 2026-05-06 · Status: Draft for implementation

## Problem Statement

A developer wants to demonstrate fitness for a senior AI/ML engineering role at Tattd (tattd.ai), an AI-powered tattoo design + artist booking marketplace. The role's job description requires production-shipped experience with diffusion models, embedding pipelines, vector databases, and retrieval-augmented generation, and explicitly excludes generic data-science backgrounds.

Without an inside referral, the developer's options are: (1) a generic CV that competes with hundreds of similar applications, or (2) an unsolicited proof — a working artifact that demonstrates the required skills *on Tattd's actual production stack*, addressing concerns Tattd specifically cares about (style fidelity, plagiarism risk, marketplace differentiation).

The developer has limited time (a 2-4 month solo timeline), no inside access to Tattd's repo or data, and needs to convert public information about Tattd's product surfaces into a credible, runnable engineering artifact.

## Solution

Build the **Tattd Studio** — a runnable POC of the AI core that powers Tattd's user journey, delivered as a local-runnable repository with deployment artifacts, comprehensive evaluation suite, and documentation framing the engineering decisions.

The Studio is a multi-turn agent that:

1. **Consults** with a prospective tattoo client to refine their design intent (style, body location, meaning, color, size), grounded in retrieved knowledge from a curated tattoo corpus (style taxonomy, anatomy/placement, aftercare, IP best-practices, cultural considerations).
2. **Generates** candidate tattoo designs using Tattd's production model (Gemini Nano Banana 2 / Pro), with a parallel comparison evaluation against open-weights alternatives (FLUX.2-dev and FLUX.2-klein, with and without a per-artist style LoRA).
3. **Self-critiques** every generation across four dimensions — anatomy/placement validity, plagiarism similarity, style coherence with stated intent, and quality — using specialized critic tools that each return structured outputs. Failures route back to refinement.
4. **Matches** the chosen design to artists in a portfolio index using a two-stage retrieval pipeline (multimodal embedding for recall, fine-grained visual similarity for rerank).
5. **Evaluates everything** through DeepEval at component, agent, and CI layers, with reproducible results committed to the repo.

The artifact's narrative engages Tattd's specific concerns directly: a `DATA_PROVENANCE.md` documents every artist source and synthetic origin, the plagiarism critic checks against both the indexed portfolio and a curated set of iconic/famous tattoos, and the per-artist LoRA artifact frames style fine-tuning as a marketplace product wedge that closed APIs structurally cannot deliver.

## User Stories

1. As a Tattd hiring decision-maker, I want to clone the repo and run the agent in under five minutes, so that I can verify the candidate ships working code without orchestrating a setup call.
2. As a Tattd hiring decision-maker, I want to read a single architecture document that explains the engineering tradeoffs the candidate made, so that I can assess the seniority of their judgment without reading every line of code.
3. As a Tattd hiring decision-maker, I want to see evaluation results in a committed report file, so that I can verify the candidate's claims about agent quality without re-running every test.
4. As a Tattd hiring decision-maker, I want to see the candidate's reasoning about Tattd's production stack reflected in the implementation, so that I trust the candidate could extend our actual infrastructure on day one.
5. As a Tattd hiring decision-maker, I want explicit treatment of plagiarism and IP risk in the data and design, so that I trust the candidate understands our domain's specific reputational concerns.
6. As a Tattd hiring decision-maker, I want a clear model-comparison artifact (production model vs open-weights), so that I can evaluate how the candidate thinks about model selection at scale.
7. As a Tattd hiring decision-maker, I want a clear "what's next at production scale" section, so that I see the candidate planning beyond the POC's bounded scope.
8. As a prospective tattoo client interacting with the agent, I want to describe my tattoo idea conversationally, so that I don't need to know the right vocabulary upfront.
9. As a prospective tattoo client, I want the agent to ask follow-up questions grounded in real tattoo knowledge (style families, placement physiology), so that the consultation feels expert rather than generic.
10. As a prospective tattoo client, I want to see multiple candidate designs at once, so that I can compare options without re-prompting.
11. As a prospective tattoo client, I want each candidate design to be filtered for anatomical plausibility before I see it, so that I am not shown impossible designs (e.g., sleeve-scale artwork on a knuckle).
12. As a prospective tattoo client, I want the agent to flag when a generated design is too similar to existing artists' work, so that I do not unknowingly request a copy.
13. As a prospective tattoo client, I want the agent to offer iterative refinement on a chosen design (color, size, sub-style adjustments), so that the conversation builds toward my final idea rather than restarting each time.
14. As a prospective tattoo client, I want artist matches to reflect the visual style of my chosen design, not just keyword overlap, so that I can see artists whose work would actually translate to my idea.
15. As a prospective tattoo client, I want the artists I am matched with to be displayed with their name and a link to their portfolio, so that I can verify their work and book them.
16. As a tattoo artist whose portfolio appears in the index, I want my source attribution preserved in the system, so that clients are directed to my actual portfolio rather than seeing my work decoupled from my identity.
17. As a tattoo artist whose portfolio appears in the index, I want the system to record my work with explicit provenance metadata, so that there is a clear trail of how my images entered the system.
18. As the developer building the POC, I want each agent capability isolated as a separately testable module, so that I can verify behavior at the unit level before stitching the agent together.
19. As the developer building the POC, I want every critic to return a typed Pydantic structure, so that downstream routing logic can branch deterministically and the eval harness has a stable schema.
20. As the developer building the POC, I want to evaluate three candidate multimodal embedding models against a held-out tattoo retrieval test set, so that the embedding choice is justified with numbers rather than vibes.
21. As the developer building the POC, I want to compare the production generation model against open-weights alternatives (with and without a per-artist LoRA) on a shared prompt set, so that the artifact demonstrates evaluation skill across model families.
22. As the developer building the POC, I want every commit to run the component-level eval suite via continuous integration, so that regressions are caught immediately.
23. As the developer building the POC, I want a documented procedure for re-indexing the vector database when the embedding model changes, so that future model upgrades do not require a from-scratch rebuild.
24. As the developer building the POC, I want every artist record and knowledge corpus chunk to carry source and provenance metadata, so that the artifact's data posture is explicitly defensible.
25. As the developer building the POC, I want the agent's failure modes (anatomy fail, plagiarism flag, style mismatch) to be observable in the trace, so that I can debug the conditional routing logic during development and demonstrate it in evaluation.
26. As the developer building the POC, I want the per-artist LoRA training to run on a hosted GPU service, so that I do not manage GPU lifecycle for a one-off training run.
27. As the developer building the POC, I want the trained LoRA artifact to be loadable by the same inference framework used in the rest of the eval, so that the comparison evaluation runs without format conversion.
28. As the developer building the POC, I want the deployment artifacts (Dockerfile, hosted-deploy config) committed to the repo, so that whoever consumes the POC can deploy it themselves without me operating it.
29. As the developer building the POC, I want the consultation knowledge corpus to be sourced from a combination of curated and synthesized content with explicit provenance per chunk, so that the corpus's defensibility is clear.
30. As the developer building the POC, I want the critic latency dominated by parallel rather than serial evaluation, so that the user-facing turnaround time stays within an acceptable budget.
31. As the developer building the POC, I want the vector database choice to support storing two embeddings (multimodal + visual-rerank) on a single record, so that artist matching does not require joining across two stores.
32. As the developer building the POC, I want explicit out-of-scope markers in the documentation for booking, payments, aftercare-as-agent, and artist-side surfaces, so that reviewers do not penalize the artifact for not solving problems it never claimed to solve.
33. As a future maintainer of the POC, I want the LangGraph version pinned with documented rationale, so that I understand why upgrades require dedicated planning rather than dependabot auto-merge.

## Implementation Decisions

### Architectural decisions

- **Agent orchestration uses LangGraph**, with a state graph composed of consultation, generation, four parallel critics, conditional routing, and matching nodes. State carries refined intent, candidate images (by URI, not bytes), critic outputs, and final artist matches.
- **Generation primary uses Gemini Nano Banana 2 / Pro** as the production model, with the FLUX.2-dev and FLUX.2-klein base models (and per-artist LoRA-adapted versions) participating in a parallel comparison evaluation but not in the runtime generation path.
- **Embedding primary uses Gemini Embedding 2** (3,072-dim, Matryoshka) as the shared text-and-image embedding for both the consultation knowledge RAG layer and the artist portfolio image retrieval layer. This unifies both retrieval surfaces into one embedding space.
- **Visual rerank uses DINOv2-ViT-B14** as a second-stage image-only fine-grained similarity model after Stage 1 multimodal recall. The full retrieval pipeline is two-stage.
- **Vector store uses Qdrant** with named-vectors per record (one named vector for the multimodal embedding tier, one for the visual rerank embedding), int8 scalar quantization configured upfront, and an alias-swap rebuild pattern for embedding-model migrations.
- **Self-critique is decomposed into four specialized modules**, not a single LLM judge. Anatomy and quality use vision-language model judges with rubrics; plagiarism uses embedding similarity against two reference corpora (artist portfolio + curated famous tattoos); style coherence uses text-image embedding alignment.
- **Each critic returns a typed Pydantic model** consumed by the routing logic. Failure on any dimension routes to a refinement step that adjusts the generation prompt before re-generating.
- **Per-artist LoRA training uses ai-toolkit (ostris) on Replicate**, with the artifact reframed as a marketplace product wedge rather than a generic open-weights checkbox. One curated artist's portfolio is the training set (with explicit permission).
- **The model comparison evaluation is a unified 5-way matrix**: production model + 2 base open-weights models + 2 LoRA-adapted versions. OpenAI Image 2 is added as a 6th column if API access is available.

### Module decomposition

- **Critic modules** — four separate deep modules, one per dimension. Interface: `(image, context) → PydanticOutput`. Stable, isolatable, swappable.
- **Embedding clients** — two modules: a multimodal client wrapping Vertex AI's Gemini Embedding 2, and a local DINOv2 client. Interface: `(input) → vector`.
- **Vector store** — wraps Qdrant client behind a named-vector-aware interface that hides quantization configuration, schema declarations, and the alias-swap rebuild pattern.
- **Knowledge retriever** — composes the multimodal embedding client with the vector store for consultation grounding. Interface: `(query, k) → chunks_with_sources`.
- **Two-stage matcher** — composes both embedding clients with the vector store for artist matching. Interface: `(design_embedding) → ranked_artists`.
- **Generation client** — wraps Nano Banana with retry, cache, and structured prompt versioning. Interface: `(prompt, n) → images`.
- **Eval harness** — composes DeepEval per layer with consistent pytest-discoverable test functions. Tier 1 component evals for each critic + retrieval + generation. Tier 2 agent-level evals for conversation, traces, and the model comparison matrix. Tier 3 stretch for synthetic data generation and drift detection.

### Data decisions

- **Artist portfolio index**: 10–20 curated real artists (public self-promoted accounts, linked with credit) plus 20–30 synthetic style-coverage artists. Per-record provenance metadata. Same dataset doubles as one of the two plagiarism reference corpora.
- **Famous tattoos corpus**: ~50 curated iconic/celebrity tattoos for a secondary plagiarism check. Captures the "obvious cultural rip" failure mode that the artist portfolio alone would miss.
- **Knowledge corpus for consultation RAG**: ~150 chunks across five areas (style taxonomy, body placement/anatomy, aftercare, IP/plagiarism best-practices, cultural/ethical considerations). Curated public sources for high-trust topics; LLM-synthesized with citations for breadth on stylistic taxonomy. Per-chunk provenance.
- **LoRA training set**: ~25–30 portfolio images from one curated real artist (with explicit permission), same provenance discipline as the broader index.
- **Eval golden sets**: paired query-to-artist data for retrieval recall@k, near-duplicate vs original pairs for plagiarism threshold calibration, valid vs invalid placement examples for anatomy precision/recall, paired (intent, image) data for style coherence correlation, stable human-rated set for quality drift detection, multi-turn fixtures for conversation eval.

### API contracts and schemas

- **Critic outputs are Pydantic models**: `AnatomyCheck`, `PlagiarismCheck`, `StyleCoherence`, `QualityScore`. Each carries dimension-specific fields plus a `confidence` or `score` field consumed by the routing layer's threshold logic.
- **Vector store schema**: each Qdrant point has a payload (artist metadata, source URL, provenance flag, style tags, location) and two named vectors (multimodal at 1,024-dim Matryoshka tier and 3,072-dim full, plus DINOv2 at its native dim).
- **Generation client returns structured envelopes**, not raw bytes: each candidate image carries a URI, the prompt used, a model identifier, and optional model-specific metadata.
- **Eval reports are committed as markdown** under a known path so reviewers can read the latest results without running tests.

### Out-of-band engineering decisions

- Python 3.12+, uv for dependency management, pytest as test runner, GitHub Actions for continuous integration, MIT license on the repo.
- LangGraph version is pinned to a specific 1.0.x release with a comment in the dependency manifest explaining the rationale (quarterly API churn).
- Images are passed through the LangGraph state by URI rather than as raw bytes, to avoid serializer issues with binary content.
- Critics run in parallel where independent (anatomy, style, quality on the image; plagiarism in parallel against the embedding-similarity vector store).
- Repo includes Dockerfile and a hosted-deploy config so reviewers can deploy themselves without the developer operating any service.

## Testing Decisions

### What makes a good test in this codebase

Tests verify observable external behavior of a module — the contract its interface promises — not implementation details. A good critic test asserts the structured output contract for known inputs (a known-anatomically-invalid image returns `placement_valid=False` with relevant issues populated); it does not assert what intermediate prompts the critic sent to the VLM. A good vector-store test asserts that named-vector queries return the right hits filtered by payload; it does not assert how Qdrant's HNSW indexed the points.

Tests should be reproducible without flaky external dependencies. Live-API tests are gated and run separately from the fast pytest suite; mocked unit tests cover happy and failure paths for the same modules at speed.

DeepEval-based evaluations are not unit tests; they are evaluations. They run on golden datasets and produce metric reports rather than pass/fail asserts. The CI runs both pytest and DeepEval; thresholds for DeepEval metrics are calibrated separately and committed alongside the test runner config.

### Modules to test

- **All four critic modules** (anatomy, plagiarism, style, quality). Each gets unit tests against fixture inputs (mocking the underlying VLM or embedding service) plus DeepEval golden-set evaluation. The "robust" claim of the agent depends on these being correct in isolation.
- **Both embedding clients** (Gemini Embedding 2 wrapper and DINOv2 wrapper). Mocked unit tests for happy and failure paths; gated live-API integration tests that verify the actual provider contract still matches.
- **Vector store module**. Integration tests against a local in-memory Qdrant instance, covering named-vector schema declaration, index creation with quantization, point upsert, named-vector query with payload filter, and the alias-swap rebuild procedure.
- **Knowledge retriever module**. Integration tests against a fixture corpus of known chunks, verifying that a known query retrieves the expected chunk with the expected source metadata.
- **Two-stage matcher module**. Integration tests against a fixture artist index with deterministic style coverage, verifying that Stage 1 recall returns the expected candidate set and Stage 2 rerank reorders by visual similarity in the expected direction.

### Prior art

No prior art in this repo since it is greenfield. The eval-style tests will be the first reference once the first critic eval lands; subsequent critic evals follow that pattern. Vector-store integration tests follow the standard pattern of a session-scoped fixture spinning up an in-memory Qdrant client. Mocked-VLM unit tests follow the standard pytest fixture-and-monkeypatch pattern.

## Out of Scope

- Booking, calendar integration, payment, and any post-match transaction flow.
- Aftercare as a dedicated agent surface (the knowledge corpus contains aftercare chunks for consultation grounding, but no dedicated aftercare conversation flow).
- Artist-side tools, dashboards, or onboarding surfaces — the artifact is single-sided (client-facing) only.
- Multi-language consultation — English-only at POC scale.
- Multi-artist LoRAs — one per-artist LoRA proves the wedge; a multi-artist roadmap belongs in the "what's next" section, not the artifact.
- Watermark detection, perceptual-hash exact-match plagiarism, or other plagiarism-detection layers beyond embedding similarity. Mentioned in the README's "what's next at production scale" section.
- Live hosted demo, video walkthrough, outreach plan, and recipient targeting — these are the developer's responsibility, not the engineering scope.
- Replacing Tattd's existing infrastructure or proposing migrations beyond what the artifact's README explicitly notes as "what I'd do at production scale."

## Further Notes

The artifact's narrative is engineering-first, not marketing-first: the README leads with architecture and tradeoffs, then evaluation results, then the per-artist LoRA wedge framing, then "what's next at production scale." The data provenance documentation, the model comparison evaluation, and the explicit out-of-scope markers are all intentional senior-engineering signals that the developer thought about the audience's likely concerns.

The acknowledged gap between the JD's literal "FLUX/SD" language and Tattd's actual production stack (Gemini) is engaged directly in the README rather than glossed over: the developer notes the discrepancy, explains the engineering call to ship on the production stack, and points to the unified comparison evaluation and per-artist LoRA artifact as the means by which the open-weights expectation is addressed.

The full architectural plan, week-by-week build sequence, eval surface table, risk inventory, and out-of-scope list live in the implementation plan file at `/Users/premnath/.claude/plans/do-you-know-the-concurrent-boot.md`. This PRD is the user-facing complement; that plan file is the executable spec.
