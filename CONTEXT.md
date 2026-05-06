# Tattd Studio — Project Context

The shared vocabulary for the Tattd Studio POC: a runnable artifact that demonstrates a multi-turn tattoo-design agent built on Tattd's production stack (Gemini Nano Banana 2 / Pro, Gemini Embedding 2) with rigorous DeepEval coverage. This document is the source of truth for terms used in code, issues, PRs, commits, and evaluation reports — every other doc defers to it.

## Language

### Agent surface

**Studio**:
The end-to-end agent that conducts one tattoo-design session from intent elicitation through artist matching.
_Avoid_: bot, assistant, agent (too generic), Tattd assistant, Concierge (deprecated — earlier name).

**Consultation**:
The multi-turn dialogue stage in which **Intent** is elicited from the human client and grounded against the **Knowledge Corpus**.
_Avoid_: chat, intake, conversation, discovery.

**Intent**:
The structured, refined description of the human client's tattoo idea consumed by the **Generation Client**.
_Avoid_: brief, request, prompt (reserved for the generation-API input string), spec.

**Refinement**:
A loop iteration in which a **Candidate Design** is regenerated with adjusted **Intent** after a **Critic** verdict or human-client feedback.
_Avoid_: retry, iteration, regeneration (the underlying mechanic, not the loop), revise.

**Routing**:
The conditional-edge logic in the LangGraph state machine that selects between **Refinement**, user-facing surfacing, and **Two-Stage Matcher** invocation based on **Critic** verdicts.
_Avoid_: branching, control flow, edge logic, dispatch.

### Generation / Critique

**Generation Client**:
The runtime wrapper around the production image-generation provider (Gemini Nano Banana 2 / Pro) that returns **Candidate Designs** as structured envelopes.
_Avoid_: model, image generator, Nano Banana wrapper (implementation leak), generator.

**Candidate Design**:
A single generated tattoo image returned by the **Generation Client**, carrying its prompt, source-model identifier, URI, and per-image metadata.
_Avoid_: output, image, generation, design (alone, ambiguous with the post-selection design).

**Critic**:
A specialized module that evaluates one dimension of a **Candidate Design** and returns a typed Pydantic verdict.
_Avoid_: judge, evaluator, validator (implies binary), checker.

**Anatomy Critic**:
The **Critic** that evaluates body-part placement plausibility for a **Candidate Design**, returning `AnatomyCheck`.
_Avoid_: placement check, anatomy validator.

**Plagiarism Critic**:
The **Critic** that scores a **Candidate Design**'s **multimodal embedding** against the **Artist Portfolio Index** and **Famous Tattoos Corpus**, returning `PlagiarismCheck`.
_Avoid_: originality check, copy check, dup check.

**Style Critic**:
The **Critic** that measures alignment between **Intent** and a **Candidate Design** via **multimodal embedding** cosine similarity, returning `StyleCoherence`.
_Avoid_: coherence judge, alignment scorer.

**Quality Critic**:
The **Critic** that scores a **Candidate Design** on composition, linework, balance, and originality via VLM rubric, returning `QualityScore`.
_Avoid_: aesthetic judge, polish scorer.

### Retrieval / Matching

**Knowledge Corpus**:
The ~150-chunk text dataset spanning style taxonomy, anatomy, aftercare, IP best-practices, and cultural considerations, indexed for **Consultation** grounding.
_Avoid_: knowledge base, RAG corpus, docs, content store.

**Knowledge Retriever**:
The module that retrieves relevant **Knowledge Corpus** chunks for a **Consultation** query via the **multimodal embedding** against the **Vector Store**.
_Avoid_: RAG, retriever (alone), document fetcher.

**Artist Portfolio Index**:
The **Vector Store** collection of indexed artist images (10–20 curated real artists + 20–30 synthetic style-coverage records) used by the **Two-Stage Matcher** and as the primary **Plagiarism Critic** reference.
_Avoid_: artist database, portfolio store, image index, gallery.

**Famous Tattoos Corpus**:
The ~50-record curated collection of iconic and celebrity tattoos serving as the secondary **Plagiarism Critic** reference.
_Avoid_: iconic set, reference set, celebrity set.

**Vector Store**:
The single Qdrant instance (named-vector schema, int8 quantization, alias-swap rebuild) hosting the **Artist Portfolio Index**, **Famous Tattoos Corpus**, and **Knowledge Corpus** as separate collections.
_Avoid_: vector DB, vector index, embedding store, Qdrant (implementation leak).

**Two-Stage Matcher**:
The post-selection retrieval pipeline that recalls candidate artists via the **multimodal embedding** (Stage 1) and reorders them via the **visual embedding** (Stage 2).
_Avoid_: matcher (alone), search pipeline, retrieval pipeline.

### Model Comparison

**LoRA Artifact**:
The per-artist style-tuned weight set, trained on FLUX.2-dev and FLUX.2-klein from one onboarded artist's portfolio with explicit permission, used exclusively inside the **Comparison Matrix**.
_Avoid_: fine-tune, adapter, LoRA (alone, ambiguous in the field), style weights.

**Comparison Matrix**:
The five-way (six with OpenAI Image 2) offline generation evaluation that scores **Candidate Designs** from FLUX.2-dev base, FLUX.2-dev + **LoRA Artifact**, FLUX.2-klein base, FLUX.2-klein + **LoRA Artifact**, and the **Generation Client** on a shared prompt set.
_Avoid_: model comparison, benchmark, eval matrix.

### Evaluation

**Eval Harness**:
The DeepEval-based runner that exposes Tier 1 (component), Tier 2 (agent), and Tier 3 (stretch) evaluations as pytest-discoverable functions wired into CI.
_Avoid_: test runner, eval framework, eval suite.

**Golden Set**:
A held-out, hand-labeled dataset used by the **Eval Harness** to measure a single **Critic**, retrieval layer, or generation property.
_Avoid_: test set, fixture, eval data, ground truth.

**Eval Report**:
The committed markdown summary of the latest **Eval Harness** results, kept in the repo so reviewers read metrics without rerunning evaluations.
_Avoid_: test report, results file, scorecard.

### Provenance

**Provenance**:
The per-record source-attribution metadata (origin URL, curator, capture date, synthetic flag, permission marker) carried by every entry in the **Artist Portfolio Index**, **Famous Tattoos Corpus**, and **Knowledge Corpus**.
_Avoid_: source, attribution, citation, credit.

**DATA_PROVENANCE.md**:
The repo-root document aggregating every **Provenance** record into a single auditable trail for the artifact's data posture.
_Avoid_: sources file, credits, attributions doc.

## Relationships

- A **Studio** session begins with a **Consultation** that produces exactly one **Intent** at a time (refined across turns).
- An **Intent** is consumed by the **Generation Client**, which produces N **Candidate Designs** per invocation.
- Each **Candidate Design** is evaluated by exactly four **Critics** in parallel: **Anatomy**, **Plagiarism**, **Style**, **Quality**.
- **Routing** consumes all four **Critic** verdicts and chooses **Refinement**, user-facing surfacing, or escalation.
- The **Knowledge Retriever** queries the **Vector Store** against the **Knowledge Corpus** only.
- The **Plagiarism Critic** queries the **Vector Store** against the **Artist Portfolio Index** and the **Famous Tattoos Corpus**.
- The **Two-Stage Matcher** queries the **Vector Store** against the **Artist Portfolio Index** only; Stage 1 uses the **multimodal embedding**, Stage 2 uses the **visual embedding**.
- A user-chosen **Candidate Design** is the input to the **Two-Stage Matcher**, which returns ranked artists.
- The **Comparison Matrix** evaluates the **Generation Client** alongside the **LoRA Artifact**; the **LoRA Artifact** never participates in runtime generation.
- The **Eval Harness** consumes one or more **Golden Sets** and emits one **Eval Report** per run.
- Every record in the **Artist Portfolio Index**, **Famous Tattoos Corpus**, and **Knowledge Corpus** carries **Provenance**, aggregated into **DATA_PROVENANCE.md**.

## Example dialogue

> **Dev:** "When the **Anatomy Critic** flags a **Candidate Design**, does **Routing** always trigger **Refinement**?"
> **Domain expert:** "**Refinement** only fires if the verdict crosses the calibrated threshold. Below that, **Routing** surfaces the **Candidate Design** with the warning attached. The threshold lives in the **Eval Harness** calibration, not in **Routing** itself."
>
> **Dev:** "And if the **Plagiarism Critic** flags it as too close to an **Artist Portfolio Index** record?"
> **Domain expert:** "**Routing** diversifies the prompt and triggers **Refinement** once. A second flag escalates to the human client — we never silently regenerate forever."
>
> **Dev:** "Where does the **LoRA Artifact** sit in this loop?"
> **Domain expert:** "Outside it. The **LoRA Artifact** only appears in the **Comparison Matrix** under the **Eval Harness**. Runtime generation always uses the **Generation Client**."

## Flagged ambiguities

- **"model"** was used during planning to mean both the runtime generation target (Nano Banana 2 / Pro) and the open-weights bases (FLUX.2-dev, FLUX.2-klein, optional OpenAI Image 2) scored offline. **Resolved:** ban "model" as a noun in code, issues, and docs. Use **Generation Client** for the runtime path, **Comparison Matrix entry** for offline candidates, and **LoRA Artifact** for fine-tuned weights.
- **"embedding"** was used for both the shared text+image vector (Gemini Embedding 2, 3,072-dim with Matryoshka) and the image-only rerank vector (DINOv2-ViT-B14). **Resolved:** always qualify. **multimodal embedding** = Gemini Embedding 2, used by the **Knowledge Retriever**, the **Plagiarism Critic**, the **Style Critic**, and Stage 1 of the **Two-Stage Matcher**. **visual embedding** = DINOv2, used only in Stage 2 rerank of the **Two-Stage Matcher**.
- **"eval"** vs **"test"** were used interchangeably. **Resolved:** **tests** are pytest assertions on module contracts with mocked dependencies; a failed test fails CI immediately. **evals** are metric-based measurements run by the **Eval Harness** against **Golden Sets**; a regression produces an **Eval Report** delta and trips a calibrated threshold rather than a binary fail. Both run in CI; the failure modes are distinct.
- **"artist"** was used for both an indexed record and a human professional. **Resolved:** **artist** alone refers to a record in the **Artist Portfolio Index** (real or synthetic), with **Provenance** linking back to the human where applicable. **onboarded artist** or **human artist** denotes the human. The human receiving the tattoo is a **human client** — never an "artist."
- **"design"** alone is ambiguous between a generation output and the user's chosen final image. **Resolved:** **Candidate Design** during generation, critique, and **Routing**; **chosen design** after user selection. "Tattoo design" stays in user-facing copy only.
