# Tattd Studio

A developer wants to demonstrate fitness for a senior AI/ML engineering role at Tattd (tattd.ai), an AI-powered tattoo design + artist booking marketplace. The role's job description requires production-shipped experience with diffusion models, embedding pipelines, vector databases, and retrieval-augmented generation, and explicitly excludes generic data-science backgrounds.

This repository is the unsolicited proof: a runnable Studio — a multi-turn tattoo-design agent built on Tattd's production stack (Gemini Nano Banana 2 / Pro, Gemini Embedding 2), with comprehensive evaluation under DeepEval. See [PRD.md](./PRD.md) for the full product requirement and [CONTEXT.md](./CONTEXT.md) for the project glossary.

## Run in 5 minutes

```bash
git clone https://github.com/techjays-prem/tattd-studio.git
cd tattd-studio
cp .env.example .env   # fill: GEMINI_API_KEY (required), VERTEX_PROJECT_ID, REPLICATE_API_TOKEN, HF_TOKEN
uv sync --extra ui --extra dev
python -m tattd_studio.main
# Gradio UI at http://localhost:7860
```

What you can do in this slice (#6):

- Type a tattoo description ("fineline minimalist mountain on inner forearm, ~3 inches")
- The Studio runs Consultation → Generation → Anatomy Critic → end
- Surfaced Candidate Designs render in a grid, each with the Anatomy Critic verdict overlaid
- The four-Critic Routing + Refinement loop arrives in slice #7

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
GEMINI_API_KEY=... \
uv run pytest
```

## Documents

- [PRD.md](./PRD.md) — product requirement
- [CONTEXT.md](./CONTEXT.md) — project glossary; vocabulary used in code, issues, PRs, and commits

## License

MIT — see [LICENSE](./LICENSE).
