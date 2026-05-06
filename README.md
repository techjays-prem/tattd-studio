# Tattd Concierge

A developer wants to demonstrate fitness for a senior AI/ML engineering role at Tattd (tattd.ai), an AI-powered tattoo design + artist booking marketplace. The role's job description requires production-shipped experience with diffusion models, embedding pipelines, vector databases, and retrieval-augmented generation, and explicitly excludes generic data-science backgrounds.

This repository is the unsolicited proof: a runnable Concierge — a multi-turn tattoo-design agent built on Tattd's production stack (Gemini Nano Banana 2 / Pro, Gemini Embedding 2), with comprehensive evaluation under DeepEval. See [PRD.md](./PRD.md) for the full product requirement and [CONTEXT.md](./CONTEXT.md) for the project glossary.

## Run in 5 minutes

> _Placeholder — populated as the repo + CI skeleton lands (issue #1)._
>
> Planned shape:
>
> ```bash
> git clone https://github.com/<user>/tattd-concierge.git
> cd tattd-concierge
> cp .env.example .env   # fill: GEMINI_API_KEY, VERTEX_PROJECT_ID, REPLICATE_API_TOKEN, HF_TOKEN
> uv sync
> python -m tattd_concierge.main
> # Gradio UI at http://localhost:7860
> ```

## Documents

- [PRD.md](./PRD.md) — product requirement
- [CONTEXT.md](./CONTEXT.md) — project glossary; vocabulary used in code, issues, PRs, and commits

## License

MIT — see [LICENSE](./LICENSE).
