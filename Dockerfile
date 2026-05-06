# Tattd Studio — runtime image.
#
# Build:  docker build -t tattd-studio .
# Run:    docker run -p 7860:7860 \
#             -e GEMINI_API_KEY=... \
#             tattd-studio
#
# The image bundles the Studio, the Knowledge Corpus, the Famous Tattoos
# Corpus, and the Artist Portfolio Index. Live providers (Gemini Embedding 2,
# Nano Banana 2 / Pro, Gemini Pro VLM) require GEMINI_API_KEY at runtime.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install uv from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra ui --no-dev

# Copy source and corpora.
COPY src/ ./src/
COPY data/ ./data/
COPY evals/ ./evals/

RUN uv sync --frozen --extra ui --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY --from=builder /app /app

EXPOSE 7860
CMD ["python", "-m", "tattd_studio.main"]
