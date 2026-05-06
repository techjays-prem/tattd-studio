"""Multimodal embedding client (text mode) for Knowledge Corpus ingest.

Knowledge Retriever queries and Knowledge Corpus chunks both go through
the *multimodal embedding* (Gemini Embedding 2 in text mode). We expose a
small protocol so tests can substitute a deterministic stub without an
API key, and production wires the real Gemini client.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Callable
from typing import Protocol


class TextEmbeddingClient(Protocol):
    """Embeds text into the multimodal embedding space (text mode)."""

    def embed(self, text: str) -> list[float]: ...


class DeterministicTextEmbeddingClient:
    """Hash-based stub used in CI tests.

    Produces a fixed-dimension unit vector seeded by SHA256 of the input.
    Cosine similarity of two embeddings is high iff the texts share many
    leading bytes after hashing — i.e. embeddings collide on the *exact*
    text, are otherwise effectively orthogonal. Good enough to verify
    retrieval wiring without an API call.
    """

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch the 32-byte digest into ``dim`` floats by repeating with
        # a counter prefix; gives a deterministic but reasonably spread
        # vector for any text input.
        out: list[float] = []
        counter = 0
        while len(out) < self._dim:
            block = hashlib.sha256(
                struct.pack(">I", counter) + digest
            ).digest()
            for i in range(0, len(block), 4):
                if len(out) >= self._dim:
                    break
                value = struct.unpack(">I", block[i : i + 4])[0]
                # Map uint32 to [-1, 1].
                out.append((value / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        # L2-normalize so cosine similarity is well-behaved.
        norm = sum(x * x for x in out) ** 0.5
        if norm > 0:
            out = [x / norm for x in out]
        return out


def build_gemini_text_embedding_client(
    *,
    api_key: str | None = None,
    embedding_model_id: str = "gemini-embedding-001",
    output_dim: int = 1024,
) -> Callable[[str], list[float]]:
    """Build a callable wrapping Gemini Embedding 2 (text mode).

    Returned object is duck-typed to TextEmbeddingClient via attribute
    access; it exposes ``embed(text)``.
    """
    from google import genai
    from google.genai import types as genai_types

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for the live text embedding client"
        )
    client = genai.Client(api_key=key)

    class _GeminiClient:
        def embed(self, text: str) -> list[float]:
            response = client.models.embed_content(
                model=embedding_model_id,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    output_dimensionality=output_dim,
                ),
            )
            return list(response.embeddings[0].values)

    return _GeminiClient()  # type: ignore[return-value]
