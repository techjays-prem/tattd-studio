"""Embedding clients.

The Studio uses two qualified embedding paths per CONTEXT.md:

- *multimodal embedding* — Gemini Embedding 2 in text or image mode;
  used by the Knowledge Retriever, the Plagiarism Critic, the Style
  Critic, and Stage 1 of the Two-Stage Matcher
- *visual embedding* — DINOv2-ViT-B14; used only in Stage 2 rerank of
  the Two-Stage Matcher

Each path has a deterministic stub for CI and a gated factory for the
real provider.
"""

from tattd_studio.embeddings.dinov2 import (
    DeterministicVisualEmbeddingClient,
    VisualEmbeddingClient,
    build_dinov2_visual_embedding_client,
)

__all__ = [
    "DeterministicVisualEmbeddingClient",
    "VisualEmbeddingClient",
    "build_dinov2_visual_embedding_client",
]
