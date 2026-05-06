"""Vector Store module.

Wraps Qdrant behind a named-vector-aware interface. Hides quantization
configuration, schema declaration, and the alias-swap rebuild pattern from
callers. Consumed by the Knowledge Retriever, the Plagiarism Critic, and
the Two-Stage Matcher.

Vocabulary discipline (CONTEXT.md): the public surface only references
*multimodal embedding* and *visual embedding* — never the bare term.
"""

from tattd_studio.vectordb.qdrant_client import (
    MULTIMODAL_EMBEDDING_DIM_1024,
    MULTIMODAL_EMBEDDING_DIM_3072,
    VISUAL_EMBEDDING_DIM,
    CollectionSchema,
    QueryHit,
    VectorStore,
)

__all__ = [
    "MULTIMODAL_EMBEDDING_DIM_1024",
    "MULTIMODAL_EMBEDDING_DIM_3072",
    "VISUAL_EMBEDDING_DIM",
    "CollectionSchema",
    "QueryHit",
    "VectorStore",
]
