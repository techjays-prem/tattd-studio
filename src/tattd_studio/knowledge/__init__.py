"""Knowledge Corpus — module exports."""

from tattd_studio.knowledge.chunks import Chunk, parse_chunks_from_markdown
from tattd_studio.knowledge.embedding import (
    DeterministicTextEmbeddingClient,
    TextEmbeddingClient,
    build_gemini_text_embedding_client,
)
from tattd_studio.knowledge.ingest import KNOWLEDGE_CORPUS_ALIAS, ingest_corpus
from tattd_studio.knowledge.retriever import (
    KnowledgeRetriever,
    RetrievedChunk,
)

__all__ = [
    "KNOWLEDGE_CORPUS_ALIAS",
    "Chunk",
    "DeterministicTextEmbeddingClient",
    "KnowledgeRetriever",
    "RetrievedChunk",
    "TextEmbeddingClient",
    "build_gemini_text_embedding_client",
    "ingest_corpus",
    "parse_chunks_from_markdown",
]
