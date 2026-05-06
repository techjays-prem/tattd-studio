"""Multi-turn ConsultationSession implementation.

A session holds:

- The running ``Intent`` (refined as the dialogue progresses)
- The full ``conversation_history`` of ``ConversationTurn`` records
- A reference to the ``KnowledgeRetriever`` for grounding each turn

Each ``advance(user_message)`` call retrieves relevant chunks, updates
the Intent, and appends a turn. The Studio response text is composed
from the retrieved chunks plus a templated synthesis pattern; tests
substitute a deterministic ``synthesize_fn`` and live mode wires a
real LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tattd_studio.knowledge import KnowledgeRetriever, RetrievedChunk
from tattd_studio.models import Intent


@dataclass(frozen=True)
class ConversationTurn:
    """One turn of the Consultation."""

    user_message: str
    studio_response: str
    grounding_chunks: list[RetrievedChunk]
    intent_after_turn: Intent


@dataclass
class ConsultationSession:
    """Stateful multi-turn consultation.

    The session can start without an Intent — set ``intent=None`` and
    the first ``advance()`` call constructs one from the user's message.
    """

    retriever: KnowledgeRetriever
    intent: Intent | None = None
    history: list[ConversationTurn] = field(default_factory=list)
    knowledge_top_k: int = 5
    synthesize_fn: Callable[[Intent, str, list[RetrievedChunk]], str] | None = None

    def advance(self, user_message: str) -> ConversationTurn:
        """Run one Consultation turn.

        Retrieves relevant Knowledge Corpus chunks for the user's
        message in the context of the running Intent, refines the Intent
        with the new constraints, and synthesizes a Studio response.
        """
        if not user_message.strip():
            raise ValueError("user_message must be non-empty")

        prior_text = self.intent.refined_description if self.intent else ""
        grounding_query = (
            f"{prior_text} | {user_message}" if prior_text else user_message
        )
        chunks = self.retriever.retrieve(grounding_query, k=self.knowledge_top_k)

        new_intent = refine_intent(self.intent, user_message, chunks)
        synthesizer = self.synthesize_fn or _default_synthesizer
        response = synthesizer(new_intent, user_message, chunks)

        turn = ConversationTurn(
            user_message=user_message,
            studio_response=response,
            grounding_chunks=chunks,
            intent_after_turn=new_intent,
        )
        self.history.append(turn)
        self.intent = new_intent
        return turn

    @property
    def conversation_text(self) -> list[str]:
        """Flattened conversation as alternating user/studio strings."""
        out: list[str] = []
        for turn in self.history:
            out.append(turn.user_message)
            out.append(turn.studio_response)
        return out

    @property
    def retrieval_context(self) -> list[str]:
        """All grounding chunk bodies seen across the session."""
        seen: dict[str, str] = {}
        for turn in self.history:
            for chunk in turn.grounding_chunks:
                seen.setdefault(chunk.chunk_id, chunk.body)
        return list(seen.values())


def refine_intent(
    intent: Intent | None,
    user_message: str,
    chunks: list[RetrievedChunk],
) -> Intent:
    """Append the user's new message into the Intent's refined_description.

    When ``intent`` is ``None`` (the start of a session) the new Intent
    is constructed from the user's message directly. Slice-#6 used a
    trivial single-turn refinement; this stays simple on purpose so the
    multi-turn machinery is the testable surface. Live mode can wire a
    real LLM-driven Intent rewriter via ``synthesize_fn``; the plan's
    stretch is to learn richer Intent structure (style, placement,
    motifs) across turns.
    """
    new = user_message.strip()
    if intent is None:
        return Intent(refined_description=new)
    base = intent.refined_description.strip()
    if not base:
        return Intent(refined_description=new)
    return Intent(refined_description=f"{base}; {new}")


def _default_synthesizer(
    intent: Intent, user_message: str, chunks: list[RetrievedChunk]
) -> str:
    """Deterministic, citation-tagged Studio response.

    Used in tests and as the CI baseline. Composes the response from the
    refined Intent and the top retrieved chunks; live mode swaps in an
    LLM-driven synthesis path.
    """
    if not chunks:
        return f"Acknowledged: {intent.refined_description}"
    citations = ", ".join(f"[{c.chunk_id}]" for c in chunks[:3])
    return (
        f"Refined intent: {intent.refined_description}. "
        f"Grounding: {citations}."
    )
