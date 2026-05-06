"""Consultation — multi-turn dialogue stage.

Per CONTEXT.md, the Consultation is the multi-turn dialogue in which an
Intent is elicited from the human client and grounded against the
Knowledge Corpus. The slice-#6 graph runs a single-turn Consultation
node (one query → one retrieval → one Intent); this module implements
the multi-turn refinement loop the plan's Eval Surface table names.

A multi-turn session unfolds as a sequence of ``ConversationTurn``
records. Each turn:

1. Captures the human client's message and the Studio's response.
2. Grounds the message against the Knowledge Corpus via the Knowledge
   Retriever and records the chunks that informed the response.
3. Updates the running Intent with the new constraints the message
   surfaced.

The full session ends when the human client commits to an Intent that
the Generation Client can run.
"""

from tattd_studio.consultation.session import (
    ConsultationSession,
    ConversationTurn,
    refine_intent,
)

__all__ = [
    "ConsultationSession",
    "ConversationTurn",
    "refine_intent",
]
