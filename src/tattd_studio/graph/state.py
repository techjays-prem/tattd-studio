"""StudioState — the typed state envelope LangGraph passes between nodes.

Per IMPLEMENTATION_PLAN.md → "Multimodal state in LangGraph", state never
carries raw image bytes; Candidate Designs are referenced by URI. The
state is intentionally additive across nodes (Consultation appends
knowledge_chunks, Generation appends candidate_designs, etc.) so that any
node can be re-run without losing prior context.
"""

from __future__ import annotations

from typing import Any, TypedDict

from tattd_studio.graph.critics import AnatomyCheck, PlacementContext
from tattd_studio.knowledge import RetrievedChunk
from tattd_studio.models import CandidateDesign, Intent


class StudioState(TypedDict):
    """End-to-end Studio session state."""

    intent: Intent
    placement_context: PlacementContext
    n_candidates: int

    knowledge_chunks: list[RetrievedChunk]
    candidate_designs: list[CandidateDesign]
    anatomy_checks: list[AnatomyCheck]

    metadata: dict[str, Any]
