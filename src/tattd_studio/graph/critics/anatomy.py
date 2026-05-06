"""Anatomy Critic.

Evaluates body-part placement plausibility for a Candidate Design and
returns an ``AnatomyCheck`` Pydantic verdict. Implements the structured-
output pattern that every subsequent Critic follows.

The VLM judge is injected via ``judge_fn`` so unit tests are deterministic
and CI never makes live API calls. The default factory
``build_gemini_anatomy_judge`` wraps Gemini Pro for production.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from tattd_studio.models.candidate_design import CandidateDesign


class PlacementContext(BaseModel):
    """The body placement that the human client has indicated for the design."""

    body_part: str = Field(default="")
    size_inches: float = Field(default=0.0, ge=0.0)
    notes: str = Field(default="")

    model_config = {"frozen": True}


class AnatomyCheck(BaseModel):
    """Verdict from the Anatomy Critic for one Candidate Design."""

    placement_valid: bool
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"frozen": True}


class _JudgeFn(Protocol):
    def __call__(
        self, candidate: CandidateDesign, context: PlacementContext
    ) -> AnatomyCheck: ...


class AnatomyCritic:
    """Wraps a VLM judge to produce an ``AnatomyCheck`` per Candidate Design.

    The wrapper is intentionally thin: it validates input, delegates to
    ``judge_fn``, and returns the verdict unchanged. All anatomy reasoning
    lives behind ``judge_fn``, which can be a deterministic fixture in
    tests or a Gemini Pro VLM call in production.
    """

    def __init__(self, *, judge_fn: _JudgeFn | Callable[..., AnatomyCheck]) -> None:
        self._judge_fn = judge_fn

    def check(
        self, candidate: CandidateDesign, context: PlacementContext
    ) -> AnatomyCheck:
        if not context.body_part.strip():
            raise ValueError(
                "PlacementContext.body_part must be supplied; the Anatomy "
                "Critic has no placement to reason about without a limb."
            )
        return self._judge_fn(candidate, context)
