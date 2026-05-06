"""Quality Critic.

Scores a Candidate Design on composition, linework, balance, and
originality via a VLM rubric. Returns ``QualityScore``. Wraps a
``judge_fn`` so unit tests inject deterministic verdicts; production
wires the Gemini Pro VLM judge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from tattd_studio.models.candidate_design import CandidateDesign


class QualityScore(BaseModel):
    """Verdict from the Quality Critic."""

    composition: float = Field(ge=0.0, le=1.0)
    linework: float = Field(ge=0.0, le=1.0)
    balance: float = Field(ge=0.0, le=1.0)
    originality: float = Field(ge=0.0, le=1.0)
    notes: str = ""

    model_config = {"frozen": True}

    @property
    def aggregate(self) -> float:
        return (self.composition + self.linework + self.balance + self.originality) / 4.0


class _QualityJudge(Protocol):
    def __call__(self, candidate: CandidateDesign) -> QualityScore: ...


class QualityCritic:
    def __init__(self, *, judge_fn: _QualityJudge | Callable[..., QualityScore]) -> None:
        self._judge_fn = judge_fn

    def check(self, candidate: CandidateDesign) -> QualityScore:
        return self._judge_fn(candidate)
