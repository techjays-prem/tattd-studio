"""Behavioral tests for the Anatomy Critic.

The Critic is a thin wrapper over a VLM judge function. Tests inject
deterministic judge implementations so CI never makes live API calls.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tattd_studio.graph.critics.anatomy import (
    AnatomyCheck,
    AnatomyCritic,
    PlacementContext,
)
from tattd_studio.models import CandidateDesign


def _candidate(prompt: str = "fineline mountain on inner forearm, ~3 inches") -> CandidateDesign:
    return CandidateDesign(
        image_uri="file:///tmp/cd-test.png",
        prompt=prompt,
        source_model_id="gemini-nano-banana-2",
        metadata={"prompt_template_version": "v1.0"},
        created_at=dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )


def test_anatomy_check_round_trips_required_fields() -> None:
    check = AnatomyCheck(
        placement_valid=True,
        issues=[],
        confidence=0.92,
    )
    assert check.placement_valid is True
    assert check.confidence == 0.92


def test_check_returns_judge_verdict_unchanged() -> None:
    expected = AnatomyCheck(placement_valid=True, issues=[], confidence=0.9)

    def judge(cd: CandidateDesign, ctx: PlacementContext) -> AnatomyCheck:
        return expected

    critic = AnatomyCritic(judge_fn=judge)
    out = critic.check(
        _candidate(),
        PlacementContext(body_part="inner forearm", size_inches=3.0),
    )
    assert out is expected


def test_check_flags_impossible_size() -> None:
    def judge(cd: CandidateDesign, ctx: PlacementContext) -> AnatomyCheck:
        return AnatomyCheck(
            placement_valid=False,
            issues=["size exceeds anatomical surface area"],
            confidence=0.95,
        )

    critic = AnatomyCritic(judge_fn=judge)
    out = critic.check(
        _candidate("huge realism portrait on knuckle"),
        PlacementContext(body_part="knuckle", size_inches=5.0),
    )
    assert out.placement_valid is False
    assert "size" in out.issues[0]


def test_check_flags_anatomy_incompatible_placement() -> None:
    def judge(cd: CandidateDesign, ctx: PlacementContext) -> AnatomyCheck:
        return AnatomyCheck(
            placement_valid=False,
            issues=["fineline detail will blur on highly mobile skin"],
            confidence=0.88,
        )

    critic = AnatomyCritic(judge_fn=judge)
    out = critic.check(
        _candidate("fineline portrait on palm"),
        PlacementContext(body_part="palm", size_inches=2.0),
    )
    assert out.placement_valid is False


def test_check_rejects_missing_limb_context() -> None:
    def judge(cd: CandidateDesign, ctx: PlacementContext) -> AnatomyCheck:
        raise AssertionError("judge must not be invoked without body_part")

    critic = AnatomyCritic(judge_fn=judge)
    with pytest.raises(ValueError, match="body_part"):
        critic.check(
            _candidate(),
            PlacementContext(body_part="", size_inches=3.0),
        )
