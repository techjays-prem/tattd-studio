"""Schema tests for CandidateDesign."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from tattd_studio.models.candidate_design import CandidateDesign


def test_candidate_design_round_trips_required_fields() -> None:
    cd = CandidateDesign(
        image_uri="file:///tmp/cd-1.png",
        prompt="fineline minimalist mountain on inner forearm, ~3 inches",
        source_model_id="gemini-nano-banana-2",
        metadata={"prompt_template_version": "v1.0"},
        created_at=dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )
    assert cd.source_model_id == "gemini-nano-banana-2"
    assert cd.metadata["prompt_template_version"] == "v1.0"


def test_candidate_design_rejects_empty_image_uri() -> None:
    with pytest.raises(ValidationError):
        CandidateDesign(
            image_uri="",
            prompt="x",
            source_model_id="gemini-nano-banana-2",
            metadata={},
            created_at=dt.datetime.now(dt.UTC),
        )
