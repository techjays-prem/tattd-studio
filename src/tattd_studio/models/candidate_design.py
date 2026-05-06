"""Candidate Design envelope.

A single image returned by the Generation Client, carrying the prompt that
produced it, the identifier of the runtime generation provider that ran,
arbitrary metadata (e.g. prompt template version), and a creation
timestamp. This is the unit that Critics evaluate and that the Two-Stage
Matcher consumes after user selection.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class CandidateDesign(BaseModel):
    """One generated tattoo image plus its provenance metadata."""

    image_uri: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    source_model_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: dt.datetime

    model_config = {"frozen": True}
