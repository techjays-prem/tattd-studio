"""Intent — the structured tattoo idea that the Generation Client consumes.

Slice #5 ships a minimal envelope. Consultation in slice #6 evolves the
shape (style, placement, size, motifs, references); fields land
incrementally as the Critic surfaces require them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """Refined description of a human client's tattoo idea."""

    refined_description: str = Field(min_length=1)

    model_config = {"frozen": True}
