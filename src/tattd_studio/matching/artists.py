"""Artist record schema + loader."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ArtistRecord(BaseModel):
    """One Artist Portfolio Index record + Provenance."""

    artist_slug: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    primary_style: str = Field(min_length=1)
    secondary_styles: list[str] = Field(default_factory=list)
    bio: str = Field(min_length=1)
    portfolio_url: str = Field(min_length=1)
    permission_marker: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    curator: str = Field(min_length=1)
    capture_date: dt.date
    synthetic: bool

    model_config = {"frozen": True}

    def style_text(self) -> str:
        """Concatenated text used as the embedding source."""
        return (
            f"{self.display_name}: {self.primary_style}. "
            f"{', '.join(self.secondary_styles)}. {self.bio}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artist_slug": self.artist_slug,
            "artist": self.display_name,  # Plagiarism Critic reads this key
            "display_name": self.display_name,
            "primary_style": self.primary_style,
            "secondary_styles": ",".join(self.secondary_styles),
            "bio": self.bio,
            "portfolio_url": self.portfolio_url,
            "source_url": self.source_url,
            "permission_marker": self.permission_marker,
            "curator": self.curator,
            "capture_date": self.capture_date.isoformat(),
            "synthetic": self.synthetic,
        }


def load_artist_records(jsonl_path: Path) -> list[ArtistRecord]:
    return [
        ArtistRecord(**json.loads(line))
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
