"""Chunk schema and Knowledge Corpus markdown parser.

Each Knowledge Corpus markdown file groups multiple chunks separated by
the unambiguous marker ``<!-- CHUNK -->`` (the marker cannot collide with
YAML frontmatter delimiters or with markdown horizontal rules). Each chunk
carries its own Provenance frontmatter and a free-form body.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import yaml
from pydantic import BaseModel, Field

CHUNK_MARKER = "<!-- CHUNK -->"
_FRONTMATTER_RE = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

VALID_AREAS = frozenset(
    {"taxonomy", "placement", "aftercare", "ip", "cultural", "famous"}
)


class Chunk(BaseModel):
    """A single Knowledge Corpus or Famous Tattoos Corpus record + Provenance."""

    chunk_id: str = Field(min_length=1)
    area: str
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    curator: str = Field(min_length=1)
    capture_date: dt.date
    synthetic: bool
    permission: str = Field(min_length=1)
    artist: str | None = None  # populated for `area: famous` records

    model_config = {"frozen": True}

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a Vector Store payload dict."""
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "area": self.area,
            "title": self.title,
            "body": self.body,
            "source_url": self.source_url,
            "curator": self.curator,
            "capture_date": self.capture_date.isoformat(),
            "synthetic": self.synthetic,
            "permission": self.permission,
        }
        if self.artist is not None:
            payload["artist"] = self.artist
        return payload


def parse_chunks_from_markdown(text: str) -> list[Chunk]:
    """Parse a multi-chunk markdown blob into typed Chunks."""
    pieces = [
        piece.strip()
        for piece in text.split(CHUNK_MARKER)
        if piece.strip()
    ]
    chunks: list[Chunk] = []
    for piece in pieces:
        match = _FRONTMATTER_RE.match(piece)
        if not match:
            raise ValueError(
                "chunk missing YAML frontmatter:\n" + piece[:200]
            )
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
        chunks.append(Chunk(body=body, **frontmatter))
    return chunks
