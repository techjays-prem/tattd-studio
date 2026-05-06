"""LoRA Artifact registry — reads `data/lora_training/artifacts.toml`."""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRAArtifact:
    """One trained LoRA Artifact entry."""

    artist_slug: str
    base_model: str  # "FLUX.2-dev" | "FLUX.2-klein"
    replicate_url: str
    trigger_word: str
    trained_on: dt.date
    permission_record: str

    @property
    def comparison_matrix_column(self) -> str:
        """The Comparison Matrix column label for this artifact."""
        return f"{self.base_model} + LoRA Artifact ({self.artist_slug})"


def load_artifacts(toml_path: Path) -> list[LoRAArtifact]:
    """Parse `artifacts.toml`. Returns an empty list when no entries exist."""
    if not toml_path.exists():
        return []
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    raw = data.get("artifacts") or []
    out: list[LoRAArtifact] = []
    for row in raw:
        out.append(
            LoRAArtifact(
                artist_slug=row["artist_slug"],
                base_model=row["base_model"],
                replicate_url=row["replicate_url"],
                trigger_word=row["trigger_word"],
                trained_on=dt.date.fromisoformat(row["trained_on"]),
                permission_record=row["permission_record"],
            )
        )
    return out
