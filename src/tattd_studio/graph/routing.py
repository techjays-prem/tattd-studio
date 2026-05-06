"""Routing — conditional-edge logic over the four Critic verdicts.

The Studio's outer behavior:

- All four Critics pass → surface to the human client.
- Any Critic fails on the first attempt → trigger Refinement with a
  prompt adjustment derived from the failing dimension.
- Any Critic fails on the second attempt → escalate (surface to the
  human client with the failure annotated; do not silently regenerate
  forever).
- Plagiarism failures get a stronger prompt-diversification adjustment
  before Refinement, per CONTEXT.md → "Routing diversifies and triggers
  Refinement once. A second flag escalates."

Thresholds live in ``evals/calibrated_thresholds.toml`` and load lazily.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from tattd_studio.graph.critics import (
    AnatomyCheck,
    PlagiarismCheck,
    QualityScore,
    StyleCoherence,
)


@dataclass(frozen=True)
class CriticVerdicts:
    """Bundle of all four Critic outputs for one Candidate Design."""

    anatomy: AnatomyCheck
    plagiarism: PlagiarismCheck
    style: StyleCoherence
    quality: QualityScore


@dataclass(frozen=True)
class RoutingDecision:
    """The Routing layer's decision for one Candidate Design."""

    action: str  # "surface" | "refine" | "escalate"
    failed_dimensions: list[str]
    refinement_hints: list[str]
    refinement_attempts_so_far: int

    @property
    def should_surface(self) -> bool:
        return self.action in {"surface", "escalate"}


@dataclass(frozen=True)
class RoutingThresholds:
    plagiarism_max_similarity: float
    style_alignment_min: float
    quality_aggregate_min: float
    anatomy_confidence_floor: float

    @classmethod
    def from_toml(cls, path: Path) -> RoutingThresholds:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls(
            plagiarism_max_similarity=float(
                data["plagiarism_critic"].get("max_similarity", 0.85)
            ),
            style_alignment_min=float(
                data["style_critic"].get("alignment_min", 0.55)
            ),
            quality_aggregate_min=float(
                data["quality_critic"].get("aggregate_min", 0.55)
            ),
            anatomy_confidence_floor=float(
                data["anatomy_critic"].get("confidence_floor", 0.5)
            ),
        )


def evaluate(
    verdicts: CriticVerdicts,
    *,
    thresholds: RoutingThresholds,
    refinement_attempts_so_far: int,
) -> RoutingDecision:
    """Decide the next action for one Candidate Design."""
    failed: list[str] = []
    hints: list[str] = []

    if not verdicts.anatomy.placement_valid:
        failed.append("anatomy")
        hints.extend(verdicts.anatomy.issues or ["adjust placement to a stable surface"])

    if verdicts.plagiarism.flagged:
        failed.append("plagiarism")
        hints.append(
            "diversify motif: avoid the top match "
            f"'{verdicts.plagiarism.top_match_artist or 'unknown'}' "
            f"(similarity {verdicts.plagiarism.top_match_similarity:.2f})"
        )

    if verdicts.style.intent_image_alignment < thresholds.style_alignment_min:
        failed.append("style")
        hints.append("re-anchor to the requested style vocabulary")

    if verdicts.quality.aggregate < thresholds.quality_aggregate_min:
        failed.append("quality")
        hints.append("strengthen composition, linework, and balance")

    if not failed:
        action = "surface"
    elif refinement_attempts_so_far < 1:
        action = "refine"
    else:
        action = "escalate"

    return RoutingDecision(
        action=action,
        failed_dimensions=failed,
        refinement_hints=hints,
        refinement_attempts_so_far=refinement_attempts_so_far,
    )
