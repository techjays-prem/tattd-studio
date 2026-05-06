"""Critics — typed Pydantic verdicts on Candidate Designs."""

from tattd_studio.graph.critics.anatomy import (
    AnatomyCheck,
    AnatomyCritic,
    PlacementContext,
)
from tattd_studio.graph.critics.anatomy_judges import (
    build_gemini_anatomy_judge,
    heuristic_anatomy_judge,
)
from tattd_studio.graph.critics.plagiarism import PlagiarismCheck, PlagiarismCritic
from tattd_studio.graph.critics.quality import QualityCritic, QualityScore
from tattd_studio.graph.critics.style import StyleCoherence, StyleCritic

__all__ = [
    "AnatomyCheck",
    "AnatomyCritic",
    "PlacementContext",
    "PlagiarismCheck",
    "PlagiarismCritic",
    "QualityCritic",
    "QualityScore",
    "StyleCoherence",
    "StyleCritic",
    "build_gemini_anatomy_judge",
    "heuristic_anatomy_judge",
]
