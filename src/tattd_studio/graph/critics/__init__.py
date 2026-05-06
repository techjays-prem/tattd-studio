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

__all__ = [
    "AnatomyCheck",
    "AnatomyCritic",
    "PlacementContext",
    "build_gemini_anatomy_judge",
    "heuristic_anatomy_judge",
]
