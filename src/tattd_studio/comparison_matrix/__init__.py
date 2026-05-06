"""Comparison Matrix.

The five-way (six with optional OpenAI Image 2) offline generation
evaluation that scores Candidate Designs from each of:

- FLUX.2-dev base
- FLUX.2-dev + LoRA Artifact
- FLUX.2-klein base
- FLUX.2-klein + LoRA Artifact
- Generation Client (Nano Banana 2 / Pro)
- (optional) OpenAI Image 2

on a shared prompt set. Per CONTEXT.md, the LoRA Artifact only appears
in the Comparison Matrix — never in runtime generation.
"""

from tattd_studio.comparison_matrix.matrix import (
    ComparisonEntry,
    ComparisonMatrix,
    ComparisonRow,
    load_comparison_prompts,
    run_matrix,
)
from tattd_studio.comparison_matrix.metrics import (
    ComparisonMetrics,
    score_entry,
)

__all__ = [
    "ComparisonEntry",
    "ComparisonMatrix",
    "ComparisonMetrics",
    "ComparisonRow",
    "load_comparison_prompts",
    "run_matrix",
    "score_entry",
]
