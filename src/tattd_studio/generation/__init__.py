"""Generation Client.

Runtime wrapper around the production image-generation provider (Gemini
Nano Banana 2 / Pro). Returns Candidate Designs as structured envelopes.
The Generation Client is the runtime path; Comparison Matrix entries are
offline only.
"""

from tattd_studio.generation.client import (
    PROMPT_TEMPLATE_VERSION,
    GenerationClient,
    GenerationError,
    TransientGenerationError,
    build_prompt,
)
from tattd_studio.generation.gemini import build_gemini_generate_fn

__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "GenerationClient",
    "GenerationError",
    "TransientGenerationError",
    "build_gemini_generate_fn",
    "build_prompt",
]
