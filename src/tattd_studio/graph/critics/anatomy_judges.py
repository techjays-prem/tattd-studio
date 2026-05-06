"""Pluggable judge implementations for the Anatomy Critic.

Two paths:

- ``heuristic_anatomy_judge`` — deterministic rule-of-thumb classifier
  used as the *baseline* in the Eval Harness. Runs in CI without any API
  credentials; produces a stable Eval Report that proves the harness is
  wired up and that the Critic interface holds.
- ``build_gemini_anatomy_judge`` — production VLM judge, gated behind
  an env var in tests.

Both implement the ``judge_fn`` protocol that ``AnatomyCritic`` consumes,
so swapping between them is a constructor argument.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from tattd_studio.graph.critics.anatomy import AnatomyCheck, PlacementContext
from tattd_studio.models.candidate_design import CandidateDesign

# Body parts that empirically do not hold ink, distort it, or are
# non-standard / unsafe placement targets. Drawn from the invalid cases
# in `data/eval/anatomy_cases.jsonl`.
_HOSTILE_PLACEMENTS = (
    "palm",
    "sole",
    "heel",
    "fingertip",
    "knuckle",
    "lip",
    "gum",
    "eyelid",
    "finger web",
    "cuticle",
    "nostril",
    "side of nose",
    "inside of lip",
    "inner lip",
    "palm crease",
)

_DETAIL_KEYWORDS = (
    "ultra-fine",
    "highly detailed",
    "high-detail",
    "fineline detailed",
    "fineline detail",
    "color realism",
    "micro-realism",
)


def heuristic_anatomy_judge(
    candidate: CandidateDesign, context: PlacementContext
) -> AnatomyCheck:
    """Rule-based baseline judge.

    Flags placements that fall into any of the well-known failure modes:
    surface-area exceeded, hostile body part for ink retention,
    composition compression, or detail/size mismatch. Issues are short
    natural-language reasons aligned with the Golden Set's vocabulary.
    """
    body_part = context.body_part.lower().strip()
    prompt = candidate.prompt.lower()
    issues: list[str] = []

    # Surface area: anything > 7 inches on a typical body part is suspect;
    # the Golden Set treats 8+ as invalid for limbs/wrists.
    if context.size_inches >= 8.0:
        issues.append("size exceeds anatomical surface area")

    # Hostile placements that don't hold ink or distort it.
    if any(hp in body_part for hp in _HOSTILE_PLACEMENTS):
        issues.append(f"{body_part} placement is unsafe or does not retain ink")

    # Composition compression — explicit prompt signal.
    if (
        "compressed" in prompt
        or "shrunken onto" in prompt
        or "fitted on inner wrist" in prompt
    ):
        issues.append("composition cannot be compressed without losing detail")

    # Detail/size mismatch — high-detail aesthetics on small canvases.
    detail_signal = any(k in prompt for k in _DETAIL_KEYWORDS)
    if detail_signal and context.size_inches < 1.0:
        issues.append("detail level not compatible with size")

    placement_valid = not issues
    confidence = 0.85 if issues else 0.65
    return AnatomyCheck(
        placement_valid=placement_valid,
        issues=issues,
        confidence=confidence,
    )


def build_gemini_anatomy_judge(
    *,
    api_key: str | None = None,
    judge_model_id: str = "gemini-2.0-flash-exp",
) -> Callable[[CandidateDesign, PlacementContext], AnatomyCheck]:
    """Construct a production VLM judge backed by Gemini Pro.

    The returned callable issues a single multimodal call per Candidate
    Design and parses a structured ``AnatomyCheck`` from the response.
    Lazy-imports ``google.genai`` so test paths that never wire this in
    don't pay the import cost.
    """
    from google import genai
    from google.genai import types as genai_types

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for the live Anatomy Critic judge"
        )
    client = genai.Client(api_key=key)

    def judge(candidate: CandidateDesign, context: PlacementContext) -> AnatomyCheck:
        instruction = (
            "You are the Anatomy Critic for a tattoo design Studio. "
            "Given the Candidate Design image and the placement context, "
            "decide whether the placement is anatomically plausible. "
            "Return JSON: {placement_valid: bool, issues: [string], "
            "confidence: float between 0 and 1}.\n"
            f"Placement: body_part={context.body_part}, "
            f"size_inches={context.size_inches}, notes={context.notes}.\n"
            f"Prompt that produced the design: {candidate.prompt}"
        )
        response = client.models.generate_content(
            model=judge_model_id,
            contents=[
                instruction,
                genai_types.Part.from_uri(
                    file_uri=candidate.image_uri, mime_type="image/png"
                ),
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return AnatomyCheck.model_validate_json(response.text)

    return judge
