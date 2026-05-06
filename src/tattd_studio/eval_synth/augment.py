"""Deterministic permutation augmenter + live DeepEval Synthesizer wrapper."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

# Body parts and size modifiers used by the deterministic Anatomy
# augmenter. Each row's `valid` flag mirrors the Anatomy Golden Set's
# expected_placement_valid for that body part.
_BODY_PARTS_VALID = (
    "inner forearm",
    "outer forearm",
    "outer upper arm",
    "shoulder blade",
    "outer ankle",
    "outer calf",
    "outer thigh",
    "outer ribs",
    "outer hip",
    "inner bicep",
    "outer collarbone",
    "outer foot",
    "spine",
)
_BODY_PARTS_INVALID = (
    "knuckle",
    "palm",
    "sole of foot",
    "fingertip",
    "inside of lip",
    "eyelid",
    "heel pad",
    "cuticle",
    "finger web",
    "gum line",
    "palm crease",
)
_VALID_PROMPTS = (
    "fineline minimalist mountain",
    "small geometric triangle",
    "fineline botanical sprig",
    "single-line drawing of a cat",
    "small dotwork heart",
    "blackwork mandala",
    "neo-traditional rose",
    "Japanese koi outline",
)
_INVALID_PROMPTS = (
    "huge realism portrait",
    "fineline detailed portrait",
    "ultra-fine portrait",
    "12-inch full-color realism",
    "highly detailed micro-realism",
)


def expand_anatomy_cases(
    seed: list[dict],
    n: int,
    *,
    seed_random: int = 42,
) -> list[dict]:
    """Permutation-augment Anatomy cases.

    ``seed`` holds the original cases (each with prompt / body_part /
    size_inches / expected_placement_valid / expected_issues). Returns
    ``n`` new cases by sampling combinations of body part + prompt + size
    consistent with the expected validity. Deterministic given
    ``seed_random``.
    """
    rng = random.Random(seed_random)
    expanded: list[dict] = []
    for i in range(n):
        valid = (i % 2 == 0)
        body_part = rng.choice(_BODY_PARTS_VALID if valid else _BODY_PARTS_INVALID)
        prompt = rng.choice(_VALID_PROMPTS if valid else _INVALID_PROMPTS)
        size = round(rng.uniform(1.0, 5.5) if valid else rng.uniform(0.5, 11.0), 1)
        expanded.append(
            {
                "id": f"syn{i + 1:03d}",
                "prompt": f"{prompt} on {body_part}, ~{size} inches",
                "body_part": body_part,
                "size_inches": size,
                "expected_placement_valid": valid,
                "expected_issues": [] if valid else [
                    "synthetic permutation case — expected invalid placement"
                ],
                "_synthetic_source": "permutation-augmenter-v1",
            }
        )
    return expanded


def synthesize_with_judge(
    seed: list[dict],
    n: int,
    *,
    contexts: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Live: use DeepEval's Synthesizer to generate semantically novel cases.

    Lazy-imports DeepEval. Each case is constructed from the seeds plus
    the provided ``contexts`` (e.g., chunks from the Knowledge Corpus
    that constrain the generation toward in-domain language).
    """
    from deepeval.synthesizer import Synthesizer

    synthesizer = Synthesizer()
    contexts_list = list(contexts) if contexts else [c["prompt"] for c in seed]
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts_list,
        max_goldens_per_context=max(1, n // max(len(contexts_list), 1)),
    )
    out: list[dict[str, Any]] = []
    for g in goldens[:n]:
        out.append(
            {
                "id": getattr(g, "id", f"synllm-{len(out) + 1}"),
                "prompt": getattr(g, "input", ""),
                "expected_output": getattr(g, "expected_output", ""),
                "_synthetic_source": "deepeval-synthesizer",
            }
        )
    return out
