"""Live integration test for GenerationClient against Nano Banana 2 / Pro.

Gated by ``RUN_LIVE_GENERATION_TESTS=1`` *and* a populated
``GEMINI_API_KEY`` so default CI runs never make a paid API call. To run:

    RUN_LIVE_GENERATION_TESTS=1 GEMINI_API_KEY=… uv run pytest \
        tests/generation/test_generation_client_live.py
"""

from __future__ import annotations

import os

import pytest

from tattd_studio.generation import GenerationClient, build_gemini_generate_fn
from tattd_studio.models import Intent

RUN_LIVE = os.environ.get("RUN_LIVE_GENERATION_TESTS") == "1"
HAS_KEY = bool(os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(
    not (RUN_LIVE and HAS_KEY),
    reason="set RUN_LIVE_GENERATION_TESTS=1 and GEMINI_API_KEY to enable",
)
def test_live_generation_returns_non_empty_image_uri() -> None:
    source_model_id = os.environ.get(
        "TATTD_GENERATION_SOURCE_MODEL_ID", "gemini-nano-banana-2"
    )
    client = GenerationClient(
        source_model_id=source_model_id,
        generate_fn=build_gemini_generate_fn(source_model_id),
    )
    candidates = client.generate(
        Intent(
            refined_description=(
                "fineline minimalist mountain on inner forearm, ~3 inches"
            )
        ),
        n=1,
    )
    assert len(candidates) == 1
    assert candidates[0].image_uri.strip() != ""
