"""Behavioral tests for GenerationClient.

The client wraps Gemini Nano Banana 2 / Pro at runtime. All tests inject
a fake `generate_fn` so no live API calls happen in CI.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tattd_studio.generation import (
    PROMPT_TEMPLATE_VERSION,
    GenerationClient,
    TransientGenerationError,
)
from tattd_studio.models import Intent


def _frozen_clock() -> dt.datetime:
    return dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC)


def _intent(text: str = "fineline minimalist mountain on inner forearm, ~3 inches") -> Intent:
    return Intent(refined_description=text)


def test_generate_returns_n_candidates_with_versioned_prompt() -> None:
    calls: list[tuple[str, int]] = []

    def fake_generate(prompt: str, n: int) -> list[str]:
        calls.append((prompt, n))
        return [f"file:///tmp/cd-{i}.png" for i in range(n)]

    client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=fake_generate,
        clock=_frozen_clock,
    )

    candidates = client.generate(_intent(), n=3)

    assert len(candidates) == 3
    assert {c.image_uri for c in candidates} == {
        "file:///tmp/cd-0.png",
        "file:///tmp/cd-1.png",
        "file:///tmp/cd-2.png",
    }
    for c in candidates:
        assert c.source_model_id == "gemini-nano-banana-2"
        assert c.metadata["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
        assert c.created_at == _frozen_clock()
    assert len(calls) == 1
    assert calls[0][1] == 3


def test_generate_retries_transient_then_succeeds() -> None:
    attempts: list[int] = []

    def flaky(prompt: str, n: int) -> list[str]:
        attempts.append(1)
        if len(attempts) < 2:
            raise TransientGenerationError("upstream 503")
        return ["file:///tmp/cd-0.png"]

    client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=flaky,
        max_retries=3,
        sleep=lambda _s: None,
        clock=_frozen_clock,
    )

    candidates = client.generate(_intent(), n=1)

    assert len(candidates) == 1
    assert len(attempts) == 2  # one fail, one success


def test_generate_propagates_after_retries_exhausted() -> None:
    def always_transient(prompt: str, n: int) -> list[str]:
        raise TransientGenerationError("upstream still 503")

    client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=always_transient,
        max_retries=2,
        sleep=lambda _s: None,
        clock=_frozen_clock,
    )

    with pytest.raises(TransientGenerationError):
        client.generate(_intent(), n=1)


def test_generate_returns_cached_candidates_on_repeat_call() -> None:
    counter = {"n": 0}

    def fake_generate(prompt: str, n: int) -> list[str]:
        counter["n"] += 1
        return [f"file:///tmp/cd-{i}.png" for i in range(n)]

    client = GenerationClient(
        source_model_id="gemini-nano-banana-2",
        generate_fn=fake_generate,
        clock=_frozen_clock,
    )

    intent = _intent()
    first = client.generate(intent, n=2)
    second = client.generate(intent, n=2)

    assert counter["n"] == 1  # provider hit only once
    assert [c.image_uri for c in first] == [c.image_uri for c in second]
